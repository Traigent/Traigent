import asyncio
from datetime import datetime

import pandas as pd

from traigent.api.types import OptimizationResult, TrialResult, TrialStatus
from traigent.config.types import TraigentConfig
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.evaluators.base import (
    BaseEvaluator,
    Dataset,
    EvaluationExample,
    EvaluationResult,
)
from traigent.optimizers.base import BaseOptimizer


def make_trial(cfg: dict, acc: float, dur: float = 1.0) -> TrialResult:
    return TrialResult(
        trial_id="t",
        config=cfg,
        metrics={"accuracy": acc},
        status=TrialStatus.COMPLETED,
        duration=dur,
        timestamp=datetime.now(),
    )


def test_to_aggregated_dataframe_basic():
    trials = [
        make_trial({"a": 1}, 0.8, 2.0),
        make_trial({"a": 1}, 0.6, 4.0),
        make_trial({"a": 2}, 0.7, 3.0),
    ]
    res = OptimizationResult(
        trials=trials,
        best_config={},
        best_score=0.0,
        optimization_id="x",
        duration=0.0,
        convergence_info={},
        status=TrialStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="dummy",
        timestamp=datetime.now(),
    )

    df = res.to_aggregated_dataframe(primary_objective="accuracy")
    assert isinstance(df, pd.DataFrame)
    # Two distinct configs
    assert len(df) == 2
    # Samples count
    row_a1 = df[df["a"] == 1].iloc[0]
    assert row_a1["samples_count"] == 2
    # Mean accuracy ~ (0.8+0.6)/2
    assert abs(row_a1["accuracy"] - 0.7) < 1e-9
    # Mean duration ~ (2+4)/2
    assert abs(row_a1["duration"] - 3.0) < 1e-9


class DummyOptimizer(BaseOptimizer):
    def __init__(self, sequence, *args, **kwargs):
        # Define config space so grouping ignores metric-only keys (like 'acc')
        super().__init__(
            config_space={"p": [1, 2, 3]}, objectives=["accuracy"], context=None
        )
        self.sequence = sequence
        self.idx = 0

    def suggest_next_trial(self, history):
        if self.idx >= len(self.sequence):
            return self.sequence[-1]
        cfg = self.sequence[self.idx]
        self.idx += 1
        self._trial_count += 1
        return cfg

    def should_stop(self, history):
        return False


class DummyEvaluator(BaseEvaluator):
    async def evaluate(self, func, config, dataset):
        acc = float(config.get("acc", 0.0))
        return EvaluationResult(
            config=config,
            example_results=[],
            aggregated_metrics={"accuracy": acc},
            total_examples=1,
            successful_examples=1,
            duration=0.1,
        )


class DummyBackend:
    def __init__(self):
        self.submissions = []

    def create_session(self, **kwargs):
        return "session_test"

    def submit_result(self, session_id, config, score, metadata=None):
        self.submissions.append(
            {
                "session_id": session_id,
                "config": config,
                "score": score,
                "metadata": metadata or {},
            }
        )

    def finalize_session_sync(self, session_id, *_args, **_kwargs):
        return {"session_id": session_id, "finalized": True}


async def _run_orchestrator(sequence, exec_mode="standard"):
    opt = DummyOptimizer(sequence)
    ev = DummyEvaluator(metrics=["accuracy"], timeout=1.0)
    oc = OptimizationOrchestrator(
        optimizer=opt,
        evaluator=ev,
        max_trials=len(sequence),
        timeout=5.0,
        config=TraigentConfig(execution_mode=exec_mode),
        parallel_trials=1,
        objectives=["accuracy"],
    )
    oc.backend_client = DummyBackend()

    async def dummy_func(**_kwargs):
        return "ok"

    ds = Dataset([EvaluationExample(input_data={"x": 1}, expected_output=None)])
    result = await oc.optimize(dummy_func, ds, function_name="f")
    return result, oc.backend_client


def test_orchestrator_end_aggregation_and_backend_submission():
    sequence = [
        {"acc": 0.5, "p": 1},
        {"acc": 0.9, "p": 1},
        {"acc": 0.6, "p": 2},
    ]

    result, backend = asyncio.run(_run_orchestrator(sequence, exec_mode="standard"))

    assert result.best_config["p"] == 1
    assert abs(result.best_score - 0.7) < 1e-9

    # Find aggregated summary submissions by checking for summary_stats with session-level aggregation
    summaries = [
        s
        for s in backend.submissions
        if s.get("metadata", {})
        .get("summary_stats", {})
        .get("metadata", {})
        .get("aggregation_level")
        == "session"
    ]
    assert len(summaries) >= 1, "Should have at least one session-level aggregation"

    # Check the new location
    last_summary = summaries[-1]
    assert "summary_stats" in last_summary["metadata"]
    summary_stats = last_summary["metadata"]["summary_stats"]
    assert "metadata" in summary_stats
    assert "aggregation_summary" in summary_stats["metadata"]

    aggregation_summary = summary_stats["metadata"]["aggregation_summary"]
    total_samples = sum(aggregation_summary.get("samples_per_config", {}).values())
    assert total_samples == len(sequence)


def test_local_mode_respects_minimize_objective_for_best_config():
    class CostOptimizer(BaseOptimizer):
        def __init__(self, seq):
            super().__init__(config_space={}, objectives=["cost"], context=None)
            self.seq = seq
            self.i = 0

        def suggest_next_trial(self, history):
            if self.i >= len(self.seq):
                return self.seq[-1]
            cfg = self.seq[self.i]
            self.i += 1
            self._trial_count += 1
            return cfg

        def should_stop(self, history):
            return False

    class CostEvaluator(BaseEvaluator):
        async def evaluate(self, func, config, dataset):
            return EvaluationResult(
                config=config,
                aggregated_metrics=(
                    {"cost": float(config["c"])} if "c" in config else {"cost": 0.0}
                ),
                example_results=[],
                total_examples=1,
                successful_examples=1,
                duration=0.01,
            )

    seq = [
        {"c": 0.8, "id": 1},
        {"c": 0.3, "id": 2},
        {"c": 0.6, "id": 3},
    ]
    opt = CostOptimizer(seq)
    ev = CostEvaluator(metrics=["cost"], timeout=1.0)
    oc = OptimizationOrchestrator(
        optimizer=opt,
        evaluator=ev,
        max_trials=len(seq),
        timeout=5.0,
        config=TraigentConfig(execution_mode="edge_analytics"),
        parallel_trials=1,
        objectives=["cost"],
    )

    class NoopBackend:
        def create_session(self, **kwargs):
            return "s"

        def submit_result(self, *args, **kwargs):
            pass

        def finalize_session_sync(self, *args, **kwargs):
            return {}

    oc.backend_client = NoopBackend()

    async def dummy(**kwargs):
        return "ok"

    ds = Dataset([EvaluationExample(input_data={})])
    result = asyncio.run(oc.optimize(dummy, ds, function_name="f"))
    assert result.best_config["id"] == 2
    assert abs(result.best_score - 0.3) < 1e-9
