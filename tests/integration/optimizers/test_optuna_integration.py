"""Integration tests for the Optuna-based optimizers."""

from __future__ import annotations

from typing import Any

import optuna
import pytest

from traigent.api.functions import get_trial_config
from traigent.api.types import TrialStatus
from traigent.config.types import TraigentConfig
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.evaluators.base import (
    BaseEvaluator,
    Dataset,
    EvaluationExample,
    EvaluationResult,
)
from traigent.evaluators.local import LocalEvaluator
from traigent.optimizers.optuna_coordinator import BatchOptimizer
from traigent.optimizers.optuna_optimizer import OptunaTPEOptimizer


def test_optuna_coordinator_handles_multi_objective_batches():
    config_space = {"model": ["alpha", "beta"]}

    def worker(config: dict[str, str]) -> dict[str, object]:
        if config["model"] == "alpha":
            return {"status": "completed", "values": [0.9, 0.25]}
        return {"status": "completed", "values": [0.7, 0.1]}

    batch_optimizer = BatchOptimizer(
        config_space=config_space,
        objectives=["accuracy", "cost"],
        n_workers=2,
        worker_fn=worker,
        coordinator_kwargs={"sampler": optuna.samplers.RandomSampler(seed=0)},
    )

    batch_optimizer.optimize_batch(n_trials=2)

    study = batch_optimizer.coordinator.study
    assert len(study.trials) == 2
    assert all(
        trial.values is not None and len(trial.values) == 2 for trial in study.trials
    )


@pytest.mark.asyncio
async def test_optuna_pruning_with_progress_callback():
    config_space = {"model": ["bad"]}

    optimizer = OptunaTPEOptimizer(
        config_space,
        ["accuracy"],
        max_trials=1,
        pruner=optuna.pruners.ThresholdPruner(lower=0.5),
        sampler=optuna.samplers.RandomSampler(seed=1),
    )

    evaluator = LocalEvaluator(metrics=["accuracy"], detailed=True)
    orchestrator = OptimizationOrchestrator(
        optimizer=optimizer,
        evaluator=evaluator,
        max_trials=1,
        config=TraigentConfig.edge_analytics_mode(),
        objectives=["accuracy"],
    )

    examples = [
        EvaluationExample(
            input_data={"text": f"example-{i}"}, expected_output="correct"
        )
        for i in range(3)
    ]
    dataset = Dataset(examples=examples, name="pruning_dataset")

    def prunable_function(
        text: str,
    ) -> str:  # pragma: no cover - executed in orchestrator
        cfg = get_trial_config()
        assert cfg["model"] == "bad"
        return "wrong"

    result = await orchestrator.optimize(prunable_function, dataset)

    assert result.trials, "Expected at least one trial result"
    trial = result.trials[0]
    assert trial.status == TrialStatus.PRUNED
    assert (
        trial.metadata.get("pruned") is True
        or trial.metadata.get("pruned_step") is not None
    )

    study = optimizer.study
    assert study.trials[0].state == optuna.trial.TrialState.PRUNED


class SyntheticCostEvaluator(BaseEvaluator):
    """Evaluator that feeds deterministic cost sequences and reports progress."""

    def __init__(self, sequences: dict[str, list[float]]) -> None:
        super().__init__(metrics=["cost"], detailed=False)
        self._sequences = sequences

    async def evaluate(
        self,
        func,
        config,
        dataset,
        *,
        progress_callback=None,
    ) -> EvaluationResult:
        model = config.get("model")
        costs = self._sequences[model]
        total_cost = 0.0
        for step, cost in enumerate(costs):
            total_cost += cost
            if progress_callback:
                progress_callback(
                    step,
                    {
                        "success": True,
                        "metrics": {"total_cost": total_cost, "cost": cost},
                        "output": str(cost),
                    },
                )

        return EvaluationResult(
            config=config,
            example_results=[],
            aggregated_metrics={"cost": total_cost},
            total_examples=len(costs),
            successful_examples=len(costs),
            duration=0.01,
            metrics={
                "cost": total_cost,
                "total_cost": total_cost,
                "examples_attempted": len(costs),
            },
        )


@pytest.mark.asyncio
async def test_optuna_bayesian_pruning_and_early_stop(monkeypatch):
    config_space = {"model": ["good", "meh", "bad"]}
    optimizer = OptunaTPEOptimizer(
        config_space,
        ["cost"],
        max_trials=10,
        sampler=optuna.samplers.TPESampler(seed=0),
    )

    # Ensure deterministic order: good -> meh -> bad
    optimizer.study.enqueue_trial({"model": "good"})
    optimizer.study.enqueue_trial({"model": "meh"})
    optimizer.study.enqueue_trial({"model": "bad"})

    evaluator = SyntheticCostEvaluator(
        {
            "good": [0.05, 0.05, 0.05],
            "meh": [0.3, 0.3, 0.3],
            "bad": [0.4, 0.4, 0.4, 0.4],
        }
    )

    orchestrator = OptimizationOrchestrator(
        optimizer=optimizer,
        evaluator=evaluator,
        max_trials=10,
        config=TraigentConfig.edge_analytics_mode(),
        objectives=["cost"],
    )

    examples = [
        EvaluationExample(input_data={"text": f"example-{i}"}, expected_output="")
        for i in range(4)
    ]
    dataset = Dataset(examples=examples, name="bayes")

    original_should_stop = OptunaTPEOptimizer.should_stop

    state = {"triggered": False}

    def patched_should_stop(self, history):
        if any(tr.status == TrialStatus.PRUNED for tr in history) and any(
            tr.status == TrialStatus.COMPLETED
            and (tr.metrics or {}).get("cost", 1.0) <= 0.2
            for tr in history
        ):
            state["triggered"] = True
            return True
        return original_should_stop(self, history)

    monkeypatch.setattr(
        OptunaTPEOptimizer, "should_stop", patched_should_stop, raising=False
    )

    def dummy_function(
        _: dict[str, Any],
    ) -> str:  # pragma: no cover - executed in orchestrator
        return "ok"

    result = await orchestrator.optimize(dummy_function, dataset)

    assert any(tr.status == TrialStatus.PRUNED for tr in result.trials)
    assert state["triggered"]
    assert optimizer.trial_count < optimizer.max_trials

    pruned_trials = [
        tr
        for tr in optimizer.study.get_trials()
        if tr.state == optuna.trial.TrialState.PRUNED
    ]
    assert pruned_trials, "Expected Optuna study to register a pruned trial"
