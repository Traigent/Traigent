"""Focused unit tests for hybrid async suggestions and privacy redaction.

These tests avoid network by stubbing the backend client and use lightweight
optimizers/evaluators to validate orchestrator behavior.
"""

from __future__ import annotations

from typing import Any

import pytest

from traigent.api.types import ExampleResult
from traigent.config.types import TraigentConfig
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.evaluators.base import (
    BaseEvaluator,
    Dataset,
    EvaluationExample,
    EvaluationResult,
)
from traigent.optimizers.base import BaseOptimizer
from traigent.optimizers.remote import MockRemoteSuggestionClient, RemoteOptimizer


class DummyAsyncBatchOptimizer(BaseOptimizer):
    """Optimizer that provides async batch candidates and tracks calls."""

    def __init__(self, config_space, objectives, **kwargs):
        super().__init__(config_space, objectives, **kwargs)
        self.async_calls = 0
        self.sync_calls = 0
        self._stop_after = kwargs.get("stop_after", 2)

    def suggest_next_trial(self, history):  # pragma: no cover - fallback path
        self.sync_calls += 1
        # Deterministic single suggestion for fallback
        return {
            k: (v[0] if isinstance(v, list) and v else None)
            for k, v in self.config_space.items()
        }

    async def suggest_next_trial_async(
        self, history, remote_context=None
    ):  # pragma: no cover - not used directly
        # Unused in this test; we provide batched API below
        return self.suggest_next_trial(history)

    async def generate_candidates_async(self, max_candidates: int, remote_context=None):
        self.async_calls += 1
        # Generate exactly max_candidates deterministic configs
        configs = []
        for i in range(max_candidates):
            cfg = {}
            for k, v in self.config_space.items():
                if isinstance(v, list) and v:
                    cfg[k] = v[min(i, len(v) - 1)]
                else:
                    cfg[k] = v
            configs.append(cfg)
        return configs

    def should_stop(self, history):
        # Stop after a couple of trials
        return len(history) >= self._stop_after


class SimpleEvaluator(BaseEvaluator):
    """Minimal async evaluator that returns fixed metrics and per-example results."""

    def __init__(self, metrics=None, **kwargs):
        super().__init__(metrics or ["accuracy"], **kwargs)

    async def evaluate(
        self, func, config: dict[str, Any], dataset: Dataset
    ) -> EvaluationResult:
        example_results = []
        for i, ex in enumerate(dataset.examples):
            # Create an ExampleResult with token/cost metrics present
            metrics = {
                "accuracy": 1.0,
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "input_cost": 0.001,
                "output_cost": 0.0005,
                "total_cost": 0.0015,
            }
            er = ExampleResult(
                example_id=f"ex{i}",
                input_data=ex.input_data,
                expected_output=ex.expected_output,
                actual_output=func(ex.input_data),
                metrics=metrics,
                execution_time=0.01,
                success=True,
                error_message=None,
            )
            example_results.append(er)

        aggregated = {"accuracy": 1.0}

        # Create summary_stats for non-cloud modes
        summary_stats = {
            "metrics": {
                "accuracy": {
                    "count": len(dataset.examples),
                    "mean": 1.0,
                    "std": 0.0,
                    "min": 1.0,
                    "25%": 1.0,
                    "50%": 1.0,
                    "75%": 1.0,
                    "max": 1.0,
                }
            },
            "execution_time": 0.02,
            "total_examples": len(dataset.examples),
            "metadata": {"aggregation_level": "trial", "sdk_version": "2.0.0"},
        }

        return EvaluationResult(
            config=config,
            example_results=example_results,
            aggregated_metrics=aggregated,
            total_examples=len(dataset.examples),
            successful_examples=len(dataset.examples),
            duration=0.02,
            summary_stats=summary_stats,
        )


class DummyBackend:
    """Backend stub capturing submissions without network."""

    def __init__(self):
        self.sessions = []
        self.submissions = []
        self.finalized = []

    def create_session(
        self,
        function_name: str,
        search_space: dict,
        optimization_goal: str,
        metadata: dict,
    ):
        sid = f"sess-{len(self.sessions)+1}"
        self.sessions.append(
            {"id": sid, "fn": function_name, "space": search_space, "meta": metadata}
        )
        return sid

    def submit_result(
        self, session_id: str, config: dict, score: float, metadata: dict
    ):
        self.submissions.append(
            {
                "session_id": session_id,
                "config": config,
                "score": score,
                "metadata": metadata,
            }
        )

    def finalize_session_sync(self, session_id: str, succeeded: bool):
        self.finalized.append({"id": session_id, "ok": succeeded})
        return {"status": "ok", "succeeded": succeeded}


def tiny_func(inp: dict[str, Any]) -> str:
    return "ok"


@pytest.mark.asyncio
async def test_hybrid_async_batch_suggestions_preferred():
    # Prepare components
    cfg = TraigentConfig(execution_mode="hybrid", privacy_enabled=False)
    optimizer = DummyAsyncBatchOptimizer(
        {"x": [1, 2, 3]}, ["accuracy"], context=cfg, stop_after=2
    )
    evaluator = SimpleEvaluator()
    orch = OptimizationOrchestrator(
        optimizer=optimizer,
        evaluator=evaluator,
        max_trials=2,
        timeout=1.0,
        config=cfg,
        parallel_trials=2,
    )
    orch.backend_client = None  # avoid network calls

    # Dataset with two examples
    ds = Dataset(
        examples=[
            EvaluationExample(input_data={"text": "a"}),
            EvaluationExample(input_data={"text": "b"}),
        ],
        name="tiny",
    )

    # Run
    await orch.optimize(func=tiny_func, dataset=ds, function_name="tiny")

    # Assert async batch suggestions were used
    assert optimizer.async_calls >= 1


@pytest.mark.asyncio
async def test_privacy_mode_sanitizes_measures():
    # Hybrid with privacy enabled
    cfg = TraigentConfig(execution_mode="hybrid", privacy_enabled=True)
    optimizer = DummyAsyncBatchOptimizer(
        {"x": [1]}, ["accuracy"], context=cfg, stop_after=1
    )
    evaluator = SimpleEvaluator()
    orch = OptimizationOrchestrator(
        optimizer=optimizer,
        evaluator=evaluator,
        max_trials=1,
        timeout=1.0,
        config=cfg,
        parallel_trials=1,
    )

    # Attach dummy backend to capture submissions
    dummy = DummyBackend()
    orch.backend_client = dummy

    ds = Dataset(
        examples=[EvaluationExample(input_data={"text": "hello"})], name="tiny"
    )

    await orch.optimize(func=tiny_func, dataset=ds, function_name="tiny_priv")

    # Should have two submissions: trial + aggregated summary for non-Edge Analytics modes
    assert len(dummy.submissions) == 2

    # First submission is the trial with measures
    trial_meta = dummy.submissions[0]["metadata"]
    assert "measures" in trial_meta and isinstance(trial_meta["measures"], list)
    assert len(trial_meta["measures"]) == 1

    m0 = trial_meta["measures"][0]
    # Privacy: should have score and response_time; may include token/cost metrics
    assert "score" in m0
    assert "response_time" in m0
    # Must not include raw inputs/outputs
    forbidden_keys = {"input", "input_data", "expected", "predicted", "actual_output"}
    assert not any(k in m0 for k in forbidden_keys)
    # Ensure raw example_results not transmitted in privacy mode
    assert "example_results" not in trial_meta

    # Aggregated summary in trial's summary_stats.metadata
    assert "summary_stats" in trial_meta
    summary_stats = trial_meta["summary_stats"]
    assert "metadata" in summary_stats
    assert "aggregation_summary" in summary_stats["metadata"]

    agg = summary_stats["metadata"]["aggregation_summary"]
    assert isinstance(agg, dict)
    assert agg.get("sanitized") is True
    # Should contain only numeric metrics and primary_objective label
    assert "primary_objective" in agg
    assert isinstance(agg.get("metrics", {}), dict)
    # No raw content keys
    forbidden = {"input", "output", "prompt", "messages", "actual_output"}
    assert not any(k in agg for k in forbidden)

    # Second submission is the session-level aggregation
    session_meta = dummy.submissions[1]["metadata"]
    assert "summary_stats" in session_meta
    assert session_meta["summary_stats"]["metadata"]["aggregation_level"] == "session"


@pytest.mark.asyncio
async def test_local_mode_includes_aggregated_summary_in_submission():
    # Edge Analytics mode with backend available should submit aggregated summary as part of trial metadata
    cfg = TraigentConfig(execution_mode="edge_analytics", privacy_enabled=False)

    class _DummyBackend:
        def __init__(self):
            self.sessions = []
            self.submissions = []
            self.finalized = []

        def create_session(self, **kwargs):
            sid = f"sess-{len(self.sessions)+1}"
            self.sessions.append({"id": sid})
            return sid

        def submit_result(
            self, session_id: str, config: dict, score: float, metadata: dict
        ):
            self.submissions.append({"session_id": session_id, "metadata": metadata})

        def finalize_session_sync(self, session_id: str, *_):
            self.finalized.append(session_id)
            return {"ok": True}

    # Use a simple optimizer/evaluator pair
    class _Opt(BaseOptimizer):
        def suggest_next_trial(self, history):
            self._trial_count += 1
            return {"x": 1}

        def should_stop(self, history):
            return len(history) >= 1

    class _Eval(BaseEvaluator):
        async def evaluate(self, func, config, dataset):
            ers = [
                ExampleResult(
                    example_id="e0",
                    input_data=dataset.examples[0].input_data,
                    expected_output=None,
                    actual_output=func(dataset.examples[0].input_data),
                    metrics={"accuracy": 1.0, "total_tokens": 10, "total_cost": 0.001},
                    execution_time=0.01,
                    success=True,
                    error_message=None,
                )
            ]
            # Create summary_stats for Edge Analytics mode
            summary_stats = {
                "metrics": {
                    "accuracy": {
                        "count": 1,
                        "mean": 1.0,
                        "std": 0.0,
                        "min": 1.0,
                        "25%": 1.0,
                        "50%": 1.0,
                        "75%": 1.0,
                        "max": 1.0,
                    }
                },
                "execution_time": 0.02,
                "total_examples": 1,
                "metadata": {"aggregation_level": "trial", "sdk_version": "2.0.0"},
            }
            return EvaluationResult(
                config=config,
                example_results=ers,
                aggregated_metrics={"accuracy": 1.0},
                total_examples=1,
                successful_examples=1,
                duration=0.02,
                summary_stats=summary_stats,
            )

    backend = _DummyBackend()
    opt = _Opt({"x": [1]}, ["accuracy"], context=cfg)
    ev = _Eval(["accuracy"])
    orch = OptimizationOrchestrator(
        optimizer=opt,
        evaluator=ev,
        max_trials=1,
        timeout=1.0,
        config=cfg,
        parallel_trials=1,
    )
    orch.backend_client = backend

    ds = Dataset([EvaluationExample(input_data={"text": "hello"})])
    await orch.optimize(func=tiny_func, dataset=ds, function_name="local_agg")

    assert len(backend.submissions) == 1
    meta = backend.submissions[0]["metadata"]

    # Check aggregated summary in summary_stats.metadata
    assert "summary_stats" in meta
    summary_stats = meta["summary_stats"]
    assert "metadata" in summary_stats
    assert "aggregation_summary" in summary_stats["metadata"]

    agg = summary_stats["metadata"]["aggregation_summary"]
    assert agg.get("sanitized") is True
    assert "metrics" in agg and isinstance(agg["metrics"], dict)


@pytest.mark.asyncio
async def test_hybrid_remote_fallback_on_failure(monkeypatch):
    # Hybrid + RemoteOptimizer with failing mock → fallback path
    cfg = TraigentConfig(execution_mode="hybrid", privacy_enabled=True)

    class FailingClient(MockRemoteSuggestionClient):
        async def suggest_batch(self, *a, **k):
            raise RuntimeError("remote unavailable")

    client = FailingClient()
    opt = RemoteOptimizer(
        {"x": [1, 2]},
        ["accuracy"],
        context=cfg,
        remote_enabled=True,
        remote_client=client,
    )
    evaluator = SimpleEvaluator()
    orch = OptimizationOrchestrator(
        optimizer=opt,
        evaluator=evaluator,
        max_trials=2,
        timeout=2.0,
        config=cfg,
        parallel_trials=2,
    )

    class _DummyBackend:
        def create_session(self, *a, **k):
            return "sess"

        def submit_result(self, *a, **k):
            return True

        def finalize_session_sync(self, *a, **k):
            return {"status": "ok"}

    orch.backend_client = _DummyBackend()

    ds = Dataset(
        examples=[EvaluationExample(input_data={"text": str(i)}) for i in range(6)],
        name="tiny",
    )

    result = await orch.optimize(func=tiny_func, dataset=ds, function_name="fallback")
    assert len(result.trials) == 2
    out_counts = [t.metadata.get("output_count") for t in result.trials]
    assert all(c == len(ds.examples) for c in out_counts)


@pytest.mark.asyncio
async def test_hybrid_remote_subset_indices_used(monkeypatch):
    # Hybrid + RemoteOptimizer with subset indices
    cfg = TraigentConfig(execution_mode="hybrid", privacy_enabled=True)

    # Remote optimizer with mock client
    client = MockRemoteSuggestionClient()
    opt = RemoteOptimizer(
        {"x": [1, 2]},
        ["accuracy"],
        context=cfg,
        remote_enabled=True,
        remote_client=client,
    )

    # Local evaluator
    evaluator = SimpleEvaluator()

    # Orchestrator with 2 parallel trials
    orch = OptimizationOrchestrator(
        optimizer=opt,
        evaluator=evaluator,
        max_trials=2,
        timeout=2.0,
        config=cfg,
        parallel_trials=2,
    )

    # Dummy backend to avoid network
    class _DummyBackend:
        def create_session(self, *a, **k):
            return "sess"

        def submit_result(self, *a, **k):
            return True

        def finalize_session_sync(self, *a, **k):
            return {"status": "ok"}

    orch.backend_client = _DummyBackend()

    # Dataset of 6 examples should be split ~3 and ~3 by subset indices
    ds = Dataset(
        examples=[EvaluationExample(input_data={"text": str(i)}) for i in range(6)],
        name="tiny",
    )

    result = await orch.optimize(func=tiny_func, dataset=ds, function_name="subset")
    # Expect 2 trials due to max_trials=2 and parallel_trials=2 (single wave)
    assert len(result.trials) == 2
    out_counts = [t.metadata.get("output_count") for t in result.trials]
    # Each trial should run a subset (non-zero, less than total, sum equals total)
    assert all(isinstance(c, int) and c > 0 for c in out_counts)
    assert sum(out_counts) == len(ds.examples)
