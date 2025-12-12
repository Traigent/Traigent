"""Integration tests for session_summary aggregation and metrics.

Tests cover:
- Hybrid mode with measures inclusion
- Session_summary containing all aggregated metrics
- Overlay metrics prefixing
- Privacy mode sanitization
- Multiple execution modes
- Edge cases for aggregation
"""

from typing import Any

import pytest

from traigent.api.types import (
    ExampleResult,
    TrialResult,
)
from traigent.config.types import TraigentConfig
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.evaluators.base import (
    BaseEvaluator,
    Dataset,
    EvaluationExample,
    EvaluationResult,
)
from traigent.optimizers.base import BaseOptimizer


class DeterministicOptimizer(BaseOptimizer):
    """Test optimizer that generates deterministic configs."""

    def __init__(self, config_space, objectives, n_trials=3, **kwargs):
        super().__init__(config_space, objectives, **kwargs)
        self.n_trials = n_trials
        self.current_trial = 0

    def suggest_next_trial(self, history: list[TrialResult]) -> dict[str, Any]:
        """Generate deterministic configurations."""
        self.current_trial += 1
        self._trial_count += 1
        config = {}
        for key, values in self.config_space.items():
            if isinstance(values, list) and values:
                # Cycle through available values
                config[key] = values[(self.current_trial - 1) % len(values)]
            else:
                config[key] = values
        return config

    def should_stop(self, history: list[TrialResult]) -> bool:
        """Stop after n_trials."""
        return len(history) >= self.n_trials


class ConfigurableEvaluator(BaseEvaluator):
    """Test evaluator that returns configurable metrics."""

    def __init__(self, metrics_pattern=None, **kwargs):
        super().__init__(metrics=["accuracy", "latency", "cost"], **kwargs)
        # Pattern: list of dicts, one per trial
        self.metrics_pattern = metrics_pattern or [
            {"accuracy": 0.8, "latency": 100, "cost": 0.01},
            {"accuracy": 0.85, "latency": 95, "cost": 0.012},
            {"accuracy": 0.75, "latency": 110, "cost": 0.008},
        ]
        self.trial_index = 0

    async def evaluate(
        self, func, config: dict[str, Any], dataset: Dataset
    ) -> EvaluationResult:
        """Return predetermined metrics based on trial index."""
        # Get metrics for this trial
        metrics = self.metrics_pattern[self.trial_index % len(self.metrics_pattern)]
        self.trial_index += 1

        # Create example results with token/cost metrics
        example_results = []
        for i, example in enumerate(dataset.examples):
            er = ExampleResult(
                example_id=f"ex_{i}",
                input_data=example.input_data,
                expected_output=example.expected_output,
                actual_output=func(example.input_data),
                metrics={
                    **metrics,
                    "input_tokens": 50,
                    "output_tokens": 30,
                    "total_tokens": 80,
                    "input_cost": 0.005,
                    "output_cost": 0.003,
                    "total_cost": 0.008,
                },
                execution_time=0.1,
                success=True,
                error_message=None,
            )
            example_results.append(er)

        # Create summary_stats as orchestrator expects
        summary_stats = {
            "metrics": {
                metric_name: {
                    "count": len(dataset.examples),
                    "mean": value,
                    "std": 0.0,
                    "min": value,
                    "25%": value,
                    "50%": value,
                    "75%": value,
                    "max": value,
                }
                for metric_name, value in metrics.items()
            },
            "execution_time": 0.1 * len(dataset.examples),
            "total_examples": len(dataset.examples),
            "metadata": {
                "aggregation_level": "trial",
                "sdk_version": "2.0.0",
            },
        }

        return EvaluationResult(
            config=config,
            example_results=example_results,
            aggregated_metrics=metrics,
            total_examples=len(dataset.examples),
            successful_examples=len(dataset.examples),
            duration=0.1 * len(dataset.examples),
            summary_stats=summary_stats,
        )


class MockBackendClient:
    """Mock backend client that captures submissions."""

    def __init__(self):
        self.sessions = []
        self.submissions = []
        self.finalized = []
        self.weighted_scores = []

    def create_session(self, **kwargs):
        session_id = f"session_{len(self.sessions) + 1}"
        self.sessions.append({"id": session_id, **kwargs})
        return session_id

    def submit_result(self, session_id, config, score, metadata=None):
        self.submissions.append(
            {
                "session_id": session_id,
                "config": config,
                "score": score,
                "metadata": metadata or {},
            }
        )

    def finalize_session_sync(self, session_id, succeeded=True):
        self.finalized.append({"id": session_id, "succeeded": succeeded})
        return {"status": "ok", "succeeded": succeeded}

    def update_trial_weighted_scores(
        self, trial_id, weighted_score=None, weighted_scores=None, **kwargs
    ):
        """Mock method to handle weighted score updates."""
        # Handle both old and new parameter names and any additional kwargs
        scores = weighted_scores or weighted_score
        self.weighted_scores.append({"trial_id": trial_id, "scores": scores, **kwargs})
        return True


def dummy_func(input_data: dict[str, Any]) -> str:
    """Dummy function for testing."""
    return f"response_for_{input_data.get('text', 'unknown')}"


@pytest.mark.asyncio
async def test_hybrid_mode_includes_measures_when_privacy_off():
    """Test that hybrid mode includes measures when privacy is disabled."""
    config = TraigentConfig(execution_mode="hybrid", privacy_enabled=False)

    optimizer = DeterministicOptimizer(
        config_space={"model": ["gpt-3.5", "gpt-4"], "temperature": [0.5, 0.7]},
        objectives=["accuracy"],
        n_trials=2,
        context=config,
    )

    evaluator = ConfigurableEvaluator(
        metrics_pattern=[
            {"accuracy": 0.8, "latency": 100, "cost": 0.01},
            {"accuracy": 0.9, "latency": 90, "cost": 0.015},
        ]
    )

    orchestrator = OptimizationOrchestrator(
        optimizer=optimizer,
        evaluator=evaluator,
        max_trials=2,
        config=config,
        parallel_trials=1,
    )

    backend = MockBackendClient()
    orchestrator.backend_client = backend

    dataset = Dataset(
        examples=[
            EvaluationExample(input_data={"text": "test1"}),
            EvaluationExample(input_data={"text": "test2"}),
        ]
    )

    await orchestrator.optimize(
        func=dummy_func, dataset=dataset, function_name="test_hybrid"
    )

    # Check that submissions include measures (not privacy mode)
    trial_submissions = [s for s in backend.submissions if "measures" in s["metadata"]]
    assert (
        len(trial_submissions) == 2
    ), "Each trial should submit measures in hybrid mode"

    for submission in trial_submissions:
        measures = submission["metadata"]["measures"]
        assert isinstance(measures, list)
        assert len(measures) > 0
        # Each measure should have metrics but no raw data
        for measure in measures:
            assert "score" in measure or any(
                k in measure for k in ["accuracy", "latency", "cost"]
            )
            # Privacy checks - no raw data even in hybrid mode when privacy is off
            assert "input_data" not in measure
            assert "actual_output" not in measure


@pytest.mark.asyncio
async def test_session_summary_contains_all_metrics():
    """Test that session_summary aggregates all metrics, not just primary objective."""
    config = TraigentConfig(execution_mode="standard", privacy_enabled=False)

    optimizer = DeterministicOptimizer(
        config_space={"model": ["gpt-3.5"]},
        objectives=["accuracy", "latency", "cost"],  # Multiple objectives
        n_trials=3,
        context=config,
    )

    evaluator = ConfigurableEvaluator(
        metrics_pattern=[
            {"accuracy": 0.8, "latency": 100, "cost": 0.01, "custom_metric": 5.0},
            {"accuracy": 0.85, "latency": 95, "cost": 0.012, "custom_metric": 4.5},
            {"accuracy": 0.9, "latency": 90, "cost": 0.015, "custom_metric": 4.0},
        ]
    )

    orchestrator = OptimizationOrchestrator(
        optimizer=optimizer,
        evaluator=evaluator,
        max_trials=3,
        config=config,
        parallel_trials=1,
        objectives=["accuracy", "latency", "cost"],
    )

    backend = MockBackendClient()
    orchestrator.backend_client = backend

    dataset = Dataset(examples=[EvaluationExample(input_data={"text": "test"})])

    await orchestrator.optimize(
        func=dummy_func, dataset=dataset, function_name="test_metrics"
    )

    # Find the session aggregation submission
    session_submissions = [
        s
        for s in backend.submissions
        if s.get("metadata", {})
        .get("summary_stats", {})
        .get("metadata", {})
        .get("aggregation_level")
        == "session"
    ]

    assert len(session_submissions) >= 1, "Should have session-level aggregation"

    # Check the aggregated metrics in session_summary
    session_meta = session_submissions[-1]["metadata"]
    assert "summary_stats" in session_meta
    assert "metadata" in session_meta["summary_stats"]
    assert "aggregation_summary" in session_meta["summary_stats"]["metadata"]

    agg_summary = session_meta["summary_stats"]["metadata"]["aggregation_summary"]
    assert "metrics" in agg_summary

    # Verify all metrics are present, not just primary objective
    metrics = agg_summary["metrics"]
    assert "accuracy" in metrics
    assert "latency" in metrics
    assert "cost" in metrics
    # Custom metrics might also be included
    if "custom_metric" in evaluator.metrics_pattern[0]:
        # If the evaluator returned custom_metric, it should be aggregated
        pass  # Custom metrics are optional

    # Verify the values are averages across trials
    expected_accuracy = sum(m["accuracy"] for m in evaluator.metrics_pattern[:3]) / 3
    expected_latency = sum(m["latency"] for m in evaluator.metrics_pattern[:3]) / 3
    expected_cost = sum(m["cost"] for m in evaluator.metrics_pattern[:3]) / 3

    assert abs(metrics["accuracy"] - expected_accuracy) < 0.01
    assert abs(metrics["latency"] - expected_latency) < 1.0
    assert abs(metrics["cost"] - expected_cost) < 0.001


@pytest.mark.asyncio
async def test_overlay_metrics_properly_prefixed():
    """Test that overlay metrics are prefixed with 'run_' to avoid collisions."""
    config = TraigentConfig(execution_mode="standard", privacy_enabled=False)

    # Create an optimizer that will cause resampling (same config twice)
    class ResamplingOptimizer(BaseOptimizer):
        def __init__(self, config_space, objectives, **kwargs):
            super().__init__(config_space, objectives, **kwargs)
            self._num_trials = 0

        def suggest_next_trial(self, history):
            self._num_trials += 1
            self._trial_count += 1
            # Always suggest the same config to trigger resampling
            return {"model": "gpt-4", "temperature": 0.7}

        def should_stop(self, history):
            return self._num_trials >= 3

    optimizer = ResamplingOptimizer(
        config_space={"model": ["gpt-4"], "temperature": [0.7]},
        objectives=["accuracy"],
        context=config,
    )

    evaluator = ConfigurableEvaluator(
        metrics_pattern=[
            {"accuracy": 0.8, "latency": 100, "cost": 0.01},
            {"accuracy": 0.85, "latency": 95, "cost": 0.012},
            {"accuracy": 0.82, "latency": 98, "cost": 0.011},
        ]
    )

    orchestrator = OptimizationOrchestrator(
        optimizer=optimizer,
        evaluator=evaluator,
        max_trials=3,
        config=config,
        parallel_trials=1,
    )

    backend = MockBackendClient()
    orchestrator.backend_client = backend

    dataset = Dataset(examples=[EvaluationExample(input_data={"text": "test"})])

    await orchestrator.optimize(
        func=dummy_func, dataset=dataset, function_name="test_overlay"
    )

    # Find session aggregation
    session_submissions = [
        s
        for s in backend.submissions
        if s.get("metadata", {})
        .get("summary_stats", {})
        .get("metadata", {})
        .get("aggregation_level")
        == "session"
    ]

    assert len(session_submissions) >= 1

    session_meta = session_submissions[-1]["metadata"]
    agg_summary = session_meta["summary_stats"]["metadata"]["aggregation_summary"]

    # Check for overlay metrics (prefixed with run_)
    metrics = agg_summary.get("metrics", {})

    # Check that if there are run-level metrics, they're properly prefixed
    for key in metrics:
        # Overlay metrics should either be standard metrics or prefixed with run_
        if key not in ["accuracy", "latency", "cost", "custom_metric"]:
            # Any additional metrics should be prefixed
            assert key.startswith("run_") or key in [
                "total_trials",
                "successful_trials",
            ], f"Metric '{key}' should be prefixed with 'run_' to avoid collisions"


@pytest.mark.asyncio
async def test_privacy_mode_excludes_raw_data():
    """Test that privacy mode excludes raw data but includes sanitized metrics."""
    config = TraigentConfig(execution_mode="hybrid", privacy_enabled=True)

    optimizer = DeterministicOptimizer(
        config_space={"model": ["gpt-3.5"]},
        objectives=["accuracy"],
        n_trials=2,
        context=config,
    )

    evaluator = ConfigurableEvaluator(
        metrics_pattern=[
            {"accuracy": 0.8, "latency": 100, "cost": 0.01},
            {"accuracy": 0.9, "latency": 90, "cost": 0.015},
        ]
    )

    orchestrator = OptimizationOrchestrator(
        optimizer=optimizer,
        evaluator=evaluator,
        max_trials=2,
        config=config,
        parallel_trials=1,
    )

    backend = MockBackendClient()
    orchestrator.backend_client = backend

    dataset = Dataset(
        examples=[
            EvaluationExample(
                input_data={"text": "sensitive input"},
                expected_output="sensitive output",
            )
        ]
    )

    await orchestrator.optimize(
        func=dummy_func, dataset=dataset, function_name="test_privacy"
    )

    # Check all submissions for privacy compliance
    for submission in backend.submissions:
        metadata = submission.get("metadata", {})

        # Should not have raw example_results in privacy mode
        assert "example_results" not in metadata

        # Check measures if present
        if "measures" in metadata:
            for measure in metadata["measures"]:
                # Should have sanitized metrics
                assert "score" in measure or any(
                    k in measure for k in ["accuracy", "latency", "cost"]
                )
                # Must not have raw data
                assert "input_data" not in measure
                assert "expected_output" not in measure
                assert "actual_output" not in measure
                assert "input" not in measure
                assert "output" not in measure

        # Check aggregation summary if present
        if "summary_stats" in metadata:
            stats_meta = metadata["summary_stats"].get("metadata", {})
            if "aggregation_summary" in stats_meta:
                agg = stats_meta["aggregation_summary"]
                assert agg.get("sanitized") is True
                # Should only have numeric metrics
                assert "metrics" in agg
                for key, value in agg["metrics"].items():
                    assert isinstance(
                        value, (int, float)
                    ), f"Metric {key} should be numeric in privacy mode"


@pytest.mark.asyncio
async def test_local_mode_aggregation():
    """Test that Edge Analytics mode properly includes aggregated summary."""
    config = TraigentConfig(execution_mode="edge_analytics", privacy_enabled=False)

    optimizer = DeterministicOptimizer(
        config_space={"model": ["gpt-3.5"]},
        objectives=["accuracy"],
        n_trials=1,
        context=config,
    )

    evaluator = ConfigurableEvaluator(
        metrics_pattern=[{"accuracy": 0.85, "latency": 95, "cost": 0.012}]
    )

    orchestrator = OptimizationOrchestrator(
        optimizer=optimizer,
        evaluator=evaluator,
        max_trials=1,
        config=config,
        parallel_trials=1,
    )

    backend = MockBackendClient()
    orchestrator.backend_client = backend

    dataset = Dataset(examples=[EvaluationExample(input_data={"text": "test"})])

    await orchestrator.optimize(
        func=dummy_func, dataset=dataset, function_name="test_local"
    )

    # Edge Analytics mode should have exactly 1 submission
    assert len(backend.submissions) == 1

    submission = backend.submissions[0]
    metadata = submission["metadata"]

    # Should have summary_stats with aggregation_summary
    assert "summary_stats" in metadata
    summary_stats = metadata["summary_stats"]
    assert "metadata" in summary_stats
    assert "aggregation_summary" in summary_stats["metadata"]

    agg = summary_stats["metadata"]["aggregation_summary"]
    assert "metrics" in agg
    assert "accuracy" in agg["metrics"]
    assert agg["metrics"]["accuracy"] == 0.85


@pytest.mark.asyncio
async def test_total_examples_calculation():
    """Test that total_examples is correctly calculated from samples_per_config."""
    config = TraigentConfig(execution_mode="standard", privacy_enabled=False)

    # Optimizer that generates different configs
    optimizer = DeterministicOptimizer(
        config_space={
            "model": ["gpt-3.5", "gpt-4", "claude"],
            "temperature": [0.5, 0.7, 0.9],
        },
        objectives=["accuracy"],
        n_trials=3,
        context=config,
    )

    evaluator = ConfigurableEvaluator()

    orchestrator = OptimizationOrchestrator(
        optimizer=optimizer,
        evaluator=evaluator,
        max_trials=3,
        config=config,
        parallel_trials=1,
    )

    backend = MockBackendClient()
    orchestrator.backend_client = backend

    # Dataset with multiple examples
    dataset = Dataset(
        examples=[EvaluationExample(input_data={"text": f"test_{i}"}) for i in range(5)]
    )

    result = await orchestrator.optimize(
        func=dummy_func, dataset=dataset, function_name="test_total_examples"
    )

    # Check the result metadata
    assert "session_summary" in result.metadata
    session_summary = result.metadata["session_summary"]

    # Calculate expected total examples
    samples_per_config = session_summary.get("samples_per_config", {})
    expected_total = sum(samples_per_config.values()) if samples_per_config else 0

    # The samples_per_config counts TRIALS per config, not individual examples
    # With 3 unique configs (each trial has different config), we expect 3 total
    assert (
        expected_total == 3
    ), f"Expected 3 total trials across configs, got {expected_total}"

    # Check that each config was tried once
    assert (
        len(samples_per_config) == 3
    ), f"Expected 3 unique configs, got {len(samples_per_config)}"
    for config_hash, count in samples_per_config.items():
        assert (
            count == 1
        ), f"Each config should be tried once, but {config_hash} was tried {count} times"

    # Also check in backend submissions
    session_submissions = [
        s
        for s in backend.submissions
        if s.get("metadata", {})
        .get("summary_stats", {})
        .get("metadata", {})
        .get("aggregation_level")
        == "session"
    ]

    if session_submissions:
        session_meta = session_submissions[-1]["metadata"]
        agg = session_meta["summary_stats"]["metadata"]["aggregation_summary"]
        backend_samples = agg.get("samples_per_config", {})
        backend_total = sum(backend_samples.values()) if backend_samples else 0
        assert (
            backend_total == 3
        ), f"Backend should also show 3 total trials, got {backend_total}"


@pytest.mark.asyncio
async def test_multiple_replicates_aggregation():
    """Test aggregation with multiple replicates of the same configuration."""
    config = TraigentConfig(execution_mode="standard", privacy_enabled=False)

    # Optimizer that repeats the same config (simulating replicates)
    class ReplicateOptimizer(BaseOptimizer):
        def __init__(self, config_space, objectives, **kwargs):
            super().__init__(config_space, objectives, **kwargs)
            self._call_count = 0

        def suggest_next_trial(self, history):
            self._call_count += 1
            self._trial_count += 1
            # Always return the same config to simulate replicates
            return {"model": "gpt-4", "temperature": 0.7}

        def should_stop(self, history):
            return self._call_count >= 5  # 5 replicates

    optimizer = ReplicateOptimizer(
        config_space={"model": ["gpt-4"], "temperature": [0.7]},
        objectives=["accuracy"],
        context=config,
    )

    # Evaluator with varying metrics to simulate variance across replicates
    evaluator = ConfigurableEvaluator(
        metrics_pattern=[
            {"accuracy": 0.80, "latency": 100, "cost": 0.010},
            {"accuracy": 0.82, "latency": 98, "cost": 0.011},
            {"accuracy": 0.85, "latency": 95, "cost": 0.012},
            {"accuracy": 0.81, "latency": 102, "cost": 0.010},
            {"accuracy": 0.83, "latency": 97, "cost": 0.011},
        ]
    )

    orchestrator = OptimizationOrchestrator(
        optimizer=optimizer,
        evaluator=evaluator,
        max_trials=5,
        config=config,
        parallel_trials=1,
    )

    backend = MockBackendClient()
    orchestrator.backend_client = backend

    dataset = Dataset(examples=[EvaluationExample(input_data={"text": "test"})])

    result = await orchestrator.optimize(
        func=dummy_func, dataset=dataset, function_name="test_replicates"
    )

    # Check that all 5 trials completed
    assert len(result.trials) == 5

    # Find session aggregation
    session_submissions = [
        s
        for s in backend.submissions
        if s.get("metadata", {})
        .get("summary_stats", {})
        .get("metadata", {})
        .get("aggregation_level")
        == "session"
    ]

    assert len(session_submissions) >= 1

    session_meta = session_submissions[-1]["metadata"]
    agg_summary = session_meta["summary_stats"]["metadata"]["aggregation_summary"]

    # Check samples_per_config
    samples_per_config = agg_summary.get("samples_per_config", {})
    assert len(samples_per_config) == 1, "Should have one unique config"

    # The single config should have 5 samples (replicates)
    config_hash = list(samples_per_config.keys())[0]
    assert (
        samples_per_config[config_hash] == 5
    ), "Should have 5 replicates of the same config"

    # Check aggregated metrics are averages
    metrics = agg_summary["metrics"]
    expected_accuracy = sum(m["accuracy"] for m in evaluator.metrics_pattern[:5]) / 5
    expected_latency = sum(m["latency"] for m in evaluator.metrics_pattern[:5]) / 5

    assert abs(metrics["accuracy"] - expected_accuracy) < 0.01
    assert abs(metrics["latency"] - expected_latency) < 1.0


@pytest.mark.asyncio
async def test_cloud_mode_no_local_aggregation():
    """Test that cloud mode doesn't perform local aggregation."""
    config = TraigentConfig(execution_mode="cloud", privacy_enabled=False)

    optimizer = DeterministicOptimizer(
        config_space={"model": ["gpt-3.5"]},
        objectives=["accuracy"],
        n_trials=2,
        context=config,
    )

    evaluator = ConfigurableEvaluator()

    orchestrator = OptimizationOrchestrator(
        optimizer=optimizer,
        evaluator=evaluator,
        max_trials=2,
        config=config,
        parallel_trials=1,
    )

    backend = MockBackendClient()
    orchestrator.backend_client = backend

    dataset = Dataset(examples=[EvaluationExample(input_data={"text": "test"})])

    await orchestrator.optimize(
        func=dummy_func, dataset=dataset, function_name="test_cloud"
    )

    # Check submissions - cloud mode now also creates session aggregations for consistency
    session_submissions = [
        s
        for s in backend.submissions
        if s.get("metadata", {})
        .get("summary_stats", {})
        .get("metadata", {})
        .get("aggregation_level")
        == "session"
    ]

    # Cloud mode now also creates session-level aggregations for consistency
    # This was changed to provide uniform behavior across all modes
    assert (
        len(session_submissions) >= 1
    ), "Cloud mode should create session aggregations for consistency"

    # Should have both trial and session submissions
    assert len(backend.submissions) > len(
        session_submissions
    ), "Should have trial submissions in addition to session aggregation"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
