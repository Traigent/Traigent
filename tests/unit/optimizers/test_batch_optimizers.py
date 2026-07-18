"""Tests for batch optimization strategies."""

import asyncio
import math
from datetime import UTC, datetime

import pytest

from traigent.api.types import TrialResult, TrialStatus
from traigent.config.types import TraigentConfig
from traigent.evaluators.base import Dataset, EvaluationExample, EvaluationResult
from traigent.invokers.base import InvocationResult
from traigent.optimizers.batch_optimizers import (
    AdaptiveBatchOptimizer,
    BatchOptimizationConfig,
    MultiObjectiveBatchOptimizer,
    ParallelBatchOptimizer,
)
from traigent.optimizers.grid import GridSearchOptimizer
from traigent.optimizers.random import RandomSearchOptimizer
from traigent.optimizers.registry import get_optimizer
from traigent.optimizers.results import Trial
from traigent.utils.multi_objective import scalarize_objectives


def _dummy_history(count: int) -> list[TrialResult]:
    """Build a ``count``-long trial history for resumed-run stop checks.

    ``should_stop`` only inspects ``len(history)`` for the resumed-history cap,
    so the trial contents are placeholders.
    """
    return [
        TrialResult(
            trial_id=f"resumed_{i}",
            config={"param1": i},
            metrics={"accuracy": 0.5},
            status=TrialStatus.COMPLETED,
            duration=0.0,
            timestamp=datetime.now(UTC),
        )
        for i in range(count)
    ]


class MockInvoker:
    """Mock invoker for testing."""

    async def invoke(self, func, config, input_data):
        """Mock single invocation."""
        return InvocationResult(
            result=f"mock_output_{input_data.get('value', 0)}",
            is_successful=True,
            execution_time=0.1,
        )

    async def invoke_batch(self, func, config, input_batch):
        """Mock batch invocation."""
        return [
            InvocationResult(
                result=f"mock_output_{input_data.get('value', 0)}",
                is_successful=True,
                execution_time=0.1,
            )
            for input_data in input_batch
        ]


class MockEvaluator:
    """Mock evaluator for testing."""

    async def evaluate(self, invocation_results, expected_outputs, dataset):
        """Mock evaluation."""
        successful_count = sum(1 for r in invocation_results if r.is_successful)
        total_count = len(invocation_results)

        accuracy = successful_count / max(1, total_count)
        success_rate = successful_count / max(1, total_count)

        return EvaluationResult(
            config={},
            total_examples=total_count,
            successful_examples=successful_count,
            duration=1.0,
            aggregated_metrics={
                "accuracy": accuracy,
                "success_rate": success_rate,
                "avg_execution_time": 0.1,
            },
        )


class TestBatchOptimizationConfig:
    """Test suite for BatchOptimizationConfig."""

    def test_init_default_values(self):
        """Test BatchOptimizationConfig initialization with defaults."""
        config = BatchOptimizationConfig()

        assert config.max_parallel_trials == 4
        assert config.batch_size == 10
        assert config.adaptive_batching is True
        assert config.early_stopping_patience == 10
        assert config.early_stopping_min_delta == 0.001
        assert config.distributed_workers == 1
        assert config.enable_checkpointing is True
        assert config.memory_limit_mb == 1000.0

    def test_min_delta_rejects_nan_negative_and_bool(self):
        """early_stopping_min_delta is validated at the config boundary.

        Now that the batch loop consumes min_delta, NaN would make every
        improvement comparison False (premature stop on improving runs) and a
        negative delta would make flat scores count as improvements (never
        stops). Mirrors OptimizationStrategy's validation.
        """
        for bad in (float("nan"), float("inf"), -0.01, True, "0.1", None):
            with pytest.raises((ValueError, TypeError)):
                BatchOptimizationConfig(early_stopping_min_delta=bad)  # type: ignore[arg-type]

        # Boundary values that must remain accepted.
        assert BatchOptimizationConfig(early_stopping_min_delta=0.0)
        assert BatchOptimizationConfig(early_stopping_min_delta=0.5)

    def test_init_custom_values(self):
        """Test BatchOptimizationConfig initialization with custom values."""
        config = BatchOptimizationConfig(
            max_parallel_trials=8,
            batch_size=20,
            adaptive_batching=False,
            early_stopping_patience=5,
            memory_limit_mb=2000.0,
        )

        assert config.max_parallel_trials == 8
        assert config.batch_size == 20
        assert config.adaptive_batching is False
        assert config.early_stopping_patience == 5
        assert config.memory_limit_mb == 2000.0


class TestParallelBatchOptimizer:
    """Test suite for ParallelBatchOptimizer."""

    def setup_method(self):
        """Set up test data."""
        self.config_space = {"param1": [1, 2, 3], "param2": [0.1, 0.2, 0.3]}
        self.objectives = ["accuracy"]

        self.base_optimizer = GridSearchOptimizer(self.config_space, self.objectives)
        self.batch_config = BatchOptimizationConfig(max_parallel_trials=2, batch_size=2)

        self.dataset = Dataset(
            examples=[
                EvaluationExample(input_data={"value": 1}, expected_output="expected1"),
                EvaluationExample(input_data={"value": 2}, expected_output="expected2"),
                EvaluationExample(input_data={"value": 3}, expected_output="expected3"),
            ],
            name="test_dataset",
        )

    def test_init(self):
        """Test ParallelBatchOptimizer initialization."""
        optimizer = ParallelBatchOptimizer(
            base_optimizer=self.base_optimizer,
            batch_config=self.batch_config,
            objectives=self.objectives,
        )

        assert optimizer.config_space == self.config_space
        assert optimizer.objectives == self.objectives
        assert optimizer.base_optimizer == self.base_optimizer
        assert optimizer.batch_config == self.batch_config
        assert optimizer.adaptive_sizer is not None

    def test_optimize_basic(self):
        """Test basic parallel batch optimization."""

        async def run_test():
            optimizer = ParallelBatchOptimizer(
                base_optimizer=self.base_optimizer, batch_config=self.batch_config
            )

            def mock_func(value, config: TraigentConfig):
                return f"result_{value}_{config.custom_params.get('param1', 0)}"

            invoker = MockInvoker()
            evaluator = MockEvaluator()

            result = await optimizer.optimize(
                func=mock_func,
                dataset=self.dataset,
                invoker=invoker,
                evaluator=evaluator,
                max_trials=4,
            )

            assert result.best_config is not None
            assert result.best_score >= 0
            assert len(result.trials) <= 4
            assert result.duration > 0
            assert "parallel_workers" in result.convergence_info
            assert result.convergence_info["parallel_workers"] == 2

        asyncio.run(run_test())

    def test_optimize_early_stopping(self):
        """Test early stopping in parallel optimization."""

        async def run_test():
            # Configure for early stopping
            early_config = BatchOptimizationConfig(
                max_parallel_trials=2, early_stopping_patience=2
            )

            optimizer = ParallelBatchOptimizer(
                base_optimizer=self.base_optimizer, batch_config=early_config
            )

            def mock_func(value, config: TraigentConfig):
                return f"result_{value}"

            invoker = MockInvoker()
            evaluator = MockEvaluator()

            result = await optimizer.optimize(
                func=mock_func,
                dataset=self.dataset,
                invoker=invoker,
                evaluator=evaluator,
                max_trials=10,  # More than needed for early stopping
            )

            # Should stop early
            assert len(result.trials) < 10
            assert "early_stopped" in result.convergence_info

        asyncio.run(run_test())

    def test_early_stopping_min_delta_ignores_sub_delta_creep(self):
        """``early_stopping_min_delta`` must gate the patience counter: score
        gains smaller than min_delta do NOT count as improvement (the field was
        previously declared but never read, so any epsilon creep deferred
        stopping forever), while material gains DO keep the run alive.
        """

        class _SequencedEvaluator:
            """Evaluator emitting a controlled accuracy sequence per call."""

            def __init__(self, start: float, step: float) -> None:
                self._value = start
                self._step = step

            async def evaluate(self, invocation_results, expected_outputs, dataset):
                self._value += self._step
                total = len(invocation_results)
                return EvaluationResult(
                    config={},
                    total_examples=total,
                    successful_examples=total,
                    duration=1.0,
                    aggregated_metrics={"accuracy": self._value},
                )

        async def run_case(step: float):
            optimizer = ParallelBatchOptimizer(
                base_optimizer=self.base_optimizer,
                batch_config=BatchOptimizationConfig(
                    max_parallel_trials=1,  # deterministic completion order
                    early_stopping_patience=3,
                    early_stopping_min_delta=0.001,
                ),
            )

            def mock_func(value, config: TraigentConfig):
                return f"result_{value}"

            return await optimizer.optimize(
                func=mock_func,
                dataset=self.dataset,
                invoker=MockInvoker(),
                evaluator=_SequencedEvaluator(start=0.5, step=step),
                max_trials=10,
            )

        # Sub-delta creep (+0.0001 per trial, min_delta=0.001): after the first
        # trial seeds the best, every later gain is ignored -> patience=3
        # consecutive non-improvements -> early stop well before max_trials.
        creep_result = asyncio.run(run_case(step=0.0001))
        assert creep_result.convergence_info["early_stopped"] is True
        assert len(creep_result.trials) < 10

        # Material gains (+0.1 per trial) reset the counter every time -> the
        # run is never early-stopped.
        improving_result = asyncio.run(run_case(step=0.1))
        assert improving_result.convergence_info["early_stopped"] is False


class TestMultiObjectiveBatchOptimizer:
    """Test suite for MultiObjectiveBatchOptimizer."""

    def setup_method(self):
        """Set up test data."""
        self.config_space = {"param1": [1, 2], "param2": [0.1, 0.2]}
        self.objectives = ["accuracy", "success_rate"]
        self.batch_config = BatchOptimizationConfig(batch_size=2)

        self.dataset = Dataset(
            examples=[
                EvaluationExample(input_data={"value": 1}, expected_output="expected1"),
                EvaluationExample(input_data={"value": 2}, expected_output="expected2"),
            ],
            name="test_dataset",
        )

    def test_init(self):
        """Test MultiObjectiveBatchOptimizer initialization."""
        optimizer = MultiObjectiveBatchOptimizer(
            configuration_space=self.config_space,
            objectives=self.objectives,
            batch_config=self.batch_config,
            pareto_frontier_size=20,
        )

        assert optimizer.config_space == self.config_space
        assert optimizer.objectives == self.objectives
        assert optimizer.batch_config == self.batch_config
        assert optimizer.pareto_frontier_size == 20
        assert len(optimizer.pareto_frontier) == 0

    def test_optimize_multi_objective(self):
        """Test multi-objective batch optimization."""

        async def run_test():
            optimizer = MultiObjectiveBatchOptimizer(
                configuration_space=self.config_space,
                objectives=self.objectives,
                batch_config=self.batch_config,
            )

            def mock_func(value, config: TraigentConfig):
                return f"result_{value}_{config.custom_params.get('param1', 0)}"

            # Mock evaluator with multiple objectives
            class MultiObjectiveEvaluator:
                async def evaluate(self, invocation_results, expected_outputs, dataset):
                    successful_count = sum(
                        1 for r in invocation_results if r.is_successful
                    )
                    total_count = len(invocation_results)

                    return EvaluationResult(
                        config={},
                        total_examples=total_count,
                        successful_examples=successful_count,
                        duration=1.0,
                        aggregated_metrics={
                            "accuracy": successful_count / max(1, total_count),
                            "success_rate": successful_count / max(1, total_count),
                        },
                    )

            invoker = MockInvoker()
            evaluator = MultiObjectiveEvaluator()

            result = await optimizer.optimize(
                func=mock_func,
                dataset=self.dataset,
                invoker=invoker,
                evaluator=evaluator,
                max_trials=5,
            )

            assert result.best_config is not None
            assert result.best_score >= 0
            # Config space has 4 combinations (2x2), so optimization stops after 4
            # even though 5 trials were requested (budget-saving early termination)
            assert len(result.trials) == 4
            assert "pareto_frontier" in result.convergence_info
            assert "objectives" in result.convergence_info
            assert result.convergence_info["objectives"] == self.objectives

        asyncio.run(run_test())

    def test_dominates(self):
        """Test dominance relationship checking."""
        optimizer = MultiObjectiveBatchOptimizer(
            configuration_space=self.config_space,
            objectives=["accuracy", "success_rate"],
            batch_config=self.batch_config,
        )

        # Test clear dominance
        scores1 = {"accuracy": 0.9, "success_rate": 0.8}
        scores2 = {"accuracy": 0.7, "success_rate": 0.6}
        assert optimizer._dominates(scores1, scores2) is True
        assert optimizer._dominates(scores2, scores1) is False

        # Test partial dominance (not dominated)
        scores3 = {"accuracy": 0.9, "success_rate": 0.5}
        scores4 = {"accuracy": 0.6, "success_rate": 0.8}
        assert optimizer._dominates(scores3, scores4) is False
        assert optimizer._dominates(scores4, scores3) is False

        # Test equal scores
        scores5 = {"accuracy": 0.8, "success_rate": 0.7}
        scores6 = {"accuracy": 0.8, "success_rate": 0.7}
        assert optimizer._dominates(scores5, scores6) is False

    def test_select_best_from_pareto_empty(self):
        """Test selecting best from empty Pareto frontier."""
        optimizer = MultiObjectiveBatchOptimizer(
            configuration_space=self.config_space,
            objectives=self.objectives,
            batch_config=self.batch_config,
        )

        assert optimizer._select_best_from_pareto() is None

    def test_should_stop_honours_max_trials(self):
        """#1916: should_stop delegates to the base optimizer's configured
        max_trials budget instead of the old hard-coded ``len(history) >= 100``.
        """
        # Large discrete space so config-space exhaustion does not fire first.
        optimizer = MultiObjectiveBatchOptimizer(
            configuration_space={"param1": list(range(50))},
            objectives=["accuracy"],
            max_trials=3,
        )

        # The caller's budget is threaded into the internal base optimizer.
        assert optimizer._base_optimizer.max_trials == 3

        # Below budget: do not stop (old code returned len([]) >= 100 == False,
        # but ignored the configured budget entirely).
        assert optimizer.should_stop([]) is False

        # Drive suggestions up to the configured budget; then it must stop.
        for _ in range(3):
            optimizer.suggest_next_trial([])
        assert optimizer.should_stop([]) is True

    def test_should_stop_default_preserves_prior_cap(self):
        """#1916: with no explicit max_trials the default budget stays 100,
        preserving the historical stop point (just no longer hard-coded).
        """
        optimizer = MultiObjectiveBatchOptimizer(
            configuration_space={"param1": list(range(200))},
            objectives=["accuracy"],
        )
        assert optimizer._base_optimizer.max_trials == 100

    def test_should_stop_honours_resumed_history(self):
        """#1916 follow-up: a resumed run passes a pre-populated history but
        drives trials through a *local* optimizer inside ``optimize``, so the
        injected base optimizer's trial counter stays at zero. The legacy
        ``len(history) >= budget`` cap must still fire; delegating solely to the
        base optimizer's trial count (the earlier fix) would never stop a
        resumed run.
        """
        optimizer = MultiObjectiveBatchOptimizer(
            configuration_space={"param1": list(range(200))},
            objectives=["accuracy"],
            max_trials=5,
        )

        # The base optimizer never saw these trials; its counter is still 0.
        assert optimizer._base_optimizer._trial_count == 0

        resumed_history = _dummy_history(5)
        assert optimizer.should_stop(resumed_history) is True
        # One short of the budget must NOT stop.
        assert optimizer.should_stop(_dummy_history(4)) is False

    def test_should_stop_none_max_trials_normalized_to_100(self):
        """#1916 follow-up: a direct-constructor ``max_trials=None`` must not
        crash ``should_stop``. RandomSearchOptimizer.should_stop evaluates
        ``count >= max_trials`` and would raise TypeError on ``None``; the
        constructor normalises it to the historical 100-trial cap.
        """
        optimizer = MultiObjectiveBatchOptimizer(
            configuration_space={"param1": list(range(500))},
            objectives=["accuracy"],
            max_trials=None,
        )

        # Normalised to 100 for both the resumed-history cap and the delegate.
        assert optimizer._max_trials == 100
        assert optimizer._base_optimizer.max_trials == 100

        # Would raise ``TypeError: '>=' not supported ...`` under a None budget.
        assert optimizer.should_stop([]) is False
        assert optimizer.should_stop(_dummy_history(100)) is True


class TestAdaptiveBatchOptimizer:
    """Test suite for AdaptiveBatchOptimizer."""

    def setup_method(self):
        """Set up test data."""
        self.config_space = {"param1": [1, 2], "param2": [0.1, 0.2]}
        self.objectives = ["accuracy"]

        self.base_optimizer = RandomSearchOptimizer(self.config_space, self.objectives)
        self.batch_config = BatchOptimizationConfig(
            batch_size=2, adaptive_batching=True
        )

        self.dataset = Dataset(
            examples=[
                EvaluationExample(input_data={"value": 1}, expected_output="expected1"),
                EvaluationExample(input_data={"value": 2}, expected_output="expected2"),
                EvaluationExample(input_data={"value": 3}, expected_output="expected3"),
            ],
            name="test_dataset",
        )

    def test_init(self):
        """Test AdaptiveBatchOptimizer initialization."""
        optimizer = AdaptiveBatchOptimizer(
            base_optimizer=self.base_optimizer, batch_config=self.batch_config
        )

        assert optimizer.config_space == self.config_space
        assert optimizer.objectives == self.objectives
        assert optimizer.base_optimizer == self.base_optimizer
        assert optimizer.batch_config == self.batch_config
        assert optimizer.adaptive_sizer is not None
        assert len(optimizer.performance_history) == 0

    def test_optimize_adaptive(self):
        """Test adaptive batch optimization."""

        async def run_test():
            optimizer = AdaptiveBatchOptimizer(
                base_optimizer=self.base_optimizer, batch_config=self.batch_config
            )

            def mock_func(value, config: TraigentConfig):
                return f"result_{value}_{config.custom_params.get('param1', 0)}"

            invoker = MockInvoker()
            evaluator = MockEvaluator()

            result = await optimizer.optimize(
                func=mock_func,
                dataset=self.dataset,
                invoker=invoker,
                evaluator=evaluator,
                max_trials=3,
            )

            assert result.best_config is not None
            assert result.best_score >= 0
            assert len(result.trials) == 3
            assert result.duration > 0
            assert "adaptive_batching" in result.convergence_info
            assert result.convergence_info["adaptive_batching"] is True
            assert "final_batch_size" in result.convergence_info
            assert "performance_history" in result.convergence_info

        asyncio.run(run_test())

    def test_update_performance_history(self):
        """Test performance history updates."""
        optimizer = AdaptiveBatchOptimizer(
            base_optimizer=self.base_optimizer, batch_config=self.batch_config
        )

        # Create mock trial
        from traigent.optimizers.results import Trial

        trial = Trial(
            configuration={"param1": 1},
            score=0.8,
            duration=1.0,
            metadata={
                "trial_index": 0,
                "batch_size": 2,
                "throughput": 3.0,
                "error_rate": 0.1,
            },
        )

        optimizer._update_performance_history(trial)

        assert len(optimizer.performance_history) == 1
        history_entry = optimizer.performance_history[0]
        assert history_entry["trial_index"] == 0
        assert history_entry["score"] == 0.8
        assert history_entry["batch_size"] == 2
        assert history_entry["throughput"] == 3.0
        assert history_entry["error_rate"] == 0.1

    def test_update_performance_history_failed_trial(self):
        """Test performance history with failed trial."""
        optimizer = AdaptiveBatchOptimizer(
            base_optimizer=self.base_optimizer, batch_config=self.batch_config
        )

        # Create failed trial
        from traigent.optimizers.results import Trial

        failed_trial = Trial(
            configuration={"param1": 1},
            score=float("-inf"),
            duration=1.0,
            metadata={"error": "Failed", "failed": True},
        )

        optimizer._update_performance_history(failed_trial)

        # Should not add failed trials to history
        assert len(optimizer.performance_history) == 0

    def test_performance_history_size_limit(self):
        """Test performance history size limit."""
        optimizer = AdaptiveBatchOptimizer(
            base_optimizer=self.base_optimizer, batch_config=self.batch_config
        )

        # Add many trials to exceed limit
        from traigent.optimizers.results import Trial

        for i in range(150):  # Exceed the 100 limit
            trial = Trial(
                configuration={"param1": 1},
                score=0.8,
                duration=1.0,
                metadata={
                    "trial_index": i,
                    "batch_size": 2,
                    "throughput": 3.0,
                    "error_rate": 0.1,
                },
            )
            optimizer._update_performance_history(trial)

        # Should maintain only last 100 entries
        assert len(optimizer.performance_history) == 100
        # Should have the most recent entries
        assert optimizer.performance_history[0]["trial_index"] == 50
        assert optimizer.performance_history[-1]["trial_index"] == 149


class TestLegacyPositionalConstructors:
    """Regression tests for legacy positional constructor calls.

    The registry-standard signature is ``(config_space, objectives, **kwargs)``,
    but pre-existing call sites use the historical positional form
    ``XxxBatchOptimizer(base_optimizer, batch_config)``. These tests pin that
    the legacy positional form still produces a correctly-wired optimizer
    instead of being misinterpreted as ``(config_space=base_optimizer,
    objectives=batch_config)``.
    """

    def setup_method(self):
        self.config_space = {"param1": [1, 2], "param2": [0.1, 0.2]}
        self.objectives = ["accuracy"]
        self.base_optimizer = GridSearchOptimizer(self.config_space, self.objectives)
        self.batch_config = BatchOptimizationConfig(max_parallel_trials=3, batch_size=7)

    def test_parallel_batch_legacy_positional(self):
        """ParallelBatchOptimizer(base_optimizer, batch_config) still works."""
        optimizer = ParallelBatchOptimizer(self.base_optimizer, self.batch_config)

        assert optimizer.base_optimizer is self.base_optimizer
        assert optimizer.batch_config is self.batch_config
        assert optimizer.config_space == self.config_space
        assert optimizer.objectives == self.objectives

    def test_parallel_batch_legacy_positional_base_only(self):
        """Single positional base_optimizer falls back to default batch_config."""
        optimizer = ParallelBatchOptimizer(self.base_optimizer)

        assert optimizer.base_optimizer is self.base_optimizer
        assert isinstance(optimizer.batch_config, BatchOptimizationConfig)
        assert optimizer.config_space == self.config_space
        assert optimizer.objectives == self.objectives

    def test_adaptive_batch_legacy_positional(self):
        """AdaptiveBatchOptimizer(base_optimizer, batch_config) still works."""
        base = RandomSearchOptimizer(self.config_space, self.objectives)
        optimizer = AdaptiveBatchOptimizer(base, self.batch_config)

        assert optimizer.base_optimizer is base
        assert optimizer.batch_config is self.batch_config
        assert optimizer.config_space == self.config_space
        assert optimizer.objectives == self.objectives

    def test_adaptive_batch_legacy_positional_base_only(self):
        """Single positional base_optimizer falls back to default batch_config."""
        base = RandomSearchOptimizer(self.config_space, self.objectives)
        optimizer = AdaptiveBatchOptimizer(base)

        assert optimizer.base_optimizer is base
        assert isinstance(optimizer.batch_config, BatchOptimizationConfig)
        assert optimizer.config_space == self.config_space
        assert optimizer.objectives == self.objectives

    def test_parallel_batch_registry_path_unchanged(self):
        """Registry-standard (config_space, objectives, **kwargs) still works."""
        optimizer = ParallelBatchOptimizer(self.config_space, self.objectives)

        assert optimizer.config_space == self.config_space
        assert optimizer.objectives == self.objectives
        assert isinstance(optimizer.base_optimizer, RandomSearchOptimizer)
        assert isinstance(optimizer.batch_config, BatchOptimizationConfig)

    def test_adaptive_batch_registry_path_unchanged(self):
        """Registry-standard (config_space, objectives, **kwargs) still works."""
        optimizer = AdaptiveBatchOptimizer(self.config_space, self.objectives)

        assert optimizer.config_space == self.config_space
        assert optimizer.objectives == self.objectives
        assert isinstance(optimizer.base_optimizer, RandomSearchOptimizer)
        assert isinstance(optimizer.batch_config, BatchOptimizationConfig)

    def test_parallel_batch_explicit_keyword_overrides_positional(self):
        """If both positional base and keyword base_optimizer provided, keyword wins."""
        other_base = RandomSearchOptimizer(self.config_space, self.objectives)
        optimizer = ParallelBatchOptimizer(
            self.base_optimizer, base_optimizer=other_base
        )

        assert optimizer.base_optimizer is other_base

    def test_adaptive_batch_explicit_keyword_overrides_positional(self):
        """If both positional base and keyword base_optimizer provided, keyword wins."""
        other_base = RandomSearchOptimizer(self.config_space, self.objectives)
        optimizer = AdaptiveBatchOptimizer(
            self.base_optimizer, base_optimizer=other_base
        )

        assert optimizer.base_optimizer is other_base


class TestBaseOptimizerKwargForwarding:
    """Regression tests that BaseOptimizer kwargs flow through to super().__init__.

    The registry-standard constructor path of ParallelBatchOptimizer and
    AdaptiveBatchOptimizer accepts ``**kwargs`` but historically dropped them
    on the floor instead of forwarding to ``BaseOptimizer.__init__``. That
    silently discarded ``objective_weights`` (and any future BaseOptimizer
    kwargs) when constructed through ``get_optimizer('parallel_batch', ...)``
    or ``get_optimizer('adaptive_batch', ...)``, leaving ``objective_weights``
    at the default equal-weight setting.
    """

    def setup_method(self):
        self.config_space = {"param1": [1, 2], "param2": [0.1, 0.2]}
        self.objectives = ["accuracy", "cost"]
        self.weights = {"accuracy": 0.7, "cost": 0.3}

    def test_parallel_batch_direct_kwargs_set_objective_weights(self):
        """Direct construction with objective_weights kwarg lands on attribute."""
        optimizer = ParallelBatchOptimizer(
            self.config_space,
            self.objectives,
            objective_weights=self.weights,
        )

        assert optimizer.objective_weights == self.weights

    def test_adaptive_batch_direct_kwargs_set_objective_weights(self):
        """Direct construction with objective_weights kwarg lands on attribute."""
        optimizer = AdaptiveBatchOptimizer(
            self.config_space,
            self.objectives,
            objective_weights=self.weights,
        )

        assert optimizer.objective_weights == self.weights

    def test_parallel_batch_via_get_optimizer_forwards_objective_weights(self):
        """get_optimizer('parallel_batch', ...) forwards objective_weights."""
        optimizer = get_optimizer(
            "parallel_batch",
            self.config_space,
            self.objectives,
            objective_weights=self.weights,
        )

        assert isinstance(optimizer, ParallelBatchOptimizer)
        assert optimizer.objective_weights == self.weights

    def test_adaptive_batch_via_get_optimizer_forwards_objective_weights(self):
        """get_optimizer('adaptive_batch', ...) forwards objective_weights."""
        optimizer = get_optimizer(
            "adaptive_batch",
            self.config_space,
            self.objectives,
            objective_weights=self.weights,
        )

        assert isinstance(optimizer, AdaptiveBatchOptimizer)
        assert optimizer.objective_weights == self.weights

    def test_parallel_batch_weights_affect_composite_score(self):
        """Forwarded objective_weights actually change composite scoring output.

        ParallelBatchOptimizer scores via scalarize_objectives over
        ``self.objective_weights``. Two optimizers with different weights must
        produce different composite scores on the same metric inputs, otherwise
        ``objective_weights`` is being silently ignored.
        """
        equal = get_optimizer(
            "parallel_batch",
            self.config_space,
            self.objectives,
            objective_weights={"accuracy": 0.5, "cost": 0.5},
        )
        skewed = get_optimizer(
            "parallel_batch",
            self.config_space,
            self.objectives,
            objective_weights={"accuracy": 0.9, "cost": 0.1},
        )

        metrics = {"accuracy": 1.0, "cost": 0.0}
        equal_score = equal._calculate_composite_score(metrics)
        skewed_score = skewed._calculate_composite_score(metrics)

        assert equal_score != skewed_score
        # Skewed weights toward the higher-value metric must score above equal weights.
        assert skewed_score > equal_score

    def test_adaptive_batch_weights_affect_composite_score(self):
        """Forwarded objective_weights actually change composite scoring output."""
        equal = get_optimizer(
            "adaptive_batch",
            self.config_space,
            self.objectives,
            objective_weights={"accuracy": 0.5, "cost": 0.5},
        )
        skewed = get_optimizer(
            "adaptive_batch",
            self.config_space,
            self.objectives,
            objective_weights={"accuracy": 0.9, "cost": 0.1},
        )

        metrics = {"accuracy": 1.0, "cost": 0.0}
        equal_score = equal._calculate_composite_score(metrics)
        skewed_score = skewed._calculate_composite_score(metrics)

        assert equal_score != skewed_score
        assert skewed_score > equal_score


class TestObjectiveOrientationInCompositeScore:
    """Regression tests for #1466.

    Batch/grid optimizers scalarized every objective as ``maximize``, so a
    declared ``minimize`` objective (cost/latency/error) contributed
    *positively* to the composite. Best-pick is ``max(score)``, so the most
    expensive / slowest config was returned as ``best_config``. These tests
    assert orientation is honored: for two configs with identical accuracy, the
    cheaper config must score strictly higher and be selected as best.

    All scoring is computed directly from in-memory metrics — no LLM calls.
    """

    config_space = {"param1": [1, 2]}
    objectives = ["accuracy", "cost"]

    def _cheap_and_expensive_metrics(self):
        """Equal accuracy, different cost. Cheaper config should win."""
        cheap = {"accuracy": 0.9, "cost": 0.1}
        expensive = {"accuracy": 0.9, "cost": 0.9}
        return cheap, expensive

    def test_grid_selection_scorer_prefers_cheaper(self):
        """Grid-track scoring ranks lower cost above higher cost.

        Post-#1682, GridSearchOptimizer owns no private composite scorer —
        grid selection routes through the shared
        ``ObjectiveSchema.compute_weighted_score``. The #1466 orientation
        guarantee therefore lives there: for equal accuracy, the cheaper
        metrics must score strictly higher.
        """
        from traigent.core.objectives import create_default_objectives

        assert "_calculate_composite_score" not in GridSearchOptimizer.__dict__

        schema = create_default_objectives(
            self.objectives, weights={"accuracy": 0.6, "cost": 0.4}
        )
        cheap, expensive = self._cheap_and_expensive_metrics()

        cheap_score = schema.compute_weighted_score(cheap)
        expensive_score = schema.compute_weighted_score(expensive)

        assert cheap_score is not None and expensive_score is not None
        # Pre-#1466 this was reversed (cost added positively → expensive higher).
        assert cheap_score > expensive_score

    def test_parallel_batch_composite_prefers_cheaper(self):
        """ParallelBatchOptimizer composite ranks lower cost above higher cost."""
        optimizer = get_optimizer("parallel_batch", self.config_space, self.objectives)
        cheap, expensive = self._cheap_and_expensive_metrics()

        assert optimizer._calculate_composite_score(
            cheap
        ) > optimizer._calculate_composite_score(expensive)

    def test_adaptive_batch_composite_prefers_cheaper(self):
        """AdaptiveBatchOptimizer composite ranks lower cost above higher cost."""
        optimizer = get_optimizer("adaptive_batch", self.config_space, self.objectives)
        cheap, expensive = self._cheap_and_expensive_metrics()

        assert optimizer._calculate_composite_score(
            cheap
        ) > optimizer._calculate_composite_score(expensive)

    def test_multi_objective_select_best_from_pareto_prefers_cheaper(self):
        """_select_best_from_pareto returns the orientation-correct optimum.

        Two configs sit on an accuracy/cost frontier with equal accuracy. Both
        are non-dominated only if accuracy differs; with equal accuracy the
        cheaper one dominates. Either way, the selected best must be the cheaper
        config, never the orientation-blind ``max(score)`` (the expensive one).
        """
        optimizer = MultiObjectiveBatchOptimizer(
            configuration_space=self.config_space,
            objectives=self.objectives,
        )

        # cost is a minimize objective: direction must be False (minimize).
        assert optimizer.objective_directions["cost"] is False
        assert optimizer.objective_directions["accuracy"] is True

        cheap_metrics = {"accuracy": 0.9, "cost": 0.1}
        expensive_metrics = {"accuracy": 0.9, "cost": 0.9}

        # Mirror the per-trial scoring path in ``_run_batch_trial`` (which uses
        # scalarize_objectives over objective_weights, honoring orientation).
        def _composite(metrics):
            return scalarize_objectives(
                metrics,
                optimizer.objective_weights,
                minimize_objectives=optimizer._minimize_objectives,
            )

        cheap_trial = Trial(
            configuration={"param1": 1},
            score=_composite(cheap_metrics),
            duration=0.0,
            metadata={"objective_scores": cheap_metrics},
        )
        expensive_trial = Trial(
            configuration={"param1": 2},
            score=_composite(expensive_metrics),
            duration=0.0,
            metadata={"objective_scores": expensive_metrics},
        )

        optimizer._update_pareto_frontier(cheap_trial)
        optimizer._update_pareto_frontier(expensive_trial)

        best = optimizer._select_best_from_pareto()
        assert best is not None
        # Pre-fix: expensive config selected (max of orientation-blind score).
        assert best.configuration == {"param1": 1}

    def test_minimize_objectives_resolved_on_base(self):
        """Optimizers expose the resolved minimize-objective list (#1466)."""
        optimizer = get_optimizer(
            "grid", self.config_space, ["accuracy", "cost", "latency", "error"]
        )
        assert set(optimizer._minimize_objectives) == {"cost", "latency", "error"}
        assert "accuracy" not in optimizer._minimize_objectives


def _trial(scores: dict, score: float = 0.0) -> Trial:
    """Build a Trial carrying objective_scores metadata for frontier tests."""
    return Trial(
        configuration={},
        score=score,
        duration=0.1,
        metadata={"objective_scores": dict(scores)},
    )


class TestBatchDominanceMissingMetric:
    """Regression for #1944: missing metric must NOT default to 0.0 (best-for-minimize)."""

    def _optimizer(self):
        return MultiObjectiveBatchOptimizer(
            configuration_space={"x": [1, 2]},
            objectives=["accuracy", "cost"],
        )

    def test_missing_minimize_metric_does_not_falsely_dominate(self):
        opt = self._optimizer()
        assert opt.objective_directions["cost"] is False  # minimize
        complete = {"accuracy": 0.90, "cost": 5.0}
        partial = {"accuracy": 0.90}  # cost MISSING
        # With the old 0.0 default, partial's cost=0.0 beat every real cost.
        assert opt._dominates(partial, complete) is False
        assert opt._dominates(complete, partial) is True

    def test_complete_cheap_point_not_evicted_by_partial(self):
        opt = self._optimizer()
        opt._update_pareto_frontier(_trial({"accuracy": 0.90, "cost": 0.5}, score=0.9))
        # A higher-accuracy trial that OMITS cost must not dominate/evict it.
        opt._update_pareto_frontier(_trial({"accuracy": 0.91}, score=0.91))
        frontier_scores = [t.metadata["objective_scores"] for t in opt.pareto_frontier]
        assert {"accuracy": 0.90, "cost": 0.5} in frontier_scores


class TestZeroWeightMissingMetricRejection:
    """Regression for the NaN scalarization leak (terra review of the #1944 fix).

    With an ALLOWED zero objective weight, a missing metric's orientation-worst
    sentinel (±inf) reached scalarization and ``0.0 * -inf`` produced NaN; the
    frontier gate only checked ``score == -inf`` (every NaN comparison is
    False), so the metric-incomplete trial entered the frontier and — inserted
    first — ``max()`` could even SELECT it (NaN comparisons keep first place).
    Metric-incomplete / non-finite trials must be rejected BEFORE
    scalarization, and the frontier must admit finite scores only.
    """

    def _optimizer(self):
        return MultiObjectiveBatchOptimizer(
            configuration_space={"x": [1, 2]},
            objectives=["accuracy", "cost"],
            objective_weights={"accuracy": 1.0, "cost": 0.0},
        )

    def test_zero_weighted_missing_metric_trial_is_rejected(self):
        opt = self._optimizer()
        # cost is MISSING and carries weight 0.0 — the old sentinel path
        # scalarized 0.0 * inf into NaN; now the trial is rejected outright.
        assert opt._compose_trial_scores({"accuracy": 0.9}) is None

    def test_non_finite_metric_values_are_rejected(self):
        opt = self._optimizer()
        assert (
            opt._compose_trial_scores({"accuracy": 0.9, "cost": float("nan")}) is None
        )
        assert (
            opt._compose_trial_scores({"accuracy": 0.9, "cost": float("inf")}) is None
        )

    def test_complete_trial_composes_finite_score(self):
        opt = self._optimizer()
        composed = opt._compose_trial_scores({"accuracy": 0.9, "cost": 5.0})
        assert composed is not None
        scores, composite = composed
        assert scores == {"accuracy": 0.9, "cost": 5.0}
        assert math.isfinite(composite)

    def test_direct_frontier_insert_rejects_incomplete_mapping(self):
        # Completeness must be enforced at _update_pareto_frontier itself
        # (the choke point), not only in _compose_trial_scores: a direct
        # caller could insert a finite-but-INCOMPLETE mapping which is
        # non-comparable under _dominates and would sit on the frontier as
        # an unmeasured, falsely-attractive point.
        opt = self._optimizer()
        opt._update_pareto_frontier(_trial({"accuracy": 0.91}, score=0.91))
        assert opt.pareto_frontier == []

        complete = _trial({"accuracy": 0.90, "cost": 0.5}, score=0.9)
        opt._update_pareto_frontier(complete)
        opt._update_pareto_frontier(_trial({"accuracy": 0.99}, score=0.99))
        assert opt.pareto_frontier == [complete]

    def test_nan_scored_trial_cannot_enter_frontier_or_win(self):
        opt = self._optimizer()
        # Inserted FIRST — under the old ``== -inf`` gate this NaN trial
        # entered the frontier and max() kept it as the selected best.
        nan_trial = _trial({"accuracy": 0.9, "cost": float("nan")}, score=float("nan"))
        opt._update_pareto_frontier(nan_trial)
        assert opt.pareto_frontier == []

        good = _trial({"accuracy": 0.8, "cost": 5.0}, score=0.8)
        opt._update_pareto_frontier(good)
        assert opt.pareto_frontier == [good]
        assert opt._select_best_from_pareto() is good


class TestParetoFrontierTrimByCrowding:
    """Regression for #1942: size cap must prune by crowding, keep the extremes."""

    def test_cost_minimal_extreme_is_retained(self):
        opt = MultiObjectiveBatchOptimizer(
            configuration_space={"x": [1, 2]},
            objectives=["accuracy", "cost"],
            pareto_frontier_size=2,
        )
        # Three mutually non-dominated trials (accuracy maximize / cost minimize).
        # Score is highest for hi_acc so the old score-sort trim would drop cheap.
        opt._update_pareto_frontier(
            _trial({"accuracy": 0.95, "cost": 90.0}, score=0.95)
        )
        opt._update_pareto_frontier(
            _trial({"accuracy": 0.80, "cost": 40.0}, score=0.80)
        )
        opt._update_pareto_frontier(_trial({"accuracy": 0.60, "cost": 5.0}, score=0.60))

        assert len(opt.pareto_frontier) == 2
        costs = {t.metadata["objective_scores"]["cost"] for t in opt.pareto_frontier}
        # The cost-minimal extreme (5.0) must survive; the interior mid (40.0) is pruned.
        assert 5.0 in costs
        assert 40.0 not in costs
