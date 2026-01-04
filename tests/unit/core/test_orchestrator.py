"""Comprehensive tests for optimization orchestrator.

Tests cover:
- Normal optimization workflows
- Edge cases and boundary conditions
- Error handling and recovery scenarios
- Timeout and stopping conditions
- Progress tracking and state management
- Integration with optimizers and evaluators
- Performance considerations
- Code quality checks
- File versioning support (OptimizationLoggerV2)
- Objective schema integration
- Backward compatibility with legacy logger
"""

import asyncio
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from traigent.api.types import (
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)
from traigent.config.types import TraigentConfig
from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.core.orchestrator_helpers import allocate_parallel_ceilings
from traigent.evaluators.base import (
    BaseEvaluator,
    Dataset,
    EvaluationExample,
    EvaluationResult,
)
from traigent.optimizers.base import BaseOptimizer
from traigent.utils.exceptions import OptimizationError, TrialPrunedError
from traigent.utils.file_versioning import FileVersionManager


class MockOptimizer(BaseOptimizer):
    """Mock optimizer for testing."""

    def __init__(self, config_space: dict[str, Any], objectives: list[str], **kwargs):
        super().__init__(config_space, objectives, **kwargs)
        self.suggested_configs = []
        self.trial_results = []
        self._suggest_count = 0
        self._max_suggestions = 5
        self._should_stop = False

    def set_max_suggestions(self, max_suggestions: int):
        """Set maximum suggestions for testing."""
        self._max_suggestions = max_suggestions
        # If max is 0, stop immediately
        if max_suggestions == 0:
            self._should_stop = True

    def suggest_next_trial(self, history: list[TrialResult]) -> dict[str, Any]:
        """Suggest next configuration to evaluate."""
        config = {
            "param1": self._suggest_count,
            "param2": f"value_{self._suggest_count}",
        }
        self.suggested_configs.append(config)
        self._suggest_count += 1

        # Set stop flag after suggesting the last config
        if self._suggest_count >= self._max_suggestions:
            self._should_stop = True

        return config

    def should_stop(self, history: list[TrialResult]) -> bool:
        """Determine if optimization should stop."""
        return self._should_stop

    def suggest(self) -> dict[str, Any]:
        """Legacy method for backward compatibility."""
        return self.suggest_next_trial([])

    def tell(self, config: dict[str, Any], result: TrialResult) -> None:
        """Record trial result."""
        self.trial_results.append((config, result))

    def is_finished(self) -> bool:
        """Check if optimization should stop."""
        return self._should_stop

    def force_stop(self):
        """Force optimizer to stop."""
        self._should_stop = True


class MockEvaluator(BaseEvaluator):
    """Mock evaluator for testing."""

    def __init__(self, **kwargs):
        self.evaluation_count = 0
        self.should_fail = False
        self.evaluation_delay = 0.0
        self.metrics_to_return = {"accuracy": 0.5, "latency": 100}

    async def evaluate(
        self,
        func: Callable,
        config: dict[str, Any],
        dataset: Dataset,
        *,
        sample_lease=None,
        progress_callback: Callable[[int, dict[str, Any]], Any] | None = None,
        **_kwargs,
    ) -> EvaluationResult:
        """Evaluate configuration for testing."""
        self.evaluation_count += 1

        if self.evaluation_delay > 0:
            await asyncio.sleep(self.evaluation_delay)

        if self.should_fail:
            raise OptimizationError(f"Evaluation failed for config: {config}")

        processed_examples: list[EvaluationExample] = []
        budget_exhausted = False
        for index, example in enumerate(dataset.examples):
            if sample_lease and not sample_lease.try_take(1):
                budget_exhausted = True
                break
            processed_examples.append(example)
            if progress_callback:
                progress_callback(index, {"success": True})

        # Create metrics based on config for deterministic testing
        metrics = self.metrics_to_return.copy()
        if "param1" in config:
            metrics["accuracy"] = 0.5 + config["param1"] * 0.1
            metrics["latency"] = 100 - config["param1"] * 10

        total_examples = len(processed_examples)
        metrics["examples_attempted"] = total_examples

        # Return proper EvaluationResult as expected by BaseEvaluator
        result = EvaluationResult(
            config=config,
            aggregated_metrics=metrics,
            total_examples=total_examples,
            successful_examples=total_examples,
            duration=self.evaluation_delay or 0.1,
            metrics=metrics,  # For backward compatibility
            outputs=[f"output_{i}" for i in range(total_examples)],
            errors=[None for _ in range(total_examples)],
        )

        result.sample_budget_exhausted = budget_exhausted
        result.examples_consumed = total_examples
        return result

    def set_failure_mode(self, should_fail: bool):
        """Set whether evaluator should fail."""
        self.should_fail = should_fail

    def set_evaluation_delay(self, delay: float):
        """Set evaluation delay for timeout testing."""
        self.evaluation_delay = delay

    def set_metrics(self, metrics: dict[str, float]):
        """Set metrics to return."""
        self.metrics_to_return = metrics


class TestOptimizationOrchestrator:
    """Comprehensive tests for OptimizationOrchestrator."""

    @pytest.fixture
    def config_space(self):
        """Sample configuration space."""
        return {"param1": (0, 10), "param2": ["value_a", "value_b", "value_c"]}

    @pytest.fixture
    def objectives(self):
        """Sample objectives."""
        return ["accuracy", "latency"]

    @pytest.fixture
    def mock_optimizer(self, config_space, objectives):
        """Create mock optimizer."""
        return MockOptimizer(config_space, objectives)

    @pytest.fixture
    def mock_evaluator(self):
        """Create mock evaluator."""
        return MockEvaluator()

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset."""
        examples = [
            EvaluationExample({"query": "Hello"}, "Hi there!"),
            EvaluationExample({"query": "Goodbye"}, "See you later!"),
        ]
        return Dataset(examples, name="test_dataset", description="Test dataset")

    @pytest.fixture
    def objective_schema(self):
        """Create sample objective schema."""
        objectives = [
            ObjectiveDefinition(name="accuracy", orientation="maximize", weight=0.7),
            ObjectiveDefinition(name="cost", orientation="minimize", weight=0.3),
        ]
        return ObjectiveSchema.from_objectives(objectives)

    @pytest.fixture
    def orchestrator(self, mock_optimizer, mock_evaluator):
        """Create orchestrator with mocks."""
        # Mock the BackendIntegratedClient at its source location (lazy imported)
        with patch(
            "traigent.cloud.backend_client.BackendIntegratedClient"
        ) as mock_backend:
            # Setup mock backend client
            mock_client = MagicMock()
            mock_client.create_session.return_value = "test_session_123"
            mock_client.submit_result.return_value = True
            mock_backend.return_value = mock_client

            orchestrator = OptimizationOrchestrator(
                optimizer=mock_optimizer,
                evaluator=mock_evaluator,
                max_trials=10,
                timeout=30.0,
            )
            # Manually set the backend_client to our mock
            orchestrator.backend_client = mock_client
            return orchestrator

    @pytest.fixture
    def mock_function(self):
        """Create a mock function to optimize."""

        async def test_function(input_data: dict[str, Any], **config) -> Any:
            """Mock function that returns based on config."""
            # Simple mock behavior - just return the input query
            return input_data.get("query", "default response")

        return test_function

    # Constructor Tests

    def test_orchestrator_creation_with_valid_params(
        self, mock_optimizer, mock_evaluator
    ):
        """Test creating orchestrator with valid parameters."""
        orchestrator = OptimizationOrchestrator(
            optimizer=mock_optimizer,
            evaluator=mock_evaluator,
            max_trials=5,
            timeout=60.0,
        )

        assert orchestrator.optimizer is mock_optimizer
        assert orchestrator.evaluator is mock_evaluator
        assert orchestrator.max_trials == 5
        assert orchestrator.timeout == 60.0
        assert orchestrator.trial_count == 0
        assert orchestrator.best_result is None
        assert orchestrator.status == OptimizationStatus.PENDING

    def test_orchestrator_creation_with_none_params(
        self, mock_optimizer, mock_evaluator
    ):
        """Test creating orchestrator with None parameters."""
        orchestrator = OptimizationOrchestrator(
            optimizer=mock_optimizer,
            evaluator=mock_evaluator,
            max_trials=None,
            timeout=None,
        )

        assert orchestrator.max_trials is None
        assert orchestrator.timeout is None

    def test_orchestrator_creation_invalid_optimizer(self, mock_evaluator):
        """Test creating orchestrator with invalid optimizer."""
        with pytest.raises(TypeError):
            OptimizationOrchestrator(
                optimizer="not_an_optimizer", evaluator=mock_evaluator
            )

    def test_orchestrator_creation_invalid_evaluator(self, mock_optimizer):
        """Test creating orchestrator with invalid evaluator."""
        with pytest.raises(TypeError):
            OptimizationOrchestrator(
                optimizer=mock_optimizer, evaluator="not_an_evaluator"
            )

    def test_orchestrator_creation_negative_max_trials(
        self, mock_optimizer, mock_evaluator
    ):
        """Test creating orchestrator with negative max_trials."""
        with pytest.raises(ValueError):
            OptimizationOrchestrator(
                optimizer=mock_optimizer, evaluator=mock_evaluator, max_trials=-1
            )

    def test_orchestrator_creation_negative_timeout(
        self, mock_optimizer, mock_evaluator
    ):
        """Test creating orchestrator with negative timeout."""
        with pytest.raises(ValueError):
            OptimizationOrchestrator(
                optimizer=mock_optimizer, evaluator=mock_evaluator, timeout=-1.0
            )

    # Main Optimization Tests

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)  # Add timeout to prevent hanging
    async def test_optimize_successful_workflow(
        self, orchestrator, mock_function, sample_dataset
    ):
        """Test successful optimization workflow."""
        # Configure optimizer to suggest 3 configs
        orchestrator.optimizer.set_max_suggestions(3)

        result = await orchestrator.optimize(mock_function, sample_dataset)

        # Verify optimization completed successfully
        assert result.status == OptimizationStatus.COMPLETED
        assert len(result.trials) == 3
        assert result.best_config is not None
        assert result.best_metrics is not None
        assert len(result.trials) == 3

        # Verify all trials were evaluated
        assert orchestrator.evaluator.evaluation_count == 3
        assert len(orchestrator.optimizer.trial_results) == 3

        # Internal bookkeeping counters should reflect observed trials
        assert orchestrator._successful_trials == 3
        assert orchestrator._failed_trials == 0
        assert orchestrator.best_result is not None

        # Verify trial results are properly structured
        for trial_result in result.trials:
            assert trial_result.status == TrialStatus.COMPLETED
            assert trial_result.config is not None

    @pytest.mark.asyncio
    async def test_default_config_is_evaluated_first(
        self, mock_optimizer, mock_evaluator, mock_function, sample_dataset
    ):
        """Ensure default_config is used as the baseline trial."""
        default_config = {"param1": 99, "param2": "value_b"}
        mock_optimizer.set_max_suggestions(10)

        with patch(
            "traigent.cloud.backend_client.BackendIntegratedClient"
        ) as mock_backend:
            mock_client = MagicMock()
            mock_client.create_session.return_value = "session-default"
            mock_backend.return_value = mock_client

            orchestrator = OptimizationOrchestrator(
                optimizer=mock_optimizer,
                evaluator=mock_evaluator,
                max_trials=3,
                timeout=5.0,
                default_config=default_config,
            )
            orchestrator.backend_client = mock_client

            result = await orchestrator.optimize(mock_function, sample_dataset)

        assert len(result.trials) == 3
        assert result.trials[0].config == default_config
        assert len(mock_optimizer.suggested_configs) == 2

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_optimize_unbounded_parallel_does_not_overflow(
        self,
        mock_evaluator,
        sample_dataset,
        mock_function,
        config_space,
        objectives,
    ):
        """Ensure infinite trial budgets with parallelism avoid OverflowError."""
        mock_optimizer = MockOptimizer(config_space, objectives)
        mock_optimizer.set_max_suggestions(2)

        with patch(
            "traigent.cloud.backend_client.BackendIntegratedClient"
        ) as mock_backend:
            mock_client = MagicMock()
            mock_client.create_session.return_value = "session-unbounded"
            mock_backend.return_value = mock_client

            orchestrator = OptimizationOrchestrator(
                optimizer=mock_optimizer,
                evaluator=mock_evaluator,
                max_trials=None,
                timeout=30.0,
                parallel_trials=2,
            )
            orchestrator.backend_client = mock_client

            result = await orchestrator.optimize(mock_function, sample_dataset)

        assert result.status == OptimizationStatus.COMPLETED
        assert len(result.trials) == 2
        assert mock_evaluator.evaluation_count == 2
        assert orchestrator._successful_trials == 2
        assert orchestrator._failed_trials == 0

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_optimize_zero_max_trials_short_circuits(
        self,
        mock_optimizer,
        mock_evaluator,
        sample_dataset,
        mock_function,
    ):
        """Ensure max_trials=0 exits without executing trials."""
        mock_optimizer.set_max_suggestions(5)

        with patch(
            "traigent.cloud.backend_client.BackendIntegratedClient"
        ) as mock_backend:
            mock_client = MagicMock()
            mock_client.create_session.return_value = "session-zero"
            mock_backend.return_value = mock_client

            orchestrator = OptimizationOrchestrator(
                optimizer=mock_optimizer,
                evaluator=mock_evaluator,
                max_trials=0,
                timeout=5.0,
            )
            orchestrator.backend_client = mock_client

            result = await orchestrator.optimize(mock_function, sample_dataset)

        assert result.status == OptimizationStatus.COMPLETED
        assert len(result.trials) == 0
        assert orchestrator._successful_trials == 0
        assert orchestrator._failed_trials == 0
        assert orchestrator._stop_reason == "max_trials_reached"

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_optimize_stops_when_sample_budget_reached(
        self,
        mock_evaluator,
        sample_dataset,
        mock_function,
        config_space,
        objectives,
    ):
        """Verify that max_total_examples halts optimization once exhausted."""
        mock_optimizer = MockOptimizer(config_space, objectives)
        mock_optimizer.set_max_suggestions(10)

        with patch(
            "traigent.cloud.backend_client.BackendIntegratedClient"
        ) as mock_backend:
            mock_client = MagicMock()
            mock_client.create_session.return_value = "session-sample-cap"
            mock_backend.return_value = mock_client

            orchestrator = OptimizationOrchestrator(
                optimizer=mock_optimizer,
                evaluator=mock_evaluator,
                max_trials=10,
                max_total_examples=3,  # dataset has 2 examples per trial
                timeout=10.0,
            )
            orchestrator.backend_client = mock_client

            result = await orchestrator.optimize(mock_function, sample_dataset)

        assert len(result.trials) == 2  # two trials -> 4 examples attempted
        assert orchestrator._stop_reason == "max_samples_reached"
        assert orchestrator._consumed_examples >= 3
        assert mock_evaluator.evaluation_count == 2

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_sample_budget_excludes_pruned_trials_when_requested(
        self,
        sample_dataset,
        mock_function,
        config_space,
        objectives,
    ):
        """Pruned trials should be ignored if samples_include_pruned=False."""

        class PruningEvaluator(MockEvaluator):
            def __init__(self):
                super().__init__()
                self._pruned_once = False

            async def evaluate(
                self,
                func: Callable,
                config: dict[str, Any],
                dataset: Dataset,
                *,
                sample_lease=None,
                progress_callback: Callable[[int, dict[str, Any]], Any] | None = None,
                **kwargs,
            ) -> EvaluationResult:
                if not self._pruned_once:
                    self._pruned_once = True
                    self.evaluation_count += 1
                    raise TrialPrunedError("pruned", step=1)
                return await super().evaluate(
                    func,
                    config,
                    dataset,
                    sample_lease=sample_lease,
                    progress_callback=progress_callback,
                    **kwargs,
                )

        pruned_evaluator = PruningEvaluator()
        mock_optimizer = MockOptimizer(config_space, objectives)
        mock_optimizer.set_max_suggestions(10)

        with patch(
            "traigent.cloud.backend_client.BackendIntegratedClient"
        ) as mock_backend:
            mock_client = MagicMock()
            mock_client.create_session.return_value = "session-sample-cap-exclude"
            mock_backend.return_value = mock_client

            orchestrator = OptimizationOrchestrator(
                optimizer=mock_optimizer,
                evaluator=pruned_evaluator,
                max_trials=10,
                max_total_examples=2,
                timeout=10.0,
                samples_include_pruned=False,
            )
            orchestrator.backend_client = mock_client

            result = await orchestrator.optimize(mock_function, sample_dataset)

        # First trial pruned, second trial successful, then budget reached.
        assert len(result.trials) == 2
        assert any(trial.status == TrialStatus.PRUNED for trial in result.trials)
        assert orchestrator._consumed_examples == 2
        assert orchestrator._stop_reason == "max_samples_reached"
        assert pruned_evaluator.evaluation_count == 2

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_optimize_with_max_trials_limit(
        self, orchestrator, mock_function, sample_dataset
    ):
        """Test optimization stops at max_trials limit."""
        # Set max trials to 2, but optimizer can suggest more
        orchestrator.max_trials = 2
        orchestrator.optimizer.set_max_suggestions(10)

        result = await orchestrator.optimize(mock_function, sample_dataset)

        assert result.status == OptimizationStatus.COMPLETED
        assert len(result.trials) == 2
        assert orchestrator.evaluator.evaluation_count == 2

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_optimize_with_timeout(
        self, orchestrator, mock_function, sample_dataset
    ):
        """Test optimization stops due to timeout."""
        # Set short timeout and slow evaluation
        orchestrator.timeout = 0.5
        orchestrator.evaluator.set_evaluation_delay(0.3)
        orchestrator.optimizer.set_max_suggestions(10)

        result = await orchestrator.optimize(mock_function, sample_dataset)

        # Should complete some trials but timeout before finishing all
        assert result.status in [
            OptimizationStatus.CANCELLED,
            OptimizationStatus.COMPLETED,
        ]
        assert len(result.trials) >= 1  # At least one trial should complete
        assert result.duration >= 0.5  # Should take at least timeout duration

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_optimize_optimizer_suggests_no_configs(
        self, orchestrator, mock_function, sample_dataset
    ):
        """Test optimization when optimizer suggests no configurations."""
        # Configure optimizer to suggest 0 configs
        orchestrator.optimizer.set_max_suggestions(0)

        result = await orchestrator.optimize(mock_function, sample_dataset)

        assert result.status == OptimizationStatus.COMPLETED
        assert len(result.trials) == 0
        assert result.best_config == {}
        assert result.best_metrics == {}
        assert result.best_score == 0.0

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_optimize_optimizer_early_stop(
        self, orchestrator, mock_function, sample_dataset
    ):
        """Test optimization when optimizer decides to stop early."""
        # Configure optimizer to stop after 2 suggestions
        orchestrator.optimizer.set_max_suggestions(2)

        result = await orchestrator.optimize(mock_function, sample_dataset)

        assert result.status == OptimizationStatus.COMPLETED
        assert len(result.trials) == 2

    # Error Handling Tests

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_optimize_evaluator_failure_single_trial(
        self, orchestrator, mock_function, sample_dataset
    ):
        """Test handling evaluator failure in single trial."""
        # Configure to fail on first evaluation
        orchestrator.evaluator.set_failure_mode(True)
        orchestrator.optimizer.set_max_suggestions(3)

        result = await orchestrator.optimize(mock_function, sample_dataset)

        # All trials should fail
        assert (
            result.status == OptimizationStatus.COMPLETED
        )  # Optimization completes even if all fail
        assert len(result.trials) == 3  # All trials are recorded, even if failed
        assert result.best_config == {}  # No successful config

        # Check that failed trials are recorded
        failed_trials = [t for t in result.trials if t.status == TrialStatus.FAILED]
        assert len(failed_trials) == 3  # All 3 trials failed

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_optimize_evaluator_intermittent_failures(
        self, orchestrator, mock_function, sample_dataset
    ):
        """Test handling intermittent evaluator failures."""
        orchestrator.optimizer.set_max_suggestions(5)

        # Mock evaluator to fail on specific trials
        original_evaluate = orchestrator.evaluator.evaluate
        evaluation_count = 0

        async def failing_evaluate(func, config, dataset):
            nonlocal evaluation_count
            evaluation_count += 1
            if evaluation_count in [2, 4]:  # Fail on 2nd and 4th evaluations
                raise OptimizationError(f"Evaluation {evaluation_count} failed")
            return await original_evaluate(func, config, dataset)

        orchestrator.evaluator.evaluate = failing_evaluate

        result = await orchestrator.optimize(mock_function, sample_dataset)

        # Should have some successful and some failed trials
        successful_trials = [
            t for t in result.trials if t.status == TrialStatus.COMPLETED
        ]
        failed_trials = [t for t in result.trials if t.status == TrialStatus.FAILED]

        assert len(successful_trials) >= 1
        assert len(failed_trials) == 2
        assert result.best_config is not None  # Should find best from successful trials

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_optimize_optimizer_exception(
        self, orchestrator, mock_function, sample_dataset
    ):
        """Test handling optimizer exceptions."""

        # Mock optimizer to throw exception
        def failing_suggest(history):
            raise OptimizationError("Optimizer failure")

        orchestrator.optimizer.suggest_next_trial = failing_suggest

        with pytest.raises(OptimizationError):
            await orchestrator.optimize(mock_function, sample_dataset)

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_optimize_dataset_none(self, orchestrator, mock_function):
        """Test optimization with None dataset."""
        with pytest.raises(ValueError, match="Dataset cannot be None"):
            await orchestrator.optimize(mock_function, None)

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_optimize_empty_dataset(self, orchestrator, mock_function):
        """Test optimization with empty dataset."""
        from traigent.evaluators.base import Dataset

        empty_dataset = Dataset([], name="empty", description="Empty dataset")

        with pytest.raises(ValueError, match="Dataset cannot be empty"):
            await orchestrator.optimize(mock_function, empty_dataset)

    # Progress Tracking Tests

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_progress_tracking_during_optimization(
        self, orchestrator, mock_function, sample_dataset
    ):
        """Test that progress is tracked correctly during optimization."""
        orchestrator.max_trials = 5
        orchestrator.optimizer.set_max_suggestions(5)

        # Track progress during optimization
        progress_values = []

        original_run_trial = orchestrator._trial_lifecycle.run_trial

        async def tracking_run_trial(*args, **kwargs):
            result = await original_run_trial(*args, **kwargs)
            progress_values.append(orchestrator.progress)
            return result

        orchestrator._trial_lifecycle.run_trial = tracking_run_trial

        await orchestrator.optimize(mock_function, sample_dataset)

        # Verify progress increased monotonically
        assert len(progress_values) == 5
        for i in range(1, len(progress_values)):
            assert progress_values[i] >= progress_values[i - 1]
        # Progress might be 0.8 (4/5) or 1.0 depending on when it's checked
        assert progress_values[-1] >= 0.8  # At least 80% complete
        # Check final progress after optimization
        assert orchestrator.progress == 1.0  # Should be 100% after completion

    def test_progress_calculation_edge_cases(self, orchestrator):
        """Test progress calculation edge cases."""
        from datetime import datetime

        # Create some dummy trials
        for i in range(5):
            trial = TrialResult(
                trial_id=f"test_{i}",
                config={},
                metrics={},
                status=TrialStatus.COMPLETED,
                duration=0.1,
                timestamp=datetime.now(),
            )
            orchestrator._trials.append(trial)

        # No max trials set
        orchestrator.max_trials = None
        assert orchestrator.progress == 0.0

        # Zero max trials
        orchestrator.max_trials = 0
        orchestrator._trials = []
        assert orchestrator.progress == 1.0

        # More trials than max (shouldn't happen but test robustness)
        orchestrator.max_trials = 3
        orchestrator._trials = [trial] * 5  # 5 trials
        assert orchestrator.progress == 1.0

    # State Management Tests

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_optimization_status_transitions(
        self, orchestrator, mock_function, sample_dataset
    ):
        """Test that optimization status transitions correctly."""
        # Initially not started
        assert orchestrator.status == OptimizationStatus.PENDING

        orchestrator.optimizer.set_max_suggestions(2)

        # Track status during optimization
        status_history = [orchestrator.status]

        original_run_trial = orchestrator._trial_lifecycle.run_trial

        async def tracking_run_trial(*args, **kwargs):
            if orchestrator.status not in status_history:
                status_history.append(orchestrator.status)
            return await original_run_trial(*args, **kwargs)

        orchestrator._trial_lifecycle.run_trial = tracking_run_trial

        await orchestrator.optimize(mock_function, sample_dataset)
        status_history.append(orchestrator.status)

        # Should transition: PENDING -> RUNNING -> COMPLETED
        assert OptimizationStatus.PENDING in status_history
        assert OptimizationStatus.RUNNING in status_history
        assert status_history[-1] == OptimizationStatus.COMPLETED

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_best_result_tracking(
        self, orchestrator, mock_function, sample_dataset
    ):
        """Test that best result is tracked correctly."""
        orchestrator.optimizer.set_max_suggestions(3)

        # Configure evaluator to return different metrics
        evaluation_count = 0
        original_evaluate = orchestrator.evaluator.evaluate

        async def varying_evaluate(func, config, dataset):
            nonlocal evaluation_count
            evaluation_count += 1

            # Return increasing accuracy values
            result = await original_evaluate(func, config, dataset)
            result.metrics["accuracy"] = 0.1 * evaluation_count  # 0.1, 0.2, 0.3
            return result

        orchestrator.evaluator.evaluate = varying_evaluate

        result = await orchestrator.optimize(mock_function, sample_dataset)

        # Best result should have highest accuracy (0.3)
        assert abs(result.best_metrics["accuracy"] - 0.3) < 1e-9
        assert abs(orchestrator.best_result.metrics["accuracy"] - 0.3) < 1e-9

    # Timeout Handling Tests

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_timeout_during_evaluation(
        self, orchestrator, mock_function, sample_dataset
    ):
        """Test timeout during long evaluation."""
        orchestrator.timeout = 0.2
        orchestrator.evaluator.set_evaluation_delay(0.5)  # Longer than timeout
        orchestrator.optimizer.set_max_suggestions(1)

        start_time = time.time()
        result = await orchestrator.optimize(mock_function, sample_dataset)
        duration = time.time() - start_time

        # Should timeout reasonably close to expected time
        assert duration < 1.0  # More lenient timeout check for CI environments
        assert result.status == OptimizationStatus.CANCELLED

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)  # Generous timeout for CI
    async def test_timeout_precision(self, orchestrator, mock_function, sample_dataset):
        """Test that timeout is enforced with reasonable precision.

        Note: This test uses generous tolerances because CI environments have
        variable load that can cause timing inconsistencies. We primarily verify
        that the timeout mechanism works, not exact timing precision.
        """
        timeout_duration = 0.5  # Longer timeout for more reliable measurement
        orchestrator.timeout = timeout_duration
        orchestrator.evaluator.set_evaluation_delay(0.1)  # Short evaluation
        orchestrator.optimizer.set_max_suggestions(10)  # Many trials

        start_time = time.time()
        result = await orchestrator.optimize(mock_function, sample_dataset)
        actual_duration = time.time() - start_time

        # Should timeout within reasonable bounds (generous tolerance for CI)
        # The key assertion is that we stopped reasonably close to timeout,
        # not that we hit it exactly
        assert (
            actual_duration < timeout_duration + 1.0
        ), f"Timeout took too long: {actual_duration:.2f}s (expected ~{timeout_duration}s)"
        assert (
            actual_duration >= timeout_duration * 0.5
        ), f"Finished too quickly: {actual_duration:.2f}s (expected ~{timeout_duration}s)"
        assert result.status == OptimizationStatus.CANCELLED

    # Performance Tests

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)  # Longer timeout for performance test
    async def test_optimization_memory_efficiency(
        self, orchestrator, mock_function, sample_dataset
    ):
        """Test that optimization doesn't leak memory with many trials."""
        import gc
        import os

        import psutil

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Run optimization with many trials
        orchestrator.max_trials = 50
        orchestrator.optimizer.set_max_suggestions(50)
        orchestrator.evaluator.set_evaluation_delay(0.01)  # Fast evaluation

        result = await orchestrator.optimize(mock_function, sample_dataset)

        # Force garbage collection
        gc.collect()

        # Check memory usage after optimization
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (< 50MB for 50 trials)
        assert memory_increase < 50 * 1024 * 1024
        assert len(result.trials) == 50

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)  # Longer timeout for concurrent test
    async def test_concurrent_optimization_safety(
        self, config_space, objectives, sample_dataset
    ):
        """Test that multiple optimizations can run concurrently safely."""
        # Create multiple orchestrators with mocked backend
        orchestrators = []
        with patch(
            "traigent.cloud.backend_client.BackendIntegratedClient"
        ) as mock_backend:
            # Setup mock backend client factory - each orchestrator gets its own mock
            def create_mock_client(*args, **kwargs):
                mock_client = MagicMock()
                # Each orchestrator gets a unique session ID to ensure unique trial IDs
                unique_session_id = f"test_session_{uuid.uuid4().hex[:8]}"
                mock_client.create_session.return_value = unique_session_id
                mock_client.submit_result.return_value = True
                return mock_client

            # Make the mock backend return a new client instance each time
            mock_backend.side_effect = create_mock_client

            for _i in range(3):
                optimizer = MockOptimizer(config_space, objectives)
                optimizer.set_max_suggestions(5)
                evaluator = MockEvaluator()
                evaluator.set_evaluation_delay(0.05)

                orchestrator = OptimizationOrchestrator(
                    optimizer=optimizer, evaluator=evaluator, max_trials=5
                )
                # Each orchestrator gets its own unique mock client
                # No need to manually set backend_client since it's created in __init__
                orchestrators.append(orchestrator)

        # Run optimizations concurrently
        # Create a mock function for each orchestrator to use
        async def mock_func(input_data: dict[str, Any], **config) -> Any:
            return input_data.get("query", "default")

        tasks = [orch.optimize(mock_func, sample_dataset) for orch in orchestrators]
        results = await asyncio.gather(*tasks)

        # All should complete successfully
        for result in results:
            assert result.status == OptimizationStatus.COMPLETED
            assert len(result.trials) == 5

        # Results should be independent (different trial IDs)
        all_trial_ids = set()
        for result in results:
            for trial in result.trials:
                assert trial.trial_id not in all_trial_ids
                all_trial_ids.add(trial.trial_id)

    # Edge Cases and Boundary Conditions

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_optimization_with_zero_timeout(
        self, orchestrator, mock_function, sample_dataset
    ):
        """Test optimization with zero timeout."""
        orchestrator.timeout = 0.0

        result = await orchestrator.optimize(mock_function, sample_dataset)

        # Should timeout immediately
        assert result.status == OptimizationStatus.CANCELLED
        assert len(result.trials) == 0

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_optimization_very_large_max_trials(
        self, orchestrator, mock_function, sample_dataset
    ):
        """Test optimization with very large max_trials."""
        orchestrator.max_trials = 1000000
        orchestrator.optimizer.set_max_suggestions(2)  # But optimizer stops early

        result = await orchestrator.optimize(mock_function, sample_dataset)

        # Should complete when optimizer finishes, not max_trials
        assert result.status == OptimizationStatus.COMPLETED
        assert len(result.trials) == 2

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_optimization_single_trial(
        self, orchestrator, mock_function, sample_dataset
    ):
        """Test optimization with exactly one trial."""
        orchestrator.max_trials = 1
        orchestrator.optimizer.set_max_suggestions(1)

        result = await orchestrator.optimize(mock_function, sample_dataset)

        assert result.status == OptimizationStatus.COMPLETED
        assert len(result.trials) == 1
        assert result.best_config is not None
        assert len(result.trials) == 1

    # Integration Tests

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_orchestrator_optimizer_integration(
        self, orchestrator, mock_function, sample_dataset
    ):
        """Test integration between orchestrator and optimizer."""
        orchestrator.optimizer.set_max_suggestions(3)

        result = await orchestrator.optimize(mock_function, sample_dataset)

        # Verify optimizer was called correctly
        assert len(orchestrator.optimizer.suggested_configs) == 3
        assert len(orchestrator.optimizer.trial_results) == 3

        # Verify all suggested configs were evaluated
        suggested_configs = {
            str(config) for config in orchestrator.optimizer.suggested_configs
        }
        evaluated_configs = {str(trial.config) for trial in result.trials}
        assert suggested_configs == evaluated_configs

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_orchestrator_evaluator_integration(
        self, orchestrator, mock_function, sample_dataset
    ):
        """Test integration between orchestrator and evaluator."""
        orchestrator.optimizer.set_max_suggestions(2)

        result = await orchestrator.optimize(mock_function, sample_dataset)

        # Verify evaluator was called for each suggested config
        assert orchestrator.evaluator.evaluation_count == 2

        # Verify all evaluations received correct dataset
        for trial in result.trials:
            assert trial.status == TrialStatus.COMPLETED
            assert "accuracy" in trial.metrics
            assert "latency" in trial.metrics

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)  # Longer timeout for end-to-end test
    async def test_end_to_end_optimization_workflow(
        self, config_space, objectives, mock_function, sample_dataset
    ):
        """Test complete end-to-end optimization workflow."""
        # Create fresh instances for integration test
        optimizer = MockOptimizer(config_space, objectives)
        optimizer.set_max_suggestions(5)

        evaluator = MockEvaluator()
        evaluator.set_metrics({"accuracy": 0.8, "f1_score": 0.75, "precision": 0.85})

        # Mock the backend to prevent connection issues
        with patch(
            "traigent.cloud.backend_client.BackendIntegratedClient"
        ) as mock_backend:
            mock_client = MagicMock()
            mock_client.create_session.return_value = "test_session"
            mock_client.submit_result.return_value = True
            mock_backend.return_value = mock_client

            orchestrator = OptimizationOrchestrator(
                optimizer=optimizer, evaluator=evaluator, max_trials=5, timeout=10.0
            )
            orchestrator.backend_client = mock_client

            # Run full optimization
            result = await orchestrator.optimize(mock_function, sample_dataset)

        # Comprehensive verification
        assert result.status == OptimizationStatus.COMPLETED
        assert len(result.trials) == 5
        assert result.duration > 0
        assert result.best_config is not None
        assert result.best_metrics is not None
        assert len(result.trials) == 5

        # Verify result structure
        assert "accuracy" in result.best_metrics
        assert "f1_score" in result.best_metrics
        assert "precision" in result.best_metrics

        # Verify all trials are properly formed
        for i, trial in enumerate(result.trials):
            assert trial.trial_id is not None
            assert trial.config is not None
            assert trial.metrics is not None
            assert trial.status == TrialStatus.COMPLETED
            assert trial.duration > 0
            assert trial.error_message is None

            # Verify config matches what optimizer suggested
            assert trial.config == optimizer.suggested_configs[i]

    # File Versioning and Logger V2 Tests

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_initialization_with_versioned_logger(
        self, mock_optimizer, mock_evaluator, objective_schema
    ):
        """Test orchestrator initialization with versioned logger."""
        config = TraigentConfig(execution_mode="edge_analytics")

        orchestrator = OptimizationOrchestrator(
            optimizer=mock_optimizer,
            evaluator=mock_evaluator,
            max_trials=10,
            config=config,
            objective_schema=objective_schema,
            use_versioned_logger=True,
            file_version="2",
        )

        assert orchestrator.use_versioned_logger is True
        assert orchestrator.file_version == "2"
        assert orchestrator._logger_v2 is None  # Not initialized yet

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_initialization_with_legacy_logger(
        self, mock_optimizer, mock_evaluator
    ):
        """Test orchestrator initialization with legacy logger."""
        config = TraigentConfig(execution_mode="edge_analytics")

        orchestrator = OptimizationOrchestrator(
            optimizer=mock_optimizer,
            evaluator=mock_evaluator,
            max_trials=10,
            config=config,
            use_versioned_logger=False,
        )

        assert orchestrator.use_versioned_logger is False
        assert orchestrator._logger_v2 is None

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_optimize_with_legacy_logger(
        self, mock_optimizer, mock_evaluator, sample_dataset, objective_schema
    ):
        """Test optimization with unified legacy logger."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TraigentConfig(
                execution_mode="edge_analytics", local_storage_path=tmpdir
            )

            orchestrator = OptimizationOrchestrator(
                optimizer=mock_optimizer,
                evaluator=mock_evaluator,
                max_trials=2,
                timeout=10.0,
                config=config,
                objective_schema=objective_schema,
                use_versioned_logger=False,
            )

            # Mock backend client to provide session ID
            with patch.object(orchestrator, "backend_client") as mock_backend:
                mock_backend.create_session.return_value = "test_session_123"
                mock_backend.submit_result.return_value = True

                # Configure optimizer
                orchestrator.optimizer.set_max_suggestions(2)

                # Run optimization
                async def test_func(input_data, **config):
                    return input_data.get("query", "response")

                await orchestrator.optimize(
                    func=test_func,
                    dataset=sample_dataset,
                    function_name="test_function",
                )

                # Check legacy logger was initialized
                assert orchestrator._logger is not None
                run_path = orchestrator._logger.run_path
                file_manager = FileVersionManager(version="2")
                legacy_manager = FileVersionManager(use_legacy=True)

                def _has_file(subdir: str, file_type: str) -> bool:
                    candidates = [
                        run_path / subdir / file_manager.get_filename(file_type),
                        run_path / subdir / legacy_manager.get_filename(file_type),
                    ]
                    return any(path.exists() for path in candidates)

                assert _has_file("meta", "session")
                assert _has_file("meta", "config")

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_log_trial_with_legacy_logger(
        self, mock_optimizer, mock_evaluator, objective_schema
    ):
        """Test logging trial with legacy logger."""
        config = TraigentConfig(execution_mode="edge_analytics")

        orchestrator = OptimizationOrchestrator(
            optimizer=mock_optimizer,
            evaluator=mock_evaluator,
            config=config,
            objective_schema=objective_schema,
            use_versioned_logger=False,
        )

        # Create mock legacy logger
        orchestrator._logger = MagicMock()

        # Create trial result
        trial_result = TrialResult(
            trial_id="trial_001",
            config={"model": "gpt-4"},
            metrics={"accuracy": 0.9},
            status=TrialStatus.COMPLETED,
            duration=5.0,
            timestamp=None,
        )

        # Log trial
        orchestrator._log_trial(trial_result)

        # Check legacy logger was called
        orchestrator._logger.log_trial_result.assert_called_once_with(trial_result)

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_log_checkpoint_with_legacy_logger(
        self, mock_optimizer, mock_evaluator
    ):
        """Test logging checkpoint with legacy logger."""
        config = TraigentConfig(execution_mode="edge_analytics")

        orchestrator = OptimizationOrchestrator(
            optimizer=mock_optimizer,
            evaluator=mock_evaluator,
            config=config,
            use_versioned_logger=False,
        )

        # Create mock legacy logger with save_checkpoint method
        orchestrator._logger = MagicMock()
        orchestrator._logger.save_checkpoint = MagicMock()

        # Log checkpoint with correct parameters for legacy logger
        optimizer_state = {"iteration": 5, "best_score": 0.85}
        trials_history = []
        orchestrator._log_checkpoint(
            optimizer_state, trials_history=trials_history, trial_count=5
        )

        # Check legacy logger was called with correct parameters
        orchestrator._logger.save_checkpoint.assert_called_once_with(
            optimizer_state=optimizer_state,
            trials_history=trials_history,
            trial_count=5,
        )

    def test_file_versioning_integration(self):
        """Test that file versioning creates expected files."""
        import tempfile
        from pathlib import Path

        from traigent.utils.optimization_logger import OptimizationLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)

            # Create logger
            logger = OptimizationLogger(
                experiment_name="test_exp",
                session_id="test_123",
                execution_mode="edge_analytics",
                base_path=temp_dir,
                buffer_size=1,
            )

            # Log some data
            logger.log_session_start(
                {"test": "config"}, objectives=["accuracy"], algorithm="test"
            )

            # Check versioned files exist
            run_path = logger.run_path
            file_manager = FileVersionManager(version="2")
            assert (run_path / "meta" / file_manager.get_filename("session")).exists()
            assert (run_path / "meta" / file_manager.get_filename("config")).exists()

            # Check version info exists
            # No version info in legacy logger

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_optimize_with_objective_schema(
        self, mock_optimizer, mock_evaluator, sample_dataset, objective_schema
    ):
        """Test optimization with objective schema."""
        config = TraigentConfig(execution_mode="edge_analytics")

        orchestrator = OptimizationOrchestrator(
            optimizer=mock_optimizer,
            evaluator=mock_evaluator,
            max_trials=2,
            config=config,
            objective_schema=objective_schema,
            use_versioned_logger=True,
        )

        # Mock backend
        with patch.object(orchestrator, "backend_client") as mock_backend:
            mock_backend.create_session.return_value = "test_session_123"
            mock_backend.submit_result.return_value = True

            orchestrator.optimizer.set_max_suggestions(2)

            async def test_func(input_data, **config):
                return input_data.get("query", "response")

            result = await orchestrator.optimize(
                func=test_func,
                dataset=sample_dataset,
                function_name="test_function",
            )

            # Verify objective schema is used
            assert orchestrator.objective_schema == objective_schema
            assert result.status == OptimizationStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_plateau_stop_triggers_early(self, sample_dataset):
        """Optimization stops when improvements plateau for configured window."""

        config_space = {"param1": (0, 1)}
        optimizer = MockOptimizer(config_space, ["accuracy"])
        optimizer.set_max_suggestions(20)

        # Force optimizer to keep suggesting the same configuration
        constant_config = {"param1": 0, "param2": "value_0"}

        def constant_suggest(_history):
            return constant_config.copy()

        optimizer.suggest_next_trial = constant_suggest  # type: ignore[assignment]

        class ConstantEvaluator(BaseEvaluator):
            async def evaluate(
                self,
                func,
                config,
                dataset,
                *,
                sample_lease=None,
                progress_callback=None,
            ) -> EvaluationResult:
                allowed = 0
                for idx, _example in enumerate(dataset.examples):
                    if sample_lease and not sample_lease.try_take(1):
                        break
                    allowed += 1
                    if progress_callback is not None:
                        progress_callback(
                            idx,
                            {
                                "success": True,
                                "metrics": {"accuracy": 0.6},
                                "output": "stub",
                            },
                        )

                metrics = {"accuracy": 0.6, "examples_attempted": allowed}

                result = EvaluationResult(
                    config=config,
                    example_results=[],
                    aggregated_metrics=metrics,
                    total_examples=allowed,
                    successful_examples=allowed,
                    duration=0.01,
                    metrics=metrics,
                )
                result.sample_budget_exhausted = bool(sample_lease) and allowed < len(
                    dataset.examples
                )
                result.examples_consumed = allowed
                return result

        evaluator = ConstantEvaluator(metrics=["accuracy"])

        orchestrator = OptimizationOrchestrator(
            optimizer=optimizer,
            evaluator=evaluator,
            max_trials=20,
            config=TraigentConfig.edge_analytics_mode(),
            objectives=["accuracy"],
            plateau_window=3,
            plateau_epsilon=0.0,
        )

        result = await orchestrator.optimize(lambda _: "stub", sample_dataset)

        assert orchestrator._stop_reason == "plateau"
        assert result.stop_reason == "plateau"  # Verify public API
        assert len(result.trials) <= 4

    @pytest.mark.asyncio
    async def test_budget_stop_condition(self, sample_dataset):
        """Budget stop condition halts optimization once limit is reached."""

        config_space = {"param1": (0, 1)}
        optimizer = MockOptimizer(config_space, ["total_cost"])
        optimizer.set_max_suggestions(20)

        class FixedCostEvaluator(BaseEvaluator):
            async def evaluate(
                self,
                func,
                config,
                dataset,
                *,
                sample_lease=None,
                progress_callback=None,
            ) -> EvaluationResult:
                allowed = 0
                for idx, _example in enumerate(dataset.examples):
                    if sample_lease and not sample_lease.try_take(1):
                        break
                    allowed += 1
                    if progress_callback is not None:
                        progress_callback(
                            idx,
                            {
                                "success": True,
                                "metrics": {"total_cost": 0.06},
                                "output": "ok",
                            },
                        )

                metrics = {"total_cost": 0.06, "examples_attempted": allowed}
                result = EvaluationResult(
                    config=config,
                    example_results=[],
                    aggregated_metrics=metrics,
                    total_examples=allowed,
                    successful_examples=allowed,
                    duration=0.01,
                    metrics=metrics,
                )
                result.sample_budget_exhausted = bool(sample_lease) and allowed < len(
                    dataset.examples
                )
                result.examples_consumed = allowed
                return result

        evaluator = FixedCostEvaluator(metrics=["total_cost"])

        orchestrator = OptimizationOrchestrator(
            optimizer=optimizer,
            evaluator=evaluator,
            max_trials=20,
            config=TraigentConfig.edge_analytics_mode(),
            objectives=["total_cost"],
            budget_limit=0.15,
        )

        result = await orchestrator.optimize(lambda _: "ok", sample_dataset)

        assert orchestrator._stop_reason == "cost_limit"
        total_cost = sum(tr.metrics.get("total_cost", 0.0) for tr in result.trials)
        assert total_cost >= 0.15
        assert len(result.trials) <= 4

    def test_abandon_optuna_trial_prunes_when_supported(self, orchestrator):
        orchestrator.optimizer.report_trial_pruned = MagicMock()

        orchestrator._abandon_optuna_trial(123, reason="trial_skipped_for_test")

        orchestrator.optimizer.report_trial_pruned.assert_called_once_with(123, 0)

    def test_abandon_optuna_trial_logs_to_trials_list(self, orchestrator):
        """Verify abandoned Optuna trials are recorded in the trials list."""
        orchestrator.optimizer.report_trial_pruned = MagicMock()
        initial_count = len(orchestrator._trials)

        orchestrator._abandon_optuna_trial(
            456, reason="budget_exhausted", pruned_step=3
        )

        # Trial should be appended to the list
        assert len(orchestrator._trials) == initial_count + 1
        abandoned_trial = orchestrator._trials[-1]
        assert abandoned_trial.trial_id == "optuna_456"
        assert abandoned_trial.status == TrialStatus.PRUNED
        assert abandoned_trial.error_message == "budget_exhausted"
        assert abandoned_trial.metadata["pruned_step"] == 3
        assert abandoned_trial.metadata["stop_reason"] == "budget_exhausted"

    def test_abandon_optuna_trial_includes_config_when_provided(self, orchestrator):
        """Verify abandoned trials include the config when provided."""
        orchestrator.optimizer.report_trial_pruned = MagicMock()
        test_config = {"model": "gpt-4", "temperature": 0.7}

        orchestrator._abandon_optuna_trial(
            789,
            reason="cache_policy_skip",
            config=test_config,
            status=TrialStatus.PRUNED,
        )

        abandoned_trial = orchestrator._trials[-1]
        assert abandoned_trial.config == test_config
        assert abandoned_trial.config["model"] == "gpt-4"
        assert abandoned_trial.metadata["abandoned"] is True

    def test_abandon_optuna_trial_does_not_count_toward_max_trials(self, orchestrator):
        """Verify abandoned trials don't trigger max_trials stop condition."""
        from traigent.core.stop_conditions import MaxTrialsStopCondition

        orchestrator.optimizer.report_trial_pruned = MagicMock()
        stop_condition = MaxTrialsStopCondition(max_trials=3)

        # Add 2 completed trials
        for i in range(2):
            trial = TrialResult(
                trial_id=f"completed_{i}",
                config={},
                metrics={},
                status=TrialStatus.COMPLETED,
                duration=0.1,
                timestamp=None,
            )
            orchestrator._trials.append(trial)

        # Add 5 abandoned/pruned trials - these should NOT count
        for i in range(5):
            orchestrator._abandon_optuna_trial(
                i, reason="cache_policy_skip", status=TrialStatus.PRUNED
            )

        # Should NOT stop because only 2 executed trials (< 3 max_trials)
        assert not stop_condition.should_stop(orchestrator._trials)

        # Add one more completed trial (now 3 total)
        trial = TrialResult(
            trial_id="completed_2",
            config={},
            metrics={},
            status=TrialStatus.COMPLETED,
            duration=0.1,
            timestamp=None,
        )
        orchestrator._trials.append(trial)

        # NOW should stop because 3 executed trials == max_trials
        assert stop_condition.should_stop(orchestrator._trials)

    def test_pruned_trial_counts_toward_max_trials(self, orchestrator):
        """Verify executed pruned trials count toward max_trials."""
        from traigent.core.stop_conditions import MaxTrialsStopCondition

        stop_condition = MaxTrialsStopCondition(max_trials=2)

        pruned_trial = TrialResult(
            trial_id="pruned_0",
            config={},
            metrics={},
            status=TrialStatus.PRUNED,
            duration=0.1,
            timestamp=None,
            metadata={"pruned": True},
        )
        orchestrator._trials.append(pruned_trial)

        assert not stop_condition.should_stop(orchestrator._trials)

        completed_trial = TrialResult(
            trial_id="completed_0",
            config={},
            metrics={},
            status=TrialStatus.COMPLETED,
            duration=0.1,
            timestamp=None,
        )
        orchestrator._trials.append(completed_trial)

        assert stop_condition.should_stop(orchestrator._trials)

    @pytest.mark.asyncio
    async def test_pruned_trial_submits_to_backend(self, orchestrator):
        """Verify pruned trials are submitted to the backend session manager."""
        orchestrator.backend_session_manager.submit_trial = AsyncMock(return_value=True)

        trial_result = TrialResult(
            trial_id="trial_pruned_0",
            config={"temperature": 0.5},
            metrics={"accuracy": 0.5},
            status=TrialStatus.PRUNED,
            duration=0.1,
            timestamp=None,
            metadata={"pruned": True},
        )

        await orchestrator._handle_trial_result(
            trial_result=trial_result,
            optimizer_config=trial_result.config,
            current_trial_index=0,
            session_id="session-123",
            optuna_trial_id=None,
            log_on_success=False,
        )

        orchestrator.backend_session_manager.submit_trial.assert_awaited_once_with(
            trial_result=trial_result,
            session_id="session-123",
        )


class TestStopReasonInResult:
    """Tests for stop_reason field in OptimizationResult."""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset."""
        examples = [
            EvaluationExample({"query": "Hello"}, "Hi there!"),
            EvaluationExample({"query": "Goodbye"}, "See you later!"),
        ]
        return Dataset(examples, name="test_dataset", description="Test dataset")

    @pytest.mark.asyncio
    async def test_stop_reason_max_trials_reached(self, sample_dataset):
        """stop_reason is 'max_trials_reached' when max_trials limit hit."""
        config_space = {"param1": (0, 1)}
        optimizer = MockOptimizer(config_space, ["accuracy"])
        optimizer.set_max_suggestions(10)  # More than max_trials

        evaluator = MockEvaluator(metrics=["accuracy"])

        orchestrator = OptimizationOrchestrator(
            optimizer=optimizer,
            evaluator=evaluator,
            max_trials=3,  # Limit to 3 trials
            config=TraigentConfig.edge_analytics_mode(),
        )

        result = await orchestrator.optimize(lambda _: "ok", sample_dataset)

        assert result.stop_reason == "max_trials_reached"
        assert len(result.trials) == 3

    @pytest.mark.asyncio
    async def test_stop_reason_timeout(self, sample_dataset):
        """stop_reason is 'timeout' when timeout is reached."""
        config_space = {"param1": (0, 1)}
        optimizer = MockOptimizer(config_space, ["accuracy"])
        optimizer.set_max_suggestions(100)

        evaluator = MockEvaluator(metrics=["accuracy"])
        evaluator.set_evaluation_delay(0.3)  # Slow evaluation

        orchestrator = OptimizationOrchestrator(
            optimizer=optimizer,
            evaluator=evaluator,
            max_trials=100,
            timeout=0.5,  # Short timeout
            config=TraigentConfig.edge_analytics_mode(),
        )

        result = await orchestrator.optimize(lambda _: "ok", sample_dataset)

        assert result.stop_reason == "timeout"

    @pytest.mark.asyncio
    async def test_stop_reason_optimizer_stop(self, sample_dataset):
        """stop_reason is 'optimizer' when optimizer requests stop."""
        config_space = {"param1": (0, 1)}
        optimizer = MockOptimizer(config_space, ["accuracy"])
        optimizer.set_max_suggestions(2)  # Optimizer stops after 2

        evaluator = MockEvaluator(metrics=["accuracy"])

        orchestrator = OptimizationOrchestrator(
            optimizer=optimizer,
            evaluator=evaluator,
            max_trials=100,
            config=TraigentConfig.edge_analytics_mode(),
        )

        result = await orchestrator.optimize(lambda _: "ok", sample_dataset)

        # Optimizer provides exactly 2 configs then stops - treated as optimizer stop
        assert result.stop_reason == "optimizer"
        assert len(result.trials) == 2

    @pytest.mark.asyncio
    async def test_stop_reason_optimizer_when_no_suggestions(self, sample_dataset):
        """stop_reason is 'optimizer' when optimizer provides no suggestions."""
        config_space = {"param1": (0, 1)}
        optimizer = MockOptimizer(config_space, ["accuracy"])
        optimizer.set_max_suggestions(0)  # No suggestions

        evaluator = MockEvaluator(metrics=["accuracy"])

        orchestrator = OptimizationOrchestrator(
            optimizer=optimizer,
            evaluator=evaluator,
            max_trials=10,
            config=TraigentConfig.edge_analytics_mode(),
        )

        result = await orchestrator.optimize(lambda _: "ok", sample_dataset)

        # When optimizer provides no suggestions, stop_reason is "optimizer"
        assert result.stop_reason == "optimizer"
        assert len(result.trials) == 0


class TestOrchestratorCodeQuality:
    """Test code quality aspects of orchestrator.py."""

    def test_no_duplicate_properties(self):
        """Ensure no duplicate property definitions in orchestrator.py."""
        file_path = Path("traigent/core/orchestrator.py")
        with open(file_path) as f:
            content = f.read()

        # Count property definitions
        status_count = content.count("def status(self)")
        optimization_id_count = content.count("def optimization_id(self)")

        assert (
            status_count == 1
        ), f"Found {status_count} status property definitions, expected 1"
        assert (
            optimization_id_count == 1
        ), f"Found {optimization_id_count} optimization_id definitions, expected 1"


def test_allocate_parallel_ceilings_distribution():
    dataset_sizes = [5, 5, 5]
    allocations = allocate_parallel_ceilings(dataset_sizes, 7)
    assert allocations == [3, 2, 2]

    dataset_sizes = [2, 10]
    allocations = allocate_parallel_ceilings(dataset_sizes, 5)
    assert allocations == [2, 3]

    dataset_sizes = [1, 1, 1]
    allocations = allocate_parallel_ceilings(dataset_sizes, 10)
    assert allocations == [1, 1, 1]


class TestCostEstimation:
    """Tests for cost estimation behavior (Issue C fix)."""

    @pytest.fixture
    def mock_optimizer(self) -> MockOptimizer:
        return MockOptimizer(
            config_space={"param": {"min": 0, "max": 1}},
            objectives=["accuracy"],
        )

    @pytest.fixture
    def mock_evaluator(self) -> MockEvaluator:
        return MockEvaluator()

    @pytest.fixture
    def small_dataset(self) -> list[EvaluationExample]:
        """Create a small dataset with 10 examples."""
        return [
            EvaluationExample(input_data={"x": i}, expected_output={"y": i * 2})
            for i in range(10)
        ]

    def test_cost_estimate_uses_budget_not_dataset_size(
        self,
        mock_optimizer: MockOptimizer,
        mock_evaluator: MockEvaluator,
        small_dataset: Dataset,
    ) -> None:
        """Issue C fix: Cost estimate uses configured budget, not clipped to dataset size.

        When max_total_examples (budget) exceeds dataset size, the estimate should
        use the full budget because:
        1. Multiple trials can re-evaluate samples with different configs
        2. The budget represents total API calls, not unique samples
        3. Clipping would underestimate cost
        """
        # Budget (1000) exceeds dataset size (10)
        orchestrator = OptimizationOrchestrator(
            optimizer=mock_optimizer,
            evaluator=mock_evaluator,
            max_trials=100,
            max_total_examples=1000,  # Much larger than dataset
        )

        # Estimate should use the full budget (1000), not clip to dataset size (10)
        estimate = orchestrator._estimate_optimization_cost(small_dataset)

        # With base_cost_per_example = 0.01, retry_factor = 1.2:
        # estimate = 1000 * 0.01 * 1.2 = 12.0
        # If it was clipped to dataset (10), it would be: 10 * 0.01 * 1.2 = 0.12
        # The estimate should be much higher than 0.12
        assert estimate > 1.0, (
            f"Expected estimate > 1.0 (using full budget), got {estimate}. "
            "Cost estimate may be incorrectly clipping to dataset size."
        )
        # More specific check: should be approximately 12.0 (1000 * 0.01 * 1.2)
        assert estimate == pytest.approx(
            12.0, rel=0.1
        ), f"Expected estimate ~12.0, got {estimate}"

    def test_cost_estimate_without_budget_uses_dataset_size(
        self,
        mock_optimizer: MockOptimizer,
        mock_evaluator: MockEvaluator,
        small_dataset: Dataset,
    ) -> None:
        """When no budget is set, estimation uses dataset size as conservative estimate."""
        orchestrator = OptimizationOrchestrator(
            optimizer=mock_optimizer,
            evaluator=mock_evaluator,
            max_trials=5,
            # No max_total_examples set
        )

        estimate = orchestrator._estimate_optimization_cost(small_dataset)

        # Without budget, uses samples_per_trial (dataset_size) * max_trials
        # = 10 * 5 * 0.01 * 1.2 = 0.60
        # But may use samples_per_trial estimate
        assert estimate > 0, "Estimate should be positive"
        # Should be reasonable for 5 trials of 10 examples each
        assert estimate < 10.0, f"Estimate {estimate} seems too high"
