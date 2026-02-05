"""Unit tests for TrialLifecycle class.

This test suite covers the trial lifecycle management methods extracted from
OptimizationOrchestrator to improve testability and reduce class complexity.

Tests cover:
- TrialLifecycle initialization
- Trial ID generation
- Progress tracking setup
- Budget lease management
- Budget metadata application
- Error result building
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from traigent.core.sample_budget import (
    LeaseClosure,
    SampleBudgetLease,
    SampleBudgetManager,
)
from traigent.core.trial_lifecycle import TrialLifecycle
from traigent.core.types import TrialResult, TrialStatus
from traigent.evaluators.base import BaseEvaluator, Dataset
from traigent.optimizers.base import BaseOptimizer
from traigent.utils.exceptions import TrialPrunedError

# =============================================================================
# Mock Classes
# =============================================================================


class MockOptimizer(BaseOptimizer):
    """Mock optimizer for testing."""

    def __init__(self):
        self.name = "mock_optimizer"

    def suggest_next_trial(self, completed_trials):
        return {"temperature": 0.5, "model": "gpt-4"}

    def should_stop(self, completed_trials):
        return False


class MockEvaluator(BaseEvaluator):
    """Mock evaluator for testing."""

    async def evaluate(self, func, config, dataset, **kwargs):
        result = MagicMock()
        result.metrics = {"accuracy": 0.85}
        result.example_results = []
        result.sample_budget_exhausted = False
        return result


class MockOrchestrator:
    """Mock orchestrator for TrialLifecycle testing."""

    def __init__(self):
        self.optimizer = MockOptimizer()
        self.evaluator = MockEvaluator()
        self._optimization_id = "test-optimization-123"
        self._sample_budget_manager = None
        self._consumed_examples = 0
        self._examples_capped = 0
        self._stop_reason = None
        self._constraints_pre_eval = []
        self._constraints_post_eval = []
        self._trials = []
        self.max_trials = 10
        self.config = {}
        self.cache_policy_handler = MagicMock()
        self.callback_manager = MagicMock()
        self.cost_enforcer = None  # Added for sequential permit enforcement
        self._default_config = None
        self._default_config_used = False
        self.objective_schema = None  # For band-based pruning support

    def _consume_default_config(self):
        return None

    async def _handle_trial_result(self, **kwargs):
        return kwargs.get("current_trial_index", 0) + 1


def create_mock_dataset(size: int = 10, name: str = "test_dataset") -> Dataset:
    """Create a mock dataset for testing."""
    dataset = MagicMock(spec=Dataset)
    dataset.name = name
    dataset.examples = [{"input": f"test_{i}"} for i in range(size)]
    dataset.__len__ = MagicMock(return_value=size)
    return dataset


# =============================================================================
# TrialLifecycle Initialization Tests
# =============================================================================


class TestTrialLifecycleInitialization:
    """Test TrialLifecycle initialization."""

    def test_initialization_with_orchestrator(self):
        """Test that TrialLifecycle initializes with orchestrator reference."""
        orchestrator = MockOrchestrator()
        lifecycle = TrialLifecycle(orchestrator)

        assert lifecycle._orchestrator is orchestrator

    def test_initialization_stores_reference(self):
        """Test that orchestrator reference is accessible."""
        orchestrator = MockOrchestrator()
        lifecycle = TrialLifecycle(orchestrator)

        # Should be able to access orchestrator properties
        assert lifecycle._orchestrator.optimizer is not None
        assert lifecycle._orchestrator.evaluator is not None


# =============================================================================
# Trial ID Generation Tests
# =============================================================================


class TestGenerateTrialId:
    """Test _generate_trial_id method."""

    def test_sequential_id_without_session(self):
        """Test sequential ID generation when no session_id is provided."""
        orchestrator = MockOrchestrator()
        lifecycle = TrialLifecycle(orchestrator)
        dataset = create_mock_dataset()

        trial_id = lifecycle._generate_trial_id(
            config={"temp": 0.5},
            trial_number=5,
            session_id=None,
            dataset=dataset,
            optuna_trial_id=None,
        )

        assert trial_id == "test-optimization-123_5"

    def test_hash_based_id_with_session(self):
        """Test hash-based ID generation when session_id is provided."""
        orchestrator = MockOrchestrator()
        lifecycle = TrialLifecycle(orchestrator)
        dataset = create_mock_dataset()

        with patch("traigent.utils.hashing.generate_trial_hash") as mock_hash:
            mock_hash.return_value = "hash-abc123"

            trial_id = lifecycle._generate_trial_id(
                config={"temp": 0.5},
                trial_number=5,
                session_id="session-xyz",
                dataset=dataset,
                optuna_trial_id=None,
            )

            assert trial_id == "hash-abc123"
            mock_hash.assert_called_once()

    def test_optuna_trial_id_included_in_hash(self):
        """Test that optuna_trial_id is included in hash config."""
        orchestrator = MockOrchestrator()
        lifecycle = TrialLifecycle(orchestrator)
        dataset = create_mock_dataset()

        with patch("traigent.utils.hashing.generate_trial_hash") as mock_hash:
            mock_hash.return_value = "hash-with-optuna"

            lifecycle._generate_trial_id(
                config={"temp": 0.5},
                trial_number=5,
                session_id="session-xyz",
                dataset=dataset,
                optuna_trial_id=42,
            )

            # Check that the config passed to hash includes optuna_trial_id
            call_args = mock_hash.call_args
            config_passed = call_args.kwargs.get("config") or call_args[1].get("config")
            assert config_passed.get("_optuna_trial_id") == 42

    def test_dataset_name_used_in_hash(self):
        """Test that dataset name is used for hash generation."""
        orchestrator = MockOrchestrator()
        lifecycle = TrialLifecycle(orchestrator)
        dataset = create_mock_dataset(name="custom_dataset")

        with patch("traigent.utils.hashing.generate_trial_hash") as mock_hash:
            mock_hash.return_value = "hash-xyz"

            lifecycle._generate_trial_id(
                config={"temp": 0.5},
                trial_number=5,
                session_id="session-xyz",
                dataset=dataset,
                optuna_trial_id=None,
            )

            call_args = mock_hash.call_args
            dataset_name = call_args.kwargs.get("dataset_name") or call_args[1].get(
                "dataset_name"
            )
            assert dataset_name == "custom_dataset"


# =============================================================================
# Progress Tracking Tests
# =============================================================================


class TestCreateProgressTracking:
    """Test _create_progress_tracking method."""

    def test_returns_none_without_optuna_trial_id(self):
        """Test that (None, None) is returned without optuna_trial_id."""
        orchestrator = MockOrchestrator()
        lifecycle = TrialLifecycle(orchestrator)
        dataset = create_mock_dataset()

        callback, state = lifecycle._create_progress_tracking(
            optuna_trial_id=None,
            dataset=dataset,
            trial_id="test-trial",
        )

        assert callback is None
        assert state is None

    def test_returns_none_without_report_capability(self):
        """Test that (None, None) is returned if optimizer lacks reporting."""
        orchestrator = MockOrchestrator()
        # Optimizer doesn't have report_intermediate_value by default

        lifecycle = TrialLifecycle(orchestrator)
        dataset = create_mock_dataset()

        callback, state = lifecycle._create_progress_tracking(
            optuna_trial_id=42,
            dataset=dataset,
            trial_id="test-trial",
        )

        assert callback is None
        assert state is None

    def test_creates_tracker_with_optuna_support(self):
        """Test that tracker is created when optimizer supports reporting."""
        orchestrator = MockOrchestrator()
        orchestrator.optimizer.report_intermediate_value = MagicMock()

        lifecycle = TrialLifecycle(orchestrator)
        dataset = create_mock_dataset()

        with patch(
            "traigent.core.trial_lifecycle.PruningProgressTracker"
        ) as mock_tracker_cls:
            mock_tracker = MagicMock()
            mock_tracker.callback = MagicMock()
            mock_tracker.state = {"evaluated": 0}
            mock_tracker_cls.return_value = mock_tracker

            callback, state = lifecycle._create_progress_tracking(
                optuna_trial_id=42,
                dataset=dataset,
                trial_id="test-trial",
            )

            assert callback is mock_tracker.callback
            assert state is mock_tracker.state
            mock_tracker_cls.assert_called_once()


# =============================================================================
# Budget Lease Management Tests
# =============================================================================


class TestSetupTrialBudgetLease:
    """Test _setup_trial_budget_lease method."""

    def test_returns_none_without_budget_manager(self):
        """Test that None is returned when no budget manager is configured."""
        orchestrator = MockOrchestrator()
        orchestrator._sample_budget_manager = None

        lifecycle = TrialLifecycle(orchestrator)
        dataset = create_mock_dataset()

        lease = lifecycle._setup_trial_budget_lease(
            dataset=dataset,
            trial_id="test-trial",
            sample_ceiling=None,
        )

        assert lease is None

    def test_creates_lease_with_budget_manager(self):
        """Test that lease is created when budget manager is configured."""
        orchestrator = MockOrchestrator()
        mock_manager = MagicMock(spec=SampleBudgetManager)
        mock_manager.remaining.return_value = 100.0
        mock_lease = MagicMock(spec=SampleBudgetLease)
        mock_manager.create_lease.return_value = mock_lease
        orchestrator._sample_budget_manager = mock_manager

        lifecycle = TrialLifecycle(orchestrator)
        dataset = create_mock_dataset(size=10)

        lease = lifecycle._setup_trial_budget_lease(
            dataset=dataset,
            trial_id="test-trial",
            sample_ceiling=None,
        )

        assert lease is mock_lease
        mock_manager.create_lease.assert_called_once_with(
            trial_id="test-trial",
            ceiling=10,  # min(dataset_size=10, remaining=100)
        )

    def test_uses_sample_ceiling_when_provided(self):
        """Test that explicit sample_ceiling is used when provided."""
        orchestrator = MockOrchestrator()
        mock_manager = MagicMock(spec=SampleBudgetManager)
        mock_manager.remaining.return_value = 100.0
        mock_lease = MagicMock(spec=SampleBudgetLease)
        mock_manager.create_lease.return_value = mock_lease
        orchestrator._sample_budget_manager = mock_manager

        lifecycle = TrialLifecycle(orchestrator)
        dataset = create_mock_dataset(size=50)

        lifecycle._setup_trial_budget_lease(
            dataset=dataset,
            trial_id="test-trial",
            sample_ceiling=25,  # Explicit ceiling
        )

        mock_manager.create_lease.assert_called_once_with(
            trial_id="test-trial",
            ceiling=25,
        )

    def test_caps_ceiling_to_remaining_budget(self):
        """Test that ceiling is capped to remaining budget."""
        orchestrator = MockOrchestrator()
        mock_manager = MagicMock(spec=SampleBudgetManager)
        mock_manager.remaining.return_value = 5.0  # Only 5 samples remaining
        mock_lease = MagicMock(spec=SampleBudgetLease)
        mock_manager.create_lease.return_value = mock_lease
        orchestrator._sample_budget_manager = mock_manager

        lifecycle = TrialLifecycle(orchestrator)
        dataset = create_mock_dataset(size=100)

        lifecycle._setup_trial_budget_lease(
            dataset=dataset,
            trial_id="test-trial",
            sample_ceiling=None,
        )

        mock_manager.create_lease.assert_called_once_with(
            trial_id="test-trial",
            ceiling=5,  # Capped to remaining
        )

    def test_handles_infinite_remaining(self):
        """Test that infinite remaining budget uses dataset size."""
        orchestrator = MockOrchestrator()
        mock_manager = MagicMock(spec=SampleBudgetManager)
        mock_manager.remaining.return_value = float("inf")
        mock_lease = MagicMock(spec=SampleBudgetLease)
        mock_manager.create_lease.return_value = mock_lease
        orchestrator._sample_budget_manager = mock_manager

        lifecycle = TrialLifecycle(orchestrator)
        dataset = create_mock_dataset(size=50)

        lifecycle._setup_trial_budget_lease(
            dataset=dataset,
            trial_id="test-trial",
            sample_ceiling=None,
        )

        mock_manager.create_lease.assert_called_once_with(
            trial_id="test-trial",
            ceiling=50,  # Uses dataset size when infinite
        )


class TestFinalizeBudgetLease:
    """Test _finalize_budget_lease method."""

    def test_returns_none_for_none_lease(self):
        """Test that None is returned when lease is None."""
        orchestrator = MockOrchestrator()
        lifecycle = TrialLifecycle(orchestrator)

        result = lifecycle._finalize_budget_lease(None)

        assert result is None

    def test_finalizes_lease_and_returns_closure(self):
        """Test that lease is finalized and closure is returned."""
        orchestrator = MockOrchestrator()
        mock_manager = MagicMock(spec=SampleBudgetManager)
        mock_manager.consumed.return_value = 50
        orchestrator._sample_budget_manager = mock_manager

        lifecycle = TrialLifecycle(orchestrator)

        mock_lease = MagicMock(spec=SampleBudgetLease)
        mock_lease.trial_id = "test-trial"
        mock_closure = LeaseClosure(
            trial_id="test-trial",
            consumed=10,
            wasted=2,
            global_remaining=40,
            exhausted=False,
        )
        mock_lease.finalize.return_value = mock_closure

        result = lifecycle._finalize_budget_lease(mock_lease)

        assert result is mock_closure
        mock_lease.finalize.assert_called_once()
        assert orchestrator._consumed_examples == 50

    def test_updates_stop_reason_when_exhausted(self):
        """Test that stop reason is set when budget is exhausted."""
        orchestrator = MockOrchestrator()
        mock_manager = MagicMock(spec=SampleBudgetManager)
        mock_manager.consumed.return_value = 100
        orchestrator._sample_budget_manager = mock_manager
        orchestrator._stop_reason = None

        lifecycle = TrialLifecycle(orchestrator)

        mock_lease = MagicMock(spec=SampleBudgetLease)
        mock_lease.trial_id = "test-trial"
        mock_closure = LeaseClosure(
            trial_id="test-trial",
            consumed=10,
            wasted=0,
            global_remaining=0,
            exhausted=True,
        )
        mock_lease.finalize.return_value = mock_closure

        lifecycle._finalize_budget_lease(mock_lease)

        assert orchestrator._stop_reason == "max_samples_reached"
        assert orchestrator._examples_capped == 1

    def test_does_not_override_existing_stop_reason(self):
        """Test that existing stop reason is not overridden."""
        orchestrator = MockOrchestrator()
        mock_manager = MagicMock(spec=SampleBudgetManager)
        mock_manager.consumed.return_value = 100
        orchestrator._sample_budget_manager = mock_manager
        orchestrator._stop_reason = "max_trials_reached"

        lifecycle = TrialLifecycle(orchestrator)

        mock_lease = MagicMock(spec=SampleBudgetLease)
        mock_lease.trial_id = "test-trial"
        mock_closure = LeaseClosure(
            trial_id="test-trial",
            consumed=10,
            wasted=0,
            global_remaining=0,
            exhausted=True,
        )
        mock_lease.finalize.return_value = mock_closure

        lifecycle._finalize_budget_lease(mock_lease)

        # Should not override existing stop reason
        assert orchestrator._stop_reason == "max_trials_reached"


# =============================================================================
# Budget Metadata Tests
# =============================================================================


class TestApplyBudgetMetadata:
    """Test _apply_budget_metadata method."""

    def test_applies_metadata_without_closure(self):
        """Test metadata application when no closure is provided."""
        orchestrator = MockOrchestrator()
        lifecycle = TrialLifecycle(orchestrator)

        trial_result = MagicMock(spec=TrialResult)
        trial_result.metadata = {}
        trial_result.metrics = {}

        lifecycle._apply_budget_metadata(trial_result, None, budget_exhausted=False)

        assert trial_result.metadata["sample_budget_exhausted"] is False

    def test_applies_metadata_with_closure(self):
        """Test metadata application with closure details."""
        orchestrator = MockOrchestrator()
        lifecycle = TrialLifecycle(orchestrator)

        trial_result = MagicMock(spec=TrialResult)
        trial_result.metadata = {}
        trial_result.metrics = {}

        closure = LeaseClosure(
            trial_id="test-trial",
            consumed=25,
            wasted=5,
            global_remaining=70,
            exhausted=False,
        )

        lifecycle._apply_budget_metadata(trial_result, closure, budget_exhausted=False)

        assert trial_result.metadata["examples_attempted"] == 25
        assert trial_result.metadata["sample_budget_remaining"] == 70
        assert trial_result.metadata["sample_budget_exhausted"] is False
        assert trial_result.metadata["sample_budget_consumed"] == 25
        assert trial_result.metadata["sample_budget_wasted"] == 5
        assert trial_result.metrics["examples_attempted"] == 25

    def test_sets_stop_reason_when_exhausted(self):
        """Test that stop_reason is set when budget is exhausted."""
        orchestrator = MockOrchestrator()
        lifecycle = TrialLifecycle(orchestrator)

        trial_result = MagicMock(spec=TrialResult)
        trial_result.metadata = {}
        trial_result.metrics = {}

        closure = LeaseClosure(
            trial_id="test-trial",
            consumed=25,
            wasted=0,
            global_remaining=0,
            exhausted=True,
        )

        lifecycle._apply_budget_metadata(trial_result, closure, budget_exhausted=True)

        assert trial_result.metadata["stop_reason"] == "sample_budget_exhausted"

    def test_preserves_existing_examples_attempted(self):
        """Test that existing examples_attempted is preserved."""
        orchestrator = MockOrchestrator()
        lifecycle = TrialLifecycle(orchestrator)

        trial_result = MagicMock(spec=TrialResult)
        trial_result.metadata = {"examples_attempted": 100}  # Pre-existing
        trial_result.metrics = {"examples_attempted": 100}

        closure = LeaseClosure(
            trial_id="test-trial",
            consumed=25,
            wasted=0,
            global_remaining=75,
            exhausted=False,
        )

        lifecycle._apply_budget_metadata(trial_result, closure, budget_exhausted=False)

        # Should keep the existing value via setdefault
        assert trial_result.metadata["examples_attempted"] == 100


# =============================================================================
# Error Result Building Tests
# =============================================================================


class TestBuildTrialErrorResult:
    """Test _build_trial_error_result method."""

    def test_builds_failed_result(self):
        """Test building a failed trial result."""
        orchestrator = MockOrchestrator()
        lifecycle = TrialLifecycle(orchestrator)

        error = ValueError("Something went wrong")
        start_time = time.time() - 1.5  # 1.5 seconds ago

        with patch("traigent.core.trial_lifecycle.build_failed_result") as mock_build:
            mock_result = MagicMock(spec=TrialResult)
            mock_result.metadata = {}
            mock_result.metrics = {}
            mock_build.return_value = mock_result

            result = lifecycle._build_trial_error_result(
                trial_id="test-trial",
                evaluation_config={"temp": 0.5},
                start_time=start_time,
                lease=None,
                progress_state=None,
                optuna_trial_id=None,
                error=error,
                is_pruned=False,
            )

            mock_build.assert_called_once()
            assert result is mock_result

    def test_builds_pruned_result(self):
        """Test building a pruned trial result."""
        orchestrator = MockOrchestrator()
        lifecycle = TrialLifecycle(orchestrator)

        error = TrialPrunedError("Trial pruned early", step=5)
        start_time = time.time() - 0.5

        with patch("traigent.core.trial_lifecycle.build_pruned_result") as mock_build:
            mock_result = MagicMock(spec=TrialResult)
            mock_result.metadata = {}
            mock_result.metrics = {}
            mock_build.return_value = mock_result

            result = lifecycle._build_trial_error_result(
                trial_id="test-trial",
                evaluation_config={"temp": 0.5},
                start_time=start_time,
                lease=None,
                progress_state={"evaluated": 5},
                optuna_trial_id=42,
                error=error,
                is_pruned=True,
            )

            mock_build.assert_called_once()
            assert result is mock_result

    def test_finalizes_lease_on_error(self):
        """Test that budget lease is finalized on error."""
        orchestrator = MockOrchestrator()
        mock_manager = MagicMock(spec=SampleBudgetManager)
        mock_manager.consumed.return_value = 50
        orchestrator._sample_budget_manager = mock_manager

        lifecycle = TrialLifecycle(orchestrator)

        mock_lease = MagicMock(spec=SampleBudgetLease)
        mock_lease.trial_id = "test-trial"
        mock_closure = LeaseClosure(
            trial_id="test-trial",
            consumed=10,
            wasted=2,
            global_remaining=40,
            exhausted=False,
        )
        mock_lease.finalize.return_value = mock_closure

        error = ValueError("Something went wrong")
        start_time = time.time()

        with patch("traigent.core.trial_lifecycle.build_failed_result") as mock_build:
            mock_result = MagicMock(spec=TrialResult)
            mock_result.metadata = {}
            mock_result.metrics = {}
            mock_build.return_value = mock_result

            lifecycle._build_trial_error_result(
                trial_id="test-trial",
                evaluation_config={"temp": 0.5},
                start_time=start_time,
                lease=mock_lease,
                progress_state=None,
                optuna_trial_id=None,
                error=error,
                is_pruned=False,
            )

            mock_lease.finalize.assert_called_once()


# =============================================================================
# Run Trial Integration Tests
# =============================================================================


class TestRunTrial:
    """Test run_trial method (integration)."""

    @pytest.mark.asyncio
    async def test_successful_trial_execution(self):
        """Test successful trial execution."""
        orchestrator = MockOrchestrator()
        lifecycle = TrialLifecycle(orchestrator)
        dataset = create_mock_dataset()

        async def mock_func(input_data):
            return "result"

        with patch.object(
            orchestrator.evaluator, "evaluate", new_callable=AsyncMock
        ) as mock_eval:
            mock_result = MagicMock()
            mock_result.metrics = {"accuracy": 0.9}
            mock_result.example_results = []
            mock_result.sample_budget_exhausted = False
            mock_eval.return_value = mock_result

            result = await lifecycle.run_trial(
                func=mock_func,
                config={"temp": 0.5},
                dataset=dataset,
                trial_number=0,
                session_id=None,
                optuna_trial_id=None,
            )

            assert result.status == TrialStatus.COMPLETED
            assert result.trial_id == "test-optimization-123_0"

    @pytest.mark.asyncio
    async def test_trial_with_evaluation_failure(self):
        """Test trial handling when evaluation fails."""
        orchestrator = MockOrchestrator()
        lifecycle = TrialLifecycle(orchestrator)
        dataset = create_mock_dataset()

        async def mock_func(input_data):
            return "result"

        with patch.object(
            orchestrator.evaluator, "evaluate", new_callable=AsyncMock
        ) as mock_eval:
            mock_eval.side_effect = RuntimeError("Evaluation failed")

            result = await lifecycle.run_trial(
                func=mock_func,
                config={"temp": 0.5},
                dataset=dataset,
                trial_number=0,
                session_id=None,
            )

            assert result.status == TrialStatus.FAILED
            assert "Evaluation failed" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_trial_with_constraint_violation(self):
        """Test trial handling when post-eval constraint fails."""
        orchestrator = MockOrchestrator()

        def bad_constraint(config, metrics):
            return metrics.get("accuracy", 0) > 0.95

        bad_constraint.__tvl_constraint__ = {
            "requires_metrics": True,
            "id": "accuracy_check",
        }
        orchestrator._constraints_post_eval = [bad_constraint]

        lifecycle = TrialLifecycle(orchestrator)
        dataset = create_mock_dataset()

        async def mock_func(input_data):
            return "result"

        with patch.object(
            orchestrator.evaluator, "evaluate", new_callable=AsyncMock
        ) as mock_eval:
            mock_result = MagicMock()
            mock_result.metrics = {"accuracy": 0.85}  # Below threshold
            mock_result.example_results = []
            mock_result.sample_budget_exhausted = False
            mock_eval.return_value = mock_result

            result = await lifecycle.run_trial(
                func=mock_func,
                config={"temp": 0.5},
                dataset=dataset,
                trial_number=0,
                session_id=None,
            )

            assert result.status == TrialStatus.FAILED


# =============================================================================
# Run Sequential Trial Tests
# =============================================================================


class TestRunSequentialTrial:
    """Test run_sequential_trial method."""

    @pytest.mark.asyncio
    async def test_stops_when_max_trials_reached(self):
        """Test that trial stops when max_trials is reached."""
        orchestrator = MockOrchestrator()
        orchestrator.max_trials = 5
        lifecycle = TrialLifecycle(orchestrator)
        dataset = create_mock_dataset()

        async def mock_func(input_data):
            return "result"

        trial_count, action = await lifecycle.run_sequential_trial(
            func=mock_func,
            dataset=dataset,
            session_id=None,
            function_name="test_func",
            trial_count=5,  # Already at max
        )

        assert action == "break"
        assert orchestrator._stop_reason == "max_trials_reached"

    @pytest.mark.asyncio
    async def test_continues_when_below_max_trials(self):
        """Test that trial continues when below max_trials."""
        orchestrator = MockOrchestrator()
        orchestrator.max_trials = 10
        lifecycle = TrialLifecycle(orchestrator)
        dataset = create_mock_dataset()

        async def mock_func(input_data):
            return "result"

        with patch.object(
            orchestrator.evaluator, "evaluate", new_callable=AsyncMock
        ) as mock_eval:
            mock_result = MagicMock()
            mock_result.metrics = {"accuracy": 0.9}
            mock_result.example_results = []
            mock_result.sample_budget_exhausted = False
            mock_eval.return_value = mock_result

            trial_count, action = await lifecycle.run_sequential_trial(
                func=mock_func,
                dataset=dataset,
                session_id=None,
                function_name="test_func",
                trial_count=3,
            )

            assert action == "continue"
            assert trial_count == 4  # Incremented by _handle_trial_result

    @pytest.mark.asyncio
    async def test_breaks_when_optimizer_fails(self):
        """Test that trial breaks when optimizer fails to suggest."""
        orchestrator = MockOrchestrator()
        orchestrator.optimizer.suggest_next_trial = MagicMock(
            side_effect=ValueError("No more configs")
        )
        lifecycle = TrialLifecycle(orchestrator)
        dataset = create_mock_dataset()

        async def mock_func(input_data):
            return "result"

        trial_count, action = await lifecycle.run_sequential_trial(
            func=mock_func,
            dataset=dataset,
            session_id=None,
            function_name="test_func",
            trial_count=0,
        )

        assert action == "break"

    @pytest.mark.asyncio
    async def test_skips_cached_config(self):
        """Test that cached config is skipped when cache policy is set."""
        orchestrator = MockOrchestrator()
        orchestrator.config = {"cache_policy": "skip_evaluated"}
        orchestrator.cache_policy_handler.apply_policy.return_value = []  # Empty = skip
        lifecycle = TrialLifecycle(orchestrator)
        dataset = create_mock_dataset()

        async def mock_func(input_data):
            return "result"

        trial_count, action = await lifecycle.run_sequential_trial(
            func=mock_func,
            dataset=dataset,
            session_id=None,
            function_name="test_func",
            trial_count=0,
        )

        assert action == "continue"
        assert trial_count == 0  # Not incremented because skipped

    @pytest.mark.asyncio
    async def test_constraint_failure_does_not_consume_trial_slot(self):
        """Test that constraint-rejected configs don't consume trial slots (issue #27)."""

        orchestrator = MockOrchestrator()
        orchestrator.max_trials = 10

        # Add a pre-constraint that always fails
        def failing_constraint(config, metrics=None):
            return False

        failing_constraint.__tvl_constraint__ = {
            "id": "test",
            "message": "Always fails",
        }
        orchestrator._constraints_pre_eval = [failing_constraint]
        orchestrator._abandon_optuna_trial = MagicMock()  # Mock the abandon method

        lifecycle = TrialLifecycle(orchestrator)
        dataset = create_mock_dataset()

        async def mock_func(input_data):
            return "result"

        trial_count, action = await lifecycle.run_sequential_trial(
            func=mock_func,
            dataset=dataset,
            session_id=None,
            function_name="test_func",
            trial_count=5,
        )

        # Key assertions: constraint failure should NOT consume a trial slot
        assert action == "continue"
        assert trial_count == 5  # NOT incremented - this is the bug fix for issue #27

        # Verify Optuna trial was abandoned with appropriate reason
        orchestrator._abandon_optuna_trial.assert_called_once()
        call_kwargs = orchestrator._abandon_optuna_trial.call_args[1]
        assert "trial_rejected_by_constraint" in call_kwargs.get("reason", "")
        assert call_kwargs.get("status") == TrialStatus.PRUNED
