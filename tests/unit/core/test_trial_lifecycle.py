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

from traigent.api.types import ExampleResult
from traigent.config.types import TraigentConfig, resolve_execution_policy
from traigent.core.backend_session_manager import BackendSessionManager
from traigent.core.sample_budget import (
    LeaseClosure,
    SampleBudgetLease,
    SampleBudgetManager,
)
from traigent.core.trial_lifecycle import TrialLifecycle
from traigent.core.types import OptimizationStatus, TrialResult, TrialStatus
from traigent.evaluators.base import (
    BaseEvaluator,
    Dataset,
    EvaluationExample,
    EvaluationResult,
)
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


class SmartPruningEvaluator(BaseEvaluator):
    """Evaluator that reports per-example progress and preserves prune partials."""

    async def evaluate(self, func, config, dataset, **kwargs):
        progress_callback = kwargs.get("progress_callback")
        example_results: list[ExampleResult] = []
        try:
            for index, example in enumerate(dataset.examples):
                result = ExampleResult(
                    example_id=f"example_{index}",
                    input_data=example.input_data,
                    expected_output=example.expected_output,
                    actual_output=f"output_{index}",
                    metrics={"accuracy": 1.0 if index == 0 else 0.0},
                    execution_time=0.01,
                    success=True,
                    error_message=None,
                    metadata={},
                )
                example_results.append(result)
                if progress_callback is not None:
                    progress_callback(
                        index,
                        {
                            "success": result.success,
                            "metrics": result.metrics,
                            "partial_cost_usd": 0.01,
                            "output": result.actual_output,
                        },
                    )
        except TrialPrunedError as exc:
            exc.example_results = example_results
            raise

        return EvaluationResult(
            config=config,
            example_results=example_results,
            aggregated_metrics={"accuracy": 0.5},
            total_examples=len(example_results),
            successful_examples=len(example_results),
            duration=0.02,
            metrics={"accuracy": 0.5, "examples_attempted": len(example_results)},
        )


class MockOrchestrator:
    """Mock orchestrator for TrialLifecycle testing."""

    def __init__(self):
        self.optimizer = MockOptimizer()
        self.evaluator = MockEvaluator()
        self.knob_resolver = None
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
        self.objectives = ["accuracy"]
        self.backend_session_manager = None

    def _apply_knob_resolution(self, config):
        """Mirror of OptimizationOrchestrator._apply_knob_resolution (RFC 0001):
        passthrough when no resolver is configured."""
        if self.knob_resolver is None:
            return config
        resolved = self.knob_resolver.resolve(config)
        return dict(resolved.config)

    def _consume_default_config(self):
        return None

    async def _handle_trial_result(self, **kwargs):
        return kwargs.get("current_trial_index", 0) + 1


class RecordingSmartPruningManager:
    """Backend-session manager test double for smart-pruning reports."""

    def __init__(self, *, enabled: bool = True, prune_after: int | None = None):
        self.enabled = enabled
        self.prune_after = prune_after
        self.reports: list[dict[str, object]] = []

    def should_report_intermediate_progress(self, session_id: str | None) -> bool:
        return self.enabled and session_id == "session-123"

    def report_intermediate_progress(self, payload: dict[str, object]):
        self.reports.append(dict(payload))
        if self.prune_after is not None and len(self.reports) >= self.prune_after:
            return {
                "prune": True,
                "prune_reason": "running score below smart-pruning threshold",
            }
        return {"prune": False, "prune_reason": None}


class RaisingIntermediateReportClient:
    """Backend client test double whose intermediate report POST fails."""

    no_egress = False

    def __init__(self) -> None:
        self.payloads: list[dict[str, object]] = []

    async def _report_intermediate_progress(self, payload: dict[str, object]):
        self.payloads.append(dict(payload))
        raise RuntimeError("report POST failed")


def create_mock_dataset(size: int = 10, name: str = "test_dataset") -> Dataset:
    """Create a mock dataset for testing."""
    dataset = MagicMock(spec=Dataset)
    dataset.name = name
    dataset.examples = [{"input": f"test_{i}"} for i in range(size)]
    dataset.__len__ = MagicMock(return_value=size)
    return dataset


def create_real_dataset(size: int = 3, name: str = "test_dataset") -> Dataset:
    return Dataset(
        examples=[
            EvaluationExample(input_data=f"question_{i}", expected_output=f"output_{i}")
            for i in range(size)
        ],
        name=name,
    )


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
            session_id=None,
        )

        assert callback is None
        assert state is None

    def test_returns_none_without_report_capability(self):
        """Test that (None, None) is returned if optimizer lacks reporting."""
        orchestrator = MockOrchestrator()

        lifecycle = TrialLifecycle(orchestrator)
        dataset = create_mock_dataset()

        callback, state = lifecycle._create_progress_tracking(
            optuna_trial_id=42,
            dataset=dataset,
            trial_id="test-trial",
            session_id="session-123",
        )

        assert callback is None
        assert state is None

    def test_returns_none_when_smart_pruning_disabled(self):
        """No smart-pruning config means no callback and no intermediate egress."""
        orchestrator = MockOrchestrator()
        manager = RecordingSmartPruningManager(enabled=False)
        orchestrator.backend_session_manager = manager
        lifecycle = TrialLifecycle(orchestrator)
        dataset = create_mock_dataset()

        callback, state = lifecycle._create_progress_tracking(
            optuna_trial_id=None,
            dataset=dataset,
            trial_id="test-trial",
            session_id="session-123",
        )

        assert callback is None
        assert state is None
        assert manager.reports == []

    @pytest.mark.asyncio
    async def test_cloud_smart_pruning_posts_and_returns_pruned_result(self):
        """prune=true stops the trial and preserves partial example results."""
        orchestrator = MockOrchestrator()
        orchestrator.evaluator = SmartPruningEvaluator()
        manager = RecordingSmartPruningManager(prune_after=2)
        orchestrator.backend_session_manager = manager
        lifecycle = TrialLifecycle(orchestrator)
        dataset = create_real_dataset(size=3)

        result = await lifecycle.run_trial(
            func=lambda value: value,
            config={"temperature": 0.2},
            dataset=dataset,
            trial_number=1,
            session_id="session-123",
        )

        assert result.status == TrialStatus.PRUNED
        assert result.error_message == "running score below smart-pruning threshold"
        assert result.metadata["pruned"] is True
        assert result.metadata["examples_attempted"] == 2
        assert len(result.metadata["example_results"]) == 2
        assert len(manager.reports) == 2
        assert manager.reports[0] == {
            "session_id": "session-123",
            "trial_id": result.trial_id,
            "running_score": 1.0,
            "examples_attempted": 1,
            "objective_name": "accuracy",
            "partial_cost_usd": 0.01,
        }
        assert manager.reports[1]["examples_attempted"] == 2
        assert manager.reports[1]["running_score"] == 0.5
        assert "output" not in manager.reports[0]

    @pytest.mark.asyncio
    async def test_cloud_smart_pruning_report_failure_continues_trial(
        self, monkeypatch
    ):
        """Report POST failures fail open at the manager boundary."""
        monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
        monkeypatch.setenv("TRAIGENT_OFFLINE", "false")
        orchestrator = MockOrchestrator()
        orchestrator.evaluator = SmartPruningEvaluator()
        traigent_config = TraigentConfig(algorithm="auto")
        traigent_config.execution_policy = resolve_execution_policy(
            algorithm="auto",
            offline=False,
        )
        failing_client = RaisingIntermediateReportClient()
        orchestrator.backend_session_manager = BackendSessionManager(
            backend_client=failing_client,
            traigent_config=traigent_config,
            objectives=["accuracy"],
            objective_schema=None,
            optimizer=orchestrator.optimizer,
            optimization_id="test-optimization-123",
            optimization_status=OptimizationStatus.RUNNING,
            smart_pruning={"label": "balanced"},
        )
        lifecycle = TrialLifecycle(orchestrator)
        dataset = create_real_dataset(size=3)

        decision = orchestrator.backend_session_manager.report_intermediate_progress(
            {
                "session_id": "session-123",
                "trial_id": "trial-direct",
                "running_score": 0.5,
                "examples_attempted": 1,
            }
        )
        result = await lifecycle.run_trial(
            func=lambda value: value,
            config={"temperature": 0.2},
            dataset=dataset,
            trial_number=1,
            session_id="session-123",
        )

        assert decision == {"prune": False, "prune_reason": None}
        assert result.status == TrialStatus.COMPLETED
        assert result.metrics["accuracy"] == 0.5
        assert len(failing_client.payloads) == 4


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

        assert len(orchestrator._trials) == 1
        rejected_trial = orchestrator._trials[0]
        assert rejected_trial.status == TrialStatus.PRUNED
        assert rejected_trial.metadata["constraint_rejected"] is True
        assert rejected_trial.metadata["stop_reason"] == "trial_rejected_by_constraint"

    @pytest.mark.asyncio
    async def test_constraint_rejection_decrements_optimizer_trial_count(self):
        """Regression: constraint-rejected configs must give back the optimizer's _trial_count slot.

        Bug: suggest_next_trial increments optimizer._trial_count before the constraint
        check in trial_lifecycle. If the constraint rejects, the optimizer still thinks
        it used a trial slot, eventually hitting max_trials before finding valid configs.
        """
        orchestrator = MockOrchestrator()
        orchestrator.max_trials = 10
        # Give the mock optimizer a _trial_count like the real RandomSearchOptimizer
        orchestrator.optimizer._trial_count = 3

        # Real suggest_next_trial increments _trial_count; simulate that
        original_suggest = orchestrator.optimizer.suggest_next_trial

        def suggest_with_increment(completed_trials):
            orchestrator.optimizer._trial_count += 1
            return original_suggest(completed_trials)

        orchestrator.optimizer.suggest_next_trial = suggest_with_increment

        def failing_constraint(config, metrics=None):
            return False

        failing_constraint.__tvl_constraint__ = {
            "id": "test",
            "message": "Always fails",
        }
        orchestrator._constraints_pre_eval = [failing_constraint]

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

        assert action == "continue"
        assert trial_count == 5
        # Key regression assertion: optimizer's _trial_count must be decremented
        # back to its pre-call value so rejected configs don't consume trial budget
        assert orchestrator.optimizer._trial_count == 3
        assert orchestrator._trials[-1].status == TrialStatus.PRUNED

    @pytest.mark.asyncio
    async def test_consecutive_constraint_rejections_get_unique_non_consuming_ids(self):
        """Regression: repeated rejected configs keep unique IDs and refund slots."""
        orchestrator = MockOrchestrator()
        orchestrator.max_trials = 10
        orchestrator.optimizer._trial_count = 3

        configs = iter(
            [
                {"temperature": 0.4, "model": "gpt-4"},
                {"temperature": 0.6, "model": "gpt-4"},
            ]
        )

        def suggest_with_increment(completed_trials):
            orchestrator.optimizer._trial_count += 1
            return next(configs)

        orchestrator.optimizer.suggest_next_trial = suggest_with_increment

        def failing_constraint(config, metrics=None):
            return False

        failing_constraint.__tvl_constraint__ = {
            "id": "test",
            "message": "Always fails",
        }
        orchestrator._constraints_pre_eval = [failing_constraint]

        lifecycle = TrialLifecycle(orchestrator)
        dataset = create_mock_dataset()

        async def mock_func(input_data):
            return "result"

        first_count, first_action = await lifecycle.run_sequential_trial(
            func=mock_func,
            dataset=dataset,
            session_id=None,
            function_name="test_func",
            trial_count=5,
        )
        second_count, second_action = await lifecycle.run_sequential_trial(
            func=mock_func,
            dataset=dataset,
            session_id=None,
            function_name="test_func",
            trial_count=5,
        )

        assert first_action == "continue"
        assert second_action == "continue"
        assert first_count == 5
        assert second_count == 5
        assert orchestrator.optimizer._trial_count == 3

        assert len(orchestrator._trials) == 2
        rejected_trials = orchestrator._trials
        assert all(trial.status == TrialStatus.PRUNED for trial in rejected_trials)
        assert all(
            trial.metadata["constraint_rejected"] is True for trial in rejected_trials
        )
        trial_ids = [trial.trial_id for trial in rejected_trials]
        assert trial_ids == [
            "test-optimization-123_5_rej1",
            "test-optimization-123_5_rej2",
        ]
        assert len(set(trial_ids)) == 2

    @pytest.mark.asyncio
    async def test_constraint_rejection_refunds_optimizer_slot_when_logging_fails(self):
        """Regression: logging failure must not leave optimizer trial slot consumed."""
        orchestrator = MockOrchestrator()
        orchestrator.optimizer._trial_count = 7
        orchestrator._log_trial = MagicMock(side_effect=RuntimeError("log failed"))

        def suggest_with_increment(completed_trials):
            orchestrator.optimizer._trial_count += 1
            return {"temperature": 0.4, "model": "gpt-4"}

        orchestrator.optimizer.suggest_next_trial = suggest_with_increment

        def failing_constraint(config, metrics=None):
            return False

        failing_constraint.__tvl_constraint__ = {
            "id": "test",
            "message": "Always fails",
        }
        orchestrator._constraints_pre_eval = [failing_constraint]

        lifecycle = TrialLifecycle(orchestrator)
        dataset = create_mock_dataset()

        async def mock_func(input_data):
            return "result"

        with pytest.raises(RuntimeError, match="log failed"):
            await lifecycle.run_sequential_trial(
                func=mock_func,
                dataset=dataset,
                session_id=None,
                function_name="test_func",
                trial_count=5,
            )

        assert orchestrator.optimizer._trial_count == 7
        assert len(orchestrator._trials) == 1
        assert orchestrator._trials[0].status == TrialStatus.PRUNED


# =============================================================================
# CancelledError Propagation Tests
# =============================================================================


class TestCancelledErrorPropagation:
    """Verify that asyncio.CancelledError is re-raised (SonarQube S7497)."""

    @pytest.mark.asyncio
    async def test_execute_trial_with_tracing_propagates_cancelled_error(self):
        """CancelledError during evaluation must propagate through _execute_trial_with_tracing."""
        import asyncio

        orchestrator = MockOrchestrator()
        # Make the evaluator raise CancelledError
        orchestrator.evaluator = MagicMock()
        orchestrator.evaluator.evaluate = AsyncMock(side_effect=asyncio.CancelledError)

        lifecycle = TrialLifecycle(orchestrator)

        with pytest.raises(asyncio.CancelledError):
            await lifecycle.run_trial(
                func=lambda x: "result",
                config={"temperature": 0.5},
                dataset=create_mock_dataset(),
                trial_number=0,
                session_id=None,
            )
