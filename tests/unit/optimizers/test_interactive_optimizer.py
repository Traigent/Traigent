"""Unit tests for interactive optimizer."""

from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pytest

from traigent.cloud.models import (
    DatasetSubsetIndices,
    NextTrialResponse,
    OptimizationFinalizationResponse,
    OptimizationSession,
    OptimizationSessionStatus,
    SessionCreationResponse,
    TrialStatus,
    TrialSuggestion,
)
from traigent.optimizers.interactive_optimizer import (
    InteractiveOptimizer,
    RemoteGuidanceService,
)
from traigent.utils.exceptions import OptimizationError


@pytest.fixture
def mock_remote_service():
    """Create a mock remote guidance service."""
    service = Mock(spec=RemoteGuidanceService)

    # Setup async mocks
    service.create_session = AsyncMock()
    service.get_next_trial = AsyncMock()
    service.submit_result = AsyncMock()
    service.finalize_session = AsyncMock()

    return service


@pytest.fixture
def optimizer(mock_remote_service):
    """Create an interactive optimizer with mock service."""
    return InteractiveOptimizer(
        config_space={"temperature": (0.0, 1.0), "model": ["gpt-3.5", "GPT-4o"]},
        objectives=["accuracy", "speed"],
        remote_service=mock_remote_service,
        dataset_metadata={"size": 1000, "type": "qa"},
    )


class TestInteractiveOptimizer:
    """Test InteractiveOptimizer functionality."""

    @pytest.mark.asyncio
    async def test_initialize_session(self, optimizer, mock_remote_service):
        """Test initializing an optimization session."""
        # Setup mock response
        mock_remote_service.create_session.return_value = SessionCreationResponse(
            session_id="session-123",
            status=OptimizationSessionStatus.ACTIVE,
            optimization_strategy={"exploration_ratio": 0.3},
        )

        # Initialize session
        session = await optimizer.initialize_session(
            function_name="test_function", max_trials=50, user_id="test_user"
        )

        assert session.session_id == "session-123"
        assert session.function_name == "test_function"
        assert session.max_trials == 50
        assert session.status == OptimizationSessionStatus.ACTIVE
        assert optimizer.session_id == "session-123"

        # Verify service was called correctly
        mock_remote_service.create_session.assert_called_once()
        call_args = mock_remote_service.create_session.call_args[0][0]
        assert call_args.function_name == "test_function"
        assert call_args.max_trials == 50

    @pytest.mark.asyncio
    async def test_initialize_session_failure(self, optimizer, mock_remote_service):
        """Test session initialization failure."""
        mock_remote_service.create_session.side_effect = Exception("Service error")

        with pytest.raises(OptimizationError, match="Session initialization failed"):
            await optimizer.initialize_session("test_function", 50)

    @pytest.mark.asyncio
    async def test_get_next_suggestion(self, optimizer, mock_remote_service):
        """Test getting next trial suggestion."""
        # Initialize session first
        mock_remote_service.create_session.return_value = SessionCreationResponse(
            session_id="session-123",
            status=OptimizationSessionStatus.ACTIVE,
            optimization_strategy={},
        )
        await optimizer.initialize_session("test_function", 50)

        # Setup mock suggestion
        suggestion = TrialSuggestion(
            trial_id="trial-001",
            session_id="session-123",
            trial_number=1,
            config={"temperature": 0.7, "model": "GPT-4o"},
            dataset_subset=DatasetSubsetIndices(
                indices=[0, 5, 10, 15, 20],
                selection_strategy="diverse_sampling",
                confidence_level=0.8,
                estimated_representativeness=0.75,
            ),
            exploration_type="exploration",
        )

        mock_remote_service.get_next_trial.return_value = NextTrialResponse(
            suggestion=suggestion,
            should_continue=True,
            session_status=OptimizationSessionStatus.ACTIVE,
        )

        # Get suggestion
        result = await optimizer.get_next_suggestion(dataset_size=1000)

        assert result is not None
        assert result.trial_id == "trial-001"
        assert result.config["temperature"] == 0.7
        assert len(result.dataset_subset.indices) == 5
        assert result.exploration_type == "exploration"

        # Verify suggestion is tracked
        assert "trial-001" in optimizer._pending_trials

    @pytest.mark.asyncio
    async def test_get_next_suggestion_no_session(self, optimizer):
        """Test getting suggestion without session raises error."""
        with pytest.raises(OptimizationError, match="No active session"):
            await optimizer.get_next_suggestion(dataset_size=1000)

    @pytest.mark.asyncio
    async def test_get_next_suggestion_complete(self, optimizer, mock_remote_service):
        """Test getting suggestion when optimization is complete."""
        # Initialize session
        mock_remote_service.create_session.return_value = SessionCreationResponse(
            session_id="session-123",
            status=OptimizationSessionStatus.ACTIVE,
            optimization_strategy={},
        )
        await optimizer.initialize_session("test_function", 50)

        # Setup response indicating completion
        mock_remote_service.get_next_trial.return_value = NextTrialResponse(
            suggestion=None,
            should_continue=False,
            reason="Max trials reached",
            session_status=OptimizationSessionStatus.COMPLETED,
        )

        result = await optimizer.get_next_suggestion(dataset_size=1000)

        assert result is None
        assert optimizer.session.status == OptimizationSessionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_report_results(self, optimizer, mock_remote_service):
        """Test reporting trial results."""
        # Initialize session
        mock_remote_service.create_session.return_value = SessionCreationResponse(
            session_id="session-123",
            status=OptimizationSessionStatus.ACTIVE,
            optimization_strategy={},
        )
        await optimizer.initialize_session("test_function", 50)

        # Get a suggestion first
        suggestion = TrialSuggestion(
            trial_id="trial-001",
            session_id="session-123",
            trial_number=1,
            config={"temperature": 0.7, "model": "GPT-4o"},
            dataset_subset=DatasetSubsetIndices(
                indices=[0, 1, 2],
                selection_strategy="random",
                confidence_level=0.5,
                estimated_representativeness=0.5,
            ),
            exploration_type="exploration",
        )
        optimizer._pending_trials["trial-001"] = suggestion

        # Report results
        await optimizer.report_results(
            trial_id="trial-001",
            metrics={"accuracy": 0.85, "speed": 0.92},
            duration=45.2,
            status=TrialStatus.COMPLETED,
        )

        # Verify submission
        mock_remote_service.submit_result.assert_called_once()
        submission = mock_remote_service.submit_result.call_args[0][0]
        assert submission.trial_id == "trial-001"
        assert submission.metrics["accuracy"] == 0.85
        assert submission.duration == 45.2

        # Verify local tracking
        assert len(optimizer._completed_trials) == 1
        assert optimizer.session.completed_trials == 1
        assert optimizer.session.best_metrics["accuracy"] == 0.85
        assert "trial-001" not in optimizer._pending_trials

    @pytest.mark.asyncio
    async def test_report_results_no_session(self, optimizer):
        """Test reporting results without session raises error."""
        with pytest.raises(OptimizationError, match="No active session"):
            await optimizer.report_results(
                trial_id="trial-001", metrics={"accuracy": 0.85}, duration=45.2
            )

    @pytest.mark.asyncio
    async def test_report_better_results(self, optimizer, mock_remote_service):
        """Test reporting better results updates best metrics."""
        # Initialize session
        mock_remote_service.create_session.return_value = SessionCreationResponse(
            session_id="session-123",
            status=OptimizationSessionStatus.ACTIVE,
            optimization_strategy={},
        )
        await optimizer.initialize_session("test_function", 50)

        # Report first result
        suggestion1 = TrialSuggestion(
            trial_id="trial-001",
            session_id="session-123",
            trial_number=1,
            config={"temperature": 0.5},
            dataset_subset=DatasetSubsetIndices([0, 1], "random", 0.5, 0.5),
            exploration_type="exploration",
        )
        optimizer._pending_trials["trial-001"] = suggestion1

        await optimizer.report_results(
            trial_id="trial-001",
            metrics={"accuracy": 0.80, "speed": 0.90},
            duration=40.0,
        )

        assert optimizer.session.best_metrics["accuracy"] == 0.80

        # Report better result
        suggestion2 = TrialSuggestion(
            trial_id="trial-002",
            session_id="session-123",
            trial_number=2,
            config={"temperature": 0.7},
            dataset_subset=DatasetSubsetIndices([0, 1], "random", 0.5, 0.5),
            exploration_type="exploitation",
        )
        optimizer._pending_trials["trial-002"] = suggestion2

        await optimizer.report_results(
            trial_id="trial-002",
            metrics={"accuracy": 0.90, "speed": 0.85},
            duration=42.0,
        )

        assert optimizer.session.best_metrics["accuracy"] == 0.90
        assert optimizer.session.best_config["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_get_optimization_status(self, optimizer, mock_remote_service):
        """Test getting optimization status."""
        # Before initialization
        status = await optimizer.get_optimization_status()
        assert status["status"] == "not_initialized"
        assert status["completed_trials"] == 0

        # After initialization
        mock_remote_service.create_session.return_value = SessionCreationResponse(
            session_id="session-123",
            status=OptimizationSessionStatus.ACTIVE,
            optimization_strategy={},
        )
        await optimizer.initialize_session("test_function", 50)

        status = await optimizer.get_optimization_status()
        assert status["status"] == "active"
        assert status["session_id"] == "session-123"
        assert status["completed_trials"] == 0
        assert status["max_trials"] == 50
        assert status["progress"] == 0.0

        # After some trials
        optimizer.session.completed_trials = 10
        optimizer.session.best_metrics = {"accuracy": 0.85}

        status = await optimizer.get_optimization_status()
        assert status["completed_trials"] == 10
        assert status["progress"] == 0.2
        assert status["best_metrics"]["accuracy"] == 0.85

    @pytest.mark.asyncio
    async def test_finalize_optimization(self, optimizer, mock_remote_service):
        """Test finalizing optimization."""
        # Initialize session
        mock_remote_service.create_session.return_value = SessionCreationResponse(
            session_id="session-123",
            status=OptimizationSessionStatus.ACTIVE,
            optimization_strategy={},
        )
        await optimizer.initialize_session("test_function", 50)

        # Setup finalization response
        mock_remote_service.finalize_session.return_value = (
            OptimizationFinalizationResponse(
                session_id="session-123",
                best_config={"temperature": 0.7, "model": "GPT-4o"},
                best_metrics={"accuracy": 0.92, "speed": 0.88},
                total_trials=45,
                successful_trials=43,
                total_duration=3600.0,
                cost_savings=0.68,
            )
        )

        # Finalize
        response = await optimizer.finalize_optimization(include_full_history=True)

        assert response.best_config["temperature"] == 0.7
        assert response.best_metrics["accuracy"] == 0.92
        assert response.successful_trials == 43
        assert response.cost_savings == 0.68

        # Verify session updated
        assert optimizer.session.status == OptimizationSessionStatus.COMPLETED
        assert optimizer.session.best_metrics["accuracy"] == 0.92

        # Verify service called
        mock_remote_service.finalize_session.assert_called_once()
        request = mock_remote_service.finalize_session.call_args[0][0]
        assert request.session_id == "session-123"
        assert request.include_full_history is True

    @pytest.mark.asyncio
    async def test_finalize_no_session(self, optimizer):
        """Test finalizing without session raises error."""
        with pytest.raises(OptimizationError, match="No active session"):
            await optimizer.finalize_optimization()

    def test_suggest_next_trial_not_implemented(self, optimizer):
        """Test synchronous suggest_next_trial raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="requires async usage"):
            optimizer.suggest_next_trial([])

    def test_should_stop(self, optimizer):
        """Test should_stop method."""
        # No session - should stop
        assert optimizer.should_stop([])

        # Create mock session
        optimizer.session = OptimizationSession(
            session_id="test",
            function_name="test",
            configuration_space={},
            objectives=[],
            max_trials=10,
            status=OptimizationSessionStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            completed_trials=5,
        )

        # Active session with trials remaining
        assert not optimizer.should_stop([])

        # Completed all trials
        optimizer.session.completed_trials = 10
        assert optimizer.should_stop([])

        # Session completed
        optimizer.session.completed_trials = 5
        optimizer.session.status = OptimizationSessionStatus.COMPLETED
        assert optimizer.should_stop([])


class TestMetricsComparison:
    """Test metrics comparison logic."""

    def test_is_better_result(self, optimizer):
        """Test _is_better_result method."""
        # No session - any result is better
        assert optimizer._is_better_result({"accuracy": 0.5})

        # Create session with best metrics
        optimizer.session = OptimizationSession(
            session_id="test",
            function_name="test",
            configuration_space={},
            objectives=["accuracy"],
            max_trials=10,
            status=OptimizationSessionStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            best_metrics={"accuracy": 0.8},
        )

        # Better result
        assert optimizer._is_better_result({"accuracy": 0.9})

        # Worse result
        assert not optimizer._is_better_result({"accuracy": 0.7})

        # Equal result
        assert not optimizer._is_better_result({"accuracy": 0.8})

        # Missing metric
        assert not optimizer._is_better_result({"other_metric": 0.9})
