"""Comprehensive tests for CloudOptimizer (cloud_optimizer.py).

This test suite covers:
- CloudOptimizer initialization and configuration
- Session management and fallback handling
- Trial suggestion workflows (basic and smart)
- Error scenarios and recovery mechanisms
- Performance tracking and statistics
- CTD (Combinatorial Test Design) scenarios
- Edge cases and boundary conditions
"""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from traigent.api.types import TrialResult, TrialStatus
from traigent.config.types import TraigentConfig
from traigent.core.objectives import create_default_objectives
from traigent.evaluators.base import EvaluationExample
from traigent.optimizers.base import BaseOptimizer
from traigent.optimizers.cloud_optimizer import CloudOptimizer
from traigent.optimizers.remote_services import (
    DatasetSubset,
    OptimizationSession,
    OptimizationSessionStatus,
    OptimizationStrategy,
    RemoteOptimizationService,
    SmartTrialSuggestion,
)
from traigent.utils.exceptions import OptimizationError, ServiceError
from traigent.utils.objectives import is_minimization_objective

# Test fixtures


@pytest.fixture
def sample_config_space():
    """Sample configuration space for testing."""
    return {
        "model": ["gpt-3.5-turbo", "gpt-4", "claude-3-sonnet"],
        "temperature": {"min": 0.0, "max": 1.0, "type": "float"},
        "max_tokens": {"min": 100, "max": 4000, "type": "int"},
        "top_p": {"min": 0.1, "max": 1.0, "type": "float"},
    }


@pytest.fixture
def sample_objectives():
    """Sample objectives for testing."""
    return ["accuracy", "cost", "latency"]


@pytest.fixture
def mock_remote_service():
    """Mock remote optimization service."""
    service = AsyncMock(spec=RemoteOptimizationService)
    service.service_name = "TestRemoteService"
    service.status.value = "connected"
    service.connect = AsyncMock()
    service.create_session = AsyncMock()
    service.suggest_configuration = AsyncMock()
    service.suggest_smart_trial = AsyncMock()
    service.should_stop_optimization = AsyncMock(return_value=False)
    service.report_trial_result = AsyncMock()
    service.close_session = AsyncMock()
    return service


@pytest.fixture
def mock_fallback_optimizer():
    """Mock fallback optimizer."""
    optimizer = AsyncMock(spec=BaseOptimizer)
    optimizer.__class__.__name__ = "MockFallbackOptimizer"
    optimizer.suggest_next_trial_async = AsyncMock()
    optimizer.should_stop_async = AsyncMock()
    return optimizer


@pytest.fixture
def sample_optimization_strategy():
    """Sample optimization strategy."""
    return OptimizationStrategy(
        max_total_evaluations=1000,
        max_cost_budget=100.0,
        max_time_budget=3600,
        early_stopping_patience=10,
        min_examples_per_trial=5,
    )


@pytest.fixture
def sample_session():
    """Sample optimization session."""
    return OptimizationSession(
        session_id="test_session_123",
        service_name="TestRemoteService",
        config_space={"temperature": {"min": 0.0, "max": 1.0}},
        objectives=["accuracy"],
        algorithm="bayesian",
        status=OptimizationSessionStatus.ACTIVE,
        created_at=datetime.now(UTC),
        optimization_strategy=OptimizationStrategy(),
    )


@pytest.fixture
def sample_trial_results():
    """Sample trial results for testing."""
    from datetime import datetime

    return [
        TrialResult(
            trial_id="trial_1",
            config={"temperature": 0.5},
            metrics={"accuracy": 0.85, "cost": 0.05},
            status=TrialStatus.COMPLETED,
            duration=1.5,
            timestamp=datetime.now(UTC),
            metadata={},
        ),
        TrialResult(
            trial_id="trial_2",
            config={"temperature": 0.7},
            metrics={"accuracy": 0.88, "cost": 0.07},
            status=TrialStatus.COMPLETED,
            duration=1.8,
            timestamp=datetime.now(UTC),
            metadata={},
        ),
        TrialResult(
            trial_id="trial_3",
            config={"temperature": 0.3},
            metrics={"accuracy": 0.82, "cost": 0.03},
            status=TrialStatus.COMPLETED,
            duration=1.2,
            timestamp=datetime.now(UTC),
            metadata={},
        ),
    ]


@pytest.fixture
def sample_dataset():
    """Sample dataset for testing."""
    return [
        EvaluationExample(
            input_data={"prompt": "What is AI?"},
            expected_output="AI is artificial intelligence",
            metadata={},
        ),
        EvaluationExample(
            input_data={"prompt": "Explain machine learning"},
            expected_output="ML is a subset of AI",
            metadata={},
        ),
        EvaluationExample(
            input_data={"prompt": "What is deep learning?"},
            expected_output="Deep learning uses neural networks",
            metadata={},
        ),
    ]


# Test Classes


class TestCloudOptimizerInitialization:
    """Test CloudOptimizer initialization and configuration."""

    def test_basic_initialization(
        self, sample_config_space, sample_objectives, mock_remote_service
    ):
        """Test basic CloudOptimizer initialization."""
        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
        )

        assert optimizer.config_space == sample_config_space
        assert optimizer.objectives == sample_objectives
        assert optimizer.remote_service == mock_remote_service
        assert optimizer.fallback_optimizer is None
        assert not optimizer._using_fallback
        assert optimizer.session_id is None
        assert optimizer._remote_successes == 0
        assert optimizer._remote_failures == 0
        assert optimizer._fallback_uses == 0

    def test_initialization_with_fallback(
        self,
        sample_config_space,
        sample_objectives,
        mock_remote_service,
        mock_fallback_optimizer,
    ):
        """Test initialization with fallback optimizer."""
        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
            fallback_optimizer=mock_fallback_optimizer,
        )

        assert optimizer.fallback_optimizer == mock_fallback_optimizer
        assert not optimizer._using_fallback

    def test_initialization_with_strategy(
        self,
        sample_config_space,
        sample_objectives,
        mock_remote_service,
        sample_optimization_strategy,
    ):
        """Test initialization with optimization strategy."""
        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
            optimization_strategy=sample_optimization_strategy,
        )

        assert optimizer.optimization_strategy == sample_optimization_strategy

    def test_initialization_with_context(
        self, sample_config_space, sample_objectives, mock_remote_service
    ):
        """Test initialization with TraigentConfig context."""
        context = TraigentConfig(
            model="gpt-4",
            temperature=0.7,
            custom_params={
                "api_key": "test_key",
                "mode": "edge_analytics",
                "execution_mode": "cloud",
            },
        )

        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
            context=context,
        )

        assert optimizer.context == context

    def test_initialization_with_all_parameters(
        self,
        sample_config_space,
        sample_objectives,
        mock_remote_service,
        mock_fallback_optimizer,
        sample_optimization_strategy,
    ):
        """Test initialization with all optional parameters."""
        context = TraigentConfig(model="gpt-4", custom_params={"api_key": "test_key"})

        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
            fallback_optimizer=mock_fallback_optimizer,
            optimization_strategy=sample_optimization_strategy,
            context=context,
            custom_param="test_value",
        )

        assert optimizer.remote_service == mock_remote_service
        assert optimizer.fallback_optimizer == mock_fallback_optimizer
        assert optimizer.optimization_strategy == sample_optimization_strategy
        assert optimizer.context == context


class TestSessionManagement:
    """Test optimization session management."""

    @pytest.mark.asyncio
    async def test_successful_session_initialization(
        self,
        sample_config_space,
        sample_objectives,
        mock_remote_service,
        sample_session,
    ):
        """Test successful session initialization."""
        mock_remote_service.create_session.return_value = sample_session

        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
        )

        session = await optimizer.initialize_session()

        assert session == sample_session
        assert optimizer.session_id == sample_session.session_id
        assert optimizer.session == sample_session
        assert not optimizer._using_fallback
        # connect() should not be called since service is already connected
        mock_remote_service.connect.assert_not_called()
        mock_remote_service.create_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_initialization_already_exists(
        self,
        sample_config_space,
        sample_objectives,
        mock_remote_service,
        sample_session,
    ):
        """Test session initialization when session already exists."""
        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
        )

        # Set existing session
        optimizer.session_id = "existing_session"
        optimizer.session = sample_session

        session = await optimizer.initialize_session()

        assert session == sample_session
        # Should not create new session
        mock_remote_service.create_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_session_initialization_service_connection_failure(
        self, sample_config_space, sample_objectives, mock_remote_service
    ):
        """Test session initialization with service connection failure."""
        mock_remote_service.status.value = "disconnected"
        mock_remote_service.connect.side_effect = Exception("Connection failed")

        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
        )

        with pytest.raises(ServiceError, match="No fallback optimizer available"):
            await optimizer.initialize_session()

    @pytest.mark.asyncio
    async def test_session_initialization_fallback_on_failure(
        self,
        sample_config_space,
        sample_objectives,
        mock_remote_service,
        mock_fallback_optimizer,
    ):
        """Test session initialization falls back on remote service failure."""
        mock_remote_service.create_session.side_effect = Exception(
            "Service unavailable"
        )

        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
            fallback_optimizer=mock_fallback_optimizer,
        )

        session = await optimizer.initialize_session()

        # Regression for #870: the public session object must be marked as
        # a fallback so callers cannot mistake it for a real remote session,
        # AND the session_id must be unique (no hardcoded "fallback_session"
        # constant). The is_fallback flag is the canonical signal.
        assert session.is_fallback is True
        assert session.session_id != "fallback_session"
        assert session.session_id.startswith("local_fallback_")
        assert session.service_name == "LocalFallback"
        assert session.metadata.get("fallback_reason", "").startswith(
            "session_creation:"
        )
        assert optimizer._using_fallback
        assert optimizer._fallback_reason.startswith("session_creation:")

    @pytest.mark.asyncio
    async def test_session_initialization_real_remote_session_not_marked_fallback(
        self,
        sample_config_space,
        sample_objectives,
        mock_remote_service,
        sample_session,
    ):
        """Counterpart to the fallback test: a successfully-created remote
        session must report is_fallback=False so callers can confidently
        treat it as a real backend session.
        """
        session = OptimizationSession(
            session_id="real-backend-uuid-1234",
            service_name=mock_remote_service.service_name,
            config_space=sample_config_space,
            objectives=sample_objectives,
            algorithm="grid",
            status=OptimizationSessionStatus.ACTIVE,
            created_at=datetime.now(UTC),
        )
        mock_remote_service.create_session.return_value = session

        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
        )

        returned = await optimizer.initialize_session()

        assert returned.is_fallback is False
        assert returned.session_id == "real-backend-uuid-1234"
        assert not returned.session_id.startswith("local_fallback_")
        assert not optimizer._using_fallback

    @pytest.mark.asyncio
    async def test_close_session_success(
        self, sample_config_space, sample_objectives, mock_remote_service
    ):
        """Test successful session closure."""
        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
        )

        optimizer.session_id = "test_session"
        optimizer.session = MagicMock()
        optimizer._smart_suggestions = [MagicMock()]

        await optimizer.close_session()

        mock_remote_service.close_session.assert_called_once_with("test_session")
        assert optimizer.session_id is None
        assert optimizer.session is None
        assert len(optimizer._smart_suggestions) == 0

    @pytest.mark.asyncio
    async def test_close_session_remote_failure(
        self, sample_config_space, sample_objectives, mock_remote_service
    ):
        """Test session closure with remote service failure."""
        mock_remote_service.close_session.side_effect = Exception("Close failed")

        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
        )

        optimizer.session_id = "test_session"

        # Should not raise exception, just log warning
        await optimizer.close_session()

        assert optimizer.session_id is None


class TestTrialSuggestions:
    """Test trial suggestion functionality."""

    @pytest.mark.asyncio
    async def test_suggest_next_trial_remote_success(
        self,
        sample_config_space,
        sample_objectives,
        mock_remote_service,
        sample_trial_results,
    ):
        """Test successful remote trial suggestion."""
        expected_config = {"temperature": 0.8, "max_tokens": 1000}
        mock_remote_service.suggest_configuration.return_value = expected_config

        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
        )

        # Mock session initialization
        optimizer.session_id = "test_session"

        config = await optimizer.suggest_next_trial_async(sample_trial_results)

        assert config == expected_config
        assert optimizer._remote_successes == 1
        assert optimizer._trial_count == 1
        mock_remote_service.suggest_configuration.assert_called_once()

    @pytest.mark.asyncio
    async def test_suggest_next_trial_fallback_on_remote_failure(
        self,
        sample_config_space,
        sample_objectives,
        mock_remote_service,
        mock_fallback_optimizer,
        sample_trial_results,
    ):
        """Test fallback to local optimizer on remote failure."""
        expected_config = {"temperature": 0.6, "max_tokens": 800}
        mock_remote_service.suggest_configuration.side_effect = Exception(
            "Remote failed"
        )
        mock_fallback_optimizer.suggest_next_trial_async.return_value = expected_config

        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
            fallback_optimizer=mock_fallback_optimizer,
        )

        optimizer.session_id = "test_session"

        config = await optimizer.suggest_next_trial_async(sample_trial_results)

        assert config == expected_config
        assert optimizer._fallback_uses == 1
        assert optimizer._remote_failures == 1
        mock_fallback_optimizer.suggest_next_trial_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_suggest_next_trial_no_fallback_available(
        self,
        sample_config_space,
        sample_objectives,
        mock_remote_service,
        sample_trial_results,
    ):
        """Test failure when no fallback optimizer available."""
        mock_remote_service.suggest_configuration.side_effect = Exception(
            "Remote failed"
        )

        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
        )

        optimizer.session_id = "test_session"

        with pytest.raises(OptimizationError, match="No available optimization method"):
            await optimizer.suggest_next_trial_async(sample_trial_results)

    @pytest.mark.asyncio
    async def test_suggest_next_trial_both_fail(
        self,
        sample_config_space,
        sample_objectives,
        mock_remote_service,
        mock_fallback_optimizer,
        sample_trial_results,
    ):
        """Test failure when both remote and fallback fail."""
        mock_remote_service.suggest_configuration.side_effect = Exception(
            "Remote failed"
        )
        mock_fallback_optimizer.suggest_next_trial_async.side_effect = Exception(
            "Fallback failed"
        )

        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
            fallback_optimizer=mock_fallback_optimizer,
        )

        optimizer.session_id = "test_session"

        with pytest.raises(
            OptimizationError, match="Both remote and fallback suggestions failed"
        ):
            await optimizer.suggest_next_trial_async(sample_trial_results)

    def test_suggest_next_trial_sync_interface(
        self,
        sample_config_space,
        sample_objectives,
        mock_remote_service,
        sample_trial_results,
    ):
        """Test synchronous interface for backward compatibility."""
        expected_config = {"temperature": 0.8}
        mock_remote_service.suggest_configuration.return_value = expected_config

        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
        )

        optimizer.session_id = "test_session"

        with patch("asyncio.run") as mock_run:
            mock_run.return_value = expected_config
            optimizer.suggest_next_trial(sample_trial_results)

        mock_run.assert_called_once()


class TestSmartTrialSuggestions:
    """Test smart trial suggestion functionality."""

    @pytest.fixture
    def sample_smart_suggestion(self, sample_dataset):
        """Sample smart trial suggestion."""
        return SmartTrialSuggestion(
            config={"temperature": 0.7, "max_tokens": 1000},
            dataset_subset=DatasetSubset(
                examples=sample_dataset[:2],
                selection_strategy="confidence_based",
                confidence_level=0.8,
                subset_id="smart_subset_1",
            ),
            exploration_type="exploitation",
            priority=2,
            metadata={"strategy": "focus_on_promising"},
        )

    @pytest.mark.asyncio
    async def test_suggest_smart_trial_remote_success(
        self,
        sample_config_space,
        sample_objectives,
        mock_remote_service,
        sample_trial_results,
        sample_dataset,
        sample_smart_suggestion,
    ):
        """Test successful remote smart trial suggestion."""
        mock_remote_service.suggest_smart_trial.return_value = sample_smart_suggestion

        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
        )

        optimizer.session_id = "test_session"

        suggestion = await optimizer.suggest_smart_trial(
            sample_trial_results, sample_dataset
        )

        assert suggestion == sample_smart_suggestion
        assert len(optimizer._smart_suggestions) == 1
        assert optimizer._smart_suggestions[0] == sample_smart_suggestion
        assert optimizer._remote_successes == 1
        mock_remote_service.suggest_smart_trial.assert_called_once()

    @pytest.mark.asyncio
    async def test_suggest_smart_trial_fallback_on_remote_failure(
        self,
        sample_config_space,
        sample_objectives,
        mock_remote_service,
        mock_fallback_optimizer,
        sample_trial_results,
        sample_dataset,
    ):
        """Test fallback smart suggestion on remote failure."""
        mock_remote_service.suggest_smart_trial.side_effect = Exception("Remote failed")
        mock_fallback_optimizer.suggest_next_trial_async.return_value = {
            "temperature": 0.5
        }

        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
            fallback_optimizer=mock_fallback_optimizer,
        )

        optimizer.session_id = "test_session"

        suggestion = await optimizer.suggest_smart_trial(
            sample_trial_results, sample_dataset
        )

        assert suggestion.config == {"temperature": 0.5}
        assert suggestion.dataset_subset.selection_strategy == "random_fallback"
        assert suggestion.dataset_subset.confidence_level == 0.3
        assert suggestion.exploration_type == "exploration"
        assert suggestion.metadata["fallback"] is True
        assert optimizer._fallback_uses == 1

    @pytest.mark.asyncio
    async def test_suggest_smart_trial_fallback_dataset_subset_size(
        self,
        sample_config_space,
        sample_objectives,
        mock_remote_service,
        mock_fallback_optimizer,
        sample_trial_results,
        sample_dataset,
        sample_optimization_strategy,
    ):
        """Test fallback smart suggestion respects dataset subset size limits."""
        mock_remote_service.suggest_smart_trial.side_effect = Exception("Remote failed")
        mock_fallback_optimizer.suggest_next_trial_async.return_value = {
            "temperature": 0.5
        }

        # Set minimum examples per trial
        sample_optimization_strategy.min_examples_per_trial = 2

        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
            fallback_optimizer=mock_fallback_optimizer,
            optimization_strategy=sample_optimization_strategy,
        )

        optimizer.session_id = "test_session"

        with patch(
            "traigent.optimizers.cloud_optimizer._SECURE_RANDOM.sample"
        ) as mock_sample:
            mock_sample.return_value = sample_dataset[:2]
            suggestion = await optimizer.suggest_smart_trial(
                sample_trial_results, sample_dataset
            )

        mock_sample.assert_called_once_with(sample_dataset, 2)
        assert len(suggestion.dataset_subset.examples) == 2

    @pytest.mark.asyncio
    async def test_suggest_smart_trial_no_fallback_available(
        self,
        sample_config_space,
        sample_objectives,
        mock_remote_service,
        sample_trial_results,
        sample_dataset,
    ):
        """Test smart suggestion failure when no fallback available."""
        mock_remote_service.suggest_smart_trial.side_effect = Exception("Remote failed")

        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
        )

        optimizer.session_id = "test_session"

        with pytest.raises(
            OptimizationError, match="No available smart optimization method"
        ):
            await optimizer.suggest_smart_trial(sample_trial_results, sample_dataset)


class TestStoppingConditions:
    """Test optimization stopping condition logic."""

    @pytest.mark.asyncio
    async def test_should_stop_remote_service_recommends_stop(
        self,
        sample_config_space,
        sample_objectives,
        mock_remote_service,
        sample_trial_results,
    ):
        """Test stopping when remote service recommends stopping."""
        mock_remote_service.should_stop_optimization.return_value = True

        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
        )

        optimizer.session_id = "test_session"

        should_stop = await optimizer.should_stop_async(sample_trial_results)

        assert should_stop is True
        mock_remote_service.should_stop_optimization.assert_called_once()

    @pytest.mark.asyncio
    async def test_should_stop_fallback_on_remote_failure(
        self,
        sample_config_space,
        sample_objectives,
        mock_remote_service,
        mock_fallback_optimizer,
        sample_trial_results,
    ):
        """Test fallback stopping condition on remote failure."""
        mock_remote_service.should_stop_optimization.side_effect = Exception(
            "Remote failed"
        )
        mock_fallback_optimizer.should_stop_async.return_value = False

        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
            fallback_optimizer=mock_fallback_optimizer,
        )

        optimizer.session_id = "test_session"

        should_stop = await optimizer.should_stop_async(sample_trial_results)

        assert should_stop is False
        mock_fallback_optimizer.should_stop_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_should_stop_strategy_based_conditions(
        self,
        sample_config_space,
        sample_objectives,
        mock_remote_service,
        sample_optimization_strategy,
    ):
        """Test strategy-based stopping conditions."""
        # Set up strategy to stop after max evaluations
        sample_optimization_strategy.max_total_evaluations = 10

        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
            optimization_strategy=sample_optimization_strategy,
        )

        # Create smart suggestions that exceed evaluation budget
        smart_suggestion = MagicMock()
        smart_suggestion.dataset_subset.examples = [
            MagicMock()
        ] * 15  # 15 examples > 10 limit
        optimizer._smart_suggestions = [smart_suggestion]

        should_stop = await optimizer.should_stop_async([])

        assert should_stop is True

    @pytest.mark.asyncio
    async def test_should_stop_cost_budget_exceeded(
        self,
        sample_config_space,
        sample_objectives,
        mock_remote_service,
        sample_optimization_strategy,
    ):
        """Test stopping when cost budget is exceeded."""
        sample_optimization_strategy.max_cost_budget = 50.0

        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
            optimization_strategy=sample_optimization_strategy,
        )

        # Create expensive smart suggestions
        expensive_suggestion = MagicMock()
        expensive_suggestion.estimated_cost = 60.0  # Exceeds budget
        optimizer._smart_suggestions = [expensive_suggestion]

        should_stop = await optimizer.should_stop_async([])

        assert should_stop is True

    @pytest.mark.asyncio
    async def test_should_stop_time_budget_exceeded(
        self,
        sample_config_space,
        sample_objectives,
        mock_remote_service,
        sample_optimization_strategy,
    ):
        """Test stopping when time budget is exceeded."""
        sample_optimization_strategy.max_time_budget = 10  # 10 seconds

        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
            optimization_strategy=sample_optimization_strategy,
        )

        # Create session that started 20 seconds ago
        old_session = MagicMock()
        old_session.created_at = datetime.now(UTC) - timedelta(seconds=20)
        optimizer.session = old_session

        should_stop = await optimizer.should_stop_async([])

        assert should_stop is True

    @pytest.mark.asyncio
    async def test_should_stop_early_stopping_patience(
        self,
        sample_config_space,
        sample_objectives,
        mock_remote_service,
        sample_optimization_strategy,
    ):
        """Test early stopping based on patience parameter."""
        sample_optimization_strategy.early_stopping_patience = 3

        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=["accuracy"],
            remote_service=mock_remote_service,
            optimization_strategy=sample_optimization_strategy,
        )

        # Create trial history with no improvement in recent trials
        history = [
            TrialResult(
                "t1",
                {"temp": 0.5},
                {"accuracy": 0.9},
                TrialStatus.COMPLETED,
                1.0,
                datetime.now(UTC),
            ),  # Best overall
            TrialResult(
                "t2",
                {"temp": 0.6},
                {"accuracy": 0.8},
                TrialStatus.COMPLETED,
                1.0,
                datetime.now(UTC),
            ),  # Recent, worse
            TrialResult(
                "t3",
                {"temp": 0.7},
                {"accuracy": 0.8},
                TrialStatus.COMPLETED,
                1.0,
                datetime.now(UTC),
            ),  # Recent, worse
            TrialResult(
                "t4",
                {"temp": 0.8},
                {"accuracy": 0.8},
                TrialStatus.COMPLETED,
                1.0,
                datetime.now(UTC),
            ),  # Recent, worse
        ]

        should_stop = await optimizer.should_stop_async(history)

        assert should_stop is True

    def test_should_stop_sync_interface(
        self, sample_config_space, sample_objectives, mock_remote_service
    ):
        """Test synchronous stopping interface."""
        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
        )

        with patch("asyncio.run") as mock_run:
            mock_run.return_value = False
            optimizer.should_stop([])

        mock_run.assert_called_once()


class TestResultReporting:
    """Test trial result reporting functionality."""

    @pytest.mark.asyncio
    async def test_report_trial_result_success(
        self,
        sample_config_space,
        sample_objectives,
        mock_remote_service,
        sample_trial_results,
    ):
        """Test successful trial result reporting."""
        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
        )

        optimizer.session_id = "test_session"
        trial_result = sample_trial_results[0]

        await optimizer.report_trial_result(trial_result)

        mock_remote_service.report_trial_result.assert_called_once_with(
            "test_session", trial_result
        )

    @pytest.mark.asyncio
    async def test_report_trial_result_failure_no_fallback_switch(
        self,
        sample_config_space,
        sample_objectives,
        mock_remote_service,
        sample_trial_results,
    ):
        """Test that reporting failures don't switch to fallback mode."""
        mock_remote_service.report_trial_result.side_effect = Exception(
            "Reporting failed"
        )

        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
        )

        optimizer.session_id = "test_session"
        trial_result = sample_trial_results[0]

        # Should not raise exception or switch to fallback
        await optimizer.report_trial_result(trial_result)

        assert not optimizer._using_fallback

    @pytest.mark.asyncio
    async def test_report_trial_result_using_fallback_mode(
        self,
        sample_config_space,
        sample_objectives,
        mock_remote_service,
        sample_trial_results,
    ):
        """Test that reporting is skipped when using fallback mode."""
        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
        )

        optimizer._using_fallback = True
        optimizer.session_id = "test_session"
        trial_result = sample_trial_results[0]

        await optimizer.report_trial_result(trial_result)

        # Should not attempt remote reporting in fallback mode
        mock_remote_service.report_trial_result.assert_not_called()


class TestPerformanceTracking:
    """Test performance tracking and statistics."""

    def test_get_optimization_stats_initial_state(
        self, sample_config_space, sample_objectives, mock_remote_service
    ):
        """Test optimization stats in initial state."""
        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
        )

        stats = optimizer.get_optimization_stats()

        assert stats["remote_service"] == "TestRemoteService"
        assert stats["using_fallback"] is False
        assert stats["fallback_reason"] is None
        assert stats["remote_successes"] == 0
        assert stats["remote_failures"] == 0
        assert stats["remote_success_rate"] == 0.0
        assert stats["fallback_uses"] == 0
        assert stats["total_trials"] == 0
        assert stats["smart_suggestions_count"] == 0
        assert stats["session_id"] is None

    def test_get_optimization_stats_with_activity(
        self, sample_config_space, sample_objectives, mock_remote_service
    ):
        """Test optimization stats after some activity."""
        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
        )

        # Simulate some activity
        optimizer._remote_successes = 7
        optimizer._remote_failures = 3
        optimizer._fallback_uses = 2
        optimizer._trial_count = 12
        optimizer.session_id = "active_session"
        optimizer._smart_suggestions = [MagicMock(), MagicMock()]

        stats = optimizer.get_optimization_stats()

        assert stats["remote_successes"] == 7
        assert stats["remote_failures"] == 3
        assert stats["remote_success_rate"] == 0.7  # 7/10
        assert stats["fallback_uses"] == 2
        assert stats["total_trials"] == 12
        assert stats["smart_suggestions_count"] == 2
        assert stats["session_id"] == "active_session"

    def test_get_smart_suggestions_history(
        self, sample_config_space, sample_objectives, mock_remote_service
    ):
        """Test getting smart suggestions history."""
        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
        )

        # Add some smart suggestions
        suggestion1 = MagicMock()
        suggestion2 = MagicMock()
        optimizer._smart_suggestions = [suggestion1, suggestion2]

        history = optimizer.get_smart_suggestions_history()

        assert len(history) == 2
        assert suggestion1 in history
        assert suggestion2 in history
        # Verify it returns a copy, not the original
        assert history is not optimizer._smart_suggestions

    def test_handle_remote_failure_tracking(
        self,
        sample_config_space,
        sample_objectives,
        mock_remote_service,
        mock_fallback_optimizer,
    ):
        """Test remote failure handling and tracking."""
        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
            fallback_optimizer=mock_fallback_optimizer,
        )

        # Simulate multiple failures
        for i in range(4):  # Exceeds failure threshold of 3
            optimizer._handle_remote_failure("test_operation", Exception(f"Error {i}"))

        assert optimizer._remote_failures == 4
        assert optimizer._using_fallback is True
        assert "Remote test_operation failed" in optimizer._fallback_reason

    def test_handle_remote_failure_no_fallback(
        self, sample_config_space, sample_objectives, mock_remote_service
    ):
        """Test remote failure handling without fallback optimizer."""
        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
        )

        # Should track failures but not switch to fallback mode
        optimizer._handle_remote_failure("test_operation", Exception("Error"))

        assert optimizer._remote_failures == 1
        assert optimizer._using_fallback is False


class TestAlgorithmInfo:
    """Test algorithm information functionality."""

    def test_get_algorithm_info_basic(
        self, sample_config_space, sample_objectives, mock_remote_service
    ):
        """Test basic algorithm info."""
        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
        )

        info = optimizer.get_algorithm_info()

        assert info["remote_service"] == "TestRemoteService"
        assert info["fallback_optimizer"] is None
        assert info["supports_smart_suggestions"] is True
        assert info["supports_adaptive_datasets"] is True
        assert info["using_fallback"] is False

    def test_get_algorithm_info_with_fallback(
        self,
        sample_config_space,
        sample_objectives,
        mock_remote_service,
        mock_fallback_optimizer,
        sample_optimization_strategy,
    ):
        """Test algorithm info with fallback and strategy."""
        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
            fallback_optimizer=mock_fallback_optimizer,
            optimization_strategy=sample_optimization_strategy,
        )

        optimizer._using_fallback = True

        info = optimizer.get_algorithm_info()

        assert info["fallback_optimizer"] == "MockFallbackOptimizer"
        assert info["using_fallback"] is True
        assert (
            info["optimization_strategy"] == sample_optimization_strategy.strategy_name
        )


class TestCTDScenarios:
    """Combinatorial Test Design scenarios for comprehensive coverage."""

    @pytest.mark.parametrize(
        "has_fallback,remote_fails,expected_fallback_use",
        [
            (True, True, True),
            (True, False, False),
            (False, True, "raises_error"),
            (False, False, False),
        ],
    )
    @pytest.mark.asyncio
    async def test_fallback_combinations(
        self,
        sample_config_space,
        sample_objectives,
        mock_remote_service,
        mock_fallback_optimizer,
        sample_trial_results,
        has_fallback,
        remote_fails,
        expected_fallback_use,
    ):
        """Test combinations of fallback availability and remote service failures."""
        if remote_fails:
            mock_remote_service.suggest_configuration.side_effect = Exception(
                "Remote failed"
            )
        else:
            mock_remote_service.suggest_configuration.return_value = {
                "temperature": 0.5
            }

        if has_fallback:
            mock_fallback_optimizer.suggest_next_trial_async.return_value = {
                "temperature": 0.6
            }
            fallback_opt = mock_fallback_optimizer
        else:
            fallback_opt = None

        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
            fallback_optimizer=fallback_opt,
        )

        optimizer.session_id = "test_session"

        if expected_fallback_use == "raises_error":
            with pytest.raises(OptimizationError):
                await optimizer.suggest_next_trial_async(sample_trial_results)
        else:
            config = await optimizer.suggest_next_trial_async(sample_trial_results)

            if expected_fallback_use:
                assert optimizer._fallback_uses > 0
                assert config == {"temperature": 0.6}
            else:
                assert optimizer._fallback_uses == 0
                if not remote_fails:
                    assert config == {"temperature": 0.5}

    @pytest.mark.parametrize(
        "budget_type,budget_exceeded",
        [
            ("evaluations", True),
            ("evaluations", False),
            ("cost", True),
            ("cost", False),
            ("time", True),
            ("time", False),
        ],
    )
    @pytest.mark.asyncio
    async def test_budget_stopping_combinations(
        self,
        sample_config_space,
        sample_objectives,
        mock_remote_service,
        budget_type,
        budget_exceeded,
    ):
        """Test different budget types and exceeded conditions."""
        strategy = OptimizationStrategy()

        if budget_type == "evaluations":
            strategy.max_total_evaluations = 5 if budget_exceeded else 100
            examples_count = 10 if budget_exceeded else 3
        elif budget_type == "cost":
            strategy.max_cost_budget = 10.0 if budget_exceeded else 100.0
            cost_amount = 15.0 if budget_exceeded else 5.0
        elif budget_type == "time":
            strategy.max_time_budget = 5 if budget_exceeded else 3600
            time_offset = 10 if budget_exceeded else 2

        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
            optimization_strategy=strategy,
        )

        if budget_type == "evaluations":
            # Create smart suggestions with examples
            suggestion = MagicMock()
            suggestion.dataset_subset.examples = [MagicMock()] * examples_count
            optimizer._smart_suggestions = [suggestion]

        elif budget_type == "cost":
            # Create smart suggestions with cost
            suggestion = MagicMock()
            suggestion.estimated_cost = cost_amount
            optimizer._smart_suggestions = [suggestion]

        elif budget_type == "time":
            # Create session with time offset
            session = MagicMock()
            session.created_at = datetime.now(UTC) - timedelta(seconds=time_offset)
            optimizer.session = session

        should_stop = await optimizer.should_stop_async([])

        assert should_stop == budget_exceeded

    @pytest.mark.parametrize(
        "service_status,session_fails,suggestion_fails,expected_mode",
        [
            ("connected", False, False, "remote"),
            ("connected", True, False, "fallback"),
            ("connected", False, True, "fallback"),
            ("disconnected", False, False, "fallback"),
            ("error", False, False, "fallback"),
        ],
    )
    @pytest.mark.asyncio
    async def test_service_state_combinations(
        self,
        sample_config_space,
        sample_objectives,
        mock_remote_service,
        mock_fallback_optimizer,
        service_status,
        session_fails,
        suggestion_fails,
        expected_mode,
    ):
        """Test combinations of service states and operation failures."""
        mock_remote_service.status.value = service_status

        if service_status != "connected":
            mock_remote_service.connect.side_effect = Exception("Cannot connect")

        if session_fails:
            mock_remote_service.create_session.side_effect = Exception("Session failed")
        else:
            session = MagicMock()
            session.session_id = "test_session"
            mock_remote_service.create_session.return_value = session

        if suggestion_fails:
            mock_remote_service.suggest_configuration.side_effect = Exception(
                "Suggestion failed"
            )
        else:
            mock_remote_service.suggest_configuration.return_value = {
                "temperature": 0.5
            }

        mock_fallback_optimizer.suggest_next_trial_async.return_value = {
            "temperature": 0.6
        }

        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
            fallback_optimizer=mock_fallback_optimizer,
        )

        try:
            await optimizer.initialize_session()
            config = await optimizer.suggest_next_trial_async([])

            if expected_mode == "remote":
                assert not optimizer._using_fallback
                assert config == {"temperature": 0.5}
            else:  # fallback
                assert optimizer._using_fallback or optimizer._fallback_uses > 0
                if optimizer._fallback_uses > 0:
                    assert config == {"temperature": 0.6}

        except ServiceError:
            # Expected for some error conditions without fallback
            pass


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_trial_history(
        self, sample_config_space, sample_objectives, mock_remote_service
    ):
        """Test operations with empty trial history."""
        mock_remote_service.suggest_configuration.return_value = {"temperature": 0.5}
        mock_remote_service.should_stop_optimization.return_value = False

        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
        )

        optimizer.session_id = "test_session"

        # Should handle empty history gracefully
        config = await optimizer.suggest_next_trial_async([])
        should_stop = await optimizer.should_stop_async([])

        assert config == {"temperature": 0.5}
        assert should_stop is False

    @pytest.mark.asyncio
    async def test_large_trial_history(
        self, sample_config_space, sample_objectives, mock_remote_service
    ):
        """Test operations with very large trial history."""
        # Create large history
        large_history = []
        for i in range(1000):
            trial = TrialResult(
                trial_id=f"trial_{i}",
                config={"temperature": 0.5 + i * 0.0001},
                metrics={"accuracy": 0.8 + i * 0.0001},
                status=TrialStatus.COMPLETED,
                duration=1.0,
                timestamp=datetime.now(UTC),
                metadata={},
            )
            large_history.append(trial)

        mock_remote_service.suggest_configuration.return_value = {"temperature": 0.5}

        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
        )

        optimizer.session_id = "test_session"

        # Should handle large history efficiently
        config = await optimizer.suggest_next_trial_async(large_history)

        assert config == {"temperature": 0.5}
        mock_remote_service.suggest_configuration.assert_called_once_with(
            "test_session", large_history, None
        )

    @pytest.mark.asyncio
    async def test_malformed_trial_results(
        self, sample_config_space, sample_objectives, mock_remote_service
    ):
        """Test handling of malformed trial results."""
        # Create malformed trial results
        malformed_history = [
            TrialResult(
                "t1",
                {"temp": 0.5},
                {},
                TrialStatus.FAILED,
                1.0,
                datetime.now(UTC),
            ),  # No metrics
            TrialResult(
                "t2",
                {},
                {"accuracy": 0.8},
                TrialStatus.COMPLETED,
                1.0,
                datetime.now(UTC),
            ),  # No config
            TrialResult(
                "t3",
                {"temp": 0.5},
                {"accuracy": None},
                TrialStatus.COMPLETED,
                1.0,
                datetime.now(UTC),
            ),  # None metric
        ]

        mock_remote_service.suggest_configuration.return_value = {"temperature": 0.5}

        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=["accuracy"],
            remote_service=mock_remote_service,
        )

        optimizer.session_id = "test_session"

        # Should handle malformed data gracefully
        config = await optimizer.suggest_next_trial_async(malformed_history)
        await optimizer.should_stop_async(malformed_history)

        assert config == {"temperature": 0.5}
        # Should not crash on malformed data

    @pytest.mark.asyncio
    async def test_concurrent_operations(
        self, sample_config_space, sample_objectives, mock_remote_service
    ):
        """Test thread safety with concurrent operations."""
        mock_remote_service.suggest_configuration.return_value = {"temperature": 0.5}

        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
        )

        optimizer.session_id = "test_session"

        # Run multiple concurrent operations
        tasks = []
        for _ in range(10):
            task = optimizer.suggest_next_trial_async([])
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # All should succeed
        assert len(results) == 10
        assert all(result == {"temperature": 0.5} for result in results)
        assert optimizer._trial_count == 10

    def test_memory_cleanup_after_close(
        self, sample_config_space, sample_objectives, mock_remote_service
    ):
        """Test memory cleanup after session close."""
        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
        )

        # Set up some state
        optimizer.session_id = "test_session"
        optimizer.session = MagicMock()
        optimizer._smart_suggestions = [MagicMock(), MagicMock()]

        # Close session
        asyncio.run(optimizer.close_session())

        # Verify cleanup
        assert optimizer.session_id is None
        assert optimizer.session is None
        assert len(optimizer._smart_suggestions) == 0

    @pytest.mark.asyncio
    async def test_network_timeout_handling(
        self, sample_config_space, sample_objectives, mock_remote_service
    ):
        """Test handling of network timeouts."""

        # Simulate timeout
        mock_remote_service.suggest_configuration.side_effect = TimeoutError(
            "Network timeout"
        )

        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
        )

        optimizer.session_id = "test_session"

        with pytest.raises(OptimizationError, match="No available optimization method"):
            await optimizer.suggest_next_trial_async([])

        # Should track as remote failure
        assert optimizer._remote_failures == 1


class TestGenerateCandidates:
    """Test candidate generation functionality."""

    @pytest.mark.asyncio
    async def test_generate_candidates_remote_batch(
        self, sample_config_space, sample_objectives, mock_remote_service
    ):
        """Test candidate generation via remote batch."""
        expected_candidates = [
            {"temperature": 0.5},
            {"temperature": 0.6},
            {"temperature": 0.7},
        ]
        mock_remote_service.suggest_batch = AsyncMock(return_value=expected_candidates)

        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
        )
        optimizer.session_id = "test_session"

        candidates = await optimizer.generate_candidates_async(3)

        assert candidates == expected_candidates
        assert optimizer._remote_successes == 1
        mock_remote_service.suggest_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_candidates_remote_sequential(
        self, sample_config_space, sample_objectives, mock_remote_service
    ):
        """Test candidate generation via sequential remote calls."""
        # Remove suggest_batch to force sequential fallback
        del mock_remote_service.suggest_batch

        mock_remote_service.suggest_configuration.side_effect = [
            {"temperature": 0.5},
            {"temperature": 0.6},
            {"temperature": 0.7},
        ]

        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
        )
        optimizer.session_id = "test_session"

        candidates = await optimizer.generate_candidates_async(3)

        assert len(candidates) == 3
        assert optimizer._remote_successes == 3

    @pytest.mark.asyncio
    async def test_generate_candidates_fallback_on_remote_failure(
        self,
        sample_config_space,
        sample_objectives,
        mock_remote_service,
        mock_fallback_optimizer,
    ):
        """Test candidate generation falls back when remote fails."""
        mock_remote_service.suggest_batch = AsyncMock(
            side_effect=Exception("Remote failed")
        )
        mock_fallback_optimizer.generate_candidates_async = AsyncMock(
            return_value=[{"temperature": 0.6}]
        )

        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
            fallback_optimizer=mock_fallback_optimizer,
        )
        optimizer.session_id = "test_session"

        candidates = await optimizer.generate_candidates_async(3)

        assert candidates == [{"temperature": 0.6}]
        assert optimizer._fallback_uses == 1

    @pytest.mark.asyncio
    async def test_generate_candidates_sequential_stops_on_exception(
        self, sample_config_space, sample_objectives, mock_remote_service
    ):
        """Test sequential generation stops when exception occurs."""
        del mock_remote_service.suggest_batch

        mock_remote_service.suggest_configuration.side_effect = [
            {"temperature": 0.5},
            Exception("Failed"),
        ]

        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
        )
        optimizer.session_id = "test_session"

        candidates = await optimizer.generate_candidates_async(3)

        # Should have only one successful candidate before failure
        assert len(candidates) == 1
        assert candidates[0] == {"temperature": 0.5}

    @pytest.mark.asyncio
    async def test_generate_candidates_fallback_failure_raises(
        self,
        sample_config_space,
        sample_objectives,
        mock_remote_service,
        mock_fallback_optimizer,
    ):
        """Test that failure in both remote and fallback raises error."""
        mock_remote_service.suggest_batch = AsyncMock(
            side_effect=Exception("Remote failed")
        )
        mock_fallback_optimizer.generate_candidates_async = AsyncMock(
            side_effect=Exception("Fallback failed")
        )

        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
            fallback_optimizer=mock_fallback_optimizer,
        )
        optimizer.session_id = "test_session"
        optimizer._using_fallback = True  # Force using fallback

        with pytest.raises(OptimizationError, match="Both remote and fallback"):
            await optimizer.generate_candidates_async(3)

    @pytest.mark.asyncio
    async def test_generate_candidates_initializes_session_if_needed(
        self,
        sample_config_space,
        sample_objectives,
        mock_remote_service,
        sample_session,
    ):
        """Test that generating candidates initializes session if needed."""
        mock_remote_service.create_session.return_value = sample_session
        mock_remote_service.suggest_batch = AsyncMock(
            return_value=[{"temperature": 0.5}]
        )

        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
        )
        # Don't set session_id - should be auto-initialized

        candidates = await optimizer.generate_candidates_async(3)

        # Session should have been initialized
        assert optimizer.session_id == sample_session.session_id
        assert candidates == [{"temperature": 0.5}]

    @pytest.mark.asyncio
    async def test_generate_candidates_empty_batch_uses_fallback(
        self,
        sample_config_space,
        sample_objectives,
        mock_remote_service,
        mock_fallback_optimizer,
    ):
        """Test that empty batch result uses fallback."""
        mock_remote_service.suggest_batch = AsyncMock(return_value=[])
        # Need to also clear suggest_configuration since it's on the mock
        del mock_remote_service.suggest_configuration
        mock_fallback_optimizer.generate_candidates_async = AsyncMock(
            return_value=[{"temperature": 0.5}]
        )

        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
            fallback_optimizer=mock_fallback_optimizer,
        )
        optimizer.session_id = "test_session"

        candidates = await optimizer.generate_candidates_async(3)

        assert candidates == [{"temperature": 0.5}]
        assert optimizer._fallback_uses == 1


class TestSessionNotInitializedErrors:
    """Test errors when session is not initialized."""

    @pytest.mark.asyncio
    async def test_generate_candidates_remote_no_session(
        self, sample_config_space, sample_objectives, mock_remote_service
    ):
        """Test _generate_candidates_remote raises when session is None."""
        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
        )
        optimizer.session_id = None

        with pytest.raises(ServiceError, match="Session not initialized"):
            await optimizer._generate_candidates_remote(3, None)

    @pytest.mark.asyncio
    async def test_generate_candidates_sequential_no_session(
        self, sample_config_space, sample_objectives, mock_remote_service
    ):
        """Test _generate_candidates_sequential raises when session is None."""
        optimizer = CloudOptimizer(
            config_space=sample_config_space,
            objectives=sample_objectives,
            remote_service=mock_remote_service,
        )
        optimizer.session_id = None

        with pytest.raises(ServiceError, match="Session not initialized"):
            await optimizer._generate_candidates_sequential(3, None)


class TestSessionConstant:
    """Test the _SESSION_NOT_INITIALIZED constant."""

    def test_session_not_initialized_constant_used(
        self, sample_config_space, sample_objectives, mock_remote_service
    ):
        """Test that _SESSION_NOT_INITIALIZED constant is defined correctly."""
        from traigent.optimizers.cloud_optimizer import _SESSION_NOT_INITIALIZED

        assert _SESSION_NOT_INITIALIZED == "Session not initialized"


class TestStrategyEarlyStoppingOrientation:
    """#1915: the patience-based early-stop must respect the primary objective's
    orientation instead of the old hard-coded maximize (``max()`` + upward test).
    """

    @staticmethod
    def _trial(trial_id: str, obj: str, value: float) -> TrialResult:
        return TrialResult(
            trial_id=trial_id,
            config={},
            metrics={obj: value},
            status=TrialStatus.COMPLETED,
            duration=1.0,
            timestamp=datetime.now(UTC),
            metadata={},
        )

    def _optimizer(self, objectives, strategy, mock_remote_service):
        return CloudOptimizer(
            config_space={"temperature": {"min": 0.0, "max": 1.0, "type": "float"}},
            objectives=objectives,
            remote_service=mock_remote_service,
            optimization_strategy=strategy,
        )

    def test_minimize_still_improving_does_not_stop(self, mock_remote_service):
        """Minimize primary objective (cost) whose recent trials keep dropping
        below the earlier best must NOT trigger the no-improvement stop.

        Under the old maximize-only code, ``max()`` treated the largest cost as
        "best" and the inverted ``<`` test fired a premature stop here.
        """
        strategy = OptimizationStrategy(early_stopping_patience=3)
        optimizer = self._optimizer(["cost"], strategy, mock_remote_service)
        history = [
            self._trial("t1", "cost", 0.5),
            self._trial("t2", "cost", 0.4),
            self._trial("t3", "cost", 0.3),
            self._trial("t4", "cost", 0.2),
            self._trial("t5", "cost", 0.1),
        ]
        assert optimizer._check_strategy_stopping_conditions(history) is False

    def test_minimize_no_improvement_stops(self, mock_remote_service):
        """Minimize primary objective whose recent trials are all worse than the
        overall best must trigger the no-improvement stop.
        """
        strategy = OptimizationStrategy(early_stopping_patience=3)
        optimizer = self._optimizer(["cost"], strategy, mock_remote_service)
        history = [
            self._trial("t1", "cost", 0.1),
            self._trial("t2", "cost", 0.1),
            self._trial("t3", "cost", 0.5),
            self._trial("t4", "cost", 0.6),
            self._trial("t5", "cost", 0.7),
        ]
        assert optimizer._check_strategy_stopping_conditions(history) is True

    def test_maximize_behaviour_preserved(self, mock_remote_service):
        """Maximize primary objective (accuracy) keeps the original semantics:
        recent best well below the overall best stops.
        """
        strategy = OptimizationStrategy(early_stopping_patience=3)
        optimizer = self._optimizer(["accuracy"], strategy, mock_remote_service)
        history = [
            self._trial("t1", "accuracy", 0.9),
            self._trial("t2", "accuracy", 0.9),
            self._trial("t3", "accuracy", 0.5),
            self._trial("t4", "accuracy", 0.4),
            self._trial("t5", "accuracy", 0.3),
        ]
        assert optimizer._check_strategy_stopping_conditions(history) is True

    def test_configurable_min_delta(self, mock_remote_service):
        """The improvement epsilon is driven by ``early_stopping_min_delta``
        (default 0.01) rather than a hard-coded literal, and it is an
        IMPROVEMENT threshold: the recent window must beat the pre-window
        baseline by MORE than min_delta to keep going, so a larger delta makes
        stopping MORE likely, never less.
        """
        # Baseline best (0.90) sits in t1, before the patience=2 recent window;
        # the recent best (0.92) rises 0.02 above it.
        history = [
            self._trial("t1", "accuracy", 0.90),
            self._trial("t2", "accuracy", 0.87),
            self._trial("t3", "accuracy", 0.92),
        ]
        # Default delta (0.01): 0.92 > 0.90 + 0.01 -> real improvement -> keep going.
        opt_default = self._optimizer(
            ["accuracy"],
            OptimizationStrategy(early_stopping_patience=2),
            mock_remote_service,
        )
        assert opt_default._check_strategy_stopping_conditions(history) is False
        # Larger delta (0.05): 0.92 <= 0.90 + 0.05 -> marginal gain ignored -> stop.
        opt_demanding = self._optimizer(
            ["accuracy"],
            OptimizationStrategy(
                early_stopping_patience=2, early_stopping_min_delta=0.05
            ),
            mock_remote_service,
        )
        assert opt_demanding._check_strategy_stopping_conditions(history) is True

    def test_minimize_configurable_min_delta_respects_direction(
        self, mock_remote_service
    ):
        """The same improvement threshold applies in the lower-is-better
        direction: the window must get more than min_delta BELOW the pre-window
        baseline to count as improving.
        """
        # Baseline best (0.100) sits in t1; the recent window's best (0.095)
        # drops 0.005 below it.
        history = [
            self._trial("t1", "cost", 0.100),
            self._trial("t2", "cost", 0.105),
            self._trial("t3", "cost", 0.095),
        ]

        # Small delta (0.001): 0.095 < 0.100 - 0.001 -> real improvement -> keep going.
        opt_lenient = self._optimizer(
            ["cost"],
            OptimizationStrategy(
                early_stopping_patience=2, early_stopping_min_delta=0.001
            ),
            mock_remote_service,
        )
        assert opt_lenient._check_strategy_stopping_conditions(history) is False

        # Larger delta (0.01): 0.095 >= 0.100 - 0.01 -> marginal gain ignored -> stop.
        opt_demanding = self._optimizer(
            ["cost"],
            OptimizationStrategy(
                early_stopping_patience=2, early_stopping_min_delta=0.01
            ),
            mock_remote_service,
        )
        assert opt_demanding._check_strategy_stopping_conditions(history) is True

    def test_flat_plateau_stops_maximize(self, mock_remote_service):
        """The canonical no-improvement case: a completely flat maximize history
        of length > patience MUST stop. Under the pre-fix arithmetic
        (window compared against the best of ALL history, window included) a
        plateau had ``best_recent == best_overall`` and never stopped.
        """
        strategy = OptimizationStrategy(early_stopping_patience=3)
        optimizer = self._optimizer(["accuracy"], strategy, mock_remote_service)
        history = [
            self._trial(f"t{i}", "accuracy", 0.8) for i in range(1, 6)
        ]  # 5 identical scores, patience window [t3..t5], baseline [t1, t2]
        assert optimizer._check_strategy_stopping_conditions(history) is True

    def test_flat_plateau_stops_minimize(self, mock_remote_service):
        """Same canonical plateau in the lower-is-better direction stops."""
        strategy = OptimizationStrategy(early_stopping_patience=3)
        optimizer = self._optimizer(["cost"], strategy, mock_remote_service)
        history = [self._trial(f"t{i}", "cost", 0.3) for i in range(1, 6)]
        assert optimizer._check_strategy_stopping_conditions(history) is True

    def test_sub_min_delta_improvement_still_stops(self, mock_remote_service):
        """A gain smaller than min_delta inside the window does NOT count as
        improvement (and must not defer stopping)."""
        strategy = OptimizationStrategy(
            early_stopping_patience=3, early_stopping_min_delta=0.01
        )
        optimizer = self._optimizer(["accuracy"], strategy, mock_remote_service)
        history = [
            self._trial("t1", "accuracy", 0.800),
            self._trial("t2", "accuracy", 0.790),
            self._trial("t3", "accuracy", 0.805),  # +0.005 <= min_delta
            self._trial("t4", "accuracy", 0.795),
            self._trial("t5", "accuracy", 0.800),
        ]
        assert optimizer._check_strategy_stopping_conditions(history) is True

    def test_material_improvement_does_not_stop(self, mock_remote_service):
        """A gain larger than min_delta inside the window IS improvement."""
        strategy = OptimizationStrategy(
            early_stopping_patience=3, early_stopping_min_delta=0.01
        )
        optimizer = self._optimizer(["accuracy"], strategy, mock_remote_service)
        history = [
            self._trial("t1", "accuracy", 0.800),
            self._trial("t2", "accuracy", 0.790),
            self._trial("t3", "accuracy", 0.850),  # +0.05 > min_delta
            self._trial("t4", "accuracy", 0.795),
            self._trial("t5", "accuracy", 0.800),
        ]
        assert optimizer._check_strategy_stopping_conditions(history) is False

    def test_history_equal_to_patience_does_not_stop(self, mock_remote_service):
        """With history no longer than the patience window there is no
        pre-window baseline to compare against -> not enough evidence to stop,
        even on a flat plateau."""
        strategy = OptimizationStrategy(early_stopping_patience=3)
        optimizer = self._optimizer(["accuracy"], strategy, mock_remote_service)
        history = [self._trial(f"t{i}", "accuracy", 0.8) for i in range(1, 4)]
        assert optimizer._check_strategy_stopping_conditions(history) is False

    def test_invalid_baseline_scores_do_not_stop(self, mock_remote_service):
        """If every pre-window score coerces to invalid, there is no usable
        baseline -> do not stop (and do not crash on min()/max() of nothing)."""
        strategy = OptimizationStrategy(early_stopping_patience=3)
        optimizer = self._optimizer(["accuracy"], strategy, mock_remote_service)
        history = [
            self._trial("t1", "accuracy", float("nan")),
            self._trial("t2", "accuracy", "bad"),
            self._trial("t3", "accuracy", 0.8),
            self._trial("t4", "accuracy", 0.8),
            self._trial("t5", "accuracy", 0.8),
        ]
        assert optimizer._check_strategy_stopping_conditions(history) is False

    def test_empty_objectives_guarded(self, mock_remote_service):
        """No objectives + a patience gate must not raise IndexError (sibling of
        #1909's ``objectives[0]`` guard)."""
        strategy = OptimizationStrategy(early_stopping_patience=2)
        optimizer = self._optimizer([], strategy, mock_remote_service)
        history = [
            self._trial("t1", "cost", 0.5),
            self._trial("t2", "cost", 0.4),
        ]
        assert optimizer._check_strategy_stopping_conditions(history) is False


class TestStrategyEarlyStoppingRobustness:
    """Follow-up hardening of the patience-based early-stop gate:

    * a ``band`` (target-range) primary objective is not directional, so the raw
      maximize/minimize plateau arithmetic must be bypassed for it; and
    * every metric value is filtered through ``coerce_finite_objective_score`` so
      NaN / infinities / bool / str / None values cannot poison or crash the min/max
      comparisons.
    """

    @staticmethod
    def _trial(trial_id: str, obj: str, value) -> TrialResult:
        return TrialResult(
            trial_id=trial_id,
            config={},
            metrics={obj: value},
            status=TrialStatus.COMPLETED,
            duration=1.0,
            timestamp=datetime.now(UTC),
            metadata={},
        )

    def _optimizer(
        self, objectives, strategy, mock_remote_service, objective_schema=None
    ):
        return CloudOptimizer(
            config_space={"temperature": {"min": 0.0, "max": 1.0, "type": "float"}},
            objectives=objectives,
            remote_service=mock_remote_service,
            optimization_strategy=strategy,
            objective_schema=objective_schema,
        )

    @staticmethod
    def _band_schema(name: str):
        from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema
        from traigent.tvl.models import BandTarget

        return ObjectiveSchema.from_objectives(
            [
                ObjectiveDefinition(
                    name=name,
                    orientation="band",
                    weight=1.0,
                    band=BandTarget(low=0.7, high=0.9),
                )
            ]
        )

    def test_band_primary_bypasses_patience_plateau_gate(self, mock_remote_service):
        """A band primary objective must NOT run the raw maximize plateau test.

        The identical plateau history stops a maximize objective, but for a band
        objective, whose "best" is closeness to a target interval, not a
        direction, the gate must be bypassed. Running the raw-max arithmetic
        would stop a run that is correctly parked inside the band.
        """
        strategy = OptimizationStrategy(early_stopping_patience=3)
        # Overall best (0.90) sits in t1; the recent window plateaus below it.
        history = [
            self._trial("t1", "accuracy", 0.90),
            self._trial("t2", "accuracy", 0.80),
            self._trial("t3", "accuracy", 0.80),
            self._trial("t4", "accuracy", 0.80),
        ]

        # Control: maximize orientation (no schema, name heuristic) DOES stop;
        # this is exactly the raw-max verdict the band path must avoid.
        maximize_opt = self._optimizer(["accuracy"], strategy, mock_remote_service)
        assert maximize_opt._check_strategy_stopping_conditions(history) is True

        # Band orientation bypasses only the plateau gate -> does not stop here.
        band_opt = self._optimizer(
            ["accuracy"],
            strategy,
            mock_remote_service,
            objective_schema=self._band_schema("accuracy"),
        )
        assert band_opt._check_strategy_stopping_conditions(history) is False

    def test_maximize_ignores_invalid_metric_values(self, mock_remote_service):
        """NaN / bool / str values among a maximize objective's recent trials are
        dropped, not allowed to poison or crash ``max()``; the valid values still
        drive a correct stop.
        """
        strategy = OptimizationStrategy(early_stopping_patience=4)
        history = [
            self._trial("t1", "accuracy", 0.90),  # valid best overall
            self._trial("t2", "accuracy", float("nan")),  # recent, dropped
            self._trial("t3", "accuracy", True),  # recent bool, dropped
            self._trial("t4", "accuracy", "oops"),  # recent str, would crash max()
            self._trial("t5", "accuracy", 0.80),  # recent, valid
        ]
        optimizer = self._optimizer(["accuracy"], strategy, mock_remote_service)
        # Valid recent best 0.80 <= baseline best 0.90 + 0.01 -> stop.
        assert optimizer._check_strategy_stopping_conditions(history) is True

    def test_minimize_ignores_invalid_metric_values(self, mock_remote_service):
        """Infinities / None among a minimize objective's recent trials are dropped
        rather than poison (``-inf`` would win ``min``) or crash (``None``) the
        comparison; the valid values still drive a correct stop.
        """
        strategy = OptimizationStrategy(early_stopping_patience=4)
        history = [
            self._trial("t1", "cost", 0.10),  # valid best overall (lowest)
            self._trial("t2", "cost", float("-inf")),  # recent, dropped
            self._trial("t3", "cost", float("inf")),  # recent, dropped
            self._trial("t4", "cost", None),  # recent, would crash min()
            self._trial("t5", "cost", 0.20),  # recent, valid
        ]
        optimizer = self._optimizer(["cost"], strategy, mock_remote_service)
        # Valid recent best 0.20 >= baseline best 0.10 - 0.01 -> stop.
        assert optimizer._check_strategy_stopping_conditions(history) is True

    def test_insufficient_valid_data_does_not_stop_or_crash(self, mock_remote_service):
        """When every recent trial's metric is invalid, the coerced recent set is
        empty and the gate falls through to "do not stop" instead of crashing on
        ``max()`` / ``min()`` of unrankable values.
        """
        strategy = OptimizationStrategy(early_stopping_patience=3)
        history = [
            self._trial("t1", "accuracy", float("nan")),
            self._trial("t2", "accuracy", "bad"),
            self._trial("t3", "accuracy", None),
        ]
        optimizer = self._optimizer(["accuracy"], strategy, mock_remote_service)
        assert optimizer._check_strategy_stopping_conditions(history) is False


class TestOptimizationStrategyPositionalABI:
    """``early_stopping_min_delta`` must be appended AFTER every pre-existing
    field so the historical positional-constructor ABI is preserved.

    The field was originally inserted between ``early_stopping_patience`` and
    ``confidence_threshold``, which silently shifted ``confidence_threshold``
    and all later positional values by one slot. This constructs the full
    legacy positional form (the field order as it shipped on ``develop``) and
    proves every legacy field keeps its old binding while the new field takes
    its default. The distinct sentinel per slot makes any off-by-one shift a
    hard failure rather than a coincidental pass.
    """

    def test_legacy_full_positional_construction_preserves_bindings(self):
        strategy = OptimizationStrategy(
            123,  # max_total_evaluations
            45.6,  # max_cost_budget
            78.9,  # max_time_budget
            0.42,  # exploration_ratio
            7,  # early_stopping_patience
            0.88,  # confidence_threshold
            3,  # min_examples_per_trial
            99,  # max_examples_per_trial
            False,  # adaptive_sample_size
            "speed",  # pareto_preference
            {"accuracy": 2.0},  # objective_weights
            "legacy_name",  # strategy_name
            {"k": "v"},  # metadata
        )

        assert strategy.max_total_evaluations == 123
        assert strategy.max_cost_budget == 45.6
        assert strategy.max_time_budget == 78.9
        assert strategy.exploration_ratio == 0.42
        assert strategy.early_stopping_patience == 7
        # The bug bound this positional slot to early_stopping_min_delta.
        assert strategy.confidence_threshold == 0.88
        assert strategy.min_examples_per_trial == 3
        assert strategy.max_examples_per_trial == 99
        assert strategy.adaptive_sample_size is False
        assert strategy.pareto_preference == "speed"
        assert strategy.objective_weights == {"accuracy": 2.0}
        assert strategy.strategy_name == "legacy_name"
        assert strategy.metadata == {"k": "v"}
        # New field is not part of the legacy positional ABI: it defaults.
        assert strategy.early_stopping_min_delta == 0.01

    def test_new_field_is_last_positional_slot(self):
        """A 14th positional argument binds to the new field, confirming it was
        appended rather than inserted mid-sequence."""
        strategy = OptimizationStrategy(
            123,
            45.6,
            78.9,
            0.42,
            7,
            0.88,
            3,
            99,
            False,
            "speed",
            {"accuracy": 2.0},
            "legacy_name",
            {"k": "v"},
            0.05,  # early_stopping_min_delta (new trailing slot)
        )
        assert strategy.early_stopping_min_delta == 0.05
        assert strategy.confidence_threshold == 0.88


class TestDeclaredOrientationOverridesHeuristic:
    """The patience-based early-stop must honour the orientation declared in the
    user's ``ObjectiveSchema``, not the name-pattern heuristic.

    ``is_minimization_objective`` guesses direction from substrings ("cost",
    "latency", "error", ...). Any lower-is-better metric whose name lacks those
    substrings is silently treated as maximize, which inverts the no-improvement
    test. The declared schema is authoritative; the heuristic is only a fallback
    for objectives no schema declares.
    """

    # Lower-is-better, but matches no _MINIMIZE_OBJECTIVE_PATTERNS substring, so
    # the heuristic misclassifies it as maximize.
    MISCLASSIFIED_OBJECTIVE = "regret"

    @staticmethod
    def _trial(trial_id: str, obj: str, value: float) -> TrialResult:
        return TrialResult(
            trial_id=trial_id,
            config={},
            metrics={obj: value},
            status=TrialStatus.COMPLETED,
            duration=1.0,
            timestamp=datetime.now(UTC),
            metadata={},
        )

    def _optimizer(self, objectives, strategy, mock_remote_service, schema=None):
        return CloudOptimizer(
            config_space={"temperature": {"min": 0.0, "max": 1.0, "type": "float"}},
            objectives=objectives,
            remote_service=mock_remote_service,
            optimization_strategy=strategy,
            objective_schema=schema,
        )

    def _plateaued_history(self, obj: str) -> list[TrialResult]:
        """Overall best (0.1) is early; the recent window is uniformly worse.

        Under minimize this is a genuine no-improvement plateau (best_recent 0.5
        is far above best_overall 0.1 -> stop). Under the heuristic's maximize
        reading, best_recent == best_overall == 0.7 -> no stop. The two
        orientations therefore disagree, which is what makes this discriminating.
        """
        return [
            self._trial("t1", obj, 0.1),
            self._trial("t2", obj, 0.1),
            self._trial("t3", obj, 0.5),
            self._trial("t4", obj, 0.6),
            self._trial("t5", obj, 0.7),
        ]

    def test_heuristic_misclassifies_the_objective_name(self):
        """Pin the premise: without an orientation the heuristic calls this
        lower-is-better metric a maximize objective."""
        assert is_minimization_objective(self.MISCLASSIFIED_OBJECTIVE) is False
        assert (
            is_minimization_objective(
                self.MISCLASSIFIED_OBJECTIVE, orientation="minimize"
            )
            is True
        )

    @pytest.mark.asyncio
    async def test_declared_minimize_reaches_production_stopping_path(
        self, mock_remote_service
    ):
        """A schema-declared minimize objective must stop through the public
        ``should_stop_async`` path, not merely in the private helper.

        The remote service is asserted untouched to prove the stop decision came
        from the strategy gate rather than from the mock's own verdict.
        """
        schema = create_default_objectives(
            [self.MISCLASSIFIED_OBJECTIVE],
            orientations={self.MISCLASSIFIED_OBJECTIVE: "minimize"},
        )
        optimizer = self._optimizer(
            [self.MISCLASSIFIED_OBJECTIVE],
            OptimizationStrategy(early_stopping_patience=3),
            mock_remote_service,
            schema=schema,
        )
        history = self._plateaued_history(self.MISCLASSIFIED_OBJECTIVE)

        assert await optimizer.should_stop_async(history) is True
        mock_remote_service.should_stop_optimization.assert_not_called()

    @pytest.mark.asyncio
    async def test_without_schema_heuristic_misses_the_plateau(
        self, mock_remote_service
    ):
        """Same history, no declared schema: the heuristic's maximize reading
        fails to see the plateau and the strategy gate does not fire.

        This is the pre-fix behaviour, retained as the backward-compatibility
        baseline for string-only objective flows and as the contrast that makes
        the test above meaningful.
        """
        optimizer = self._optimizer(
            [self.MISCLASSIFIED_OBJECTIVE],
            OptimizationStrategy(early_stopping_patience=3),
            mock_remote_service,
        )
        history = self._plateaued_history(self.MISCLASSIFIED_OBJECTIVE)

        assert optimizer._check_strategy_stopping_conditions(history) is False

    def test_declared_maximize_overrides_minimize_heuristic(self, mock_remote_service):
        """Inverse direction: "cost" reads as minimize to the heuristic, but a
        schema declaring maximize must win."""
        schema = create_default_objectives(["cost"], orientations={"cost": "maximize"})
        optimizer = self._optimizer(
            ["cost"],
            OptimizationStrategy(early_stopping_patience=3),
            mock_remote_service,
            schema=schema,
        )
        # Rising values: improving under maximize (no stop); under the minimize
        # heuristic these would look like a plateau and stop.
        history = self._plateaued_history("cost")

        assert optimizer._check_strategy_stopping_conditions(history) is False

    def test_objective_absent_from_schema_falls_back_to_heuristic(
        self, mock_remote_service
    ):
        """A schema that does not declare the primary objective genuinely lacks
        an explicit value, so the heuristic still applies."""
        schema = create_default_objectives(["accuracy"])
        optimizer = self._optimizer(
            ["cost"],
            OptimizationStrategy(early_stopping_patience=3),
            mock_remote_service,
            schema=schema,
        )
        assert optimizer._primary_objective_orientation("cost") is None
        # Heuristic minimize semantics: recent trials keep dropping -> no stop.
        history = [
            self._trial("t1", "cost", 0.5),
            self._trial("t2", "cost", 0.4),
            self._trial("t3", "cost", 0.3),
            self._trial("t4", "cost", 0.2),
            self._trial("t5", "cost", 0.1),
        ]
        assert optimizer._check_strategy_stopping_conditions(history) is False
