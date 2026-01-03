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

        assert session.session_id == "fallback_session"
        assert session.service_name == "LocalFallback"
        assert optimizer._using_fallback
        assert optimizer._fallback_reason.startswith("session_creation:")

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

        with patch("random.sample") as mock_sample:
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
