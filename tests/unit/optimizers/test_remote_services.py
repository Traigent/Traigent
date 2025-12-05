"""Comprehensive tests for remote optimization services (remote_services.py).

This test suite covers:
- Service status and info management
- Optimization session lifecycle
- Dataset subset management
- Smart trial suggestions
- Service metrics and health checks
- Error scenarios and edge cases
- CTD (Combinatorial Test Design) scenarios
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from traigent.api.types import TrialResult, TrialStatus
from traigent.evaluators.base import EvaluationExample
from traigent.optimizers.remote_services import (
    DatasetSubset,
    OptimizationSession,
    OptimizationSessionStatus,
    OptimizationStrategy,
    RemoteOptimizationService,
    ServiceInfo,
    ServiceMetrics,
    ServiceStatus,
    SmartTrialSuggestion,
)
from traigent.utils.exceptions import ServiceError

# Test fixtures


@pytest.fixture
def sample_service_info():
    """Sample service information."""
    return ServiceInfo(
        name="TestService",
        version="1.0.0",
        supported_algorithms=["bayesian", "grid", "random"],
        max_concurrent_sessions=10,
        capabilities={
            "smart_suggestions": True,
            "multi_objective": True,
            "adaptive_datasets": True,
        },
        endpoints={
            "optimize": "/v1/optimize",
            "sessions": "/v1/sessions",
            "health": "/v1/health",
        },
        status=ServiceStatus.CONNECTED,
    )


@pytest.fixture
def sample_evaluation_examples():
    """Sample evaluation examples."""
    return [
        EvaluationExample(
            input_data={"prompt": "What is AI?"},
            expected_output="AI is artificial intelligence",
            metadata={"difficulty": "easy", "category": "definition"},
        ),
        EvaluationExample(
            input_data={"prompt": "Explain quantum computing"},
            expected_output="Quantum computing uses quantum mechanics",
            metadata={"difficulty": "hard", "category": "technical"},
        ),
        EvaluationExample(
            input_data={"prompt": "What is machine learning?"},
            expected_output="ML is a subset of AI",
            metadata={"difficulty": "medium", "category": "definition"},
        ),
        EvaluationExample(
            input_data={"prompt": "How do neural networks work?"},
            expected_output="Neural networks process information through layers",
            metadata={"difficulty": "hard", "category": "technical"},
        ),
        EvaluationExample(
            input_data={"prompt": "What is data science?"},
            expected_output="Data science extracts insights from data",
            metadata={"difficulty": "easy", "category": "definition"},
        ),
    ]


@pytest.fixture
def sample_dataset_subset(sample_evaluation_examples):
    """Sample dataset subset."""
    return DatasetSubset(
        examples=sample_evaluation_examples[:3],
        selection_strategy="confidence_based",
        confidence_level=0.8,
        subset_id="subset_123",
        metadata={
            "original_indices": [0, 1, 2],
            "selection_criteria": {"min_difficulty": "easy", "max_difficulty": "hard"},
            "diversity_score": 0.85,
        },
    )


@pytest.fixture
def sample_optimization_strategy():
    """Sample optimization strategy."""
    return OptimizationStrategy(
        max_total_evaluations=1000,
        max_cost_budget=50.0,
        max_time_budget=3600,
        exploration_ratio=0.3,
        early_stopping_patience=10,
        confidence_threshold=0.95,
        min_examples_per_trial=5,
        max_examples_per_trial=50,
        adaptive_sample_size=True,
        pareto_preference="balanced",
        objective_weights={"accuracy": 0.6, "cost": 0.4},
        strategy_name="test_strategy",
        metadata={"version": "1.0", "creator": "test"},
    )


@pytest.fixture
def sample_optimization_session(sample_optimization_strategy):
    """Sample optimization session."""
    return OptimizationSession(
        session_id="session_abc123",
        service_name="TestService",
        config_space={
            "temperature": {"min": 0.0, "max": 1.0, "type": "float"},
            "max_tokens": {"min": 100, "max": 4000, "type": "int"},
        },
        objectives=["accuracy", "cost"],
        algorithm="bayesian",
        status=OptimizationSessionStatus.ACTIVE,
        created_at=datetime.now(),
        metadata={"project": "test", "user": "test_user"},
        max_trials=100,
        timeout=3600.0,
        optimization_strategy=sample_optimization_strategy,
        trials_completed=5,
        evaluations_performed=25,
        total_cost=12.5,
        total_time=120.0,
        best_score=0.85,
        best_config={"temperature": 0.7, "max_tokens": 1000},
        pareto_frontier=[
            {"config": {"temperature": 0.7}, "accuracy": 0.85, "cost": 0.1},
            {"config": {"temperature": 0.5}, "accuracy": 0.82, "cost": 0.08},
        ],
        confidence_in_optimum=0.75,
    )


@pytest.fixture
def sample_smart_suggestion(sample_dataset_subset):
    """Sample smart trial suggestion."""
    return SmartTrialSuggestion(
        config={"temperature": 0.8, "max_tokens": 1500},
        dataset_subset=sample_dataset_subset,
        exploration_type="exploitation",
        expected_value=0.87,
        uncertainty=0.05,
        estimated_cost=2.5,
        estimated_duration=30.0,
        priority=3,
        suggestion_id="sugg_456",
        metadata={
            "reasoning": "High-confidence region exploitation",
            "similar_configs": 2,
            "exploration_depth": 3,
        },
    )


@pytest.fixture
def sample_trial_results():
    """Sample trial results."""
    return [
        TrialResult(
            trial_id="trial_1",
            config={"temperature": 0.5, "max_tokens": 1000},
            metrics={"accuracy": 0.85, "cost": 0.05, "latency": 0.8},
            status=TrialStatus.COMPLETED,
            duration=25.0,
            timestamp=datetime.now(),
            metadata={"dataset_size": 10},
        ),
        TrialResult(
            trial_id="trial_2",
            config={"temperature": 0.7, "max_tokens": 1500},
            metrics={"accuracy": 0.88, "cost": 0.08, "latency": 1.2},
            status=TrialStatus.COMPLETED,
            duration=40.0,
            timestamp=datetime.now(),
            metadata={"dataset_size": 15},
        ),
        TrialResult(
            trial_id="trial_3",
            config={"temperature": 0.3, "max_tokens": 800},
            metrics={"accuracy": 0.82, "cost": 0.03, "latency": 0.6},
            status=TrialStatus.COMPLETED,
            duration=20.0,
            timestamp=datetime.now(),
            metadata={"dataset_size": 8},
        ),
        TrialResult(
            trial_id="trial_4",
            config={"temperature": 0.9, "max_tokens": 2000},
            metrics={},
            status=TrialStatus.FAILED,
            duration=60.0,
            timestamp=datetime.now(),
            error_message="timeout",
            metadata={},
        ),
    ]


# Mock service implementation for testing


class MockRemoteOptimizationService(RemoteOptimizationService):
    """Mock implementation of RemoteOptimizationService for testing."""

    def __init__(self, service_name: str = "MockService", **kwargs):
        super().__init__(service_name, "http://mock.service", **kwargs)
        self._connected = False
        self._sessions = {}
        self._health_status = {"status": "healthy", "uptime": 3600}

    async def connect(self) -> ServiceInfo:
        self._status = ServiceStatus.CONNECTED
        self._connected = True
        return ServiceInfo(
            name=self.service_name,
            version="1.0.0",
            supported_algorithms=["bayesian", "random"],
            max_concurrent_sessions=5,
            status=ServiceStatus.CONNECTED,
        )

    async def disconnect(self) -> None:
        self._status = ServiceStatus.DISCONNECTED
        self._connected = False
        self._sessions.clear()

    async def health_check(self) -> Dict[str, Any]:
        return self._health_status

    async def create_session(
        self, config_space, objectives, algorithm="bayesian", **kwargs
    ) -> OptimizationSession:
        if not self._connected:
            raise ServiceError("Service not connected")

        session_id = f"mock_session_{uuid.uuid4().hex[:8]}"
        session = OptimizationSession(
            session_id=session_id,
            service_name=self.service_name,
            config_space=config_space,
            objectives=objectives,
            algorithm=algorithm,
            status=OptimizationSessionStatus.ACTIVE,
            created_at=datetime.now(),
            **kwargs,
        )
        self._sessions[session_id] = session
        return session

    async def get_session(self, session_id: str) -> OptimizationSession:
        if session_id not in self._sessions:
            raise ServiceError(f"Session {session_id} not found")
        return self._sessions[session_id]

    async def close_session(self, session_id: str) -> None:
        if session_id in self._sessions:
            self._sessions[session_id].status = OptimizationSessionStatus.COMPLETED
            del self._sessions[session_id]

    async def suggest_configuration(
        self,
        session_id: str,
        trial_history: List[TrialResult],
        remote_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self._status != ServiceStatus.CONNECTED:
            raise ServiceError("Service not connected")
        if session_id not in self._sessions:
            raise ServiceError(f"Session {session_id} not found")
        return {"temperature": 0.6, "max_tokens": 1200}

    async def report_trial_result(
        self, session_id: str, trial_result: TrialResult
    ) -> None:
        if session_id not in self._sessions:
            raise ServiceError(f"Session {session_id} not found")
        # Update session metrics
        session = self._sessions[session_id]
        session.trials_completed += 1

    async def should_stop_optimization(
        self, session_id: str, trial_history: List[TrialResult]
    ) -> bool:
        if session_id not in self._sessions:
            raise ServiceError(f"Session {session_id} not found")
        # Simple stopping condition for testing
        return len(trial_history) >= 10


# Test Classes


class TestServiceStatus:
    """Test service status enumeration."""

    def test_service_status_values(self):
        """Test ServiceStatus enum values."""
        assert ServiceStatus.DISCONNECTED == "disconnected"
        assert ServiceStatus.CONNECTING == "connecting"
        assert ServiceStatus.CONNECTED == "connected"
        assert ServiceStatus.ERROR == "error"
        assert ServiceStatus.UNAVAILABLE == "unavailable"

    def test_service_status_membership(self):
        """Test ServiceStatus membership."""
        status_values = {status.value for status in ServiceStatus}
        assert "connected" in status_values
        assert "invalid" not in status_values


class TestOptimizationSessionStatus:
    """Test optimization session status enumeration."""

    def test_session_status_values(self):
        """Test OptimizationSessionStatus enum values."""
        assert OptimizationSessionStatus.INITIALIZING == "initializing"
        assert OptimizationSessionStatus.ACTIVE == "active"
        assert OptimizationSessionStatus.PAUSED == "paused"
        assert OptimizationSessionStatus.COMPLETED == "completed"
        assert OptimizationSessionStatus.FAILED == "failed"
        assert OptimizationSessionStatus.CANCELLED == "cancelled"


class TestServiceInfo:
    """Test ServiceInfo dataclass."""

    def test_service_info_creation(self, sample_service_info):
        """Test ServiceInfo creation with all fields."""
        assert sample_service_info.name == "TestService"
        assert sample_service_info.version == "1.0.0"
        assert "bayesian" in sample_service_info.supported_algorithms
        assert sample_service_info.max_concurrent_sessions == 10
        assert sample_service_info.capabilities["smart_suggestions"] is True
        assert sample_service_info.status == ServiceStatus.CONNECTED

    def test_service_info_minimal_creation(self):
        """Test ServiceInfo creation with minimal fields."""
        info = ServiceInfo(
            name="MinimalService",
            version="0.1.0",
            supported_algorithms=["random"],
            max_concurrent_sessions=1,
        )

        assert info.name == "MinimalService"
        assert info.capabilities == {}
        assert info.endpoints == {}
        assert info.status == ServiceStatus.DISCONNECTED


class TestOptimizationSession:
    """Test OptimizationSession dataclass."""

    def test_session_creation(self, sample_optimization_session):
        """Test OptimizationSession creation with all fields."""
        session = sample_optimization_session

        assert session.session_id == "session_abc123"
        assert session.service_name == "TestService"
        assert "temperature" in session.config_space
        assert "accuracy" in session.objectives
        assert session.algorithm == "bayesian"
        assert session.status == OptimizationSessionStatus.ACTIVE
        assert session.trials_completed == 5
        assert session.best_score == 0.85
        assert len(session.pareto_frontier) == 2

    def test_session_minimal_creation(self):
        """Test OptimizationSession creation with minimal fields."""
        session = OptimizationSession(
            session_id="minimal_session",
            service_name="TestService",
            config_space={"temp": {"min": 0, "max": 1}},
            objectives=["accuracy"],
            algorithm="random",
            status=OptimizationSessionStatus.INITIALIZING,
            created_at=datetime.now(),
        )

        assert session.session_id == "minimal_session"
        assert session.trials_completed == 0
        assert session.best_score is None
        assert session.pareto_frontier == []
        assert session.confidence_in_optimum == 0.0


class TestServiceMetrics:
    """Test ServiceMetrics dataclass."""

    def test_service_metrics_creation(self):
        """Test ServiceMetrics creation."""
        now = datetime.now()
        metrics = ServiceMetrics(
            response_time_ms=150.5,
            success_rate=0.95,
            total_requests=100,
            failed_requests=5,
            last_request_time=now,
            average_session_duration=300.0,
            active_sessions=3,
        )

        assert metrics.response_time_ms == 150.5
        assert metrics.success_rate == 0.95
        assert metrics.total_requests == 100
        assert metrics.failed_requests == 5
        assert metrics.last_request_time == now
        assert metrics.average_session_duration == 300.0
        assert metrics.active_sessions == 3


class TestDatasetSubset:
    """Test DatasetSubset dataclass and functionality."""

    def test_dataset_subset_creation(
        self, sample_dataset_subset, sample_evaluation_examples
    ):
        """Test DatasetSubset creation."""
        subset = sample_dataset_subset

        assert len(subset.examples) == 3
        assert subset.selection_strategy == "confidence_based"
        assert subset.confidence_level == 0.8
        assert subset.subset_id == "subset_123"
        assert "original_indices" in subset.metadata

    def test_dataset_subset_size_property(self, sample_dataset_subset):
        """Test DatasetSubset.size property."""
        assert sample_dataset_subset.size == 3

    def test_dataset_subset_indices_property(self, sample_dataset_subset):
        """Test DatasetSubset.indices property."""
        indices = sample_dataset_subset.indices
        assert indices == [0, 1, 2]

    def test_dataset_subset_indices_default(self, sample_evaluation_examples):
        """Test DatasetSubset.indices default behavior."""
        subset = DatasetSubset(
            examples=sample_evaluation_examples[:2],
            selection_strategy="random",
            confidence_level=0.5,
            subset_id="test",
        )

        indices = subset.indices
        assert indices == [0, 1]  # Default range

    def test_dataset_subset_empty(self):
        """Test DatasetSubset with empty examples."""
        subset = DatasetSubset(
            examples=[],
            selection_strategy="empty",
            confidence_level=0.0,
            subset_id="empty",
        )

        assert subset.size == 0
        assert subset.indices == []


class TestSmartTrialSuggestion:
    """Test SmartTrialSuggestion dataclass."""

    def test_smart_suggestion_creation(self, sample_smart_suggestion):
        """Test SmartTrialSuggestion creation with all fields."""
        suggestion = sample_smart_suggestion

        assert suggestion.config == {"temperature": 0.8, "max_tokens": 1500}
        assert suggestion.dataset_subset.size == 3
        assert suggestion.exploration_type == "exploitation"
        assert suggestion.expected_value == 0.87
        assert suggestion.uncertainty == 0.05
        assert suggestion.estimated_cost == 2.5
        assert suggestion.estimated_duration == 30.0
        assert suggestion.priority == 3
        assert suggestion.suggestion_id == "sugg_456"
        assert "reasoning" in suggestion.metadata

    def test_smart_suggestion_minimal_creation(self, sample_dataset_subset):
        """Test SmartTrialSuggestion creation with minimal fields."""
        suggestion = SmartTrialSuggestion(
            config={"temperature": 0.5},
            dataset_subset=sample_dataset_subset,
            exploration_type="exploration",
        )

        assert suggestion.config == {"temperature": 0.5}
        assert suggestion.expected_value is None
        assert suggestion.uncertainty is None
        assert suggestion.priority == 0
        assert len(suggestion.suggestion_id) == 8  # Default UUID hex[:8]
        assert suggestion.metadata == {}

    def test_smart_suggestion_auto_id_generation(self, sample_dataset_subset):
        """Test automatic suggestion ID generation."""
        suggestion1 = SmartTrialSuggestion(
            config={"temp": 0.5},
            dataset_subset=sample_dataset_subset,
            exploration_type="exploration",
        )

        suggestion2 = SmartTrialSuggestion(
            config={"temp": 0.7},
            dataset_subset=sample_dataset_subset,
            exploration_type="exploitation",
        )

        assert suggestion1.suggestion_id != suggestion2.suggestion_id
        assert len(suggestion1.suggestion_id) == 8
        assert len(suggestion2.suggestion_id) == 8


class TestOptimizationStrategy:
    """Test OptimizationStrategy dataclass."""

    def test_optimization_strategy_creation(self, sample_optimization_strategy):
        """Test OptimizationStrategy creation with all fields."""
        strategy = sample_optimization_strategy

        assert strategy.max_total_evaluations == 1000
        assert strategy.max_cost_budget == 50.0
        assert strategy.max_time_budget == 3600
        assert strategy.exploration_ratio == 0.3
        assert strategy.early_stopping_patience == 10
        assert strategy.confidence_threshold == 0.95
        assert strategy.min_examples_per_trial == 5
        assert strategy.max_examples_per_trial == 50
        assert strategy.adaptive_sample_size is True
        assert strategy.pareto_preference == "balanced"
        assert strategy.objective_weights == {"accuracy": 0.6, "cost": 0.4}
        assert strategy.strategy_name == "test_strategy"

    def test_optimization_strategy_defaults(self):
        """Test OptimizationStrategy default values."""
        strategy = OptimizationStrategy()

        assert strategy.max_total_evaluations is None
        assert strategy.max_cost_budget is None
        assert strategy.max_time_budget is None
        assert strategy.exploration_ratio == 0.3
        assert strategy.early_stopping_patience == 10
        assert strategy.confidence_threshold == 0.95
        assert strategy.min_examples_per_trial == 5
        assert strategy.max_examples_per_trial is None
        assert strategy.adaptive_sample_size is True
        assert strategy.pareto_preference is None
        assert strategy.objective_weights == {}
        assert strategy.strategy_name == "smart_optimization"
        assert strategy.metadata == {}


class TestRemoteOptimizationServiceAbstract:
    """Test RemoteOptimizationService abstract base class."""

    def test_abstract_service_initialization(self):
        """Test RemoteOptimizationService initialization."""
        # Can't instantiate abstract class directly
        with pytest.raises(TypeError):
            RemoteOptimizationService("test", "http://test.com")

    def test_mock_service_initialization(self):
        """Test mock service initialization."""
        service = MockRemoteOptimizationService(
            service_name="TestMockService",
            api_key="test_key",
            timeout=60.0,
            custom_param="test_value",
        )

        assert service.service_name == "TestMockService"
        assert service.endpoint == "http://mock.service"
        assert service.api_key == "test_key"
        assert service.timeout == 60.0
        assert service.config["custom_param"] == "test_value"
        assert service._status == ServiceStatus.DISCONNECTED
        assert service._max_active_sessions == 100

    def test_service_metrics_initialization(self):
        """Test service metrics initialization."""
        service = MockRemoteOptimizationService()

        assert service._metrics.response_time_ms == 0.0
        assert service._metrics.success_rate == 1.0
        assert service._metrics.total_requests == 0
        assert service._metrics.failed_requests == 0
        assert isinstance(service._metrics.last_request_time, datetime)


class TestMockServiceOperations:
    """Test MockRemoteOptimizationService operations."""

    @pytest.mark.asyncio
    async def test_service_connection_lifecycle(self):
        """Test service connection and disconnection."""
        service = MockRemoteOptimizationService()

        # Initially disconnected
        assert not service._connected
        assert service._status == ServiceStatus.DISCONNECTED

        # Connect
        info = await service.connect()
        assert service._connected
        assert service._status == ServiceStatus.CONNECTED
        assert info.name == "MockService"
        assert "bayesian" in info.supported_algorithms

        # Disconnect
        await service.disconnect()
        assert not service._connected
        assert service._status == ServiceStatus.DISCONNECTED
        assert len(service._sessions) == 0

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test service health check."""
        service = MockRemoteOptimizationService()

        health = await service.health_check()
        assert health["status"] == "healthy"
        assert health["uptime"] == 3600

    @pytest.mark.asyncio
    async def test_session_lifecycle(self):
        """Test optimization session lifecycle."""
        service = MockRemoteOptimizationService()
        await service.connect()

        # Create session
        config_space = {"temperature": {"min": 0.0, "max": 1.0}}
        objectives = ["accuracy"]

        session = await service.create_session(config_space, objectives)

        assert session.service_name == "MockService"
        assert session.config_space == config_space
        assert session.objectives == objectives
        assert session.algorithm == "bayesian"
        assert session.status == OptimizationSessionStatus.ACTIVE
        assert session.session_id in service._sessions

        # Get session
        retrieved_session = await service.get_session(session.session_id)
        assert retrieved_session.session_id == session.session_id

        # Close session
        await service.close_session(session.session_id)
        assert session.session_id not in service._sessions

    @pytest.mark.asyncio
    async def test_session_operations_without_connection(self):
        """Test session operations fail without connection."""
        service = MockRemoteOptimizationService()

        with pytest.raises(ServiceError, match="Service not connected"):
            await service.create_session({}, [])

    @pytest.mark.asyncio
    async def test_nonexistent_session_operations(self):
        """Test operations on nonexistent sessions."""
        service = MockRemoteOptimizationService()
        await service.connect()

        with pytest.raises(ServiceError, match="Session fake_session not found"):
            await service.get_session("fake_session")

        with pytest.raises(ServiceError, match="Session fake_session not found"):
            await service.suggest_configuration("fake_session", [])

        with pytest.raises(ServiceError, match="Session fake_session not found"):
            await service.report_trial_result("fake_session", MagicMock())

        with pytest.raises(ServiceError, match="Session fake_session not found"):
            await service.should_stop_optimization("fake_session", [])

    @pytest.mark.asyncio
    async def test_optimization_operations(self, sample_trial_results):
        """Test optimization operations."""
        service = MockRemoteOptimizationService()
        await service.connect()

        session = await service.create_session(
            {"temp": {"min": 0, "max": 1}}, ["accuracy"]
        )

        # Suggest configuration
        config = await service.suggest_configuration(
            session.session_id, sample_trial_results
        )
        assert config == {"temperature": 0.6, "max_tokens": 1200}

        # Report trial result
        await service.report_trial_result(session.session_id, sample_trial_results[0])
        updated_session = await service.get_session(session.session_id)
        assert updated_session.trials_completed == 1

        # Check stopping condition
        short_history = sample_trial_results[:3]
        should_stop = await service.should_stop_optimization(
            session.session_id, short_history
        )
        assert should_stop is False

        long_history = sample_trial_results * 3  # 12 trials
        should_stop = await service.should_stop_optimization(
            session.session_id, long_history
        )
        assert should_stop is True


class TestCTDScenarios:
    """Combinatorial Test Design scenarios for comprehensive coverage."""

    @pytest.mark.parametrize(
        "service_connected,session_exists,expected_result",
        [
            (True, True, "success"),
            (True, False, "session_error"),
            (False, True, "service_error"),
            (False, False, "service_error"),
        ],
    )
    @pytest.mark.asyncio
    async def test_operation_combinations(
        self, service_connected, session_exists, expected_result
    ):
        """Test combinations of service and session states."""
        service = MockRemoteOptimizationService()

        if service_connected:
            await service.connect()

        session_id = "test_session"
        if session_exists and service_connected:
            session = await service.create_session(
                {"temp": {"min": 0, "max": 1}}, ["accuracy"]
            )
            session_id = session.session_id

        if expected_result == "success":
            config = await service.suggest_configuration(session_id, [])
            assert config == {"temperature": 0.6, "max_tokens": 1200}
        elif expected_result == "session_error":
            with pytest.raises(ServiceError, match="not found"):
                await service.suggest_configuration(session_id, [])
        elif expected_result == "service_error":
            with pytest.raises(ServiceError, match="Service not connected"):
                await service.suggest_configuration(session_id, [])

    @pytest.mark.parametrize(
        "strategy_field,value,expected_behavior",
        [
            ("max_total_evaluations", 10, "budget_constraint"),
            ("max_cost_budget", 5.0, "budget_constraint"),
            ("max_time_budget", 60, "budget_constraint"),
            ("exploration_ratio", 0.0, "pure_exploitation"),
            ("exploration_ratio", 1.0, "pure_exploration"),
            ("early_stopping_patience", 1, "early_stop"),
            ("confidence_threshold", 0.99, "high_confidence"),
            ("min_examples_per_trial", 100, "large_minimum"),
            ("adaptive_sample_size", False, "fixed_sample_size"),
        ],
    )
    def test_strategy_parameter_combinations(
        self, strategy_field, value, expected_behavior
    ):
        """Test different optimization strategy parameter combinations."""
        strategy = OptimizationStrategy()
        setattr(strategy, strategy_field, value)

        if expected_behavior == "budget_constraint":
            assert getattr(strategy, strategy_field) == value
            # Strategy should enforce constraints
        elif expected_behavior == "pure_exploitation":
            assert strategy.exploration_ratio == 0.0
        elif expected_behavior == "pure_exploration":
            assert strategy.exploration_ratio == 1.0
        elif expected_behavior == "early_stop":
            assert strategy.early_stopping_patience == 1
        elif expected_behavior == "high_confidence":
            assert strategy.confidence_threshold == 0.99
        elif expected_behavior == "large_minimum":
            assert strategy.min_examples_per_trial == 100
        elif expected_behavior == "fixed_sample_size":
            assert strategy.adaptive_sample_size is False

    @pytest.mark.parametrize(
        "subset_size,confidence,strategy,expected_type",
        [
            (5, 0.9, "confidence_based", "high_confidence"),
            (50, 0.3, "random", "low_confidence"),
            (1, 0.8, "hard_examples", "challenging"),
            (100, 0.5, "diverse", "comprehensive"),
            (0, 0.0, "empty", "invalid"),
        ],
    )
    def test_dataset_subset_combinations(
        self,
        sample_evaluation_examples,
        subset_size,
        confidence,
        strategy,
        expected_type,
    ):
        """Test different dataset subset combinations."""
        if subset_size > len(sample_evaluation_examples):
            examples = sample_evaluation_examples * (
                subset_size // len(sample_evaluation_examples) + 1
            )
            examples = examples[:subset_size]
        else:
            examples = sample_evaluation_examples[:subset_size]

        subset = DatasetSubset(
            examples=examples,
            selection_strategy=strategy,
            confidence_level=confidence,
            subset_id=f"test_{expected_type}",
        )

        assert subset.size == subset_size
        assert subset.confidence_level == confidence
        assert subset.selection_strategy == strategy

        if expected_type == "high_confidence":
            assert subset.confidence_level >= 0.8
        elif expected_type == "low_confidence":
            assert subset.confidence_level < 0.5
        elif expected_type == "challenging":
            assert subset.size == 1
        elif expected_type == "comprehensive":
            assert subset.size >= 50
        elif expected_type == "invalid":
            assert subset.size == 0


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""

    @pytest.mark.asyncio
    async def test_session_creation_with_invalid_parameters(self):
        """Test session creation with invalid parameters."""
        service = MockRemoteOptimizationService()
        await service.connect()

        # Empty config space
        session = await service.create_session({}, ["accuracy"])
        assert session.config_space == {}

        # Empty objectives
        session = await service.create_session({"temp": {"min": 0, "max": 1}}, [])
        assert session.objectives == []

    def test_dataset_subset_with_large_metadata(self, sample_evaluation_examples):
        """Test DatasetSubset with large metadata."""
        large_metadata = {f"key_{i}": f"value_{i}" for i in range(1000)}

        subset = DatasetSubset(
            examples=sample_evaluation_examples,
            selection_strategy="test",
            confidence_level=0.5,
            subset_id="large_metadata_test",
            metadata=large_metadata,
        )

        assert len(subset.metadata) == 1000
        assert subset.metadata["key_999"] == "value_999"

    def test_smart_suggestion_with_extreme_values(self, sample_dataset_subset):
        """Test SmartTrialSuggestion with extreme values."""
        suggestion = SmartTrialSuggestion(
            config={"temperature": -1.0, "max_tokens": 1000000},  # Extreme values
            dataset_subset=sample_dataset_subset,
            exploration_type="extreme",
            expected_value=2.0,  # Above typical range
            uncertainty=-0.1,  # Negative uncertainty
            estimated_cost=1000000.0,  # Very high cost
            estimated_duration=-10.0,  # Negative duration
            priority=-5,  # Negative priority
        )

        # Should handle extreme values gracefully
        assert suggestion.config["temperature"] == -1.0
        assert suggestion.expected_value == 2.0
        assert suggestion.uncertainty == -0.1
        assert suggestion.priority == -5

    @pytest.mark.asyncio
    async def test_concurrent_session_operations(self):
        """Test concurrent session operations."""
        service = MockRemoteOptimizationService()
        await service.connect()

        # Create multiple sessions concurrently
        import asyncio

        tasks = []
        for i in range(5):
            task = service.create_session(
                {"param": {"min": i, "max": i + 1}}, [f"objective_{i}"]
            )
            tasks.append(task)

        sessions = await asyncio.gather(*tasks)

        assert len(sessions) == 5
        assert len({s.session_id for s in sessions}) == 5  # All unique IDs
        assert len(service._sessions) == 5

        # Close all sessions concurrently
        close_tasks = [service.close_session(s.session_id) for s in sessions]
        await asyncio.gather(*close_tasks)

        assert len(service._sessions) == 0

    def test_optimization_strategy_edge_values(self):
        """Test OptimizationStrategy with edge values."""
        strategy = OptimizationStrategy(
            max_total_evaluations=0,
            max_cost_budget=0.0,
            max_time_budget=0.0,
            exploration_ratio=2.0,  # Above 1.0
            early_stopping_patience=0,
            confidence_threshold=1.5,  # Above 1.0
            min_examples_per_trial=0,
            max_examples_per_trial=0,
        )

        # Should store values as provided (validation may be done elsewhere)
        assert strategy.max_total_evaluations == 0
        assert strategy.exploration_ratio == 2.0
        assert strategy.confidence_threshold == 1.5
        assert strategy.min_examples_per_trial == 0

    @pytest.mark.asyncio
    async def test_service_memory_management(self):
        """Test service memory management with many sessions."""
        service = MockRemoteOptimizationService()
        await service.connect()

        # Create sessions up to the limit
        sessions = []
        for i in range(service._max_active_sessions):
            session = await service.create_session({"p": i}, ["obj"])
            sessions.append(session)

        assert len(service._sessions) == service._max_active_sessions

        # Creating one more should still work (mock doesn't enforce limit)
        extra_session = await service.create_session({"p": 999}, ["obj"])
        sessions.append(extra_session)

        assert len(service._sessions) == service._max_active_sessions + 1

    def test_service_metrics_edge_cases(self):
        """Test ServiceMetrics with edge case values."""
        metrics = ServiceMetrics(
            response_time_ms=0.0,
            success_rate=0.0,
            total_requests=0,
            failed_requests=1000,  # More failures than requests (edge case)
            last_request_time=datetime.min,  # Minimum datetime
            average_session_duration=float("inf"),  # Infinite duration
            active_sessions=-1,  # Negative sessions
        )

        assert metrics.response_time_ms == 0.0
        assert metrics.success_rate == 0.0
        assert metrics.failed_requests == 1000
        assert metrics.average_session_duration == float("inf")
        assert metrics.active_sessions == -1

    def test_trial_result_edge_cases(self):
        """Test handling of edge case trial results."""
        # Trial with no metrics
        trial_no_metrics = TrialResult(
            trial_id="no_metrics",
            config={"temp": 0.5},
            metrics={},
            status=TrialStatus.FAILED,
            duration=1.0,
            timestamp=datetime.now(),
            metadata={},
        )

        # Trial with None values
        trial_none_values = TrialResult(
            trial_id="none_values",
            config={"temp": None},
            metrics={},
            status=TrialStatus.COMPLETED,
            duration=1.0,
            timestamp=datetime.now(),
            metadata={},
        )

        # Trial with extreme metrics
        trial_extreme = TrialResult(
            trial_id="extreme",
            config={"temp": float("inf")},
            metrics={"accuracy": -float("inf"), "cost": float("nan")},
            status=TrialStatus.COMPLETED,
            duration=1.0,
            timestamp=datetime.now(),
            metadata={},
        )

        # Should handle all edge cases without crashing
        assert trial_no_metrics.metrics == {}
        assert trial_none_values.config["temp"] is None
        assert trial_extreme.config["temp"] == float("inf")
        assert (
            trial_extreme.metrics["cost"] != trial_extreme.metrics["cost"]
        )  # NaN check
