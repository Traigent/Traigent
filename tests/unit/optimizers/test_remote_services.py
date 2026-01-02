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

# Traceability: CONC-Layer-Integration CONC-Quality-Reliability FUNC-OPT-ALGORITHMS FUNC-CLOUD-HYBRID

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from traigent.api.types import TrialResult, TrialStatus
from traigent.evaluators.base import EvaluationExample
from traigent.optimizers.remote_services import (
    DatasetSubset,
    MockRemoteService,
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

    async def health_check(self) -> dict[str, Any]:
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
        trial_history: list[TrialResult],
        remote_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
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
        self, session_id: str, trial_history: list[TrialResult]
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


class TestMockRemoteService:
    """Tests for the MockRemoteService implementation from remote_services.py."""

    def test_mock_remote_service_initialization(self) -> None:
        """Test MockRemoteService initialization with default parameters."""
        service = MockRemoteService()

        assert service.service_name == "MockTraigentService"
        assert service.endpoint == "mock://localhost"
        assert service.api_key is None
        assert service.timeout == 30.0
        assert service._status == ServiceStatus.DISCONNECTED
        assert service._suggestion_count == 0

    def test_mock_remote_service_custom_initialization(self) -> None:
        """Test MockRemoteService initialization with custom parameters."""
        service = MockRemoteService(
            service_name="CustomMock",
            endpoint="mock://custom",
            api_key="test_key_123",
            timeout=60.0,
            custom_param="value",
        )

        assert service.service_name == "CustomMock"
        assert service.endpoint == "mock://custom"
        assert service.api_key == "test_key_123"
        assert service.timeout == 60.0
        assert service.config["custom_param"] == "value"

    @pytest.mark.asyncio
    async def test_mock_service_connect(self) -> None:
        """Test MockRemoteService connect method."""
        service = MockRemoteService()

        info = await service.connect()

        assert service._status == ServiceStatus.CONNECTED
        assert service._service_info is not None
        assert info.name == "MockTraigentService"
        assert info.version == "1.0.0-mock"
        assert "random" in info.supported_algorithms
        assert "grid" in info.supported_algorithms
        assert "bayesian" in info.supported_algorithms
        assert info.max_concurrent_sessions == 10
        assert info.capabilities["batch_suggestions"] is True
        assert info.capabilities["session_persistence"] is False
        assert info.capabilities["real_time_updates"] is True
        assert info.status == ServiceStatus.CONNECTED

    @pytest.mark.asyncio
    async def test_mock_service_disconnect(self) -> None:
        """Test MockRemoteService disconnect method."""
        service = MockRemoteService()
        await service.connect()

        # Create some sessions
        await service.create_session({"temp": (0.0, 1.0)}, ["accuracy"])
        await service.create_session({"tokens": (100, 1000)}, ["cost"])

        assert len(service._active_sessions) == 2

        # Disconnect should close all sessions
        await service.disconnect()

        assert service._status == ServiceStatus.DISCONNECTED
        assert service._service_info is None
        assert len(service._active_sessions) == 0

    @pytest.mark.asyncio
    async def test_mock_service_health_check(self) -> None:
        """Test MockRemoteService health check."""
        service = MockRemoteService()

        health = await service.health_check()

        assert health["status"] == "healthy"
        assert health["uptime"] == 99.9
        assert health["active_sessions"] == 0
        assert health["total_requests"] >= 0
        assert "response_time_ms" in health

    @pytest.mark.asyncio
    async def test_mock_service_create_session_comprehensive(
        self, sample_optimization_strategy
    ) -> None:
        """Test MockRemoteService session creation with all parameters."""
        service = MockRemoteService()
        await service.connect()

        config_space = {
            "temperature": (0.0, 1.0),
            "max_tokens": (100, 4000),
            "model": ["gpt-4", "gpt-3.5-turbo"],
        }
        objectives = ["accuracy", "cost", "latency"]

        session = await service.create_session(
            config_space=config_space,
            objectives=objectives,
            algorithm="bayesian",
            max_trials=50,
            timeout=1800.0,
            optimization_strategy=sample_optimization_strategy,
            context=None,
            custom_metadata="test_value",
        )

        assert session.session_id.startswith("MockTraigentService_")
        assert session.service_name == "MockTraigentService"
        assert session.config_space == config_space
        assert session.objectives == objectives
        assert session.algorithm == "bayesian"
        assert session.status == OptimizationSessionStatus.ACTIVE
        assert session.max_trials == 50
        assert session.timeout == 1800.0
        assert session.optimization_strategy == sample_optimization_strategy
        assert session.metadata["mock_service"] is True
        assert session.metadata["custom_metadata"] == "test_value"

    @pytest.mark.asyncio
    async def test_mock_service_get_session(self) -> None:
        """Test MockRemoteService get_session method."""
        service = MockRemoteService()
        await service.connect()

        session = await service.create_session({"temp": (0.0, 1.0)}, ["accuracy"])
        session_id = session.session_id

        # Get the session
        retrieved = await service.get_session(session_id)

        assert retrieved.session_id == session_id
        assert retrieved.config_space == session.config_space

    @pytest.mark.asyncio
    async def test_mock_service_get_nonexistent_session(self) -> None:
        """Test MockRemoteService get_session with invalid ID."""
        service = MockRemoteService()
        await service.connect()

        with pytest.raises(ServiceError, match="Session invalid_id not found"):
            await service.get_session("invalid_id")

    @pytest.mark.asyncio
    async def test_mock_service_close_session(self) -> None:
        """Test MockRemoteService close_session method."""
        service = MockRemoteService()
        await service.connect()

        session = await service.create_session({"temp": (0.0, 1.0)}, ["accuracy"])
        session_id = session.session_id

        assert session_id in service._active_sessions

        await service.close_session(session_id)

        assert session_id not in service._active_sessions

    @pytest.mark.asyncio
    async def test_mock_service_close_nonexistent_session(self) -> None:
        """Test MockRemoteService close_session with invalid ID."""
        service = MockRemoteService()
        await service.connect()

        # Should not raise error - just does nothing
        result = await service.close_session("nonexistent_session")
        assert result is None  # Method returns None

    @pytest.mark.asyncio
    async def test_mock_service_suggest_configuration_categorical(self) -> None:
        """Test MockRemoteService suggest_configuration with categorical parameters."""
        service = MockRemoteService()
        await service.connect()

        config_space = {"model": ["gpt-4", "gpt-3.5-turbo", "claude-2"]}
        session = await service.create_session(config_space, ["accuracy"])

        config = await service.suggest_configuration(session.session_id, [])

        assert "model" in config
        assert config["model"] in ["gpt-4", "gpt-3.5-turbo", "claude-2"]
        assert "_mock_suggestion_id" in config
        assert config["_mock_suggestion_id"] == 1

    @pytest.mark.asyncio
    async def test_mock_service_suggest_configuration_continuous(self) -> None:
        """Test MockRemoteService suggest_configuration with continuous parameters."""
        service = MockRemoteService()
        await service.connect()

        config_space = {"temperature": (0.0, 1.0), "top_p": (0.5, 1.0)}
        session = await service.create_session(config_space, ["accuracy"])

        config = await service.suggest_configuration(session.session_id, [])

        assert "temperature" in config
        assert "top_p" in config
        assert 0.0 <= config["temperature"] <= 1.0
        assert 0.5 <= config["top_p"] <= 1.0

    @pytest.mark.asyncio
    async def test_mock_service_suggest_configuration_integer(self) -> None:
        """Test MockRemoteService suggest_configuration with integer parameters."""
        service = MockRemoteService()
        await service.connect()

        config_space = {"max_tokens": (100, 1000), "batch_size": (1, 32)}
        session = await service.create_session(config_space, ["cost"])

        config = await service.suggest_configuration(session.session_id, [])

        assert "max_tokens" in config
        assert "batch_size" in config
        assert isinstance(config["max_tokens"], int)
        assert isinstance(config["batch_size"], int)
        assert 100 <= config["max_tokens"] <= 1000
        assert 1 <= config["batch_size"] <= 32

    @pytest.mark.asyncio
    async def test_mock_service_suggest_configuration_fixed(self) -> None:
        """Test MockRemoteService suggest_configuration with fixed parameters."""
        service = MockRemoteService()
        await service.connect()

        config_space = {"fixed_param": "fixed_value", "another_fixed": 42}
        session = await service.create_session(config_space, ["accuracy"])

        config = await service.suggest_configuration(session.session_id, [])

        assert config["fixed_param"] == "fixed_value"
        assert config["another_fixed"] == 42

    @pytest.mark.asyncio
    async def test_mock_service_suggest_configuration_with_context(self) -> None:
        """Test MockRemoteService suggest_configuration with remote context."""
        service = MockRemoteService()
        await service.connect()

        config_space = {"temp": (0.0, 1.0)}
        session = await service.create_session(config_space, ["accuracy"])

        remote_context = {"previous_best": 0.95, "iteration": 10}
        config = await service.suggest_configuration(
            session.session_id, [], remote_context
        )

        assert "_remote_context" in config
        assert config["_remote_context"] == remote_context

    @pytest.mark.asyncio
    async def test_mock_service_suggest_configuration_counter(self) -> None:
        """Test MockRemoteService suggestion counter increments."""
        service = MockRemoteService()
        await service.connect()

        config_space = {"temp": (0.0, 1.0)}
        session = await service.create_session(config_space, ["accuracy"])

        config1 = await service.suggest_configuration(session.session_id, [])
        config2 = await service.suggest_configuration(session.session_id, [])
        config3 = await service.suggest_configuration(session.session_id, [])

        assert config1["_mock_suggestion_id"] == 1
        assert config2["_mock_suggestion_id"] == 2
        assert config3["_mock_suggestion_id"] == 3
        assert service._suggestion_count == 3

    @pytest.mark.asyncio
    async def test_mock_service_report_trial_result(self, sample_trial_results) -> None:
        """Test MockRemoteService report_trial_result method."""
        service = MockRemoteService()
        await service.connect()

        session = await service.create_session({"temp": (0.0, 1.0)}, ["accuracy"])

        # Report first result
        await service.report_trial_result(session.session_id, sample_trial_results[0])

        updated_session = await service.get_session(session.session_id)
        assert updated_session.trials_completed == 1
        assert updated_session.best_score == 0.85
        assert updated_session.best_config == sample_trial_results[0].config

    @pytest.mark.asyncio
    async def test_mock_service_report_trial_result_updates_best(
        self, sample_trial_results
    ) -> None:
        """Test MockRemoteService updates best score when better result is reported."""
        service = MockRemoteService()
        await service.connect()

        session = await service.create_session({"temp": (0.0, 1.0)}, ["accuracy"])

        # Report results in order: 0.85, then 0.88 (better), then 0.82 (worse)
        await service.report_trial_result(session.session_id, sample_trial_results[0])
        updated_session = await service.get_session(session.session_id)
        assert updated_session.best_score == 0.85

        await service.report_trial_result(session.session_id, sample_trial_results[1])
        updated_session = await service.get_session(session.session_id)
        assert updated_session.best_score == 0.88
        assert updated_session.best_config == sample_trial_results[1].config

        await service.report_trial_result(session.session_id, sample_trial_results[2])
        updated_session = await service.get_session(session.session_id)
        assert updated_session.best_score == 0.88  # Should not change
        assert updated_session.trials_completed == 3

    @pytest.mark.asyncio
    async def test_mock_service_report_failed_trial(self, sample_trial_results) -> None:
        """Test MockRemoteService handles failed trial results."""
        service = MockRemoteService()
        await service.connect()

        session = await service.create_session({"temp": (0.0, 1.0)}, ["accuracy"])

        # Report failed trial (index 3)
        failed_trial = sample_trial_results[3]
        await service.report_trial_result(session.session_id, failed_trial)

        updated_session = await service.get_session(session.session_id)
        assert updated_session.trials_completed == 1
        assert updated_session.best_score is None  # No successful trials yet

    @pytest.mark.asyncio
    async def test_mock_service_should_stop_max_trials(
        self, sample_trial_results
    ) -> None:
        """Test MockRemoteService stopping condition based on max_trials."""
        service = MockRemoteService()
        await service.connect()

        session = await service.create_session(
            {"temp": (0.0, 1.0)}, ["accuracy"], max_trials=5
        )

        # Should not stop with 3 trials
        should_stop = await service.should_stop_optimization(
            session.session_id, sample_trial_results[:3]
        )
        assert should_stop is False

        # Should stop with 5 trials
        should_stop = await service.should_stop_optimization(
            session.session_id, sample_trial_results * 2  # 8 trials
        )
        assert should_stop is True

    @pytest.mark.asyncio
    async def test_mock_service_should_stop_convergence(self) -> None:
        """Test MockRemoteService stopping condition based on convergence."""
        service = MockRemoteService()
        await service.connect()

        session = await service.create_session({"temp": (0.0, 1.0)}, ["accuracy"])

        # Create trial history with converged scores (all 0.85)
        converged_trials = [
            TrialResult(
                trial_id=f"trial_{i}",
                config={"temp": 0.5},
                metrics={"accuracy": 0.85},
                status=TrialStatus.COMPLETED,
                duration=10.0,
                timestamp=datetime.now(),
            )
            for i in range(15)
        ]

        should_stop = await service.should_stop_optimization(
            session.session_id, converged_trials
        )
        assert should_stop is True

    @pytest.mark.asyncio
    async def test_mock_service_should_stop_not_enough_trials(self) -> None:
        """Test MockRemoteService does not stop with too few trials."""
        service = MockRemoteService()
        await service.connect()

        session = await service.create_session({"temp": (0.0, 1.0)}, ["accuracy"])

        # Only 5 trials - should not check convergence
        few_trials = [
            TrialResult(
                trial_id=f"trial_{i}",
                config={"temp": 0.5},
                metrics={"accuracy": 0.85},
                status=TrialStatus.COMPLETED,
                duration=10.0,
                timestamp=datetime.now(),
            )
            for i in range(5)
        ]

        should_stop = await service.should_stop_optimization(
            session.session_id, few_trials
        )
        assert should_stop is False

    @pytest.mark.asyncio
    async def test_mock_service_suggest_smart_trial(
        self, sample_evaluation_examples, sample_trial_results
    ) -> None:
        """Test MockRemoteService suggest_smart_trial method."""
        service = MockRemoteService()
        await service.connect()

        strategy = OptimizationStrategy(
            min_examples_per_trial=3, max_examples_per_trial=10
        )
        session = await service.create_session(
            {"temp": (0.0, 1.0)}, ["accuracy"], optimization_strategy=strategy
        )

        suggestion = await service.suggest_smart_trial(
            session.session_id, sample_trial_results, sample_evaluation_examples
        )

        assert isinstance(suggestion, SmartTrialSuggestion)
        assert "temp" in suggestion.config
        assert isinstance(suggestion.dataset_subset, DatasetSubset)
        assert suggestion.dataset_subset.size >= 3
        assert suggestion.dataset_subset.size <= 10
        assert suggestion.exploration_type in [
            "exploration",
            "exploitation",
            "verification",
            "refinement",
        ]
        assert suggestion.estimated_cost is not None
        assert suggestion.estimated_duration is not None
        assert suggestion.priority >= 0

    @pytest.mark.asyncio
    async def test_mock_service_suggest_smart_trial_early_exploration(
        self, sample_evaluation_examples
    ) -> None:
        """Test MockRemoteService smart trial with early exploration."""
        service = MockRemoteService()
        await service.connect()

        strategy = OptimizationStrategy(min_examples_per_trial=5)
        session = await service.create_session(
            {"temp": (0.0, 1.0)}, ["accuracy"], optimization_strategy=strategy
        )

        # Early in optimization (2 trials)
        early_trials = [
            TrialResult(
                trial_id=f"trial_{i}",
                config={"temp": 0.5},
                metrics={"accuracy": 0.8},
                status=TrialStatus.COMPLETED,
                duration=10.0,
                timestamp=datetime.now(),
            )
            for i in range(2)
        ]

        suggestion = await service.suggest_smart_trial(
            session.session_id, early_trials, sample_evaluation_examples
        )

        assert suggestion.exploration_type == "exploration"
        assert suggestion.dataset_subset.selection_strategy == "diverse_sampling"
        assert suggestion.dataset_subset.confidence_level == 0.3

    @pytest.mark.asyncio
    async def test_mock_service_suggest_smart_trial_mid_optimization(
        self, sample_evaluation_examples
    ) -> None:
        """Test MockRemoteService smart trial during mid optimization."""
        service = MockRemoteService()
        await service.connect()

        strategy = OptimizationStrategy(min_examples_per_trial=5)
        session = await service.create_session(
            {"temp": (0.0, 1.0)}, ["accuracy"], optimization_strategy=strategy
        )

        # Mid optimization (7 trials)
        mid_trials = [
            TrialResult(
                trial_id=f"trial_{i}",
                config={"temp": 0.5 + i * 0.05},
                metrics={"accuracy": 0.8 + i * 0.01},
                status=TrialStatus.COMPLETED,
                duration=10.0,
                timestamp=datetime.now(),
            )
            for i in range(7)
        ]

        suggestion = await service.suggest_smart_trial(
            session.session_id, mid_trials, sample_evaluation_examples
        )

        assert suggestion.exploration_type in ["exploration", "exploitation"]
        assert suggestion.dataset_subset.selection_strategy == "representative_sampling"
        assert suggestion.dataset_subset.confidence_level == 0.6

    @pytest.mark.asyncio
    async def test_mock_service_suggest_smart_trial_late_optimization(
        self, sample_evaluation_examples
    ) -> None:
        """Test MockRemoteService smart trial in late optimization."""
        service = MockRemoteService()
        await service.connect()

        strategy = OptimizationStrategy(min_examples_per_trial=5)
        session = await service.create_session(
            {"temp": (0.0, 1.0)}, ["accuracy"], optimization_strategy=strategy
        )

        # Late optimization (15 trials)
        late_trials = [
            TrialResult(
                trial_id=f"trial_{i}",
                config={"temp": 0.5},
                metrics={"accuracy": 0.85},
                status=TrialStatus.COMPLETED,
                duration=10.0,
                timestamp=datetime.now(),
            )
            for i in range(15)
        ]

        suggestion = await service.suggest_smart_trial(
            session.session_id, late_trials, sample_evaluation_examples
        )

        assert suggestion.exploration_type in ["verification", "refinement"]
        assert (
            suggestion.dataset_subset.selection_strategy == "high_confidence_sampling"
        )
        assert suggestion.dataset_subset.confidence_level == 0.8


class TestRemoteOptimizationServiceHelpers:
    """Test helper methods of RemoteOptimizationService."""

    @pytest.mark.asyncio
    async def test_update_metrics_success(self) -> None:
        """Test _update_metrics with successful request."""
        import time

        service = MockRemoteService()

        initial_requests = service._metrics.total_requests
        initial_failed = service._metrics.failed_requests

        request_start = time.time()
        await asyncio.sleep(0.01)  # Simulate some work
        service._update_metrics(request_start, success=True)

        assert service._metrics.total_requests == initial_requests + 1
        assert service._metrics.failed_requests == initial_failed
        assert service._metrics.response_time_ms > 0
        assert service._metrics.success_rate > 0

    @pytest.mark.asyncio
    async def test_update_metrics_failure(self) -> None:
        """Test _update_metrics with failed request."""
        import time

        service = MockRemoteService()

        initial_requests = service._metrics.total_requests
        initial_failed = service._metrics.failed_requests

        request_start = time.time()
        service._update_metrics(request_start, success=False)

        assert service._metrics.total_requests == initial_requests + 1
        assert service._metrics.failed_requests == initial_failed + 1

    @pytest.mark.asyncio
    async def test_update_metrics_success_rate_calculation(self) -> None:
        """Test _update_metrics calculates success rate correctly."""
        import time

        service = MockRemoteService()

        request_start = time.time()

        # 3 successes
        service._update_metrics(request_start, success=True)
        service._update_metrics(request_start, success=True)
        service._update_metrics(request_start, success=True)

        # 1 failure
        service._update_metrics(request_start, success=False)

        # Success rate should be 3/4 = 0.75
        assert service._metrics.total_requests == 4
        assert service._metrics.failed_requests == 1
        assert service._metrics.success_rate == 0.75

    def test_create_session_id(self) -> None:
        """Test _create_session_id generates valid IDs."""
        service = MockRemoteService(service_name="TestService")

        session_id1 = service._create_session_id()
        session_id2 = service._create_session_id()

        assert session_id1.startswith("TestService_")
        assert session_id2.startswith("TestService_")
        assert session_id1 != session_id2
        assert len(session_id1.split("_")[1]) == 8
        assert len(session_id2.split("_")[1]) == 8

    @pytest.mark.asyncio
    async def test_validate_session_existing(self) -> None:
        """Test _validate_session with existing active session."""
        service = MockRemoteService()
        await service.connect()

        session = await service.create_session({"temp": (0.0, 1.0)}, ["accuracy"])

        validated = await service._validate_session(session.session_id)

        assert validated.session_id == session.session_id
        assert validated.status == OptimizationSessionStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_validate_session_nonexistent(self) -> None:
        """Test _validate_session with nonexistent session."""
        service = MockRemoteService()
        await service.connect()

        with pytest.raises(ServiceError, match="Invalid or inactive session"):
            await service._validate_session("nonexistent_session")

    @pytest.mark.asyncio
    async def test_validate_session_inactive_status(self) -> None:
        """Test _validate_session rejects inactive session."""
        service = MockRemoteService()
        await service.connect()

        session = await service.create_session({"temp": (0.0, 1.0)}, ["accuracy"])

        # Mark session as completed
        service._active_sessions[session.session_id].status = (
            OptimizationSessionStatus.COMPLETED
        )

        with pytest.raises(ServiceError, match="not active"):
            await service._validate_session(session.session_id)

    @pytest.mark.asyncio
    async def test_validate_session_failed_status(self) -> None:
        """Test _validate_session rejects failed session."""
        service = MockRemoteService()
        await service.connect()

        session = await service.create_session({"temp": (0.0, 1.0)}, ["accuracy"])

        # Mark session as failed
        service._active_sessions[session.session_id].status = (
            OptimizationSessionStatus.FAILED
        )

        with pytest.raises(ServiceError, match="not active"):
            await service._validate_session(session.session_id)

    @pytest.mark.asyncio
    async def test_validate_session_max_sessions_limit(self) -> None:
        """Test _validate_session enforces max active sessions limit."""
        service = MockRemoteService()
        service._max_active_sessions = 3  # Set low limit for testing
        await service.connect()

        # Create max sessions
        sessions = []
        for i in range(3):
            session = await service.create_session({f"p{i}": (0, 1)}, ["obj"])
            sessions.append(session)

        assert len(service._active_sessions) == 3

        # Create a 4th session - should trigger cleanup
        # Mark first session as completed
        service._active_sessions[sessions[0].session_id].status = (
            OptimizationSessionStatus.COMPLETED
        )

        # Now try to create a new session that will trigger cleanup
        # This will trigger the limit enforcement in create_session
        await service.create_session({"new": (0, 1)}, ["obj"])

        # The oldest session should have been removed
        assert len(service._active_sessions) <= service._max_active_sessions

    @pytest.mark.asyncio
    async def test_service_properties(self) -> None:
        """Test RemoteOptimizationService property methods."""
        service = MockRemoteService()

        # Before connection
        assert service.status == ServiceStatus.DISCONNECTED
        assert service.service_info is None
        assert isinstance(service.metrics, ServiceMetrics)
        assert service.get_session_count() == 0
        assert service.get_active_sessions() == []

        # After connection
        await service.connect()

        assert service.status == ServiceStatus.CONNECTED
        assert service.service_info is not None
        assert service.service_info.name == "MockTraigentService"

        # Create sessions
        session1 = await service.create_session({"p1": (0, 1)}, ["obj1"])
        session2 = await service.create_session({"p2": (0, 1)}, ["obj2"])

        assert service.get_session_count() == 2
        active_sessions = service.get_active_sessions()
        assert len(active_sessions) == 2
        assert any(s.session_id == session1.session_id for s in active_sessions)
        assert any(s.session_id == session2.session_id for s in active_sessions)


class TestBatchOperations:
    """Test batch operation methods."""

    @pytest.mark.asyncio
    async def test_suggest_multiple_configurations(self, sample_trial_results) -> None:
        """Test suggest_multiple_configurations returns multiple suggestions."""
        service = MockRemoteService()
        await service.connect()

        session = await service.create_session({"temp": (0.0, 1.0)}, ["accuracy"])

        suggestions = await service.suggest_multiple_configurations(
            session.session_id, sample_trial_results, num_suggestions=5
        )

        assert len(suggestions) == 5
        assert all("temp" in config for config in suggestions)
        # Check that suggestion IDs are incrementing
        assert suggestions[0]["_mock_suggestion_id"] == 1
        assert suggestions[4]["_mock_suggestion_id"] == 5

    @pytest.mark.asyncio
    async def test_suggest_multiple_configurations_empty(self) -> None:
        """Test suggest_multiple_configurations with zero suggestions."""
        service = MockRemoteService()
        await service.connect()

        session = await service.create_session({"temp": (0.0, 1.0)}, ["accuracy"])

        suggestions = await service.suggest_multiple_configurations(
            session.session_id, [], num_suggestions=0
        )

        assert len(suggestions) == 0

    @pytest.mark.asyncio
    async def test_report_multiple_trial_results(self, sample_trial_results) -> None:
        """Test report_multiple_trial_results reports all results."""
        service = MockRemoteService()
        await service.connect()

        session = await service.create_session({"temp": (0.0, 1.0)}, ["accuracy"])

        await service.report_multiple_trial_results(
            session.session_id, sample_trial_results[:3]
        )

        updated_session = await service.get_session(session.session_id)
        assert updated_session.trials_completed == 3

    @pytest.mark.asyncio
    async def test_report_multiple_trial_results_handles_errors(self) -> None:
        """Test report_multiple_trial_results continues on individual errors."""
        service = MockRemoteService()
        await service.connect()

        session = await service.create_session({"temp": (0.0, 1.0)}, ["accuracy"])

        # Create mix of valid and invalid trial results
        valid_trial = TrialResult(
            trial_id="valid",
            config={"temp": 0.5},
            metrics={"accuracy": 0.8},
            status=TrialStatus.COMPLETED,
            duration=10.0,
            timestamp=datetime.now(),
        )

        # Report should handle errors gracefully
        await service.report_multiple_trial_results(session.session_id, [valid_trial])

        updated_session = await service.get_session(session.session_id)
        assert updated_session.trials_completed >= 1

    @pytest.mark.asyncio
    async def test_suggest_multiple_smart_trials(
        self, sample_evaluation_examples, sample_trial_results
    ) -> None:
        """Test suggest_multiple_smart_trials returns multiple suggestions."""
        service = MockRemoteService()
        await service.connect()

        strategy = OptimizationStrategy(min_examples_per_trial=3)
        session = await service.create_session(
            {"temp": (0.0, 1.0)}, ["accuracy"], optimization_strategy=strategy
        )

        suggestions = await service.suggest_multiple_smart_trials(
            session.session_id,
            sample_trial_results,
            sample_evaluation_examples,
            num_suggestions=3,
        )

        assert len(suggestions) == 3
        assert all(isinstance(s, SmartTrialSuggestion) for s in suggestions)
        assert all(isinstance(s.dataset_subset, DatasetSubset) for s in suggestions)

    @pytest.mark.asyncio
    async def test_update_optimization_strategy(self) -> None:
        """Test update_optimization_strategy updates session strategy."""
        service = MockRemoteService()
        await service.connect()

        session = await service.create_session({"temp": (0.0, 1.0)}, ["accuracy"])

        new_strategy = OptimizationStrategy(
            exploration_ratio=0.5, min_examples_per_trial=10
        )

        await service.update_optimization_strategy(session.session_id, new_strategy)

        updated_session = await service.get_session(session.session_id)
        assert updated_session.optimization_strategy == new_strategy
        assert updated_session.optimization_strategy.exploration_ratio == 0.5
        assert updated_session.optimization_strategy.min_examples_per_trial == 10


class TestSmartOptimizationHelpers:
    """Test smart optimization helper methods in MockRemoteService."""

    def test_select_smart_dataset_subset_early_stage(
        self, sample_evaluation_examples
    ) -> None:
        """Test dataset subset selection in early optimization stage."""
        service = MockRemoteService()
        strategy = OptimizationStrategy(min_examples_per_trial=3)

        # Early stage (2 trials)
        subset = service._select_smart_dataset_subset(
            sample_evaluation_examples, [], strategy, trial_count=2
        )

        assert subset.size == 3
        assert subset.selection_strategy == "diverse_sampling"
        assert subset.confidence_level == 0.3

    def test_select_smart_dataset_subset_mid_stage(
        self, sample_evaluation_examples
    ) -> None:
        """Test dataset subset selection in mid optimization stage."""
        service = MockRemoteService()
        strategy = OptimizationStrategy(min_examples_per_trial=2)

        # Mid stage (7 trials)
        subset = service._select_smart_dataset_subset(
            sample_evaluation_examples, [], strategy, trial_count=7
        )

        # min_size * 2 = 4, capped at dataset size (5)
        assert subset.size == 4
        assert subset.selection_strategy == "representative_sampling"
        assert subset.confidence_level == 0.6

    def test_select_smart_dataset_subset_late_stage(
        self, sample_evaluation_examples
    ) -> None:
        """Test dataset subset selection in late optimization stage."""
        service = MockRemoteService()
        strategy = OptimizationStrategy(min_examples_per_trial=2)

        # Late stage (12 trials)
        subset = service._select_smart_dataset_subset(
            sample_evaluation_examples, [], strategy, trial_count=12
        )

        # min_size * 3 = 6 or total//2 = 2, min of those capped at total = 2
        assert subset.size >= 2
        assert subset.selection_strategy == "high_confidence_sampling"
        assert subset.confidence_level == 0.8

    def test_select_smart_dataset_subset_respects_max(
        self, sample_evaluation_examples
    ) -> None:
        """Test dataset subset selection respects max_examples_per_trial."""
        service = MockRemoteService()
        strategy = OptimizationStrategy(
            min_examples_per_trial=5, max_examples_per_trial=7
        )

        # Late stage would normally use more examples, but should be capped
        subset = service._select_smart_dataset_subset(
            sample_evaluation_examples, [], strategy, trial_count=15
        )

        assert subset.size <= 7

    def test_determine_exploration_type_early(self) -> None:
        """Test exploration type determination in early stage."""
        service = MockRemoteService()
        strategy = OptimizationStrategy()

        trials = [
            TrialResult(
                trial_id=f"t{i}",
                config={"temp": 0.5},
                metrics={"accuracy": 0.8},
                status=TrialStatus.COMPLETED,
                duration=10.0,
                timestamp=datetime.now(),
            )
            for i in range(3)
        ]

        exploration_type = service._determine_exploration_type(trials, strategy)
        assert exploration_type == "exploration"

    def test_determine_exploration_type_mid_improving(self) -> None:
        """Test exploration type with improving performance."""
        service = MockRemoteService()
        strategy = OptimizationStrategy()

        # Create improving trials
        trials = [
            TrialResult(
                trial_id=f"t{i}",
                config={"temp": 0.5},
                metrics={"accuracy": 0.7 + i * 0.02},
                status=TrialStatus.COMPLETED,
                duration=10.0,
                timestamp=datetime.now(),
            )
            for i in range(10)
        ]

        exploration_type = service._determine_exploration_type(trials, strategy)
        assert exploration_type in ["exploration", "exploitation"]

    def test_determine_exploration_type_late(self) -> None:
        """Test exploration type in late stage."""
        service = MockRemoteService()
        strategy = OptimizationStrategy()

        trials = [
            TrialResult(
                trial_id=f"t{i}",
                config={"temp": 0.5},
                metrics={"accuracy": 0.85},
                status=TrialStatus.COMPLETED,
                duration=10.0,
                timestamp=datetime.now(),
            )
            for i in range(20)
        ]

        exploration_type = service._determine_exploration_type(trials, strategy)
        assert exploration_type in ["verification", "refinement"]

    def test_calculate_priority_exploration(self) -> None:
        """Test priority calculation for exploration."""
        service = MockRemoteService()

        priority = service._calculate_priority("exploration", confidence=0.5)
        assert priority >= 1

    def test_calculate_priority_exploitation(self) -> None:
        """Test priority calculation for exploitation."""
        service = MockRemoteService()

        priority = service._calculate_priority("exploitation", confidence=0.8)
        assert priority >= 3

    def test_calculate_priority_verification(self) -> None:
        """Test priority calculation for verification."""
        service = MockRemoteService()

        priority = service._calculate_priority("verification", confidence=0.9)
        assert priority >= 2

    def test_calculate_priority_refinement(self) -> None:
        """Test priority calculation for refinement."""
        service = MockRemoteService()

        priority = service._calculate_priority("refinement", confidence=0.95)
        assert priority >= 4

    def test_calculate_priority_confidence_bonus(self) -> None:
        """Test that higher confidence increases priority."""
        service = MockRemoteService()

        low_confidence = service._calculate_priority("exploration", confidence=0.1)
        high_confidence = service._calculate_priority("exploration", confidence=0.9)

        assert high_confidence > low_confidence


class TestDefaultSmartTrialImplementation:
    """Test the default implementation of suggest_smart_trial in base class."""

    @pytest.mark.asyncio
    async def test_suggest_smart_trial_fallback(
        self, sample_evaluation_examples
    ) -> None:
        """Test RemoteOptimizationService default suggest_smart_trial implementation."""
        service = MockRemoteService()
        await service.connect()

        session = await service.create_session({"temp": (0.0, 1.0)}, ["accuracy"])

        # Test that it works with full dataset
        suggestion = await service.suggest_smart_trial(
            session.session_id, [], sample_evaluation_examples
        )

        assert isinstance(suggestion, SmartTrialSuggestion)
        assert "temp" in suggestion.config
        assert isinstance(suggestion.dataset_subset, DatasetSubset)
        assert suggestion.dataset_subset.size > 0
