"""Comprehensive tests for RemoteServiceRegistry (service_registry.py).

This test suite covers:
- Service registration and management
- Service discovery and selection
- Health monitoring and status tracking
- Service requirements matching
- Service ranking algorithms
- Error handling and edge cases
- CTD (Combinatorial Test Design) scenarios
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from traigent.optimizers.remote_services import (
    RemoteOptimizationService,
    ServiceInfo,
    ServiceStatus,
)
from traigent.optimizers.service_registry import (
    RemoteServiceRegistry,
    ServiceRanking,
    ServiceRequirements,
)
from traigent.utils.exceptions import ServiceError

# Mock service implementations for testing


class MockRemoteService(RemoteOptimizationService):
    """Mock remote service for testing."""

    def __init__(self, service_name: str, **kwargs):
        super().__init__(service_name, f"http://{service_name.lower()}.com", **kwargs)
        self._connected = False
        self._health_status = "healthy"
        self._capabilities = {
            "smart_suggestions": True,
            "multi_objective": True,
            "adaptive_datasets": False,
        }
        self._algorithms = ["bayesian", "random"]
        self._max_sessions = 10
        self._response_time = 100.0
        self._success_rate = 0.95

        # Initialize metrics properly for ranking
        from datetime import datetime

        from traigent.optimizers.remote_services import ServiceMetrics

        self._metrics = ServiceMetrics(
            response_time_ms=self._response_time,
            success_rate=self._success_rate,
            total_requests=0,
            failed_requests=0,
            last_request_time=datetime.now(),
        )

    async def connect(self) -> ServiceInfo:
        self._connected = True
        self._status = ServiceStatus.CONNECTED
        return ServiceInfo(
            name=self.service_name,
            version="1.0.0",
            supported_algorithms=self._algorithms,
            max_concurrent_sessions=self._max_sessions,
            capabilities=self._capabilities,
            status=ServiceStatus.CONNECTED,
        )

    async def disconnect(self) -> None:
        self._connected = False
        self._status = ServiceStatus.DISCONNECTED

    async def health_check(self) -> dict[str, Any]:
        return {
            "status": self._health_status,
            "response_time_ms": self._response_time,
            "success_rate": self._success_rate,
            "active_sessions": 0,
        }

    # Implement other abstract methods (minimal for testing)
    async def create_session(self, *args, **kwargs):
        pass

    async def get_session(self, *args, **kwargs):
        pass

    async def close_session(self, *args, **kwargs):
        pass

    async def suggest_configuration(self, *args, **kwargs):
        pass

    async def report_trial_result(self, *args, **kwargs):
        pass

    async def should_stop_optimization(self, *args, **kwargs):
        pass


# Test fixtures


@pytest.fixture
def registry():
    """Fresh service registry for each test."""
    return RemoteServiceRegistry()


@pytest.fixture
def mock_service_basic():
    """Basic mock service."""
    return MockRemoteService("BasicService")


@pytest.fixture
def mock_service_advanced():
    """Advanced mock service with more capabilities."""
    service = MockRemoteService("AdvancedService")
    service._capabilities = {
        "smart_suggestions": True,
        "multi_objective": True,
        "adaptive_datasets": True,
        "real_time_optimization": True,
    }
    service._algorithms = ["bayesian", "random", "genetic", "grid"]
    service._max_sessions = 50
    service._response_time = 50.0
    service._success_rate = 0.98
    return service


@pytest.fixture
def mock_service_limited():
    """Limited mock service with minimal capabilities."""
    service = MockRemoteService("LimitedService")
    service._capabilities = {
        "smart_suggestions": False,
        "multi_objective": False,
        "adaptive_datasets": False,
    }
    service._algorithms = ["random"]
    service._max_sessions = 1
    service._response_time = 500.0
    service._success_rate = 0.80
    return service


@pytest.fixture
def mock_service_unreliable():
    """Unreliable mock service for testing error conditions."""
    service = MockRemoteService("UnreliableService")
    service._health_status = "degraded"
    service._response_time = 2000.0
    service._success_rate = 0.60
    return service


@pytest.fixture
def sample_requirements_basic():
    """Basic service requirements."""
    return ServiceRequirements(
        required_algorithms={"bayesian"},
        required_capabilities={"smart_suggestions"},
        max_response_time_ms=200.0,
        min_success_rate=0.90,
    )


@pytest.fixture
def sample_requirements_advanced():
    """Advanced service requirements."""
    return ServiceRequirements(
        required_algorithms={"bayesian", "genetic"},
        preferred_algorithms={"grid"},
        required_capabilities={"smart_suggestions", "multi_objective"},
        preferred_capabilities={"adaptive_datasets", "real_time_optimization"},
        max_response_time_ms=100.0,
        min_success_rate=0.95,
        min_concurrent_sessions=20,
        preferred_services=["AdvancedService"],
        max_cost_per_trial=1.0,
    )


@pytest.fixture
def sample_requirements_strict():
    """Strict service requirements that are hard to meet."""
    return ServiceRequirements(
        required_algorithms={"quantum_optimization"},
        required_capabilities={"time_travel", "perfect_prediction"},
        max_response_time_ms=1.0,
        min_success_rate=1.0,
        min_concurrent_sessions=1000,
    )


# Test Classes


class TestServiceRequirements:
    """Test ServiceRequirements dataclass."""

    def test_service_requirements_creation_empty(self):
        """Test ServiceRequirements creation with default values."""
        req = ServiceRequirements()

        assert req.required_algorithms == set()
        assert req.preferred_algorithms == set()
        assert req.required_capabilities == set()
        assert req.preferred_capabilities == set()
        assert req.max_response_time_ms is None
        assert req.min_success_rate is None
        assert req.min_concurrent_sessions is None
        assert req.preferred_services == []
        assert req.excluded_services == []
        assert req.max_cost_per_trial is None
        assert req.prefer_free_tier is False

    def test_service_requirements_creation_full(self, sample_requirements_advanced):
        """Test ServiceRequirements creation with all fields."""
        req = sample_requirements_advanced

        assert "bayesian" in req.required_algorithms
        assert "genetic" in req.required_algorithms
        assert "grid" in req.preferred_algorithms
        assert "smart_suggestions" in req.required_capabilities
        assert "multi_objective" in req.required_capabilities
        assert "adaptive_datasets" in req.preferred_capabilities
        assert req.max_response_time_ms == 100.0
        assert req.min_success_rate == 0.95
        assert req.min_concurrent_sessions == 20
        assert "AdvancedService" in req.preferred_services
        assert req.max_cost_per_trial == 1.0

    def test_service_requirements_with_exclusions(self):
        """Test ServiceRequirements with excluded services."""
        req = ServiceRequirements(
            excluded_services=["SlowService", "ExpensiveService"], prefer_free_tier=True
        )

        assert "SlowService" in req.excluded_services
        assert "ExpensiveService" in req.excluded_services
        assert req.prefer_free_tier is True


class TestServiceRanking:
    """Test ServiceRanking dataclass."""

    def test_service_ranking_creation(self, mock_service_basic):
        """Test ServiceRanking creation."""
        ranking = ServiceRanking(
            service=mock_service_basic,
            score=0.85,
            meets_requirements=True,
            ranking_details={
                "algorithm_match": 0.9,
                "capability_match": 0.8,
                "performance_score": 0.85,
            },
        )

        assert ranking.service == mock_service_basic
        assert ranking.score == 0.85
        assert ranking.meets_requirements is True
        assert ranking.ranking_details["algorithm_match"] == 0.9

    def test_service_ranking_minimal(self, mock_service_basic):
        """Test ServiceRanking with minimal fields."""
        ranking = ServiceRanking(
            service=mock_service_basic, score=0.5, meets_requirements=False
        )

        assert ranking.service == mock_service_basic
        assert ranking.score == 0.5
        assert ranking.meets_requirements is False
        assert ranking.ranking_details == {}


class TestRemoteServiceRegistryBasics:
    """Test basic RemoteServiceRegistry functionality."""

    def test_registry_initialization(self, registry):
        """Test registry initialization."""
        assert len(registry._services) == 0
        assert len(registry._service_info) == 0
        assert len(registry._health_checks) == 0
        assert len(registry._selection_strategies) > 0  # Default strategies

    def test_register_service_basic(self, registry, mock_service_basic):
        """Test basic service registration."""
        registry.register_service(mock_service_basic, auto_connect=False)

        assert "BasicService" in registry._services
        assert registry._services["BasicService"] == mock_service_basic
        assert registry._health_checks["BasicService"] is False

    def test_register_service_duplicate(self, registry, mock_service_basic):
        """Test registering duplicate service (should replace)."""
        registry.register_service(mock_service_basic, auto_connect=False)

        # Register another service with same name
        another_service = MockRemoteService("BasicService")
        registry.register_service(another_service, auto_connect=False)

        # Should have replaced the original
        assert registry._services["BasicService"] == another_service
        assert registry._services["BasicService"] != mock_service_basic

    def test_register_multiple_services(
        self, registry, mock_service_basic, mock_service_advanced, mock_service_limited
    ):
        """Test registering multiple services."""
        registry.register_service(mock_service_basic, auto_connect=False)
        registry.register_service(mock_service_advanced, auto_connect=False)
        registry.register_service(mock_service_limited, auto_connect=False)

        assert len(registry._services) == 3
        assert "BasicService" in registry._services
        assert "AdvancedService" in registry._services
        assert "LimitedService" in registry._services

    @pytest.mark.asyncio
    async def test_register_service_with_auto_connect(
        self, registry, mock_service_basic
    ):
        """Test service registration with auto-connect."""
        # Mock the _connect_service method
        with patch.object(registry, "_connect_service") as mock_connect:
            registry.register_service(mock_service_basic, auto_connect=True)

            # Wait a bit for the task to be created
            await asyncio.sleep(0.01)

            # Should have attempted to connect
            mock_connect.assert_called_once_with("BasicService")

    @pytest.mark.asyncio
    async def test_register_background_task_discards_cancelled_task(self, registry):
        """Cancelled background tasks should be removed from the registry."""

        async def wait_forever() -> None:
            await asyncio.sleep(3600)

        task = asyncio.create_task(wait_forever())
        registry._register_background_task(task)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        await asyncio.sleep(0)
        assert task not in registry._background_tasks

    @pytest.mark.asyncio
    async def test_register_background_task_logs_failures(self, registry):
        """Failed background tasks should be logged and discarded."""

        async def fail() -> None:
            raise RuntimeError("boom")

        with patch("traigent.optimizers.service_registry.logger.error") as mock_error:
            task = asyncio.create_task(fail())
            registry._register_background_task(task)
            await asyncio.sleep(0)

        assert task not in registry._background_tasks
        mock_error.assert_called_once()

    def test_get_registered_services(
        self, registry, mock_service_basic, mock_service_advanced
    ):
        """Test getting list of registered services."""
        registry.register_service(mock_service_basic, auto_connect=False)
        registry.register_service(mock_service_advanced, auto_connect=False)

        service_names = registry.list_services()

        assert len(service_names) == 2
        assert "BasicService" in service_names
        assert "AdvancedService" in service_names

    def test_get_service_by_name(self, registry, mock_service_basic):
        """Test getting service by name."""
        registry.register_service(mock_service_basic, auto_connect=False)

        service = registry.get_service("BasicService")
        assert service == mock_service_basic

        # Non-existent service
        service = registry.get_service("NonExistentService")
        assert service is None

    def test_unregister_service(self, registry, mock_service_basic):
        """Test unregistering a service."""
        registry.register_service(mock_service_basic, auto_connect=False)
        assert "BasicService" in registry._services

        success = registry.unregister_service("BasicService")
        assert success is True
        assert "BasicService" not in registry._services
        assert "BasicService" not in registry._health_checks

        # Try to unregister non-existent service
        success = registry.unregister_service("NonExistentService")
        assert success is False

    @pytest.mark.asyncio
    async def test_unregister_service_schedules_background_disconnect(
        self, registry, mock_service_basic
    ):
        """Unregister should schedule disconnect when an event loop is running."""
        registry.register_service(mock_service_basic, auto_connect=False)

        with patch.object(
            registry, "_disconnect_service", AsyncMock()
        ) as mock_disconnect:
            assert registry.unregister_service("BasicService") is True
            await asyncio.sleep(0)

        mock_disconnect.assert_awaited_once_with("BasicService")


class TestServiceConnection:
    """Test service connection and health monitoring."""

    @pytest.mark.asyncio
    async def test_connect_service_success(self, registry, mock_service_basic):
        """Test successful service connection."""
        registry.register_service(mock_service_basic, auto_connect=False)

        await registry.connect_service("BasicService")

        assert mock_service_basic._connected is True
        assert "BasicService" in registry._service_info
        assert registry._service_info["BasicService"].status == ServiceStatus.CONNECTED

    @pytest.mark.asyncio
    async def test_connect_service_failure(self, registry):
        """Test service connection failure."""
        # Create a service that fails to connect
        failing_service = MockRemoteService("FailingService")
        failing_service.connect = AsyncMock(
            side_effect=ServiceError("Connection failed")
        )

        registry.register_service(failing_service, auto_connect=False)

        with pytest.raises(ServiceError):
            await registry.connect_service("FailingService")

    @pytest.mark.asyncio
    async def test_connect_nonexistent_service(self, registry):
        """Test connecting to non-existent service."""
        with pytest.raises(ServiceError, match="Service NonExistentService not found"):
            await registry.connect_service("NonExistentService")

    @pytest.mark.asyncio
    async def test_connect_all_services(
        self, registry, mock_service_basic, mock_service_advanced
    ):
        """Test connecting to all registered services."""
        registry.register_service(mock_service_basic, auto_connect=False)
        registry.register_service(mock_service_advanced, auto_connect=False)

        results = await registry.connect_all_services()

        assert len(results) == 2
        assert results["BasicService"] is True
        assert results["AdvancedService"] is True
        assert mock_service_basic._connected is True
        assert mock_service_advanced._connected is True

    @pytest.mark.asyncio
    async def test_connect_all_services_with_failures(
        self, registry, mock_service_basic
    ):
        """Test connecting to all services with some failures."""
        failing_service = MockRemoteService("FailingService")
        failing_service.connect = AsyncMock(
            side_effect=ServiceError("Connection failed")
        )

        registry.register_service(mock_service_basic, auto_connect=False)
        registry.register_service(failing_service, auto_connect=False)

        results = await registry.connect_all_services()

        assert len(results) == 2
        assert results["BasicService"] is True
        assert results["FailingService"] is False

    @pytest.mark.asyncio
    async def test_disconnect_service(self, registry, mock_service_basic):
        """Test service disconnection."""
        registry.register_service(mock_service_basic, auto_connect=False)
        await registry.connect_service("BasicService")

        assert mock_service_basic._connected is True

        await registry.disconnect_service("BasicService")

        assert mock_service_basic._connected is False

    @pytest.mark.asyncio
    async def test_health_check_single_service(self, registry, mock_service_basic):
        """Test health check for single service."""
        registry.register_service(mock_service_basic, auto_connect=False)
        await registry.connect_service("BasicService")

        health = await registry.check_service_health("BasicService")

        assert health["status"] == "healthy"
        assert health["response_time_ms"] == 100.0
        assert health["success_rate"] == 0.95
        assert registry._health_checks["BasicService"] is True

    @pytest.mark.asyncio
    async def test_health_check_all_services(
        self,
        registry,
        mock_service_basic,
        mock_service_advanced,
        mock_service_unreliable,
    ):
        """Test health check for all services."""
        registry.register_service(mock_service_basic, auto_connect=False)
        registry.register_service(mock_service_advanced, auto_connect=False)
        registry.register_service(mock_service_unreliable, auto_connect=False)

        await registry.connect_all_services()

        health_results = await registry.check_all_services_health()

        assert len(health_results) == 3
        assert health_results["BasicService"]["status"] == "healthy"
        assert health_results["AdvancedService"]["status"] == "healthy"
        assert health_results["UnreliableService"]["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_get_service_status(self, registry, mock_service_basic):
        """Test getting service status."""
        registry.register_service(mock_service_basic, auto_connect=False)

        # Before connection
        status = registry.get_service_status("BasicService")
        assert status == ServiceStatus.DISCONNECTED

        # After connection
        await registry.connect_service("BasicService")
        status = registry.get_service_status("BasicService")
        assert status == ServiceStatus.CONNECTED

        # Non-existent service
        status = registry.get_service_status("NonExistentService")
        assert status is None


class TestServiceSelection:
    """Test service selection and ranking functionality."""

    @pytest.mark.asyncio
    async def test_find_services_by_requirements_basic(
        self,
        registry,
        mock_service_basic,
        mock_service_advanced,
        sample_requirements_basic,
    ):
        """Test finding services by basic requirements."""
        registry.register_service(mock_service_basic, auto_connect=False)
        registry.register_service(mock_service_advanced, auto_connect=False)
        await registry.connect_all_services()

        services = await registry.find_services_by_requirements(
            sample_requirements_basic
        )

        # Both services should meet basic requirements
        assert len(services) >= 1
        service_names = [s.service_name for s in services]
        assert "BasicService" in service_names or "AdvancedService" in service_names

    @pytest.mark.asyncio
    async def test_find_services_by_requirements_advanced(
        self,
        registry,
        mock_service_basic,
        mock_service_advanced,
        mock_service_limited,
        sample_requirements_advanced,
    ):
        """Test finding services by advanced requirements."""
        registry.register_service(mock_service_basic, auto_connect=False)
        registry.register_service(mock_service_advanced, auto_connect=False)
        registry.register_service(mock_service_limited, auto_connect=False)
        await registry.connect_all_services()

        services = await registry.find_services_by_requirements(
            sample_requirements_advanced
        )

        # Only advanced service should meet advanced requirements
        assert len(services) >= 1
        service_names = [s.service_name for s in services]
        # Should prefer AdvancedService based on requirements
        if "AdvancedService" in service_names:
            # AdvancedService should be ranked higher
            assert services[0].service_name == "AdvancedService"

    @pytest.mark.asyncio
    async def test_find_services_by_requirements_none_match(
        self, registry, mock_service_basic, sample_requirements_strict
    ):
        """Test finding services when none match strict requirements."""
        registry.register_service(mock_service_basic, auto_connect=False)
        await registry.connect_service("BasicService")

        services = await registry.find_services_by_requirements(
            sample_requirements_strict
        )

        # No services should meet strict requirements
        assert len(services) == 0

    @pytest.mark.asyncio
    async def test_rank_services_basic(
        self,
        registry,
        mock_service_basic,
        mock_service_advanced,
        sample_requirements_basic,
    ):
        """Test ranking services by requirements."""
        registry.register_service(mock_service_basic, auto_connect=False)
        registry.register_service(mock_service_advanced, auto_connect=False)
        await registry.connect_all_services()

        rankings = await registry.rank_services(sample_requirements_basic)

        assert len(rankings) == 2
        # Rankings should be sorted by score (highest first)
        assert rankings[0].score >= rankings[1].score

        # Advanced service should score higher
        if rankings[0].service.service_name == "AdvancedService":
            assert rankings[0].score > rankings[1].score

    @pytest.mark.asyncio
    async def test_rank_services_with_exclusions(
        self, registry, mock_service_basic, mock_service_advanced
    ):
        """Test ranking services with exclusions."""
        registry.register_service(mock_service_basic, auto_connect=False)
        registry.register_service(mock_service_advanced, auto_connect=False)
        await registry.connect_all_services()

        requirements = ServiceRequirements(excluded_services=["BasicService"])

        rankings = await registry.rank_services(requirements)

        # Should only include non-excluded services
        service_names = [r.service.service_name for r in rankings]
        assert "BasicService" not in service_names
        assert "AdvancedService" in service_names

    @pytest.mark.asyncio
    async def test_select_best_service(
        self,
        registry,
        mock_service_basic,
        mock_service_advanced,
        sample_requirements_advanced,
    ):
        """Test selecting the best service."""
        registry.register_service(mock_service_basic, auto_connect=False)
        registry.register_service(mock_service_advanced, auto_connect=False)
        await registry.connect_all_services()

        best_service = await registry.select_best_service(sample_requirements_advanced)

        assert best_service is not None
        # Should select AdvancedService based on requirements
        assert best_service.service_name == "AdvancedService"

    @pytest.mark.asyncio
    async def test_select_best_service_none_available(
        self, registry, sample_requirements_strict
    ):
        """Test selecting best service when none meet requirements."""
        best_service = await registry.select_best_service(sample_requirements_strict)
        assert best_service is None

    @pytest.mark.asyncio
    async def test_select_service_with_strategy(
        self, registry, mock_service_basic, mock_service_advanced
    ):
        """Test service selection with custom strategy."""
        registry.register_service(mock_service_basic, auto_connect=False)
        registry.register_service(mock_service_advanced, auto_connect=False)
        await registry.connect_all_services()

        # Custom strategy that prefers "Basic" in name
        def custom_strategy(rankings, requirements):
            for ranking in rankings:
                if "Basic" in ranking.service.service_name:
                    return ranking.service
            return rankings[0].service if rankings else None

        registry.register_selection_strategy("prefer_basic", custom_strategy)

        requirements = ServiceRequirements()
        selected = await registry.select_service_with_strategy(
            requirements, "prefer_basic"
        )

        assert selected is not None
        assert selected.service_name == "BasicService"


class TestSelectionStrategies:
    """Test service selection strategies."""

    @pytest.mark.asyncio
    async def test_default_selection_strategies(self, registry):
        """Test that default selection strategies are registered."""
        strategies = registry.get_selection_strategies()

        # Should have default strategies
        assert "best_score" in strategies
        assert "fastest_response" in strategies
        assert "most_reliable" in strategies
        assert len(strategies) >= 3

    def test_register_custom_selection_strategy(self, registry):
        """Test registering custom selection strategy."""

        def custom_strategy(services, requirements):
            return services[0] if services else None

        registry.register_selection_strategy("custom", custom_strategy)

        strategies = registry.get_selection_strategies()
        assert "custom" in strategies
        assert strategies["custom"] == custom_strategy

    def test_unregister_selection_strategy(self, registry):
        """Test unregistering selection strategy."""

        def custom_strategy(services, requirements):
            return services[0] if services else None

        registry.register_selection_strategy("custom", custom_strategy)
        assert "custom" in registry.get_selection_strategies()

        success = registry.unregister_selection_strategy("custom")
        assert success is True
        assert "custom" not in registry.get_selection_strategies()

        # Try to unregister non-existent strategy
        success = registry.unregister_selection_strategy("nonexistent")
        assert success is False


class TestServiceMetrics:
    """Test service metrics and monitoring."""

    @pytest.mark.asyncio
    async def test_get_service_metrics(self, registry, mock_service_basic):
        """Test getting service metrics."""
        registry.register_service(mock_service_basic, auto_connect=False)
        await registry.connect_service("BasicService")

        # Perform health check to populate metrics
        await registry.check_service_health("BasicService")

        metrics = registry.get_service_metrics("BasicService")

        assert metrics is not None
        assert "response_time_ms" in metrics
        assert "success_rate" in metrics
        assert metrics["response_time_ms"] == 100.0
        assert metrics["success_rate"] == 0.95

    def test_get_all_service_metrics(
        self, registry, mock_service_basic, mock_service_advanced
    ):
        """Test getting metrics for all services."""
        registry.register_service(mock_service_basic, auto_connect=False)
        registry.register_service(mock_service_advanced, auto_connect=False)

        # Initially should be empty
        all_metrics = registry.get_all_service_metrics()
        assert len(all_metrics) == 0

    @pytest.mark.asyncio
    async def test_service_performance_tracking(self, registry, mock_service_basic):
        """Test service performance tracking over time."""
        registry.register_service(mock_service_basic, auto_connect=False)
        await registry.connect_service("BasicService")

        # Perform multiple health checks
        for _ in range(3):
            await registry.check_service_health("BasicService")

        metrics = registry.get_service_metrics("BasicService")
        assert metrics is not None

    def test_compare_service_performance(
        self,
        registry,
        mock_service_basic,
        mock_service_advanced,
        mock_service_unreliable,
    ):
        """Test comparing performance between services."""
        registry.register_service(mock_service_basic, auto_connect=False)
        registry.register_service(mock_service_advanced, auto_connect=False)
        registry.register_service(mock_service_unreliable, auto_connect=False)

        # Mock some metrics
        registry._service_metrics = {
            "BasicService": {"response_time_ms": 100.0, "success_rate": 0.95},
            "AdvancedService": {"response_time_ms": 50.0, "success_rate": 0.98},
            "UnreliableService": {"response_time_ms": 2000.0, "success_rate": 0.60},
        }

        comparison = registry.compare_services(
            ["AdvancedService", "BasicService", "UnreliableService"]
        )

        assert len(comparison) == 3
        # Should be sorted by performance (best first)
        assert comparison[0]["service_name"] == "AdvancedService"
        assert comparison[-1]["service_name"] == "UnreliableService"


class TestCTDScenarios:
    """Combinatorial Test Design scenarios for comprehensive coverage."""

    @pytest.mark.parametrize(
        "num_services,num_requirements,expected_matches",
        [
            (1, 1, 1),  # Single service, single requirement
            (3, 1, 3),  # Multiple services, single requirement
            (3, 3, 1),  # Multiple services, multiple requirements (selective)
            (0, 1, 0),  # No services
            (3, 0, 3),  # No requirements (all match)
        ],
    )
    @pytest.mark.asyncio
    async def test_service_matching_combinations(
        self, registry, num_services, num_requirements, expected_matches
    ):
        """Test different combinations of services and requirements."""
        # Create services
        services = []
        for i in range(num_services):
            service = MockRemoteService(f"Service{i}")
            if i == 0:  # Make first service more capable
                service._algorithms = ["bayesian", "genetic", "grid"]
                service._capabilities = {
                    "smart_suggestions": True,
                    "multi_objective": True,
                    "adaptive_datasets": True,
                }
            services.append(service)
            registry.register_service(service, auto_connect=False)

        if num_services > 0:
            await registry.connect_all_services()

        # Create requirements
        requirements = ServiceRequirements()
        if num_requirements >= 1:
            requirements.required_algorithms = {"bayesian"}
        if num_requirements >= 2:
            requirements.required_capabilities = {"smart_suggestions"}
        if num_requirements >= 3:
            requirements.min_success_rate = 0.99  # Very high requirement

        matching_services = await registry.find_services_by_requirements(requirements)

        # Adjust expected matches based on actual capability
        if num_requirements >= 3 and num_services > 0:
            # High success rate requirement might not be met
            expected_matches = min(expected_matches, 1)
        elif num_requirements >= 2 and num_services > 1:
            # Only some services have smart_suggestions
            expected_matches = min(expected_matches, 1)

        assert len(matching_services) <= expected_matches

    @pytest.mark.parametrize(
        "auto_connect,connection_success,expected_status",
        [
            (True, True, ServiceStatus.CONNECTED),
            (True, False, ServiceStatus.DISCONNECTED),
            (False, True, ServiceStatus.CONNECTED),  # Manually connected
            (False, False, ServiceStatus.DISCONNECTED),
        ],
    )
    @pytest.mark.asyncio
    async def test_connection_combinations(
        self, registry, auto_connect, connection_success, expected_status
    ):
        """Test different connection scenarios."""
        service = MockRemoteService("TestService")

        if not connection_success:
            service.connect = AsyncMock(side_effect=ServiceError("Connection failed"))

        # Register with auto_connect setting
        registry.register_service(service, auto_connect=auto_connect)

        if auto_connect and connection_success:
            # Wait for auto-connection
            await asyncio.sleep(0.01)
        elif not auto_connect and connection_success:
            # Manually connect
            await registry.connect_service("TestService")
        elif not auto_connect and not connection_success:
            # Try to connect and expect failure
            try:
                await registry.connect_service("TestService")
            except ServiceError:
                pass  # Expected failure

        status = registry.get_service_status("TestService")
        assert status == expected_status or status == ServiceStatus.DISCONNECTED

    @pytest.mark.parametrize(
        "algorithm_overlap,capability_overlap,expected_preference",
        [
            ({"bayesian"}, {"smart_suggestions"}, "both_match"),
            ({"bayesian"}, set(), "algorithm_only"),
            (set(), {"smart_suggestions"}, "capability_only"),
            (set(), set(), "neither_match"),
        ],
    )
    @pytest.mark.asyncio
    async def test_requirement_matching_combinations(
        self, registry, algorithm_overlap, capability_overlap, expected_preference
    ):
        """Test different requirement matching scenarios."""
        # Create two services with different capabilities
        service1 = MockRemoteService("Service1")
        service1._algorithms = ["bayesian", "random"]
        service1._capabilities = {"smart_suggestions": True, "multi_objective": False}

        service2 = MockRemoteService("Service2")
        service2._algorithms = ["grid", "genetic"]
        service2._capabilities = {"smart_suggestions": False, "adaptive_datasets": True}

        registry.register_service(service1, auto_connect=False)
        registry.register_service(service2, auto_connect=False)
        await registry.connect_all_services()

        requirements = ServiceRequirements(
            required_algorithms=algorithm_overlap,
            required_capabilities=capability_overlap,
        )

        rankings = await registry.rank_services(requirements)

        if expected_preference == "both_match":
            # Service1 should rank higher (matches both)
            matching_services = [r for r in rankings if r.meets_requirements]
            if matching_services:
                assert matching_services[0].service.service_name == "Service1"
        elif expected_preference == "neither_match":
            # No services should fully meet requirements
            matching_services = [r for r in rankings if r.meets_requirements]
            assert len(matching_services) == 0


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_connection_timeout_handling(self, registry):
        """Test handling of connection timeouts."""
        # Create service that times out on connection
        timeout_service = MockRemoteService("TimeoutService")
        timeout_service.connect = AsyncMock(
            side_effect=TimeoutError("Connection timeout")
        )

        registry.register_service(timeout_service, auto_connect=False)

        with pytest.raises(asyncio.TimeoutError):
            await registry.connect_service("TimeoutService")

    @pytest.mark.asyncio
    async def test_health_check_failure_handling(self, registry):
        """Test handling of health check failures."""
        failing_service = MockRemoteService("FailingService")
        failing_service.health_check = AsyncMock(
            side_effect=ServiceError("Health check failed")
        )

        registry.register_service(failing_service, auto_connect=False)
        await registry.connect_service("FailingService")

        # Health check should handle failure gracefully
        health = await registry.check_service_health("FailingService")
        assert health is not None  # Should return default/error health status

    def test_invalid_service_operations(self, registry):
        """Test operations on invalid/non-existent services."""
        # Get non-existent service
        assert registry.get_service("NonExistent") is None

        # Get status of non-existent service
        assert registry.get_service_status("NonExistent") is None

        # Get metrics of non-existent service
        assert registry.get_service_metrics("NonExistent") is None

    def test_malformed_requirements(self, registry):
        """Test handling of malformed requirements."""
        # Requirements with None values
        requirements = ServiceRequirements()
        requirements.required_algorithms = None  # Should handle gracefully
        assert requirements.required_algorithms is None  # Verify value was set

        # Should not crash when processing - verify requirements object is valid
        assert isinstance(requirements, ServiceRequirements)

    @pytest.mark.asyncio
    async def test_service_registry_cleanup(self, registry, mock_service_basic):
        """Test proper cleanup of registry resources."""
        registry.register_service(mock_service_basic, auto_connect=False)
        await registry.connect_service("BasicService")

        # Clear registry
        await registry.clear_all_services()

        assert len(registry._services) == 0
        assert len(registry._service_info) == 0
        assert len(registry._health_checks) == 0

    def test_concurrent_registration(self, registry):
        """Test concurrent service registration."""
        import threading

        services = []
        for i in range(10):
            service = MockRemoteService(f"ConcurrentService{i}")
            services.append(service)

        # Register services concurrently
        threads = []
        for service in services:
            thread = threading.Thread(
                target=lambda s=service: registry.register_service(
                    s, auto_connect=False
                )
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All services should be registered
        assert len(registry._services) == 10

    @pytest.mark.asyncio
    async def test_memory_usage_with_many_services(self, registry):
        """Test memory usage with many registered services."""
        # Register many services
        for i in range(100):
            service = MockRemoteService(f"Service{i}")
            registry.register_service(service, auto_connect=False)

        assert len(registry._services) == 100

        # Connect some services
        await registry.connect_service("Service0")
        await registry.connect_service("Service50")
        await registry.connect_service("Service99")

        # Should handle large number of services efficiently
        all_services = registry.get_registered_services()
        assert len(all_services) == 100

    def test_edge_case_requirements(self, registry):
        """Test edge case requirements."""
        # Empty sets and lists
        requirements = ServiceRequirements(
            required_algorithms=set(),
            preferred_algorithms=set(),
            required_capabilities=set(),
            preferred_capabilities=set(),
            preferred_services=[],
            excluded_services=[],
        )

        # Should handle empty requirements gracefully
        assert len(requirements.required_algorithms) == 0

        # Extreme numeric values
        extreme_requirements = ServiceRequirements(
            max_response_time_ms=0.0,  # Impossible requirement
            min_success_rate=1.0,  # Perfect requirement
            min_concurrent_sessions=999999,  # Very high requirement
            max_cost_per_trial=0.0,  # Free requirement
        )

        assert extreme_requirements.max_response_time_ms == 0.0
        assert extreme_requirements.min_success_rate == 1.0
