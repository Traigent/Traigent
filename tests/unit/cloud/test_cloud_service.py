"""Tests for Traigent Cloud Service."""

import time
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from traigent.cloud.service import (
    OptimizationRequest,
    OptimizationResponse,
    TraigentCloudService,
)
from traigent.evaluators.base import Dataset, EvaluationExample


@pytest.fixture
def sample_dataset():
    """Create sample dataset for testing."""
    examples = [
        EvaluationExample(
            input_data={"query": "What is AI?"},
            expected_output="AI is artificial intelligence.",
            metadata={"difficulty": "easy"},
        ),
        EvaluationExample(
            input_data={"query": "What is ML?"},
            expected_output="ML is machine learning.",
            metadata={"difficulty": "medium"},
        ),
        EvaluationExample(
            input_data={"query": "What is DL?"},
            expected_output="DL is deep learning.",
            metadata={"difficulty": "hard"},
        ),
        EvaluationExample(
            input_data={"query": "What is NLP?"},
            expected_output="NLP is natural language processing.",
            metadata={"difficulty": "medium"},
        ),
        EvaluationExample(
            input_data={"query": "What is CV?"},
            expected_output="CV is computer vision.",
            metadata={"difficulty": "medium"},
        ),
    ]
    return Dataset(examples=examples, name="test_dataset", description="Test dataset")


@pytest.fixture
def optimization_request(sample_dataset):
    """Create optimization request for testing."""
    return OptimizationRequest(
        function_name="llm_function",
        dataset=sample_dataset,
        configuration_space={
            "model": ["gpt-3.5", "gpt-4"],
            "temperature": [0.0, 0.5, 1.0],
            "max_tokens": [100, 200, 300],
        },
        objectives=["accuracy", "cost"],
        max_trials=50,
        target_cost_reduction=0.65,
        user_id="test_user_123",
        billing_tier="standard",
    )


@pytest.fixture
def cloud_service():
    """Create cloud service instance for testing."""
    return TraigentCloudService()


class TestOptimizationRequest:
    """Test OptimizationRequest dataclass."""

    def test_optimization_request_creation(self, sample_dataset):
        """Test creating optimization request."""
        request = OptimizationRequest(
            function_name="test_function",
            dataset=sample_dataset,
            configuration_space={"param": [1, 2, 3]},
            objectives=["accuracy"],
            max_trials=25,
            target_cost_reduction=0.7,
            user_id="user_123",
            billing_tier="professional",
        )

        assert request.function_name == "test_function"
        assert request.dataset == sample_dataset
        assert request.configuration_space == {"param": [1, 2, 3]}
        assert request.objectives == ["accuracy"]
        assert request.max_trials == 25
        assert request.target_cost_reduction == 0.7
        assert request.user_id == "user_123"
        assert request.billing_tier == "professional"

    def test_optimization_request_defaults(self, sample_dataset):
        """Test optimization request with default values."""
        request = OptimizationRequest(
            function_name="test_function",
            dataset=sample_dataset,
            configuration_space={"param": [1, 2, 3]},
            objectives=["accuracy"],
        )

        assert request.max_trials == 50
        assert request.target_cost_reduction == 0.65
        assert request.user_id is None
        assert request.billing_tier == "standard"


class TestOptimizationResponse:
    """Test OptimizationResponse dataclass."""

    def test_optimization_response_creation(self):
        """Test creating optimization response."""
        response = OptimizationResponse(
            request_id="opt_123456",
            best_config={"param": "value"},
            best_metrics={"accuracy": 0.85},
            trials_count=25,
            optimization_time=120.5,
            cost_reduction=0.6,
            subset_used=True,
            billing_info={"credits_used": 10.0},
            status="completed",
        )

        assert response.request_id == "opt_123456"
        assert response.best_config == {"param": "value"}
        assert response.best_metrics == {"accuracy": 0.85}
        assert response.trials_count == 25
        assert response.optimization_time == 120.5
        assert response.cost_reduction == 0.6
        assert response.subset_used is True
        assert response.billing_info == {"credits_used": 10.0}
        assert response.status == "completed"

    def test_optimization_response_default_status(self):
        """Test optimization response with default status."""
        response = OptimizationResponse(
            request_id="opt_123",
            best_config={},
            best_metrics={},
            trials_count=0,
            optimization_time=0.0,
            cost_reduction=0.0,
            subset_used=False,
            billing_info={},
        )

        assert response.status == "completed"


class TestTraigentCloudService:
    """Test Traigent Cloud Service functionality."""

    def test_service_initialization(self, cloud_service):
        """Test service initialization."""
        assert cloud_service.subset_selector is not None
        assert cloud_service.usage_tracker is not None
        assert cloud_service.billing_manager is not None
        assert cloud_service.total_optimizations == 0
        assert cloud_service.total_cost_savings == 0.0
        assert cloud_service.uptime_start <= time.time()

    def test_process_optimization_request_success(
        self, cloud_service, optimization_request
    ):
        """Test successful optimization request processing."""

        async def run_test():
            with (
                patch.object(
                    cloud_service.billing_manager, "check_usage_limits"
                ) as mock_limits,
                patch.object(
                    cloud_service.subset_selector, "select_optimal_subset"
                ) as mock_subset,
                patch.object(
                    cloud_service, "_run_enhanced_optimization"
                ) as mock_optimize,
                patch.object(
                    cloud_service.usage_tracker, "record_optimization"
                ) as mock_record,
            ):

                # Mock billing limits check (allowed)
                mock_limits.return_value = {
                    "allowed": True,
                    "estimated_cost": 5.0,
                    "remaining_credits": 95.0,
                }

                # Mock subset selection (50% reduction)
                subset_examples = optimization_request.dataset.examples[
                    :3
                ]  # 3 out of 5
                mock_subset.return_value = Dataset(
                    examples=subset_examples, name="subset"
                )

                # Mock optimization results
                mock_optimize.return_value = {
                    "best_config": {"temperature": 0.7, "max_tokens": 150},
                    "best_metrics": {"accuracy": 0.88, "cost": 0.05},
                    "trials_count": 25,
                }

                # Process request
                response = await cloud_service.process_optimization_request(
                    optimization_request
                )

                # Verify response
                assert response.status == "completed"
                assert response.best_config == {"temperature": 0.7, "max_tokens": 150}
                assert response.best_metrics == {"accuracy": 0.88, "cost": 0.05}
                assert response.trials_count == 25
                assert response.cost_reduction == 0.4  # 3/5 = 60%, so reduction = 40%
                assert response.subset_used is True
                assert response.billing_info["credits_used"] == 5.0
                assert response.billing_info["remaining_credits"] == 95.0
                assert response.billing_info["billing_tier"] == "standard"

                # Verify service statistics updated
                assert cloud_service.total_optimizations == 1
                assert cloud_service.total_cost_savings == 0.4

                # Verify method calls
                mock_limits.assert_called_once_with(50, 5)  # max_trials, dataset_size
                mock_subset.assert_called_once_with(
                    optimization_request.dataset, target_reduction=0.65
                )
                mock_optimize.assert_called_once()
                mock_record.assert_called_once()

        import asyncio

        asyncio.run(run_test())

    def test_process_optimization_request_empty_dataset(self, cloud_service):
        """Ensure empty datasets do not cause division errors."""

        async def run_test():
            empty_dataset = Dataset(examples=[], name="empty", description="Empty")
            empty_request = OptimizationRequest(
                function_name="llm_function",
                dataset=empty_dataset,
                configuration_space={"model": ["gpt-3.5"], "temperature": [0.0, 0.5]},
                objectives=["accuracy"],
                max_trials=10,
                target_cost_reduction=0.5,
                user_id="user",
                billing_tier="standard",
            )

            with (
                patch.object(
                    cloud_service.billing_manager, "check_usage_limits"
                ) as mock_limits,
                patch.object(
                    cloud_service.subset_selector, "select_optimal_subset"
                ) as mock_subset,
                patch.object(
                    cloud_service, "_run_enhanced_optimization"
                ) as mock_optimize,
                patch.object(
                    cloud_service.usage_tracker, "record_optimization"
                ) as mock_record,
            ):

                mock_limits.return_value = {
                    "allowed": True,
                    "estimated_cost": 0.0,
                    "remaining_credits": 100.0,
                }

                mock_optimize.return_value = {
                    "best_config": {},
                    "best_metrics": {},
                    "trials_count": 0,
                }

                response = await cloud_service.process_optimization_request(
                    empty_request
                )

                assert response.status == "completed"
                assert response.cost_reduction == 0.0
                assert response.subset_used is False
                assert cloud_service.total_cost_savings == 0.0

                mock_limits.assert_called_once_with(10, 0)
                mock_subset.assert_not_called()
                mock_optimize.assert_called_once()
                mock_record.assert_called_once()

        import asyncio

        asyncio.run(run_test())

    def test_process_optimization_request_billing_limits_exceeded(
        self, cloud_service, optimization_request
    ):
        """Test optimization request with billing limits exceeded."""

        async def run_test():
            with patch.object(
                cloud_service.billing_manager, "check_usage_limits"
            ) as mock_limits:
                # Mock billing limits check (not allowed)
                mock_limits.return_value = {
                    "allowed": False,
                    "reason": "Credit limit exceeded",
                    "remaining_credits": 0,
                }

                response = await cloud_service.process_optimization_request(
                    optimization_request
                )

                # Verify failure response
                assert response.status == "failed_limits"
                assert response.best_config == {}
                assert response.best_metrics == {}
                assert response.trials_count == 0
                assert response.optimization_time == 0.0
                assert response.cost_reduction == 0.0
                assert response.subset_used is False
                assert "reason" in response.billing_info

        import asyncio

        asyncio.run(run_test())

    def test_process_optimization_request_exception_handling(
        self, cloud_service, optimization_request
    ):
        """Test optimization request with exception handling."""

        async def run_test():
            with patch.object(
                cloud_service.billing_manager,
                "check_usage_limits",
                side_effect=Exception("Billing error"),
            ):
                response = await cloud_service.process_optimization_request(
                    optimization_request
                )

                # Verify error response
                assert response.status == "failed"
                assert response.best_config == {}
                assert response.best_metrics == {}
                assert response.trials_count == 0
                assert response.cost_reduction == 0.0
                assert response.subset_used is False
                assert response.billing_info["error"] == "Billing error"

        import asyncio

        asyncio.run(run_test())

    def test_run_enhanced_optimization_professional_tier(
        self, cloud_service, sample_dataset
    ):
        """Test enhanced optimization for professional tier."""

        async def run_test():
            with (
                patch("traigent.cloud.service.get_optimizer") as mock_get_optimizer,
                patch("traigent.cloud.service.LocalEvaluator") as mock_evaluator_class,
            ):

                # Mock Bayesian optimizer
                mock_optimizer = MagicMock()
                mock_optimizer.optimize = AsyncMock(
                    return_value=MagicMock(
                        best_config={"param": "value"},
                        best_metrics={"accuracy": 0.9},
                        trials=[MagicMock() for _ in range(30)],  # 30 trials
                    )
                )
                mock_get_optimizer.return_value = mock_optimizer

                # Mock evaluator
                mock_evaluator_class.return_value = MagicMock()

                result = await cloud_service._run_enhanced_optimization(
                    dataset=sample_dataset,
                    configuration_space={"param": [1, 2, 3]},
                    objectives=["accuracy"],
                    max_trials=20,
                    billing_tier="professional",
                )

                # Verify Bayesian optimizer was requested
                mock_get_optimizer.assert_called_with(
                    "bayesian",
                    {"param": [1, 2, 3]},
                    ["accuracy"],
                    max_trials=30,
                )

                # Verify max_trials was adjusted (1.5x for professional)
                optimize_call = mock_optimizer.optimize.call_args
                assert optimize_call[1]["max_trials"] == 30  # 20 * 1.5

                # Verify result
                assert result["best_config"] == {"param": "value"}
                assert result["best_metrics"] == {"accuracy": 0.9}
                assert result["trials_count"] == 30

        import asyncio

        asyncio.run(run_test())

    def test_run_enhanced_optimization_standard_tier(
        self, cloud_service, sample_dataset
    ):
        """Test enhanced optimization for standard tier."""

        async def run_test():
            with (
                patch("traigent.cloud.service.get_optimizer") as mock_get_optimizer,
                patch("traigent.cloud.service.LocalEvaluator") as mock_evaluator_class,
            ):

                # Mock random optimizer
                mock_optimizer = MagicMock()
                mock_optimizer.optimize = AsyncMock(
                    return_value=MagicMock(
                        best_config={"param": "value"},
                        best_metrics={"accuracy": 0.8},
                        trials=[MagicMock() for _ in range(25)],
                    )
                )
                mock_get_optimizer.return_value = mock_optimizer
                mock_evaluator_class.return_value = MagicMock()

                await cloud_service._run_enhanced_optimization(
                    dataset=sample_dataset,
                    configuration_space={"param": [1, 2, 3]},
                    objectives=["accuracy"],
                    max_trials=25,
                    billing_tier="standard",
                )

                # Verify random optimizer was used
                mock_get_optimizer.assert_called_with(
                    "random",
                    {"param": [1, 2, 3]},
                    ["accuracy"],
                    max_trials=25,
                )

                # Verify max_trials not adjusted (1.0x for standard)
                optimize_call = mock_optimizer.optimize.call_args
                assert optimize_call[1]["max_trials"] == 25

        import asyncio

        asyncio.run(run_test())

    def test_run_enhanced_optimization_enterprise_tier(
        self, cloud_service, sample_dataset
    ):
        """Test enhanced optimization for enterprise tier."""

        async def run_test():
            with (
                patch("traigent.cloud.service.get_optimizer") as mock_get_optimizer,
                patch("traigent.cloud.service.LocalEvaluator") as mock_evaluator_class,
            ):

                mock_optimizer = MagicMock()
                mock_optimizer.optimize = AsyncMock(
                    return_value=MagicMock(
                        best_config={"param": "value"},
                        best_metrics={"accuracy": 0.95},
                        trials=[MagicMock() for _ in range(40)],
                    )
                )
                mock_get_optimizer.return_value = mock_optimizer
                mock_evaluator_class.return_value = MagicMock()

                await cloud_service._run_enhanced_optimization(
                    dataset=sample_dataset,
                    configuration_space={"param": [1, 2, 3]},
                    objectives=["accuracy"],
                    max_trials=20,
                    billing_tier="enterprise",
                )

                # Verify Bayesian optimizer was requested
                mock_get_optimizer.assert_called_with(
                    "bayesian",
                    {"param": [1, 2, 3]},
                    ["accuracy"],
                    max_trials=40,
                )

                # Verify max_trials was adjusted (2.0x for enterprise)
                optimize_call = mock_optimizer.optimize.call_args
                assert optimize_call[1]["max_trials"] == 40  # 20 * 2.0

        import asyncio

        asyncio.run(run_test())

    def test_run_enhanced_optimization_free_tier(self, cloud_service, sample_dataset):
        """Test enhanced optimization for free tier."""

        async def run_test():
            with (
                patch("traigent.cloud.service.get_optimizer") as mock_get_optimizer,
                patch("traigent.cloud.service.LocalEvaluator") as mock_evaluator_class,
            ):

                mock_optimizer = MagicMock()
                mock_optimizer.optimize = AsyncMock(
                    return_value=MagicMock(
                        best_config={"param": "value"},
                        best_metrics={"accuracy": 0.75},
                        trials=[MagicMock() for _ in range(10)],
                    )
                )
                mock_get_optimizer.return_value = mock_optimizer
                mock_evaluator_class.return_value = MagicMock()

                await cloud_service._run_enhanced_optimization(
                    dataset=sample_dataset,
                    configuration_space={"param": [1, 2, 3]},
                    objectives=["accuracy"],
                    max_trials=20,
                    billing_tier="free",
                )

                # Verify random optimizer was used
                mock_get_optimizer.assert_called_with(
                    "random",
                    {"param": [1, 2, 3]},
                    ["accuracy"],
                    max_trials=10,
                )

                # Verify max_trials was adjusted (0.5x for free)
                optimize_call = mock_optimizer.optimize.call_args
                assert optimize_call[1]["max_trials"] == 10  # 20 * 0.5

        import asyncio

        asyncio.run(run_test())

    def test_run_enhanced_optimization_fallback_optimizer(
        self, cloud_service, sample_dataset
    ):
        """Test fallback to random optimizer when Bayesian fails."""

        async def run_test():
            with (
                patch("traigent.cloud.service.get_optimizer") as mock_get_optimizer,
                patch("traigent.cloud.service.LocalEvaluator") as mock_evaluator_class,
            ):

                # First call (Bayesian) raises exception, second call (random) succeeds
                mock_optimizer = MagicMock()
                mock_optimizer.optimize = AsyncMock(
                    return_value=MagicMock(
                        best_config={"param": "fallback"},
                        best_metrics={"accuracy": 0.8},
                        trials=[MagicMock() for _ in range(15)],
                    )
                )

                mock_get_optimizer.side_effect = [
                    Exception("Bayesian not available"),
                    mock_optimizer,
                ]
                mock_evaluator_class.return_value = MagicMock()

                result = await cloud_service._run_enhanced_optimization(
                    dataset=sample_dataset,
                    configuration_space={"param": [1, 2, 3]},
                    objectives=["accuracy"],
                    max_trials=10,
                    billing_tier="professional",  # Should try Bayesian first
                )

                # Verify both optimizer calls
                assert mock_get_optimizer.call_count == 2
                mock_get_optimizer.assert_has_calls(
                    [
                        call(
                            "bayesian",
                            {"param": [1, 2, 3]},
                            ["accuracy"],
                            max_trials=15,
                        ),
                        call(
                            "random",
                            {"param": [1, 2, 3]},
                            ["accuracy"],
                            max_trials=15,
                        ),
                    ]
                )

                # Verify result from fallback optimizer
                assert result["best_config"] == {"param": "fallback"}

        import asyncio

        asyncio.run(run_test())

    def test_get_service_health(self, cloud_service):
        """Test getting service health information."""

        async def run_test():
            # Simulate some service usage
            cloud_service.total_optimizations = 5
            cloud_service.total_cost_savings = 2.5  # 50% average cost reduction

            health = await cloud_service.get_service_health()

            assert health["status"] == "healthy"
            assert health["total_optimizations"] == 5
            assert health["average_cost_reduction"] == 50.0  # 2.5/5 * 100
            assert health["service_version"] == "1.0.0"
            assert "random" in health["available_algorithms"]
            assert "grid" in health["available_algorithms"]
            assert "bayesian" in health["available_algorithms"]
            assert "accuracy" in health["supported_objectives"]
            assert health["max_dataset_size"] == 10000
            assert health["max_trials"] == 1000
            assert "uptime_hours" in health
            assert health["uptime_hours"] >= 0

        import asyncio

        asyncio.run(run_test())

    def test_get_service_health_no_optimizations(self, cloud_service):
        """Test service health with no optimizations."""

        async def run_test():
            health = await cloud_service.get_service_health()

            assert health["total_optimizations"] == 0
            assert health["average_cost_reduction"] == 0.0  # No division by zero

        import asyncio

        asyncio.run(run_test())

    def test_get_optimization_history(self, cloud_service):
        """Test getting optimization history."""

        async def run_test():
            history = await cloud_service.get_optimization_history(
                user_id="test_user", limit=5
            )

            assert len(history) == 5
            for record in history:
                assert "request_id" in record
                assert "function_name" in record
                assert "timestamp" in record
                assert "trials_count" in record
                assert "cost_reduction" in record
                assert "status" in record
                assert record["status"] == "completed"

        import asyncio

        asyncio.run(run_test())

    def test_get_optimization_history_large_limit(self, cloud_service):
        """Test optimization history with large limit."""

        async def run_test():
            history = await cloud_service.get_optimization_history(limit=100)

            # Should return max 10 records (mock data limit)
            assert len(history) == 10

        import asyncio

        asyncio.run(run_test())

    def test_get_optimization_history_no_user_id(self, cloud_service):
        """Test optimization history without user ID."""

        async def run_test():
            history = await cloud_service.get_optimization_history(limit=3)

            assert len(history) == 3
            # Should work without user_id (demo mode)

        import asyncio

        asyncio.run(run_test())

    def test_create_optimization_request_from_dict(self, cloud_service):
        """Test creating optimization request from dictionary data."""
        dataset_data = {
            "name": "test_dataset",
            "examples": [
                {
                    "input_data": {"query": "What is AI?"},
                    "expected_output": "Artificial Intelligence",
                    "metadata": {"difficulty": "easy"},
                },
                {
                    "input_data": {"query": "What is ML?"},
                    "expected_output": "Machine Learning",
                    "metadata": {"difficulty": "medium"},
                },
            ],
        }

        request = cloud_service.create_optimization_request(
            function_name="llm_function",
            dataset_data=dataset_data,
            configuration_space={"temperature": [0.5, 0.7, 1.0]},
            objectives=["accuracy", "cost"],
            max_trials=30,
            target_cost_reduction=0.7,
            user_id="test_user",
            billing_tier="professional",
        )

        assert isinstance(request, OptimizationRequest)
        assert request.function_name == "llm_function"
        assert len(request.dataset.examples) == 2
        assert request.dataset.name == "test_dataset"
        assert request.configuration_space == {"temperature": [0.5, 0.7, 1.0]}
        assert request.objectives == ["accuracy", "cost"]
        assert request.max_trials == 30
        assert request.target_cost_reduction == 0.7
        assert request.user_id == "test_user"
        assert request.billing_tier == "professional"

        # Verify example conversion
        first_example = request.dataset.examples[0]
        assert first_example.input_data == {"query": "What is AI?"}
        assert first_example.expected_output == "Artificial Intelligence"
        assert first_example.metadata == {"difficulty": "easy"}

    def test_create_optimization_request_minimal_dataset(self, cloud_service):
        """Test creating optimization request with minimal dataset data."""
        dataset_data = {
            "examples": [
                {"input_data": "Simple input", "expected_output": "Simple output"}
            ]
        }

        request = cloud_service.create_optimization_request(
            function_name="simple_function",
            dataset_data=dataset_data,
            configuration_space={"param": [1, 2, 3]},
            objectives=["accuracy"],
        )

        assert request.dataset.name == "cloud_dataset"  # Default name
        assert len(request.dataset.examples) == 1
        assert request.dataset.examples[0].metadata == {}  # Default metadata
        assert request.max_trials == 50  # Default
        assert request.billing_tier == "standard"  # Default

    def test_create_optimization_request_empty_dataset(self, cloud_service):
        """Test creating optimization request with empty dataset."""
        dataset_data = {"examples": []}

        request = cloud_service.create_optimization_request(
            function_name="empty_function",
            dataset_data=dataset_data,
            configuration_space={"param": [1, 2, 3]},
            objectives=["accuracy"],
        )

        assert len(request.dataset.examples) == 0
        assert request.dataset.name == "cloud_dataset"


class TestServiceStatistics:
    """Test service statistics tracking."""

    def test_statistics_initialization(self, cloud_service):
        """Test statistics are properly initialized."""
        assert cloud_service.total_optimizations == 0
        assert cloud_service.total_cost_savings == 0.0
        assert isinstance(cloud_service.uptime_start, float)
        assert cloud_service.uptime_start <= time.time()

    def test_statistics_updates_during_optimization(
        self, cloud_service, optimization_request
    ):
        """Test statistics are updated during optimization."""

        async def run_test():
            with (
                patch.object(
                    cloud_service.billing_manager, "check_usage_limits"
                ) as mock_limits,
                patch.object(
                    cloud_service.subset_selector, "select_optimal_subset"
                ) as mock_subset,
                patch.object(
                    cloud_service, "_run_enhanced_optimization"
                ) as mock_optimize,
                patch.object(cloud_service.usage_tracker, "record_optimization"),
            ):

                # Setup mocks
                mock_limits.return_value = {
                    "allowed": True,
                    "estimated_cost": 5.0,
                    "remaining_credits": 95.0,
                }
                subset_examples = optimization_request.dataset.examples[
                    :2
                ]  # 2 out of 5 = 60% reduction
                mock_subset.return_value = Dataset(
                    examples=subset_examples, name="subset"
                )
                mock_optimize.return_value = {
                    "best_config": {},
                    "best_metrics": {},
                    "trials_count": 25,
                }

                # Initial state
                initial_optimizations = cloud_service.total_optimizations
                initial_savings = cloud_service.total_cost_savings

                # Process request
                await cloud_service.process_optimization_request(optimization_request)

                # Verify statistics updated
                assert cloud_service.total_optimizations == initial_optimizations + 1
                assert (
                    cloud_service.total_cost_savings == initial_savings + 0.6
                )  # 60% reduction

        import asyncio

        asyncio.run(run_test())

    def test_multiple_optimizations_statistics(
        self, cloud_service, optimization_request
    ):
        """Test statistics accumulation over multiple optimizations."""

        async def run_test():
            with (
                patch.object(
                    cloud_service.billing_manager, "check_usage_limits"
                ) as mock_limits,
                patch.object(
                    cloud_service.subset_selector, "select_optimal_subset"
                ) as mock_subset,
                patch.object(
                    cloud_service, "_run_enhanced_optimization"
                ) as mock_optimize,
                patch.object(cloud_service.usage_tracker, "record_optimization"),
            ):

                # Setup mocks
                mock_limits.return_value = {
                    "allowed": True,
                    "estimated_cost": 5.0,
                    "remaining_credits": 95.0,
                }
                mock_optimize.return_value = {
                    "best_config": {},
                    "best_metrics": {},
                    "trials_count": 25,
                }

                # First optimization: 50% reduction
                mock_subset.return_value = Dataset(
                    examples=optimization_request.dataset.examples[:3], name="subset1"
                )  # 3 out of 5 = 40% reduction
                await cloud_service.process_optimization_request(optimization_request)

                # Second optimization: 80% reduction
                mock_subset.return_value = Dataset(
                    examples=optimization_request.dataset.examples[:1], name="subset2"
                )  # 1 out of 5 = 80% reduction
                await cloud_service.process_optimization_request(optimization_request)

                # Verify accumulated statistics
                assert cloud_service.total_optimizations == 2
                assert (
                    abs(cloud_service.total_cost_savings - 1.2) < 0.001
                )  # 0.4 + 0.8, with floating point tolerance

        import asyncio

        asyncio.run(run_test())


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_optimization_request_empty_dataset(self, cloud_service):
        """Test optimization with empty dataset."""

        async def run_test():
            empty_dataset = Dataset(examples=[], name="empty")
            request = OptimizationRequest(
                function_name="empty_function",
                dataset=empty_dataset,
                configuration_space={"param": [1, 2, 3]},
                objectives=["accuracy"],
            )

            with patch.object(
                cloud_service.billing_manager, "check_usage_limits"
            ) as mock_limits:
                mock_limits.return_value = {"allowed": True}

                # Should handle empty dataset gracefully
                await cloud_service.process_optimization_request(request)

                # Billing check should be called with 0 dataset size
                mock_limits.assert_called_once_with(50, 0)

        import asyncio

        asyncio.run(run_test())

    def test_optimization_unknown_billing_tier(self, cloud_service, sample_dataset):
        """Test optimization with unknown billing tier."""

        async def run_test():
            with (
                patch("traigent.cloud.service.get_optimizer") as mock_get_optimizer,
                patch("traigent.cloud.service.LocalEvaluator") as mock_evaluator_class,
            ):

                mock_optimizer = MagicMock()
                mock_optimizer.optimize = AsyncMock(
                    return_value=MagicMock(best_config={}, best_metrics={}, trials=[])
                )
                mock_get_optimizer.return_value = mock_optimizer
                mock_evaluator_class.return_value = MagicMock()

                await cloud_service._run_enhanced_optimization(
                    dataset=sample_dataset,
                    configuration_space={"param": [1, 2, 3]},
                    objectives=["accuracy"],
                    max_trials=20,
                    billing_tier="unknown_tier",
                )

                # Should use standard tier multiplier (1.0) for unknown tier
                optimize_call = mock_optimizer.optimize.call_args
                assert optimize_call[1]["max_trials"] == 20  # 20 * 1.0

        import asyncio

        asyncio.run(run_test())

    def test_create_request_missing_dataset_fields(self, cloud_service):
        """Test creating request with missing dataset fields."""
        dataset_data = {
            "examples": [
                {
                    "input_data": "test input"
                    # Missing expected_output and metadata
                }
            ]
        }

        # Should handle missing fields gracefully - this will raise KeyError
        with pytest.raises(KeyError):
            cloud_service.create_optimization_request(
                function_name="test_function",
                dataset_data=dataset_data,
                configuration_space={"param": [1, 2, 3]},
                objectives=["accuracy"],
            )

    def test_dataset_conversion_edge_cases(self, cloud_service):
        """Test dataset conversion with various edge cases."""
        # Test with no examples key
        dataset_data_no_examples = {}
        request = cloud_service.create_optimization_request(
            function_name="test_function",
            dataset_data=dataset_data_no_examples,
            configuration_space={"param": [1, 2, 3]},
            objectives=["accuracy"],
        )
        assert len(request.dataset.examples) == 0

        # Test with no name key
        dataset_data_no_name = {
            "examples": [{"input_data": "test", "expected_output": "output"}]
        }
        request = cloud_service.create_optimization_request(
            function_name="test_function",
            dataset_data=dataset_data_no_name,
            configuration_space={"param": [1, 2, 3]},
            objectives=["accuracy"],
        )
        assert request.dataset.name == "cloud_dataset"
