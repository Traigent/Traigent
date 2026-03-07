"""Tests for Traigent Cloud Client."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from traigent.cloud.client import (
    CloudOptimizationResult,
    CloudServiceError,
    TraigentCloudClient,
)
from traigent.config.backend_config import BackendConfig
from traigent.evaluators.base import Dataset, EvaluationExample


@pytest.fixture
def sample_dataset():
    """Create sample dataset for testing."""
    examples = [
        EvaluationExample(
            input_data={"text": "Test input 1"}, expected_output="output1"
        ),
        EvaluationExample(
            input_data={"text": "Test input 2"}, expected_output="output2"
        ),
        EvaluationExample(
            input_data={"text": "Test input 3"}, expected_output="output3"
        ),
    ]
    return Dataset(examples=examples, name="test_dataset")


@pytest.fixture
def mock_cloud_client():
    """Create mock cloud client for testing."""
    return TraigentCloudClient(
        api_key="tg_test_" + "x" * 56,  # pragma: allowlist secret
        base_url="http://localhost:8000",
        enable_fallback=True,
    )


class TestTraigentCloudClient:
    """Test cases for Traigent Cloud Client."""

    def test_client_initialization(self):
        """Test client initialization with different parameters."""
        client = TraigentCloudClient(
            api_key="tg_test_key",  # pragma: allowlist secret
            base_url="http://localhost:5000",
            enable_fallback=False,
            max_retries=5,
            timeout=60.0,
        )

        assert client.base_url == "http://localhost:5000"
        assert client.api_base_url == "http://localhost:5000/api/v1"
        assert client.enable_fallback is False
        assert client.max_retries == 5
        assert client.timeout == 60.0

    def test_client_default_initialization(self, monkeypatch):
        """Test client initialization with defaults."""
        # Ensure backend resolution does not depend on external environment configuration
        for var in [
            "TRAIGENT_BACKEND_URL",
            "TRAIGENT_API_URL",
            "TRAIGENT_DEFAULT_LOCAL_URL",
        ]:
            monkeypatch.delenv(var, raising=False)

        monkeypatch.setenv("TRAIGENT_ENV", "production")

        client = TraigentCloudClient()

        assert client.base_url == BackendConfig.DEFAULT_PROD_URL
        assert client.api_base_url == BackendConfig.get_backend_api_url()
        assert client.enable_fallback is True
        assert client.max_retries == 3
        assert client.timeout == 30.0

    def test_context_manager(self, mock_cloud_client):
        """Test async context manager functionality."""

        async def run_test():
            async with mock_cloud_client as client:
                assert client._session is not None

            # Session should be closed after exit
            assert mock_cloud_client._session is None

        asyncio.run(run_test())

    def test_close_clears_shared_session(self, mock_cloud_client):
        """Public close() should release the shared HTTP session."""

        async def run_test():
            session = MagicMock()
            session.close = AsyncMock()
            mock_cloud_client._session = session

            await mock_cloud_client.close()

            session.close.assert_awaited_once()
            assert mock_cloud_client._session is None

        asyncio.run(run_test())

    def test_optimize_function_success(self, mock_cloud_client, sample_dataset):
        """Test successful optimization with cloud service."""

        async def run_test():
            # Mock the auth and submission
            with (
                patch.object(
                    mock_cloud_client.auth, "is_authenticated", return_value=True
                ),
                patch.object(
                    mock_cloud_client.auth,
                    "get_headers",
                    return_value={
                        "Authorization": "Bearer test_token",
                        "Content-Type": "application/json",
                    },
                ),
                patch.object(mock_cloud_client, "_submit_optimization") as mock_submit,
            ):

                mock_submit.return_value = {
                    "best_config": {"param1": "value1", "param2": 0.5},
                    "best_metrics": {"accuracy": 0.85, "speed": 0.9},
                    "trials_count": 25,
                }

                async with mock_cloud_client as client:
                    result = await client.optimize_function(
                        function_name="test_function",
                        dataset=sample_dataset,
                        configuration_space={
                            "param1": ["a", "b"],
                            "param2": [0.1, 0.5, 0.9],
                        },
                        objectives=["accuracy", "speed"],
                        max_trials=50,
                    )

                assert isinstance(result, CloudOptimizationResult)
                assert result.best_config == {"param1": "value1", "param2": 0.5}
                assert result.best_metrics == {"accuracy": 0.85, "speed": 0.9}
                assert result.trials_count == 25
                assert result.subset_used is True
                assert result.cost_reduction > 0

        asyncio.run(run_test())

    def test_optimize_function_auth_failure(self, mock_cloud_client, sample_dataset):
        """Test optimization failure due to authentication."""

        async def run_test():
            from traigent.cloud.auth import AuthenticationError

            with (
                patch.object(
                    mock_cloud_client.auth, "is_authenticated", return_value=False
                ),
                patch.object(
                    mock_cloud_client.auth,
                    "get_headers",
                    side_effect=AuthenticationError("Not authenticated"),
                ),
            ):

                with pytest.raises(AuthenticationError, match="Not authenticated"):
                    async with mock_cloud_client:
                        pass  # The exception should be raised in __aenter__

        asyncio.run(run_test())

    def test_optimize_function_with_fallback(self, mock_cloud_client, sample_dataset):
        """Test fallback to local optimization when cloud fails."""

        async def local_function(text: str, param: int = 1) -> str:
            return f"{text}:{param}"

        async def run_test():
            with (
                patch.object(
                    mock_cloud_client.auth, "is_authenticated", return_value=True
                ),
                patch.object(
                    mock_cloud_client.auth,
                    "get_headers",
                    return_value={
                        "Authorization": "Bearer test_token",
                        "Content-Type": "application/json",
                    },
                ),
                patch.object(
                    mock_cloud_client,
                    "_submit_optimization",
                    side_effect=Exception("Network error"),
                ),
                patch(
                    "traigent.optimizers.registry.get_optimizer"
                ) as mock_get_optimizer,
                patch(
                    "traigent.core.orchestrator.OptimizationOrchestrator"
                ) as mock_orchestrator_class,
            ):

                mock_optimizer = MagicMock()
                mock_get_optimizer.return_value = mock_optimizer

                mock_orchestrator = mock_orchestrator_class.return_value
                mock_orchestrator.optimize = AsyncMock(
                    return_value=MagicMock(
                        best_config={"param": 1},
                        best_metrics={"accuracy": 0.8},
                        trials=[object(), object()],
                    )
                )

                async with mock_cloud_client as client:
                    result = await client.optimize_function(
                        function_name="test_function",
                        dataset=sample_dataset,
                        configuration_space={"param": [1, 2, 3]},
                        objectives=["accuracy"],
                        local_function=local_function,
                    )

                # Should get fallback result
                assert isinstance(result, CloudOptimizationResult)
                assert result.best_config == {"param": 1}
                assert result.best_metrics == {"accuracy": 0.8}
                assert result.trials_count == 2
                assert result.subset_used is False
                assert result.cost_reduction == 0.0

        asyncio.run(run_test())

    def test_optimize_function_no_fallback(self, sample_dataset):
        """Test optimization failure without fallback."""

        async def run_test():
            client = TraigentCloudClient(
                enable_fallback=False, api_key="tg_test_" + "x" * 56  # pragma: allowlist secret
            )

            with (
                patch.object(client.auth, "is_authenticated", return_value=True),
                patch.object(
                    client.auth,
                    "get_headers",
                    return_value={
                        "Authorization": "Bearer test_token",
                        "Content-Type": "application/json",
                    },
                ),
                patch.object(
                    client,
                    "_submit_optimization",
                    side_effect=Exception("Network error"),
                ),
            ):

                async with client as c:
                    with pytest.raises(
                        CloudServiceError, match="Cloud optimization failed"
                    ):
                        await c.optimize_function(
                            function_name="test_function",
                            dataset=sample_dataset,
                            configuration_space={},
                            objectives=["accuracy"],
                        )

        asyncio.run(run_test())

    def test_submit_optimization_success(self, mock_cloud_client):
        """Test successful optimization submission."""

        async def run_test():
            mock_response = MagicMock()
            mock_response.status = 200
            # The response.json() returns the actual dict that retry_http_request will return
            mock_response.json = AsyncMock(
                return_value={
                    "best_config": {"param": "value"},
                    "best_metrics": {"accuracy": 0.9},
                    "trials_count": 10,
                }
            )

            mock_session = MagicMock()
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_cloud_client._session = mock_session

            result = await mock_cloud_client._submit_optimization(
                {
                    "function_name": "test",
                    "dataset": {},
                    "configuration_space": {},
                    "objectives": ["accuracy"],
                }
            )

            assert result["best_config"] == {"param": "value"}
            assert result["best_metrics"] == {"accuracy": 0.9}
            assert result["trials_count"] == 10

        asyncio.run(run_test())

    def test_submit_optimization_rate_limited(self, mock_cloud_client):
        """Test optimization submission with rate limiting."""

        async def run_test():
            # First request: rate limited (429)
            mock_response_429 = MagicMock()
            mock_response_429.status = 429

            # Second request: success
            mock_response_200 = MagicMock()
            mock_response_200.status = 200
            mock_response_200.json = AsyncMock(
                return_value={
                    "best_config": {"param": "value"},
                    "best_metrics": {"accuracy": 0.9},
                    "trials_count": 10,
                }
            )

            mock_session = MagicMock()
            mock_session.post.return_value.__aenter__.side_effect = [
                mock_response_429,
                mock_response_200,
            ]
            mock_cloud_client._session = mock_session

            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await mock_cloud_client._submit_optimization({})

            assert result["trials_count"] == 10

        asyncio.run(run_test())

    def test_submit_optimization_max_retries(self, mock_cloud_client):
        """Test optimization submission exceeding max retries."""

        async def run_test():
            mock_response = MagicMock()
            mock_response.status = 500
            mock_response.text = AsyncMock(return_value="Internal Server Error")

            mock_session = MagicMock()
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_cloud_client._session = mock_session

            with pytest.raises(CloudServiceError, match="HTTP 500"):
                await mock_cloud_client._submit_optimization({})

        asyncio.run(run_test())

    def test_serialize_dataset(self, mock_cloud_client, sample_dataset):
        """Test dataset serialization for cloud transmission."""
        serialized = mock_cloud_client._serialize_dataset(sample_dataset)

        assert serialized["name"] == "test_dataset"
        assert len(serialized["examples"]) == 3

        first_example = serialized["examples"][0]
        assert first_example["input_data"] == {"text": "Test input 1"}
        assert first_example["expected_output"] == "output1"
        assert "metadata" in first_example

    def test_get_usage_stats(self, mock_cloud_client):
        """Test getting usage statistics."""

        async def run_test():
            with patch.object(
                mock_cloud_client.usage_tracker, "get_usage_stats"
            ) as mock_stats:
                mock_stats.return_value = {
                    "total_optimizations": 5,
                    "total_credits": 12.5,
                    "total_time": 120.0,
                }

                stats = await mock_cloud_client.get_usage_stats()
                assert stats["total_optimizations"] == 5
                assert stats["total_credits"] == 12.5
                assert stats["total_time"] == 120.0

        asyncio.run(run_test())

    def test_check_service_status_healthy(self, mock_cloud_client):
        """Test service status check when healthy."""

        async def run_test():
            mock_response = MagicMock()
            mock_response.json = AsyncMock(
                return_value={"status": "healthy", "uptime": "24h", "version": "1.0.0"}
            )

            mock_session = MagicMock()
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_cloud_client._session = mock_session

            status = await mock_cloud_client.check_service_status()
            assert status["status"] == "healthy"
            assert "uptime" in status

        asyncio.run(run_test())

    def test_check_service_status_unavailable(self, mock_cloud_client):
        """Test service status check when unavailable."""

        async def run_test():
            mock_session = MagicMock()
            mock_session.get.side_effect = Exception("Connection failed")
            mock_cloud_client._session = mock_session

            status = await mock_cloud_client.check_service_status()
            assert status["status"] == "unavailable"
            assert "error" in status

        asyncio.run(run_test())


class TestCloudOptimizationResult:
    """Test cases for CloudOptimizationResult dataclass."""

    def test_cloud_optimization_result_creation(self):
        """Test creation of CloudOptimizationResult."""
        result = CloudOptimizationResult(
            best_config={"param": "value"},
            best_metrics={"accuracy": 0.9},
            trials_count=25,
            cost_reduction=0.65,
            optimization_time=120.5,
            subset_used=True,
            subset_size=150,
        )

        assert result.best_config == {"param": "value"}
        assert result.best_metrics == {"accuracy": 0.9}
        assert result.trials_count == 25
        assert result.cost_reduction == 0.65
        assert result.optimization_time == 120.5
        assert result.subset_used is True
        assert result.subset_size == 150

    def test_cloud_optimization_result_defaults(self):
        """Test CloudOptimizationResult with default values."""
        result = CloudOptimizationResult(
            best_config={},
            best_metrics={},
            trials_count=0,
            cost_reduction=0.0,
            optimization_time=0.0,
            subset_used=False,
        )

        assert result.subset_size is None


class TestCloudServiceError:
    """Test cases for CloudServiceError exception."""

    def test_cloud_service_error_creation(self):
        """Test CloudServiceError creation."""
        error = CloudServiceError("Test error message")
        assert str(error) == "Test error message"

    def test_cloud_service_error_inheritance(self):
        """Test CloudServiceError inheritance from Exception."""
        error = CloudServiceError("Test error")
        assert isinstance(error, Exception)
