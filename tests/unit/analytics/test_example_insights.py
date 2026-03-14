"""Unit tests for ExampleInsightsClient.

Tests for the analytics client that retrieves example-level insights
and dataset quality metrics from the backend.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Observability CONC-Quality-Maintainability

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Check if httpx is available
try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

# Import module to test (handles missing httpx)
from traigent.analytics import example_insights as ei_module


class TestExampleInsightsClientInit:
    """Tests for ExampleInsightsClient initialization."""

    @pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed")
    def test_init_defaults(self) -> None:
        """Test initialization with default values."""
        from traigent.analytics.example_insights import ExampleInsightsClient

        with patch("traigent.utils.env_config.get_api_key", return_value="test_key"):
            client = ExampleInsightsClient()

            assert client.backend_url == "http://localhost:5000"
            assert client.timeout == 30.0
            assert client.api_key == "test_key"
            assert client._client is None

    @pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed")
    def test_init_with_custom_values(self) -> None:
        """Test initialization with custom values."""
        from traigent.analytics.example_insights import ExampleInsightsClient

        client = ExampleInsightsClient(
            backend_url="https://custom.api.com",
            api_key="custom_key",
            timeout=60.0,
        )

        assert client.backend_url == "https://custom.api.com"
        assert client.api_key == "custom_key"
        assert client.timeout == 60.0

    @pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed")
    def test_init_strips_trailing_slash(self) -> None:
        """Test that trailing slash is stripped from URL."""
        from traigent.analytics.example_insights import ExampleInsightsClient

        client = ExampleInsightsClient(
            backend_url="http://localhost:5000/",
            api_key="key",
        )

        assert client.backend_url == "http://localhost:5000"

    @pytest.mark.skipif(HTTPX_AVAILABLE, reason="Test for httpx not available")
    def test_init_raises_without_httpx(self) -> None:
        """Test that ImportError is raised when httpx not available."""
        with pytest.raises(ImportError, match="httpx is required"):
            ei_module.ExampleInsightsClient()


@pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed")
class TestExampleInsightsClientAsyncContextManager:
    """Tests for async context manager functionality."""

    @pytest.mark.asyncio
    async def test_async_context_manager_entry(self) -> None:
        """Test async context manager entry."""
        from traigent.analytics.example_insights import ExampleInsightsClient

        async with ExampleInsightsClient(api_key="test") as client:
            assert client is not None

    @pytest.mark.asyncio
    async def test_async_context_manager_closes_client(self) -> None:
        """Test that async context manager closes client on exit."""
        from traigent.analytics.example_insights import ExampleInsightsClient

        client = ExampleInsightsClient(api_key="test")
        # Manually create a mock client
        mock_http = AsyncMock()
        client._client = mock_http

        await client.close()

        mock_http.aclose.assert_called_once()
        assert client._client is None


@pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed")
class TestExampleInsightsClientGetClient:
    """Tests for _get_client method."""

    @pytest.mark.asyncio
    async def test_get_client_creates_client(self) -> None:
        """Test that _get_client creates a new client."""
        from traigent.analytics.example_insights import ExampleInsightsClient

        client = ExampleInsightsClient(api_key="test_token")

        # Mock httpx.AsyncClient
        with patch("httpx.AsyncClient") as mock_async_client:
            mock_instance = MagicMock()
            mock_async_client.return_value = mock_instance

            result = client._get_client()

            assert result == mock_instance
            mock_async_client.assert_called_once()
            # Check headers include auth
            call_kwargs = mock_async_client.call_args.kwargs
            assert "Authorization" in call_kwargs["headers"]
            assert call_kwargs["headers"]["Authorization"] == "Bearer test_token"

    @pytest.mark.asyncio
    async def test_get_client_reuses_existing(self) -> None:
        """Test that _get_client reuses existing client."""
        from traigent.analytics.example_insights import ExampleInsightsClient

        client = ExampleInsightsClient(api_key="test")

        # Set up existing mock client
        mock_existing = MagicMock()
        client._client = mock_existing

        result = client._get_client()

        assert result == mock_existing

    @pytest.mark.asyncio
    async def test_get_client_without_api_key(self) -> None:
        """Test _get_client without API key."""
        from traigent.analytics.example_insights import ExampleInsightsClient

        client = ExampleInsightsClient(api_key=None)
        client.api_key = None  # Ensure no key

        with patch("httpx.AsyncClient") as mock_async_client:
            mock_instance = MagicMock()
            mock_async_client.return_value = mock_instance

            client._get_client()

            call_kwargs = mock_async_client.call_args.kwargs
            # Authorization header should not be present without key
            assert "Authorization" not in call_kwargs["headers"]


@pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed")
class TestComputeScores:
    """Tests for compute_scores method."""

    @pytest.mark.asyncio
    async def test_compute_scores_success(self) -> None:
        """Test successful compute_scores call."""
        from traigent.analytics.example_insights import ExampleInsightsClient

        client = ExampleInsightsClient(api_key="test")

        # Use MagicMock for response since httpx Response.json() is synchronous
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "accepted",
            "job_id": "job_123",
            "poll_url": "/jobs/job_123",
        }
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        client._client = mock_http

        result = await client.compute_scores(experiment_run_id="run_123")

        assert result["status"] == "accepted"
        assert result["job_id"] == "job_123"
        mock_http.post.assert_called_once_with(
            "/analytics/example-scoring/run_123/compute"
        )


@pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed")
class TestGetExampleScores:
    """Tests for get_example_scores method."""

    @pytest.mark.asyncio
    async def test_get_example_scores_immediate_success(self) -> None:
        """Test get_example_scores when scores are immediately available."""
        from traigent.analytics.example_insights import ExampleInsightsClient

        client = ExampleInsightsClient(api_key="test")

        # Use MagicMock for response since httpx Response.json() is synchronous
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "example_1": {
                "content_uniqueness": 0.8,
                "content_novelty": 0.6,
                "composite_score": 0.7,
            },
            "example_2": {
                "content_uniqueness": 0.9,
                "content_novelty": 0.7,
                "composite_score": 0.8,
            },
        }
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get.return_value = mock_response
        client._client = mock_http

        result = await client.get_example_scores(experiment_run_id="run_123")

        assert "example_1" in result
        assert "example_2" in result
        assert result["example_1"]["content_uniqueness"] == 0.8
        mock_http.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_example_scores_with_filter(self) -> None:
        """Test get_example_scores with example_ids filter."""
        from traigent.analytics.example_insights import ExampleInsightsClient

        client = ExampleInsightsClient(api_key="test")

        # Use MagicMock for response since httpx Response.json() is synchronous
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "example_1": {"composite_score": 0.7},
        }
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get.return_value = mock_response
        client._client = mock_http

        result = await client.get_example_scores(
            experiment_run_id="run_123",
            example_ids=["example_1"],
        )

        assert "example_1" in result
        # Check that params were passed
        call_kwargs = mock_http.get.call_args.kwargs
        assert call_kwargs["params"]["example_ids"] == ["example_1"]

    @pytest.mark.asyncio
    async def test_get_example_scores_polling(self) -> None:
        """Test get_example_scores polls on 404."""
        from traigent.analytics.example_insights import ExampleInsightsClient

        client = ExampleInsightsClient(api_key="test")

        # First call returns 404, second returns success
        mock_error_response = MagicMock()
        mock_error_response.status_code = 404

        # Use MagicMock for response since httpx Response.json() is synchronous
        mock_success_response = MagicMock()
        mock_success_response.json.return_value = {"example_1": {"score": 0.9}}
        mock_success_response.raise_for_status = MagicMock()

        # Use side_effect to simulate 404 then success
        call_count = 0

        async def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                error = httpx.HTTPStatusError(
                    "Not Found", request=MagicMock(), response=mock_error_response
                )
                raise error
            return mock_success_response

        mock_http = AsyncMock()
        mock_http.get.side_effect = mock_get
        client._client = mock_http

        client.timeout = 5.0
        result = await client.get_example_scores(
            experiment_run_id="run_123",
            poll_interval=0.01,  # Fast polling for test
        )

        assert "example_1" in result
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_get_example_scores_timeout(self) -> None:
        """Test get_example_scores raises TimeoutError after timeout."""
        from traigent.analytics.example_insights import ExampleInsightsClient

        client = ExampleInsightsClient(api_key="test")

        mock_error_response = MagicMock()
        mock_error_response.status_code = 404

        async def mock_get(*args, **kwargs):
            error = httpx.HTTPStatusError(
                "Not Found", request=MagicMock(), response=mock_error_response
            )
            raise error

        mock_http = AsyncMock()
        mock_http.get.side_effect = mock_get
        client._client = mock_http

        client.timeout = 0.05  # Very short timeout
        with pytest.raises(TimeoutError, match="Scores not ready"):
            await client.get_example_scores(
                experiment_run_id="run_123",
                poll_interval=0.01,
            )

    @pytest.mark.asyncio
    async def test_get_example_scores_reraises_non_404_error(self) -> None:
        """Test get_example_scores re-raises non-404 HTTP errors."""
        from traigent.analytics.example_insights import ExampleInsightsClient

        client = ExampleInsightsClient(api_key="test")

        mock_error_response = MagicMock()
        mock_error_response.status_code = 500

        async def mock_get(*args, **kwargs):
            error = httpx.HTTPStatusError(
                "Server Error", request=MagicMock(), response=mock_error_response
            )
            raise error

        mock_http = AsyncMock()
        mock_http.get.side_effect = mock_get
        client._client = mock_http

        with pytest.raises(httpx.HTTPStatusError):
            await client.get_example_scores(experiment_run_id="run_123")


@pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed")
class TestGetDatasetQuality:
    """Tests for get_dataset_quality method."""

    @pytest.mark.asyncio
    async def test_get_dataset_quality_success(self) -> None:
        """Test successful get_dataset_quality call."""
        from traigent.analytics.example_insights import ExampleInsightsClient

        client = ExampleInsightsClient(api_key="test")

        # Use MagicMock for response since httpx Response.json() is synchronous
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "dataset_quality": 0.85,
            "coverage_score": 0.9,
            "diversity_score": 0.8,
            "efficiency_score": 0.85,
            "top_informative_ids": ["ex_1", "ex_2"],
            "recommendations": ["Add more diverse examples"],
        }
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get.return_value = mock_response
        client._client = mock_http

        result = await client.get_dataset_quality(experiment_run_id="run_123")

        assert result["dataset_quality"] == 0.85
        assert result["coverage_score"] == 0.9
        mock_http.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_dataset_quality_polling(self) -> None:
        """Test get_dataset_quality polls on 404."""
        from traigent.analytics.example_insights import ExampleInsightsClient

        client = ExampleInsightsClient(api_key="test")

        mock_error_response = MagicMock()
        mock_error_response.status_code = 404

        # Use MagicMock for response since httpx Response.json() is synchronous
        mock_success_response = MagicMock()
        mock_success_response.json.return_value = {"dataset_quality": 0.85}
        mock_success_response.raise_for_status = MagicMock()

        call_count = 0

        async def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                error = httpx.HTTPStatusError(
                    "Not Found", request=MagicMock(), response=mock_error_response
                )
                raise error
            return mock_success_response

        mock_http = AsyncMock()
        mock_http.get.side_effect = mock_get
        client._client = mock_http

        client.timeout = 5.0
        result = await client.get_dataset_quality(
            experiment_run_id="run_123",
            poll_interval=0.01,
        )

        assert result["dataset_quality"] == 0.85
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_get_dataset_quality_timeout(self) -> None:
        """Test get_dataset_quality raises TimeoutError after timeout."""
        from traigent.analytics.example_insights import ExampleInsightsClient

        client = ExampleInsightsClient(api_key="test")

        mock_error_response = MagicMock()
        mock_error_response.status_code = 404

        async def mock_get(*args, **kwargs):
            error = httpx.HTTPStatusError(
                "Not Found", request=MagicMock(), response=mock_error_response
            )
            raise error

        mock_http = AsyncMock()
        mock_http.get.side_effect = mock_get
        client._client = mock_http

        client.timeout = 0.05  # Very short timeout
        with pytest.raises(TimeoutError, match="Quality metrics not ready"):
            await client.get_dataset_quality(
                experiment_run_id="run_123",
                poll_interval=0.01,
            )


@pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed")
class TestGetJobStatus:
    """Tests for get_job_status method."""

    @pytest.mark.asyncio
    async def test_get_job_status_pending(self) -> None:
        """Test get_job_status for pending job."""
        from traigent.analytics.example_insights import ExampleInsightsClient

        client = ExampleInsightsClient(api_key="test")

        # Use MagicMock for response since httpx Response.json() is synchronous
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "pending",
        }
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get.return_value = mock_response
        client._client = mock_http

        result = await client.get_job_status(job_id="job_123")

        assert result["status"] == "pending"
        mock_http.get.assert_called_once_with("/jobs/job_123")

    @pytest.mark.asyncio
    async def test_get_job_status_completed(self) -> None:
        """Test get_job_status for completed job."""
        from traigent.analytics.example_insights import ExampleInsightsClient

        client = ExampleInsightsClient(api_key="test")

        # Use MagicMock for response since httpx Response.json() is synchronous
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "completed",
            "result": {"scores_computed": 100},
        }
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get.return_value = mock_response
        client._client = mock_http

        result = await client.get_job_status(job_id="job_123")

        assert result["status"] == "completed"
        assert result["result"]["scores_computed"] == 100

    @pytest.mark.asyncio
    async def test_get_job_status_failed(self) -> None:
        """Test get_job_status for failed job."""
        from traigent.analytics.example_insights import ExampleInsightsClient

        client = ExampleInsightsClient(api_key="test")

        # Use MagicMock for response since httpx Response.json() is synchronous
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "failed",
            "error": "Insufficient data for scoring",
        }
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get.return_value = mock_response
        client._client = mock_http

        result = await client.get_job_status(job_id="job_123")

        assert result["status"] == "failed"
        assert result["error"] == "Insufficient data for scoring"


class TestExampleInsightsModuleFlags:
    """Tests for module-level flags and availability."""

    def test_httpx_availability_flag(self) -> None:
        """Test HTTPX_AVAILABLE flag matches actual availability."""
        assert ei_module.HTTPX_AVAILABLE == HTTPX_AVAILABLE
