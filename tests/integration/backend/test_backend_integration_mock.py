"""Backend integration tests with mocked HTTP calls.

Tests backend client integration using mocked HTTP responses to avoid
requiring a running backend server during test execution.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from traigent.cloud.backend_client import BackendClientConfig, BackendIntegratedClient


@pytest.mark.integration
class TestBackendIntegrationMocked:
    """Integration tests for backend client with mocked HTTP calls."""

    @pytest.fixture
    def backend_config(self):
        """Create backend configuration for tests."""
        return BackendClientConfig(
            backend_base_url="http://localhost:5000", enable_session_sync=True
        )

    @pytest.fixture
    def backend_client(self, backend_config):
        """Create backend client instance."""
        return BackendIntegratedClient(
            api_key="tg_" + "a" * 61,  # Valid 64-char format
            backend_config=backend_config,
            enable_fallback=True,
        )

    @pytest.fixture
    def mock_aiohttp_session(self):
        """Create mocked aiohttp session with proper async context managers."""
        # Create mock response that works as an async context manager
        mock_response = AsyncMock()
        mock_response.status = 201
        mock_response.json = AsyncMock()
        mock_response.text = AsyncMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        # Create mock for post that returns an async context manager
        mock_post_context = AsyncMock()
        mock_post_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post_context.__aexit__ = AsyncMock(return_value=None)

        # Create mock session
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_post_context)
        mock_session.put = MagicMock(return_value=mock_post_context)
        mock_session.get = MagicMock(return_value=mock_post_context)
        mock_session.delete = MagicMock(return_value=mock_post_context)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        # Create session class mock
        mock_session_class = MagicMock(return_value=mock_session)

        return mock_session_class, mock_response

    @pytest.mark.asyncio
    async def test_create_hybrid_session(self, backend_client, mock_aiohttp_session):
        """Test creating a hybrid session with mocked responses."""
        mock_session_class, mock_response = mock_aiohttp_session

        with (
            patch(
                "traigent.cloud.backend_client.aiohttp.ClientSession",
                mock_session_class,
            ),
            patch("traigent.cloud.backend_client.aiohttp.ClientTimeout"),
            patch("traigent.cloud.backend_client.AIOHTTP_AVAILABLE", True),
        ):
            # Set up mock response for session creation endpoint
            mock_response.json.return_value = {
                "session_id": "test_session_001",
                "status": "CREATED",
                "token": "test_token_001",
                "optimizer_endpoint": "http://localhost:5000/optimizer",
                "metadata": {
                    "experiment_id": "test_exp_001",
                    "experiment_run_id": "test_run_001",
                    "total_configurations": 9,
                },
            }

            # Use backend client as async context manager
            async with backend_client:
                session_id, token, optimizer_endpoint = (
                    await backend_client.create_hybrid_session(
                        problem_statement="test_function",
                        search_space={
                            "temperature": [0.1, 0.5, 0.9],
                            "max_tokens": [100, 150, 200],
                        },
                        optimization_config={
                            "objectives": ["maximize"],
                            "max_trials": 10,
                        },
                        metadata={"dataset_metadata": {"size": 100}},
                    )
                )

            # Should return valid session details
            assert session_id is not None
            assert isinstance(session_id, str)
            assert token is not None
            assert isinstance(token, str)
            assert optimizer_endpoint is not None
            assert isinstance(optimizer_endpoint, str)

            # Should match our mock values
            assert session_id == "test_session_001"
            assert token == "test_token_001"
            assert optimizer_endpoint == "http://localhost:5000/optimizer"

    @pytest.mark.asyncio
    async def test_submit_trial_result_with_mock(
        self, backend_client, mock_aiohttp_session
    ):
        """Test submitting trial results with mocked responses."""
        mock_session_class, mock_response = mock_aiohttp_session

        with (
            patch(
                "traigent.cloud.backend_client.aiohttp.ClientSession",
                mock_session_class,
            ),
            patch("traigent.cloud.backend_client.aiohttp.ClientTimeout"),
            patch("traigent.cloud.backend_client.AIOHTTP_AVAILABLE", True),
        ):
            # Set up mock response for trial submission
            mock_response.status = 200
            mock_response.json.return_value = {"status": "success"}

            # Should not raise exception
            result = await backend_client.submit_privacy_trial_results(
                session_id="test_session_123",
                trial_id="trial_001",
                config={"temperature": 0.7, "max_tokens": 150},
                metrics={"score": 0.85, "cost": 0.002, "latency": 1.23},
                duration=1.5,
            )

            # Should return a boolean (True for success, False for failure)
            assert isinstance(result, bool)
            assert result is True  # Should succeed with our mock

    @pytest.mark.asyncio
    async def test_handle_backend_error(self, backend_client, mock_aiohttp_session):
        """Test error handling with mocked error responses."""
        mock_session_class, mock_response = mock_aiohttp_session

        with (
            patch(
                "traigent.cloud.backend_client.aiohttp.ClientSession",
                mock_session_class,
            ),
            patch("traigent.cloud.backend_client.aiohttp.ClientTimeout"),
            patch("traigent.cloud.backend_client.AIOHTTP_AVAILABLE", True),
        ):
            # Set up mock error response
            mock_response.status = 500
            mock_response.text = AsyncMock(return_value="Internal server error")

            # Without session endpoint succeeding, should raise CloudServiceError
            from traigent.cloud.client import CloudServiceError

            # Use backend client as async context manager
            async with backend_client:
                with pytest.raises(CloudServiceError) as exc_info:
                    await backend_client.create_hybrid_session(
                        problem_statement="test_function",
                        search_space={"temperature": [0.1, 0.5, 0.9]},
                        optimization_config={
                            "objectives": ["maximize"],
                            "max_trials": 10,
                        },
                        metadata={"dataset_metadata": {"size": 100}},
                    )

            # Should have proper error message
            assert "Failed to create hybrid session" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_session_creation_request_dto(
        self, backend_client, mock_aiohttp_session
    ):
        """Test that session creation request DTO is properly formatted."""
        mock_session_class, mock_response = mock_aiohttp_session

        with (
            patch(
                "traigent.cloud.backend_client.aiohttp.ClientSession",
                mock_session_class,
            ),
            patch("traigent.cloud.backend_client.aiohttp.ClientTimeout"),
            patch("traigent.cloud.backend_client.AIOHTTP_AVAILABLE", True),
        ):
            # Set up mock response for session endpoint
            mock_response.json.return_value = {
                "session_id": "test_session_001",
                "status": "CREATED",
                "token": "test_token_001",
                "optimizer_endpoint": "http://localhost:5000/optimizer",
                "metadata": {
                    "experiment_id": "test_exp_001",
                    "experiment_run_id": "test_run_001",
                },
            }

            # Capture the actual request sent
            request_data = None

            def capture_request(*args, **kwargs):
                nonlocal request_data
                request_data = kwargs.get("json", {})
                # Return the mock_post_context, not mock_response directly
                mock_post_context = AsyncMock()
                mock_post_context.__aenter__ = AsyncMock(return_value=mock_response)
                mock_post_context.__aexit__ = AsyncMock(return_value=None)
                return mock_post_context

            # Override post method to capture request
            mock_session = mock_session_class.return_value
            mock_session.post.side_effect = capture_request

            # Use backend client as async context manager
            async with backend_client:
                session_id, token, optimizer_endpoint = (
                    await backend_client.create_hybrid_session(
                        problem_statement="test_function",
                        search_space={
                            "temperature": [0.1, 0.5, 0.9],
                            "max_tokens": [100, 150, 200],
                        },
                        optimization_config={
                            "objectives": ["maximize"],
                            "max_trials": 10,
                        },
                        metadata={"dataset_metadata": {"size": 100}},
                    )
                )

            # Verify request structure for session endpoint
            # Note: optimization_config is used locally for session info but not sent in the request
            assert request_data is not None
            assert "problem_statement" in request_data
            assert request_data["problem_statement"] == "test_function"
            assert "search_space" in request_data
            assert "metadata" in request_data
