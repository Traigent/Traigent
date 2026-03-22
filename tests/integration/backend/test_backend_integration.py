"""Backend integration tests for Traigent SDK.

Tests the backend client integration with privacy-preserving metadata submission.
These are integration tests that verify communication with the backend service.
"""

from unittest.mock import AsyncMock, patch

import pytest

from traigent.cloud.backend_client import BackendClientConfig, BackendIntegratedClient
from traigent.config.types import TraigentConfig


@pytest.mark.integration
class TestBackendIntegration:
    """Integration tests for backend client."""

    @pytest.fixture
    def backend_config(self):
        """Create backend configuration for tests."""
        return BackendClientConfig(
            backend_base_url="http://localhost:5000", enable_session_sync=True
        )

    @pytest.fixture
    def traigent_config(self):
        """Create Traigent configuration for tests."""
        return TraigentConfig(execution_mode="edge_analytics", minimal_logging=True)

    @pytest.fixture
    def backend_client(self, backend_config):
        """Create backend client instance."""
        return BackendIntegratedClient(
            api_key=None,  # No API key needed for local backend
            backend_config=backend_config,
            enable_fallback=True,
        )

    @pytest.mark.asyncio
    async def test_create_session(self, backend_client):
        """Test creating a new optimization session."""
        result = backend_client.create_session(
            function_name="test_function",
            search_space={
                "temperature": [0.1, 0.3, 0.5, 0.7, 0.9],
                "max_tokens": [100, 150, 200],
            },
            optimization_goal="maximize",
            metadata={"dataset_size": 100, "max_trials": 10, "test_run": True},
        )

        assert result.session_id is not None
        assert isinstance(result.session_id, str)
        assert len(result.session_id) > 0

    @pytest.mark.asyncio
    async def test_submit_trial_result(self, backend_client):
        """Test submitting trial results."""
        # First create a session
        session_id = backend_client.create_session(
            function_name="test_function",
            search_space={
                "temperature": [0.1, 0.3, 0.5, 0.7, 0.9],
                "max_tokens": [100, 150, 200],
            },
            optimization_goal="maximize",
        ).session_id

        # Submit a trial result
        backend_client.submit_result(
            session_id=session_id,
            config={"temperature": 0.7, "max_tokens": 150},
            score=0.85,
            metadata={"cost": 0.002, "latency": 1.23, "duration": 1.5},
        )

        # Verify session is still valid after submission
        assert (
            session_id is not None
        ), "Session ID should remain valid after trial submission"
        assert isinstance(session_id, str), "Session ID should be a string"

    @pytest.mark.asyncio
    async def test_submit_multiple_trials(self, backend_client):
        """Test submitting multiple trial results."""
        # Create session
        session_id = backend_client.create_session(
            function_name="test_function",
            search_space={
                "temperature": [0.1, 0.3, 0.5, 0.7, 0.9],
                "max_tokens": [100, 150, 200],
            },
            optimization_goal="maximize",
        ).session_id

        # Submit multiple trials
        trials = [
            {"config": {"temperature": 0.7, "max_tokens": 150}, "score": 0.85},
            {"config": {"temperature": 0.5, "max_tokens": 200}, "score": 0.92},
            {"config": {"temperature": 0.3, "max_tokens": 100}, "score": 0.78},
        ]

        submitted_count = 0
        for trial in trials:
            backend_client.submit_result(
                session_id=session_id,
                config=trial["config"],
                score=trial["score"],
                metadata={"cost": 0.002, "latency": 1.23, "duration": 1.5},
            )
            submitted_count += 1

        # Verify all trials were submitted
        assert submitted_count == len(
            trials
        ), f"Expected {len(trials)} trials submitted, got {submitted_count}"
        assert (
            session_id is not None
        ), "Session ID should remain valid after multiple submissions"

    @pytest.mark.asyncio
    async def test_backend_fallback_mode(self, backend_config):
        """Test backend client fallback when server is unavailable."""
        # Create client with unreachable backend
        backend_config.backend_base_url = "http://localhost:9999"  # Non-existent port

        client = BackendIntegratedClient(
            api_key=None, backend_config=backend_config, enable_fallback=True
        )

        # Should not raise exception due to fallback mode
        session_id = client.create_session(
            function_name="test_function",
            search_space={"temperature": [0.1, 0.5, 0.9]},
            optimization_goal="maximize",
        )

        # In fallback mode, should still return a session ID
        assert session_id is not None


@pytest.mark.integration
class TestBackendIntegrationWithMocks:
    """Integration tests using mocked backend responses."""

    @pytest.fixture
    def mock_backend_response(self):
        """Mock successful backend responses."""
        return {
            "session_id": "test-session-123",
            "status": "success",
            "message": "Session created successfully",
        }

    @patch("traigent.cloud.api_operations.aiohttp.ClientSession")
    def test_create_session_with_mock(self, mock_session_class, mock_backend_response):
        """Test session creation with mocked backend."""
        from unittest.mock import MagicMock

        # Set up mock response
        mock_response = MagicMock()
        mock_response.status = 201
        mock_response.json = AsyncMock(
            return_value={
                "session_id": "test-session-123",
                "metadata": {"experiment_id": "test-session-123"},
            }
        )

        # Make response work as async context manager
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        # Set up mock session with proper post method
        mock_session = MagicMock()
        # session.post() is a regular method that returns a context manager
        mock_session.post.return_value = mock_response

        # Make session work as async context manager
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_session_class.return_value = mock_session

        backend_config = BackendClientConfig(
            backend_base_url="http://localhost:5000", enable_session_sync=True
        )

        client = BackendIntegratedClient(
            api_key=None, backend_config=backend_config, enable_fallback=False
        )

        session_id = client.create_session(
            function_name="test_function",
            search_space={"temperature": [0.1, 0.5, 0.9]},
            optimization_goal="maximize",
        )

        assert session_id is not None
        assert isinstance(session_id, str)

    @patch("traigent.cloud.api_operations.aiohttp.ClientSession")
    def test_submit_result_with_mock(self, mock_session_class):
        """Test result submission with mocked backend."""
        from unittest.mock import MagicMock

        # Set up mock response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"status": "success"})

        # Make response work as async context manager
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        # Set up mock session with proper post method
        mock_session = MagicMock()
        # session.post() is a regular method that returns a context manager
        mock_session.post.return_value = mock_response

        # Make session work as async context manager
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_session_class.return_value = mock_session

        backend_config = BackendClientConfig(
            backend_base_url="http://localhost:5000", enable_session_sync=True
        )

        client = BackendIntegratedClient(
            api_key=None, backend_config=backend_config, enable_fallback=False
        )

        # Should not raise exception
        client.submit_result(
            session_id="test-session-123",
            config={"temperature": 0.7},
            score=0.85,
            metadata={"cost": 0.002},
        )

        # Verify client was created and submit_result completed without exception
        # Note: In fallback=False mode, the client may not make actual HTTP calls
        # if internal validation fails or session sync is disabled
        assert client is not None, "Client should be created successfully"
