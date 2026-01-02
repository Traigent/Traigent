"""Unit tests for session finalization functionality."""

from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest

from traigent.cloud.backend_client import BackendIntegratedClient


class TestSessionFinalization:
    """Test session finalization API calls and logic."""

    @pytest.fixture
    def client(self):
        """Create a mock BackendIntegratedClient."""
        with patch("traigent.cloud.backend_client.AIOHTTP_AVAILABLE", True):
            client = BackendIntegratedClient(
                api_key="test-api-key", base_url="http://localhost:5000"
            )
            # Mock the auth manager
            client.auth_manager = Mock()
            client.auth_manager.augment_headers = AsyncMock(
                return_value={"Authorization": "Bearer test-key"}
            )
            return client

    @pytest.mark.asyncio
    async def test_finalize_session_via_api_success(self, client):
        """Test successful session finalization via backend API."""
        session_id = "test-session-123"
        experiment_run_id = "run-456"

        # Add session mapping
        client.session_bridge.create_session_mapping(
            session_id=session_id,
            experiment_id="exp-789",
            experiment_run_id=experiment_run_id,
            function_name="test_func",
            configuration_space={},
            objectives=["maximize"],
        )

        # Mock the HTTP response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.post = Mock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            # Call finalize
            response = await client._session_ops.finalize_session(session_id)

            # Verify API was called
            mock_session.post.assert_called_once()
            call_args = mock_session.post.call_args
            assert f"/sessions/{session_id}/finalize" in call_args[0][0]

            # Verify response metadata
            assert response.session_id == session_id
            assert response.metadata["finalized_via_api"] is True

    @pytest.mark.asyncio
    async def test_finalize_session_via_api_not_available(self, client):
        """Test finalization when backend API endpoint is not available."""
        session_id = "test-session-123"
        experiment_run_id = "run-456"

        # Add session mapping
        client.session_bridge.create_session_mapping(
            session_id=session_id,
            experiment_id="exp-789",
            experiment_run_id=experiment_run_id,
            function_name="test_func",
            configuration_space={},
            objectives=["maximize"],
        )

        # Mock the HTTP response with 404
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.text = AsyncMock(return_value="Not found")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.post = Mock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            # Call finalize - should not raise error
            response = await client._session_ops.finalize_session(session_id)

            # Verify it still returns a response (relying on auto-finalization)
            assert response.session_id == session_id
            assert response.metadata["finalized_via_api"] is False

    @pytest.mark.asyncio
    async def test_finalize_session_idempotent(self, client):
        """Test that calling finalize multiple times is safe."""
        session_id = "test-session-123"
        experiment_run_id = "run-456"

        # Add session mapping
        client.session_bridge.create_session_mapping(
            session_id=session_id,
            experiment_id="exp-789",
            experiment_run_id=experiment_run_id,
            function_name="test_func",
            configuration_space={},
            objectives=["maximize"],
        )

        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.post = Mock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            # Call finalize first time
            response1 = await client._session_ops.finalize_session(session_id)
            assert response1.session_id == session_id

            # Call finalize second time - should not error
            response2 = await client._session_ops.finalize_session(session_id)
            assert response2.session_id == session_id

            # Both should succeed
            assert mock_session.post.call_count >= 1

    @pytest.mark.asyncio
    async def test_finalize_session_without_mapping(self, client):
        """Test finalization when no session mapping exists."""
        session_id = "test-session-no-mapping"

        # Don't add session mapping

        # Call finalize - should not crash
        response = await client._session_ops.finalize_session(session_id)

        # Verify it returns a response
        assert response.session_id == session_id
        assert response.metadata["experiment_run_id"] is None

    @pytest.mark.asyncio
    async def test_finalize_session_network_error(self, client):
        """Test finalization handles network errors gracefully."""
        session_id = "test-session-123"
        experiment_run_id = "run-456"

        # Add session mapping
        client.session_bridge.create_session_mapping(
            session_id=session_id,
            experiment_id="exp-789",
            experiment_run_id=experiment_run_id,
            function_name="test_func",
            configuration_space={},
            objectives=["maximize"],
        )

        # Mock network error
        mock_session = AsyncMock()
        mock_session.post = Mock(side_effect=aiohttp.ClientError("Network error"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            # Call finalize - should not raise error
            response = await client._session_ops.finalize_session(session_id)

            # Verify it still returns a response
            assert response.session_id == session_id
            # finalized_via_api should be False due to error
            assert response.metadata["finalized_via_api"] is False


class TestAutoFinalizationDetection:
    """Test detection of backend auto-finalization."""

    @pytest.fixture
    def client(self):
        """Create a mock BackendIntegratedClient."""
        with patch("traigent.cloud.backend_client.AIOHTTP_AVAILABLE", True):
            client = BackendIntegratedClient(
                api_key="test-api-key", base_url="http://localhost:5000"
            )
            client.auth_manager = Mock()
            client.auth_manager.augment_headers = AsyncMock(
                return_value={"Authorization": "Bearer test-key"}
            )
            # Mock the _update_config_run_status method
            client._update_config_run_status = AsyncMock(return_value=True)
            client._update_config_run_measures = AsyncMock(return_value=True)
            return client

    @pytest.mark.asyncio
    async def test_submit_results_detects_auto_finalization(self, client):
        """Test that SDK detects auto-finalization from backend response."""
        session_id = "test-session-123"
        trial_id = "trial-456"

        # Mock response with continue_optimization=false
        mock_response = AsyncMock()
        mock_response.status = 201
        mock_response.json = AsyncMock(
            return_value={
                "success": True,
                "continue_optimization": False,  # Backend auto-finalized
                "message": "Session finalized",
            }
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.post = Mock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            # Submit trial results
            result = await client._trial_ops.submit_trial_result_via_session(
                session_id=session_id,
                trial_id=trial_id,
                config={"temp": 0.5},
                metrics={"accuracy": 0.9},
                status="COMPLETED",
            )

            # Verify success
            assert result is True

            # Verify API was called
            mock_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_submit_results_continue_optimization_true(self, client):
        """Test that SDK handles continue_optimization=true correctly."""
        session_id = "test-session-123"
        trial_id = "trial-456"

        # Mock response with continue_optimization=true
        mock_response = AsyncMock()
        mock_response.status = 201
        mock_response.json = AsyncMock(
            return_value={
                "success": True,
                "continue_optimization": True,  # More trials available
                "message": "Results submitted",
            }
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.post = Mock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            # Submit trial results
            result = await client._trial_ops.submit_trial_result_via_session(
                session_id=session_id,
                trial_id=trial_id,
                config={"temp": 0.5},
                metrics={"accuracy": 0.9},
                status="COMPLETED",
            )

            # Verify success
            assert result is True

    @pytest.mark.asyncio
    async def test_submit_results_no_continue_flag_defaults_true(self, client):
        """Test that missing continue_optimization flag defaults to True."""
        session_id = "test-session-123"
        trial_id = "trial-456"

        # Mock response without continue_optimization field
        mock_response = AsyncMock()
        mock_response.status = 201
        mock_response.json = AsyncMock(
            return_value={
                "success": True,
                # No continue_optimization field
                "message": "Results submitted",
            }
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.post = Mock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            # Submit trial results
            result = await client._trial_ops.submit_trial_result_via_session(
                session_id=session_id,
                trial_id=trial_id,
                config={"temp": 0.5},
                metrics={"accuracy": 0.9},
                status="COMPLETED",
            )

            # Verify success (defaults to continue=true)
            assert result is True
