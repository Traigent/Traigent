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

        # Mock the HTTP response — backend returns no JSON body (legacy
        # path). The SDK should still treat finalization as successful but
        # mark summary_available=False because the backend didn't return
        # any summary fields. See #890.
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            side_effect=Exception("no JSON body in legacy response")
        )
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
            # Backend returned no summary → flag must be False so callers
            # don't treat the empty best_config / best_metrics as truth.
            assert response.metadata["summary_available"] is False
            assert response.best_config == {}
            assert response.best_metrics == {}

    @pytest.mark.asyncio
    async def test_finalize_session_preserves_backend_summary(self, client):
        """Regression for #890: when the backend returns a full summary
        payload, the SDK response must preserve best_config, best_metrics,
        total_trials, duration, savings, stop_reason, and convergence_history
        verbatim — and summary_available must be True.
        """
        session_id = "test-session-summary"
        experiment_run_id = "run-summary"

        client.session_bridge.create_session_mapping(
            session_id=session_id,
            experiment_id="exp-summary",
            experiment_run_id=experiment_run_id,
            function_name="summary_func",
            configuration_space={},
            objectives=["accuracy"],
        )

        backend_payload = {
            "best_config": {"model": "gpt-4o", "temperature": 0.2},
            "best_metrics": {"accuracy": 0.91},
            "total_trials": 12,
            "successful_trials": 10,
            "total_duration": 42.5,
            "cost_savings": 1.23,
            "stop_reason": "max_trials_reached",
            "convergence_history": [
                {"trial": 1, "score": 0.7},
                {"trial": 2, "score": 0.81},
            ],
        }

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=backend_payload)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.post = Mock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            response = await client._session_ops.finalize_session(session_id)

        assert response.session_id == session_id
        assert response.metadata["finalized_via_api"] is True
        assert response.metadata["summary_available"] is True

        # Backend payload fields preserved verbatim
        assert response.best_config == {"model": "gpt-4o", "temperature": 0.2}
        assert response.best_metrics == {"accuracy": 0.91}
        assert response.total_trials == 12
        assert response.successful_trials == 10
        assert response.total_duration == 42.5
        assert response.cost_savings == 1.23
        assert response.stop_reason == "max_trials_reached"
        assert response.convergence_history == [
            {"trial": 1, "score": 0.7},
            {"trial": 2, "score": 0.81},
        ]

    @pytest.mark.asyncio
    async def test_finalize_session_legacy_payload_marks_summary_unavailable(
        self, client
    ):
        """Regression for #890: a backend response whose shape doesn't
        include any of the documented summary fields (e.g. a legacy
        {best_trial, all_results} response) MUST NOT set
        summary_available=True. Otherwise callers would treat the SDK's
        fallback empty best_config/best_metrics as authoritative.
        """
        session_id = "test-session-legacy"
        experiment_run_id = "run-legacy"

        client.session_bridge.create_session_mapping(
            session_id=session_id,
            experiment_id="exp-legacy",
            experiment_run_id=experiment_run_id,
            function_name="legacy_func",
            configuration_space={},
            objectives=["accuracy"],
        )

        legacy_payload = {
            "best_trial": {"trial_id": "t1", "metrics": {"accuracy": 0.9}},
            "all_results": [{"trial_id": "t1"}],
            "metadata": {"some": "value"},
        }

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=legacy_payload)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.post = Mock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            response = await client._session_ops.finalize_session(session_id)

        # Finalize call succeeded (200), but the payload didn't include any
        # documented summary fields → summary_available must be False so
        # callers don't read the empty best_config / best_metrics as truth.
        assert response.metadata["finalized_via_api"] is True
        assert response.metadata["summary_available"] is False
        assert response.best_config == {}
        assert response.best_metrics == {}

    @pytest.mark.asyncio
    async def test_finalize_session_endpoint_unavailable_marks_summary_unavailable(
        self, client
    ):
        """Regression for #890: when the backend finalize endpoint is
        unavailable (404 etc.), summary_available must be False so callers
        don't treat empty best_config/best_metrics as backend-authoritative.
        """
        session_id = "test-session-unavail"
        experiment_run_id = "run-unavail"

        client.session_bridge.create_session_mapping(
            session_id=session_id,
            experiment_id="exp-unavail",
            experiment_run_id=experiment_run_id,
            function_name="unavail_func",
            configuration_space={},
            objectives=["accuracy"],
        )

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
            response = await client._session_ops.finalize_session(session_id)

        assert response.metadata["finalized_via_api"] is False
        assert response.metadata["summary_available"] is False

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
    def client(self, monkeypatch):
        """Create a mock BackendIntegratedClient."""
        # Disable offline mode so backend calls are actually made
        monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
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
