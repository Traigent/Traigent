"""Unit tests for API operations module."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from traigent.cloud.api_operations import (
    _JSON_CONTENT_TYPE,
    _LOCAL_FALLBACK_MSG,
    AIOHTTP_AVAILABLE,
    ApiOperations,
)
from traigent.cloud.client import CloudServiceError
from traigent.cloud.models import (
    AgentExecutionRequest,
    AgentOptimizationRequest,
    NextTrialRequest,
    SessionCreationRequest,
    TrialResultSubmission,
)


class TestApiOperationsInit:
    """Test ApiOperations initialization."""

    def test_init_with_client(self):
        """Test initialization with a client."""
        mock_client = Mock()
        ops = ApiOperations(mock_client)
        assert ops.client is mock_client


class TestValidateAndSanitizeUrl:
    """Test URL validation and sanitization."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_client = Mock()
        self.ops = ApiOperations(mock_client)

    def test_empty_url_raises(self):
        """Test that empty URL raises ValueError."""
        with pytest.raises(ValueError, match="URL cannot be empty"):
            self.ops.validate_and_sanitize_url("")

    def test_invalid_scheme_raises(self):
        """Test that invalid scheme raises ValueError."""
        with pytest.raises(ValueError, match="URL scheme must be one of"):
            self.ops.validate_and_sanitize_url("ftp://example.com")

    def test_no_host_raises(self):
        """Test that URL without host raises ValueError."""
        with pytest.raises(ValueError, match="URL must have a valid host"):
            self.ops.validate_and_sanitize_url("http://")

    def test_valid_https_url(self):
        """Test valid HTTPS URL passes validation."""
        url = "https://api.example.com/v1"
        result = self.ops.validate_and_sanitize_url(url)
        assert result == "https://api.example.com/v1"

    def test_valid_http_url(self):
        """Test valid HTTP URL passes validation."""
        url = "http://api.example.com/v1"
        result = self.ops.validate_and_sanitize_url(url)
        assert result == "http://api.example.com/v1"

    def test_localhost_url_allowed(self):
        """Test localhost URL is allowed."""
        url = "http://localhost:8000/api"
        result = self.ops.validate_and_sanitize_url(url)
        assert result == "http://localhost:8000/api"

    def test_127_0_0_1_url_allowed(self):
        """Test 127.0.0.1 URL is allowed."""
        url = "http://127.0.0.1:8000/api"
        result = self.ops.validate_and_sanitize_url(url)
        assert result == "http://127.0.0.1:8000/api"

    def test_private_ip_192_168_logs_warning(self):
        """Test that private IP logs warning but is allowed."""
        url = "http://192.168.1.1:8000/api"
        result = self.ops.validate_and_sanitize_url(url)
        assert result == "http://192.168.1.1:8000/api"

    def test_private_ip_10_logs_warning(self):
        """Test that private IP 10.x logs warning but is allowed."""
        url = "http://10.0.0.1:8000/api"
        result = self.ops.validate_and_sanitize_url(url)
        assert result == "http://10.0.0.1:8000/api"

    def test_url_with_query_preserved(self):
        """Test that query parameters are preserved."""
        url = "https://api.example.com/v1?key=value"
        result = self.ops.validate_and_sanitize_url(url)
        assert result == "https://api.example.com/v1?key=value"

    def test_trailing_slash_stripped(self):
        """Test that trailing slash is stripped."""
        url = "https://api.example.com/v1/"
        result = self.ops.validate_and_sanitize_url(url)
        assert result == "https://api.example.com/v1"

    def test_fragment_stripped(self):
        """Test that fragment is stripped."""
        url = "https://api.example.com/v1#section"
        result = self.ops.validate_and_sanitize_url(url)
        assert result == "https://api.example.com/v1"


class TestMapToBackendStatus:
    """Test status mapping."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_client = Mock()
        self.ops = ApiOperations(mock_client)

    def test_pending_maps_to_active(self):
        """Test pending maps to ACTIVE."""
        assert self.ops.map_to_backend_status("pending") == "ACTIVE"

    def test_running_maps_to_active(self):
        """Test running maps to ACTIVE."""
        assert self.ops.map_to_backend_status("running") == "ACTIVE"

    def test_in_progress_maps_to_active(self):
        """Test in_progress maps to ACTIVE."""
        assert self.ops.map_to_backend_status("in_progress") == "ACTIVE"

    def test_completed_maps_to_completed(self):
        """Test completed maps to COMPLETED."""
        assert self.ops.map_to_backend_status("completed") == "COMPLETED"

    def test_failed_maps_to_failed(self):
        """Test failed maps to FAILED."""
        assert self.ops.map_to_backend_status("failed") == "FAILED"

    def test_not_started_maps_to_created(self):
        """Test not_started maps to CREATED."""
        assert self.ops.map_to_backend_status("not_started") == "CREATED"

    def test_pruned_maps_to_pruned(self):
        """Test pruned maps to PRUNED."""
        assert self.ops.map_to_backend_status("pruned") == "PRUNED"

    def test_cancelled_maps_to_cancelled(self):
        """Test cancelled maps to CANCELLED."""
        assert self.ops.map_to_backend_status("cancelled") == "CANCELLED"

    def test_uppercase_status_preserved(self):
        """Test uppercase status preserved."""
        assert self.ops.map_to_backend_status("COMPLETED") == "COMPLETED"
        assert self.ops.map_to_backend_status("FAILED") == "FAILED"

    def test_unknown_status_uppercased(self):
        """Test unknown status is uppercased."""
        assert self.ops.map_to_backend_status("custom") == "CUSTOM"


class TestSanitizeErrorMessage:
    """Test error message sanitization."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_client = Mock()
        self.ops = ApiOperations(mock_client)

    def test_none_returns_none(self):
        """Test None input returns None."""
        assert self.ops.sanitize_error_message(None) is None

    def test_empty_string_returns_none(self):
        """Test empty string returns None."""
        assert self.ops.sanitize_error_message("") is None

    def test_api_key_redacted(self):
        """Test API keys are redacted."""
        msg = "Error with api_key=example-key-1234"
        result = self.ops.sanitize_error_message(msg)
        assert "example-key-1234" not in result
        assert "[REDACTED]" in result

    def test_token_redacted(self):
        """Test tokens are redacted."""
        msg = "Invalid token: eyJhbGciOiJIUzI1NiJ9"
        result = self.ops.sanitize_error_message(msg)
        assert "[REDACTED]" in result

    def test_password_redacted(self):
        """Test passwords are redacted."""
        msg = "password=secret123"
        result = self.ops.sanitize_error_message(msg)
        assert "[REDACTED]" in result

    def test_home_path_redacted(self):
        """Test home paths are redacted."""
        msg = "Error in /home/username/secret/file.py"
        result = self.ops.sanitize_error_message(msg)
        assert "/home/username" not in result
        assert "[PATH]" in result

    def test_users_path_redacted(self):
        """Test Users paths are redacted (macOS)."""
        msg = "Error in /Users/username/secret/file.py"
        result = self.ops.sanitize_error_message(msg)
        assert "/Users/username" not in result
        assert "[PATH]" in result

    def test_long_message_truncated(self):
        """Test long messages are truncated."""
        msg = "x" * 2000
        result = self.ops.sanitize_error_message(msg)
        assert len(result) <= 1020  # 1000 + "[truncated]" suffix
        assert "[truncated]" in result

    def test_normal_message_preserved(self):
        """Test normal message is preserved."""
        msg = "Normal error message"
        result = self.ops.sanitize_error_message(msg)
        assert result == msg


class TestCreateTraigentSessionViaApi:
    """Test session creation via API."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_client = Mock()
        mock_client.auth_manager = Mock()
        mock_client.auth_manager.augment_headers = AsyncMock(
            return_value={
                "Content-Type": _JSON_CONTENT_TYPE,
                "Authorization": "Bearer test",
            }
        )
        mock_client.backend_config = Mock()
        mock_client.backend_config.api_base_url = "https://api.example.com"
        self.ops = ApiOperations(mock_client)

    @pytest.mark.asyncio
    async def test_mock_mode_returns_local_ids(self):
        """Test mock mode returns local session IDs."""
        with patch("traigent.cloud.api_operations.is_mock_mode", return_value=True):
            request = SessionCreationRequest(
                function_name="test_func",
                configuration_space={"param": [1, 2, 3]},
                objectives=["maximize"],
                max_trials=10,
            )
            session_id, exp_id, run_id = await self.ops.create_traigent_session_via_api(
                request
            )
            assert "mock_session_" in session_id
            assert "mock_exp_" in exp_id
            assert "mock_run_" in run_id


class TestResolveMaxTrials:
    """Test _resolve_max_trials method."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_client = Mock()
        self.ops = ApiOperations(mock_client)

    def test_none_max_trials_returns_default(self):
        """Test None max_trials returns default 10."""
        request = Mock()
        request.max_trials = None
        result = self.ops._resolve_max_trials(request)
        assert result == 10

    def test_explicit_max_trials_preserved(self):
        """Test explicit max_trials is preserved."""
        request = Mock()
        request.max_trials = 50
        result = self.ops._resolve_max_trials(request)
        assert result == 50


class TestBuildSessionPayload:
    """Test _build_session_payload method."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_client = Mock()
        self.ops = ApiOperations(mock_client)

    def test_basic_payload(self):
        """Test basic payload structure."""
        request = Mock()
        request.function_name = "test_func"
        request.metadata = None
        request.dataset_metadata = {"size": 100}
        request.configuration_space = {"param": [1, 2, 3]}
        request.objectives = ["maximize"]

        payload = self.ops._build_session_payload(request, 10)

        assert payload["problem_statement"] == "test_func"
        assert payload["dataset"]["metadata"] == {"size": 100}
        assert payload["search_space"] == {"param": [1, 2, 3]}
        assert payload["optimization_config"]["max_trials"] == 10
        assert payload["optimization_config"]["optimization_goal"] == "maximize"

    def test_payload_with_metadata(self):
        """Test payload with metadata."""
        request = Mock()
        request.function_name = "test_func"
        request.metadata = {"evaluation_set": "train", "custom_key": "value"}
        request.dataset_metadata = {}
        request.configuration_space = {}
        request.objectives = []

        payload = self.ops._build_session_payload(request, 20)

        assert payload["metadata"]["evaluation_set"] == "train"
        assert payload["metadata"]["custom_key"] == "value"
        assert payload["metadata"]["function_name"] == "test_func"

    def test_payload_empty_objectives(self):
        """Test payload with empty objectives uses default."""
        request = Mock()
        request.function_name = "test_func"
        request.metadata = None
        request.dataset_metadata = {}
        request.configuration_space = {}
        request.objectives = []

        payload = self.ops._build_session_payload(request, 10)

        assert payload["optimization_config"]["optimization_goal"] == "maximize"


class TestBuildConnector:
    """Test _build_connector method."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_client = Mock()
        self.ops = ApiOperations(mock_client)

    def test_returns_none(self):
        """Test returns None (no custom connector)."""
        result = self.ops._build_connector()
        assert result is None


class TestHandleSessionError:
    """Test _handle_session_error method."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_client = Mock()
        self.ops = ApiOperations(mock_client)

    def test_500_error_raises_cloud_service_error(self):
        """Test 500 error raises CloudServiceError."""
        with pytest.raises(CloudServiceError, match="server error"):
            self.ops._handle_session_error(500, "Internal Server Error")

    def test_503_error_raises_cloud_service_error(self):
        """Test 503 error raises CloudServiceError."""
        with pytest.raises(CloudServiceError, match="service unavailable"):
            self.ops._handle_session_error(503, "Service Unavailable")

    def test_502_error_raises_cloud_service_error(self):
        """Test 502 error raises CloudServiceError."""
        with pytest.raises(CloudServiceError, match="gateway error"):
            self.ops._handle_session_error(502, "Bad Gateway")

    def test_504_error_raises_cloud_service_error(self):
        """Test 504 error raises CloudServiceError."""
        with pytest.raises(CloudServiceError, match="gateway error"):
            self.ops._handle_session_error(504, "Gateway Timeout")

    def test_other_error_raises_cloud_service_error(self):
        """Test other errors raise CloudServiceError."""
        with pytest.raises(CloudServiceError, match="Failed to create session"):
            self.ops._handle_session_error(400, "Bad Request")


class TestHandleClientError:
    """Test _handle_client_error method."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_client = Mock()
        self.ops = ApiOperations(mock_client)

    def test_raises_cloud_service_error(self):
        """Test raises CloudServiceError with network error message."""
        mock_error = Mock()
        mock_error.__str__ = Mock(return_value="Connection failed")
        with pytest.raises(CloudServiceError, match="Network error"):
            self.ops._handle_client_error(mock_error)


class TestHandleGenericSessionException:
    """Test _handle_generic_session_exception method."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_client = Mock()
        self.ops = ApiOperations(mock_client)

    def test_500_in_message_raises_specific_error(self):
        """Test 500 in message raises specific CloudServiceError."""
        error = Exception("HTTP 500 Internal Server Error")
        with pytest.raises(CloudServiceError, match="Backend temporarily unavailable"):
            self.ops._handle_generic_session_exception(error)

    def test_connection_refused_raises_specific_error(self):
        """Test connection refused raises specific CloudServiceError."""
        error = Exception("Connection refused by server")
        with pytest.raises(
            CloudServiceError, match="Cloud backend service not accessible"
        ):
            self.ops._handle_generic_session_exception(error)

    def test_generic_error_is_reraised(self):
        """Test generic errors are re-raised."""
        error = ValueError("Some other error")
        with pytest.raises(ValueError, match="Some other error"):
            self.ops._handle_generic_session_exception(error)


class TestUpdateConfigRunStatus:
    """Test update_config_run_status method."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_client = Mock()
        mock_client.auth_manager = Mock()
        mock_client.auth_manager.augment_headers = AsyncMock(
            return_value={"Content-Type": _JSON_CONTENT_TYPE}
        )
        mock_client.backend_config = Mock()
        mock_client.backend_config.api_base_url = "https://api.example.com"
        self.ops = ApiOperations(mock_client)

    @pytest.mark.asyncio
    async def test_aiohttp_not_available_returns_false(self):
        """Test returns False when aiohttp not available."""
        with patch("traigent.cloud.api_operations.AIOHTTP_AVAILABLE", False):
            result = await self.ops.update_config_run_status("config_123", "COMPLETED")
            assert result is False


class TestUpdateConfigRunMeasures:
    """Test update_config_run_measures method."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_client = Mock()
        mock_client.auth_manager = Mock()
        mock_client.auth_manager.augment_headers = AsyncMock(
            return_value={"Content-Type": _JSON_CONTENT_TYPE}
        )
        mock_client.backend_config = Mock()
        mock_client.backend_config.api_base_url = "https://api.example.com"
        self.ops = ApiOperations(mock_client)

    @pytest.mark.asyncio
    async def test_aiohttp_not_available_returns_false(self):
        """Test returns False when aiohttp not available."""
        with patch("traigent.cloud.api_operations.AIOHTTP_AVAILABLE", False):
            result = await self.ops.update_config_run_measures(
                "config_123", {"accuracy": 0.95}
            )
            assert result is False

    @pytest.mark.asyncio
    async def test_empty_metrics_returns_false(self):
        """Test returns False when metrics is empty."""
        result = await self.ops.update_config_run_measures("config_123", {})
        assert result is False


class TestUpdateExperimentRunStatusOnCompletion:
    """Test update_experiment_run_status_on_completion method."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_client = Mock()
        mock_client.auth_manager = Mock()
        mock_client.auth_manager.augment_headers = AsyncMock(
            return_value={"Content-Type": _JSON_CONTENT_TYPE}
        )
        mock_client.backend_config = Mock()
        mock_client.backend_config.api_base_url = "https://api.example.com"
        self.ops = ApiOperations(mock_client)

    @pytest.mark.asyncio
    async def test_aiohttp_not_available_returns_early(self):
        """Test returns early when aiohttp not available."""
        with patch("traigent.cloud.api_operations.AIOHTTP_AVAILABLE", False):
            # Should not raise
            await self.ops.update_experiment_run_status_on_completion(
                "run_123", "completed"
            )


class TestCreateCloudSession:
    """Test create_cloud_session method."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_client = Mock()
        self.ops = ApiOperations(mock_client)

    @pytest.mark.asyncio
    async def test_returns_session_response(self):
        """Test returns valid session response."""
        request = SessionCreationRequest(
            function_name="test_func",
            configuration_space={"param": [1, 2, 3]},
            objectives=["maximize"],
            max_trials=10,
            billing_tier="standard",
        )
        response = await self.ops.create_cloud_session(request)
        assert "cloud_session_" in response.session_id
        assert response.metadata["billing_tier"] == "standard"


class TestGetCloudTrialSuggestion:
    """Test get_cloud_trial_suggestion method."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_client = Mock()
        self.ops = ApiOperations(mock_client)

    @pytest.mark.asyncio
    async def test_returns_trial_suggestion(self):
        """Test returns valid trial suggestion."""
        request = NextTrialRequest(
            session_id="session_123",
            previous_results=None,
        )
        response = await self.ops.get_cloud_trial_suggestion(request)
        assert response.suggestion is not None
        assert "trial_" in response.suggestion.trial_id
        assert response.should_continue is True


class TestSubmitCloudTrialResults:
    """Test submit_cloud_trial_results method."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_client = Mock()
        self.ops = ApiOperations(mock_client)

    @pytest.mark.asyncio
    async def test_submits_without_error(self):
        """Test submits without raising errors."""
        from traigent.cloud.models import TrialStatus

        submission = TrialResultSubmission(
            session_id="session_123",
            trial_id="trial_123",
            metrics={"accuracy": 0.95},
            duration=1.5,
            status=TrialStatus.COMPLETED,
        )
        # Should not raise
        await self.ops.submit_cloud_trial_results(submission)


class TestSubmitAgentOptimization:
    """Test submit_agent_optimization method."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_client = Mock()
        self.ops = ApiOperations(mock_client)

    @pytest.mark.asyncio
    async def test_returns_optimization_response(self):
        """Test returns valid optimization response."""
        agent_spec = Mock()
        agent_spec.name = "test_agent"
        request = AgentOptimizationRequest(
            agent_spec=agent_spec,
            dataset=Mock(),
            configuration_space={"param": [1, 2, 3]},
            objectives=["maximize"],
            max_trials=10,
        )
        response = await self.ops.submit_agent_optimization(request)
        assert "agent_session_" in response.session_id
        assert "opt_" in response.optimization_id
        assert response.status == "started"


class TestExecuteCloudAgent:
    """Test execute_cloud_agent method."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_client = Mock()
        self.ops = ApiOperations(mock_client)

    @pytest.mark.asyncio
    async def test_returns_execution_response(self):
        """Test returns valid execution response."""
        agent_spec = Mock()
        agent_spec.name = "test_agent"
        agent_spec.id = "agent_123"
        request = AgentExecutionRequest(
            agent_spec=agent_spec,
            input_data={"query": "test"},
        )
        response = await self.ops.execute_cloud_agent(request)
        assert response.output == "Mock agent response"
        assert abs(response.duration - 1.5) < 0.01  # Avoid floating point comparison
        assert response.tokens_used == 50


class TestDeprecatedMethods:
    """Test deprecated methods raise NotImplementedError."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_client = Mock()
        self.ops = ApiOperations(mock_client)

    @pytest.mark.asyncio
    async def test_create_backend_experiment_tracking_raises(self):
        """Test create_backend_experiment_tracking raises NotImplementedError."""
        request = Mock()
        with pytest.raises(
            NotImplementedError, match="SDK must use session endpoints only"
        ):
            await self.ops.create_backend_experiment_tracking(request)

    @pytest.mark.asyncio
    async def test_create_backend_agent_experiment_raises(self):
        """Test create_backend_agent_experiment raises NotImplementedError."""
        with pytest.raises(
            NotImplementedError, match="SDK must use session endpoints only"
        ):
            await self.ops.create_backend_agent_experiment(
                agent_spec=Mock(),
                dataset=Mock(),
                configuration_space={},
                objectives=[],
                max_trials=10,
            )


class TestAiohttpPlaceholder:
    """Test aiohttp placeholder behavior when library is not available."""

    def test_placeholder_classes_exist(self):
        """Test placeholder classes exist when aiohttp not available."""
        if not AIOHTTP_AVAILABLE:
            from traigent.cloud.api_operations import aiohttp

            assert hasattr(aiohttp, "ClientConnectorError")
            assert hasattr(aiohttp, "ClientError")
            assert hasattr(aiohttp, "ClientTimeout")
            assert hasattr(aiohttp, "TCPConnector")
            assert hasattr(aiohttp, "ClientSession")


class TestConstants:
    """Test module constants."""

    def test_json_content_type(self):
        """Test JSON content type constant."""
        assert _JSON_CONTENT_TYPE == "application/json"

    def test_local_fallback_msg(self):
        """Test local fallback message constant."""
        assert "fall back" in _LOCAL_FALLBACK_MSG.lower()


class TestAiohttpNotAvailable:
    """Test behavior when aiohttp is not available."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_client = Mock()
        mock_client.auth_manager = Mock()
        mock_client.auth_manager.augment_headers = AsyncMock(
            return_value={"Content-Type": _JSON_CONTENT_TYPE}
        )
        mock_client.backend_config = Mock()
        mock_client.backend_config.api_base_url = "https://api.example.com"
        self.ops = ApiOperations(mock_client)

    @pytest.mark.asyncio
    async def test_create_session_without_aiohttp_returns_fallback_ids(self):
        """Test session creation returns fallback IDs when aiohttp not available."""
        with (
            patch("traigent.cloud.api_operations.AIOHTTP_AVAILABLE", False),
            patch("traigent.cloud.api_operations.is_mock_mode", return_value=False),
        ):
            request = SessionCreationRequest(
                function_name="test_func",
                configuration_space={"param": [1, 2, 3]},
                objectives=["maximize"],
                max_trials=10,
            )
            session_id, exp_id, run_id = await self.ops.create_traigent_session_via_api(
                request
            )
            assert "session_" in session_id
            assert "exp_" in exp_id
            assert "run_" in run_id


class TestHandleConnectorError:
    """Test _handle_connector_error method."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_client = Mock()
        self.ops = ApiOperations(mock_client)

    def test_mock_mode_logs_debug(self):
        """Test mock mode logs debug message."""
        import traigent.cloud.api_operations as api_ops

        api_ops._backend_unavailable_warned = False
        with patch("traigent.cloud.api_operations.is_mock_mode", return_value=True):
            error = RuntimeError("Connection failed")
            with pytest.raises(CloudServiceError, match="Backend unavailable"):
                self.ops._handle_connector_error(error)

    def test_first_warning_shows_full_message(self):
        """Test first warning shows full message."""
        import traigent.cloud.api_operations as api_ops

        api_ops._backend_unavailable_warned = False
        with patch("traigent.cloud.api_operations.is_mock_mode", return_value=False):
            error = RuntimeError("Connection failed to api.example.com")
            with pytest.raises(CloudServiceError, match="Backend unavailable"):
                self.ops._handle_connector_error(error)
            assert api_ops._backend_unavailable_warned is True

    def test_localhost_shows_development_hint(self):
        """Test localhost error shows development hint."""
        import traigent.cloud.api_operations as api_ops

        api_ops._backend_unavailable_warned = False
        with patch("traigent.cloud.api_operations.is_mock_mode", return_value=False):
            error = RuntimeError("Connection failed to localhost:8000")
            with pytest.raises(CloudServiceError, match="Backend unavailable"):
                self.ops._handle_connector_error(error)

    def test_127_0_0_1_shows_development_hint(self):
        """Test 127.0.0.1 error shows development hint."""
        import traigent.cloud.api_operations as api_ops

        api_ops._backend_unavailable_warned = False
        with patch("traigent.cloud.api_operations.is_mock_mode", return_value=False):
            error = RuntimeError("Connection failed to 127.0.0.1:8000")
            with pytest.raises(CloudServiceError, match="Backend unavailable"):
                self.ops._handle_connector_error(error)

    def test_second_warning_logs_debug_only(self):
        """Test subsequent warnings only log debug."""
        import traigent.cloud.api_operations as api_ops

        api_ops._backend_unavailable_warned = True
        with patch("traigent.cloud.api_operations.is_mock_mode", return_value=False):
            error = RuntimeError("Connection failed again")
            with pytest.raises(CloudServiceError, match="Backend unavailable"):
                self.ops._handle_connector_error(error)


class TestPostSessionCreation:
    """Test _post_session_creation method."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_client = Mock()
        mock_client.auth_manager = Mock()
        mock_client.auth_manager.augment_headers = AsyncMock(
            return_value={"Content-Type": _JSON_CONTENT_TYPE}
        )
        mock_client.backend_config = Mock()
        mock_client.backend_config.api_base_url = "https://api.example.com"
        self.ops = ApiOperations(mock_client)

    @pytest.mark.asyncio
    async def test_successful_session_creation(self):
        """Test successful session creation returns IDs."""
        mock_response = AsyncMock()
        mock_response.status = 201
        mock_response.json = AsyncMock(
            return_value={
                "session_id": "session_123",
                "metadata": {
                    "experiment_id": "exp_123",
                    "experiment_run_id": "run_123",
                },
            }
        )

        mock_session = AsyncMock()
        mock_session.post = Mock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock()
            )
        )

        with patch("traigent.cloud.api_operations.aiohttp") as mock_aiohttp:
            mock_aiohttp.ClientSession = Mock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(return_value=mock_session),
                    __aexit__=AsyncMock(),
                )
            )
            mock_aiohttp.ClientTimeout = Mock()

            session_id, exp_id, run_id = await self.ops._post_session_creation(
                payload={"test": "data"},
                headers={"Content-Type": "application/json"},
                connector=None,
            )
            assert session_id == "session_123"
            assert exp_id == "exp_123"
            assert run_id == "run_123"


class TestParseSessionResponse:
    """Test _parse_session_response method."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_client = Mock()
        self.ops = ApiOperations(mock_client)

    @pytest.mark.asyncio
    async def test_parse_response_with_all_ids(self):
        """Test parsing response with all IDs present."""
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(
            return_value={
                "session_id": "session_123",
                "metadata": {
                    "experiment_id": "exp_123",
                    "experiment_run_id": "run_123",
                },
            }
        )
        session_id, exp_id, run_id = await self.ops._parse_session_response(
            mock_response
        )
        assert session_id == "session_123"
        assert exp_id == "exp_123"
        assert run_id == "run_123"

    @pytest.mark.asyncio
    async def test_parse_response_with_fallback_ids(self):
        """Test parsing response with missing experiment IDs falls back to session ID."""
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(
            return_value={
                "session_id": "session_456",
                "metadata": {},
            }
        )
        session_id, exp_id, run_id = await self.ops._parse_session_response(
            mock_response
        )
        assert session_id == "session_456"
        assert exp_id == "session_456"
        assert run_id == "session_456"


class TestUpdateConfigRunStatusSuccess:
    """Test update_config_run_status with successful responses."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_client = Mock()
        mock_client.auth_manager = Mock()
        mock_client.auth_manager.augment_headers = AsyncMock(
            return_value={"Content-Type": _JSON_CONTENT_TYPE}
        )
        mock_client.backend_config = Mock()
        mock_client.backend_config.api_base_url = "https://api.example.com"
        self.ops = ApiOperations(mock_client)

    @pytest.mark.asyncio
    async def test_successful_status_update_200(self):
        """Test successful status update returns True."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="OK")

        mock_session = AsyncMock()
        mock_session.put = Mock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock()
            )
        )

        with patch("traigent.cloud.api_operations.aiohttp") as mock_aiohttp:
            mock_aiohttp.ClientSession = Mock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(return_value=mock_session),
                    __aexit__=AsyncMock(),
                )
            )
            mock_aiohttp.ClientTimeout = Mock()

            result = await self.ops.update_config_run_status("config_123", "COMPLETED")
            assert result is True

    @pytest.mark.asyncio
    async def test_successful_status_update_204(self):
        """Test successful status update with 204 returns True."""
        mock_response = AsyncMock()
        mock_response.status = 204

        mock_session = AsyncMock()
        mock_session.put = Mock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock()
            )
        )

        with patch("traigent.cloud.api_operations.aiohttp") as mock_aiohttp:
            mock_aiohttp.ClientSession = Mock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(return_value=mock_session),
                    __aexit__=AsyncMock(),
                )
            )
            mock_aiohttp.ClientTimeout = Mock()

            result = await self.ops.update_config_run_status("config_123", "COMPLETED")
            assert result is True

    @pytest.mark.asyncio
    async def test_failed_status_update_returns_false(self):
        """Test failed status update returns False."""
        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.text = AsyncMock(return_value="Bad Request")

        mock_session = AsyncMock()
        mock_session.put = Mock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock()
            )
        )

        with patch("traigent.cloud.api_operations.aiohttp") as mock_aiohttp:
            mock_aiohttp.ClientSession = Mock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(return_value=mock_session),
                    __aexit__=AsyncMock(),
                )
            )
            mock_aiohttp.ClientTimeout = Mock()

            result = await self.ops.update_config_run_status("config_123", "COMPLETED")
            assert result is False

    @pytest.mark.asyncio
    async def test_exception_returns_false(self):
        """Test exception during update returns False."""
        with patch("traigent.cloud.api_operations.aiohttp") as mock_aiohttp:
            mock_aiohttp.ClientSession = Mock(
                side_effect=Exception("Connection failed")
            )

            result = await self.ops.update_config_run_status("config_123", "COMPLETED")
            assert result is False


class TestUpdateConfigRunMeasuresSuccess:
    """Test update_config_run_measures with various metric mappings."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_client = Mock()
        mock_client.auth_manager = Mock()
        mock_client.auth_manager.augment_headers = AsyncMock(
            return_value={"Content-Type": _JSON_CONTENT_TYPE}
        )
        mock_client.backend_config = Mock()
        mock_client.backend_config.api_base_url = "https://api.example.com"
        self.ops = ApiOperations(mock_client)

    @pytest.mark.asyncio
    async def test_successful_measures_update(self):
        """Test successful measures update returns True."""
        mock_response = AsyncMock()
        mock_response.status = 200

        mock_session = AsyncMock()
        mock_session.put = Mock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock()
            )
        )

        with patch("traigent.cloud.api_operations.aiohttp") as mock_aiohttp:
            mock_aiohttp.ClientSession = Mock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(return_value=mock_session),
                    __aexit__=AsyncMock(),
                )
            )
            mock_aiohttp.ClientTimeout = Mock()

            result = await self.ops.update_config_run_measures(
                "config_123", {"accuracy": 0.95, "score": 0.9}
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_measures_update_with_execution_time(self):
        """Test measures update with execution time."""
        mock_response = AsyncMock()
        mock_response.status = 200

        mock_session = AsyncMock()
        mock_session.put = Mock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock()
            )
        )

        with patch("traigent.cloud.api_operations.aiohttp") as mock_aiohttp:
            mock_aiohttp.ClientSession = Mock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(return_value=mock_session),
                    __aexit__=AsyncMock(),
                )
            )
            mock_aiohttp.ClientTimeout = Mock()

            result = await self.ops.update_config_run_measures(
                "config_123", {"accuracy": 0.95}, execution_time=1.5
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_measures_update_with_all_standard_metrics(self):
        """Test measures update with all standard OptiGen metrics."""
        mock_response = AsyncMock()
        mock_response.status = 200

        mock_session = AsyncMock()
        mock_session.put = Mock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock()
            )
        )

        with patch("traigent.cloud.api_operations.aiohttp") as mock_aiohttp:
            mock_aiohttp.ClientSession = Mock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(return_value=mock_session),
                    __aexit__=AsyncMock(),
                )
            )
            mock_aiohttp.ClientTimeout = Mock()

            result = await self.ops.update_config_run_measures(
                "config_123",
                {
                    "accuracy": 0.95,
                    "faithfulness": 0.9,
                    "relevance": 0.85,
                    "latency": 100.0,
                    "cost": 0.01,
                    "context_precision": 0.88,
                    "context_recall": 0.92,
                    "custom_metric": 0.75,
                },
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_measures_update_uses_accuracy_as_score_fallback(self):
        """Test measures update uses accuracy as score when no explicit score."""
        mock_response = AsyncMock()
        mock_response.status = 200

        mock_session = AsyncMock()
        mock_session.put = Mock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock()
            )
        )

        with patch("traigent.cloud.api_operations.aiohttp") as mock_aiohttp:
            mock_aiohttp.ClientSession = Mock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(return_value=mock_session),
                    __aexit__=AsyncMock(),
                )
            )
            mock_aiohttp.ClientTimeout = Mock()

            # Only accuracy, no score - should use accuracy as score
            result = await self.ops.update_config_run_measures(
                "config_123", {"accuracy": 0.95}
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_measures_update_handles_none_values(self):
        """Test measures update handles None values gracefully."""
        mock_response = AsyncMock()
        mock_response.status = 200

        mock_session = AsyncMock()
        mock_session.put = Mock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock()
            )
        )

        with patch("traigent.cloud.api_operations.aiohttp") as mock_aiohttp:
            mock_aiohttp.ClientSession = Mock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(return_value=mock_session),
                    __aexit__=AsyncMock(),
                )
            )
            mock_aiohttp.ClientTimeout = Mock()

            result = await self.ops.update_config_run_measures(
                "config_123", {"accuracy": 0.95, "score": None, "latency": None}
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_measures_update_handles_invalid_values(self):
        """Test measures update converts invalid values to 0.0."""
        mock_response = AsyncMock()
        mock_response.status = 200

        mock_session = AsyncMock()
        mock_session.put = Mock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock()
            )
        )

        with patch("traigent.cloud.api_operations.aiohttp") as mock_aiohttp:
            mock_aiohttp.ClientSession = Mock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(return_value=mock_session),
                    __aexit__=AsyncMock(),
                )
            )
            mock_aiohttp.ClientTimeout = Mock()

            # Test with non-numeric value
            result = await self.ops.update_config_run_measures(
                "config_123", {"accuracy": 0.95, "custom": "invalid"}
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_measures_update_failure_returns_false(self):
        """Test measures update failure returns False."""
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")

        mock_session = AsyncMock()
        mock_session.put = Mock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock()
            )
        )

        with patch("traigent.cloud.api_operations.aiohttp") as mock_aiohttp:
            mock_aiohttp.ClientSession = Mock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(return_value=mock_session),
                    __aexit__=AsyncMock(),
                )
            )
            mock_aiohttp.ClientTimeout = Mock()

            result = await self.ops.update_config_run_measures(
                "config_123", {"accuracy": 0.95}
            )
            assert result is False

    @pytest.mark.asyncio
    async def test_measures_update_exception_returns_false(self):
        """Test measures update exception returns False."""
        with patch("traigent.cloud.api_operations.aiohttp") as mock_aiohttp:
            mock_aiohttp.ClientSession = Mock(side_effect=Exception("Network error"))

            result = await self.ops.update_config_run_measures(
                "config_123", {"accuracy": 0.95}
            )
            assert result is False


class TestUpdateExperimentRunStatusSuccess:
    """Test update_experiment_run_status_on_completion with various scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_client = Mock()
        mock_client.auth_manager = Mock()
        mock_client.auth_manager.augment_headers = AsyncMock(
            return_value={"Content-Type": _JSON_CONTENT_TYPE}
        )
        mock_client.backend_config = Mock()
        mock_client.backend_config.api_base_url = "https://api.example.com"
        self.ops = ApiOperations(mock_client)

    @pytest.mark.asyncio
    async def test_successful_status_update(self):
        """Test successful experiment run status update."""
        mock_response = AsyncMock()
        mock_response.status = 200

        mock_session = AsyncMock()
        mock_session.put = Mock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock()
            )
        )

        with patch("traigent.cloud.api_operations.aiohttp") as mock_aiohttp:
            mock_aiohttp.ClientSession = Mock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(return_value=mock_session),
                    __aexit__=AsyncMock(),
                )
            )
            mock_aiohttp.ClientTimeout = Mock()

            # Should not raise
            await self.ops.update_experiment_run_status_on_completion(
                "run_123", "completed"
            )

    @pytest.mark.asyncio
    async def test_status_update_with_204(self):
        """Test status update with 204 response."""
        mock_response = AsyncMock()
        mock_response.status = 204

        mock_session = AsyncMock()
        mock_session.put = Mock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock()
            )
        )

        with patch("traigent.cloud.api_operations.aiohttp") as mock_aiohttp:
            mock_aiohttp.ClientSession = Mock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(return_value=mock_session),
                    __aexit__=AsyncMock(),
                )
            )
            mock_aiohttp.ClientTimeout = Mock()

            await self.ops.update_experiment_run_status_on_completion(
                "run_123", "failed"
            )

    @pytest.mark.asyncio
    async def test_invalid_status_uses_completed_default(self):
        """Test invalid status defaults to COMPLETED."""
        mock_response = AsyncMock()
        mock_response.status = 200

        mock_session = AsyncMock()
        mock_session.put = Mock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock()
            )
        )

        with patch("traigent.cloud.api_operations.aiohttp") as mock_aiohttp:
            mock_aiohttp.ClientSession = Mock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(return_value=mock_session),
                    __aexit__=AsyncMock(),
                )
            )
            mock_aiohttp.ClientTimeout = Mock()

            await self.ops.update_experiment_run_status_on_completion(
                "run_123", "unknown_status"
            )

    @pytest.mark.asyncio
    async def test_failed_update_logs_warning(self):
        """Test failed update logs warning."""
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")

        mock_session = AsyncMock()
        mock_session.put = Mock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock()
            )
        )

        with patch("traigent.cloud.api_operations.aiohttp") as mock_aiohttp:
            mock_aiohttp.ClientSession = Mock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(return_value=mock_session),
                    __aexit__=AsyncMock(),
                )
            )
            mock_aiohttp.ClientTimeout = Mock()

            # Should not raise, just log warning
            await self.ops.update_experiment_run_status_on_completion(
                "run_123", "completed"
            )

    @pytest.mark.asyncio
    async def test_exception_logs_warning(self):
        """Test exception logs warning."""
        with patch("traigent.cloud.api_operations.aiohttp") as mock_aiohttp:
            mock_aiohttp.ClientSession = Mock(side_effect=Exception("Network error"))

            # Should not raise, just log warning
            await self.ops.update_experiment_run_status_on_completion(
                "run_123", "completed"
            )

    @pytest.mark.asyncio
    async def test_pruned_status_accepted(self):
        """Test pruned status is accepted."""
        mock_response = AsyncMock()
        mock_response.status = 200

        mock_session = AsyncMock()
        mock_session.put = Mock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock()
            )
        )

        with patch("traigent.cloud.api_operations.aiohttp") as mock_aiohttp:
            mock_aiohttp.ClientSession = Mock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(return_value=mock_session),
                    __aexit__=AsyncMock(),
                )
            )
            mock_aiohttp.ClientTimeout = Mock()

            await self.ops.update_experiment_run_status_on_completion(
                "run_123", "pruned"
            )

    @pytest.mark.asyncio
    async def test_cancelled_status_accepted(self):
        """Test cancelled status is accepted."""
        mock_response = AsyncMock()
        mock_response.status = 200

        mock_session = AsyncMock()
        mock_session.put = Mock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock()
            )
        )

        with patch("traigent.cloud.api_operations.aiohttp") as mock_aiohttp:
            mock_aiohttp.ClientSession = Mock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(return_value=mock_session),
                    __aexit__=AsyncMock(),
                )
            )
            mock_aiohttp.ClientTimeout = Mock()

            await self.ops.update_experiment_run_status_on_completion(
                "run_123", "cancelled"
            )
