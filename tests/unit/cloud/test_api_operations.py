"""Unit tests for API operations module."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from traigent.cloud.api_operations import (
    _JSON_CONTENT_TYPE,
    AIOHTTP_AVAILABLE,
    ApiOperations,
    _typed_configuration_space,
)
from traigent.cloud.client import (
    CloudRemoteExecutionUnavailableError,
    CloudServiceError,
)
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
    """Test status mapping (issue #1302 — run-lifecycle wire vocab).

    Offline producer witness: these assert ``map_to_backend_status`` emits ONLY
    backend-canonical run-lifecycle members and NEVER the session-lifecycle
    ACTIVE / CREATED on the experiment-run / configuration-run status PUT path.
    They FAIL against the pre-#1302 mapper which emitted ACTIVE / CREATED.
    """

    def setup_method(self):
        """Set up test fixtures."""
        mock_client = Mock()
        self.ops = ApiOperations(mock_client)

    def test_pending_maps_to_pending(self):
        """pending -> PENDING (was ACTIVE before #1302)."""
        assert self.ops.map_to_backend_status("pending") == "PENDING"

    def test_running_maps_to_running(self):
        """running -> RUNNING (was ACTIVE before #1302)."""
        assert self.ops.map_to_backend_status("running") == "RUNNING"

    def test_in_progress_maps_to_running(self):
        """in_progress -> RUNNING (was ACTIVE before #1302)."""
        assert self.ops.map_to_backend_status("in_progress") == "RUNNING"

    def test_not_started_maps_to_not_started(self):
        """not_started -> NOT_STARTED (was CREATED before #1302)."""
        assert self.ops.map_to_backend_status("not_started") == "NOT_STARTED"

    def test_completed_maps_to_completed(self):
        """Test completed maps to COMPLETED."""
        assert self.ops.map_to_backend_status("completed") == "COMPLETED"

    def test_failed_maps_to_failed(self):
        """Test failed maps to FAILED."""
        assert self.ops.map_to_backend_status("failed") == "FAILED"

    def test_cancelled_maps_to_cancelled(self):
        """Test cancelled maps to CANCELLED."""
        assert self.ops.map_to_backend_status("cancelled") == "CANCELLED"

    def test_never_emits_session_lifecycle_vocab_for_experiment_runs(self):
        """No SDK in-flight status maps to the session-lifecycle ACTIVE/CREATED.

        This is the core #1302 regression: in-flight states (pending/running/
        not_started) must persist on the run PUT path, which they cannot if the
        SDK emits ACTIVE/CREATED (backend 400-rejects those).
        """
        from traigent.cloud.api_operations import EXPERIMENT_RUN_WIRE_STATUSES

        forbidden = {"ACTIVE", "CREATED"}
        for sdk_status in [
            "pending",
            "running",
            "in_progress",
            "not_started",
            "completed",
            "failed",
            "cancelled",
            "pruned",
        ]:
            mapped = self.ops.map_to_backend_status(
                sdk_status, endpoint="experiment_run"
            )
            assert mapped not in forbidden, (
                f"{sdk_status!r} mapped to forbidden session-lifecycle value {mapped!r}"
            )
            assert mapped in EXPERIMENT_RUN_WIRE_STATUSES

    def test_config_run_vocab_for_config_endpoint(self):
        """Every config-run mapping is a member of CONFIGURATION_RUN_WIRE_STATUSES."""
        from traigent.cloud.api_operations import CONFIGURATION_RUN_WIRE_STATUSES

        for sdk_status in [
            "not_started",
            "running",
            "in_progress",
            "completed",
            "failed",
            "cancelled",
            "pruned",
        ]:
            mapped = self.ops.map_to_backend_status(sdk_status, endpoint="config_run")
            assert mapped not in {"ACTIVE", "CREATED"}
            assert mapped in CONFIGURATION_RUN_WIRE_STATUSES

    def test_pruned_preserved_for_config_runs(self):
        """pruned -> PRUNED for config-runs (they accept PRUNED)."""
        assert (
            self.ops.map_to_backend_status("pruned", endpoint="config_run") == "PRUNED"
        )

    def test_pruned_folds_to_unknown_for_experiment_runs(self):
        """pruned -> UNKNOWN for experiment-runs (no PRUNED member).

        Must NOT emit PRUNED on the experiment-run endpoint, and must NOT claim
        COMPLETED — a pruned run is not a successful completion, so emitting
        COMPLETED would be fake completion. UNKNOWN is the honest neutral fold.
        """
        assert (
            self.ops.map_to_backend_status("pruned", endpoint="experiment_run")
            == "UNKNOWN"
        )

    def test_uppercase_status_preserved(self):
        """Already-canonical wire values pass through (case-insensitive)."""
        assert self.ops.map_to_backend_status("COMPLETED") == "COMPLETED"
        assert self.ops.map_to_backend_status("FAILED") == "FAILED"
        assert self.ops.map_to_backend_status("RUNNING") == "RUNNING"

    def test_unknown_status_falls_back_to_unknown_not_failed(self):
        """Unrecognized statuses fall back to UNKNOWN, never FAILED/CUSTOM.

        Pre-#1302 the gate defaulted unexpected statuses to FAILED (asserting
        failure on something we did not understand) and the mapper uppercased
        arbitrary strings into invalid backend members.
        """
        assert self.ops.map_to_backend_status("custom") == "UNKNOWN"
        assert (
            self.ops.map_to_backend_status("custom", endpoint="config_run") == "UNKNOWN"
        )

    def test_paused_supported_for_experiment_runs_only(self):
        """paused -> PAUSED for experiment-runs; config-runs lack PAUSED."""
        assert (
            self.ops.map_to_backend_status("paused", endpoint="experiment_run")
            == "PAUSED"
        )
        # config-runs have no PAUSED member -> neutral fallback, not a 400.
        assert (
            self.ops.map_to_backend_status("paused", endpoint="config_run") == "UNKNOWN"
        )


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
        msg = "Invalid token: eyJhbGciOiJIUzI1NiJ9"  # noqa: S105 - test JWT fragment
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
        """Test offline mode returns local session IDs."""
        with patch(
            "traigent.cloud.api_operations.is_backend_offline", return_value=True
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
            assert "mock_session_" in session_id
            assert "mock_exp_" in exp_id
            assert "mock_run_" in run_id

    @pytest.mark.asyncio
    async def test_auth_error_on_create_propagates_as_authentication_error(self):
        """#1278: a 401/403 during session creation must reach the caller AS an
        AuthenticationError — NOT be demoted to a plain CloudServiceError by the
        generic ``except Exception`` ladder.

        Before the fix, AuthenticationError (raised by ``_handle_session_error``
        with the structured 403 detail) fell through to ``except Exception`` and
        was re-wrapped as CloudServiceError, so session_operations classified it
        SESSION_FAILED -> BACKEND_UNREACHABLE ("check your network/URL") instead
        of an auth/permission error. This test fails on the pre-fix ladder
        (raises CloudServiceError) and passes once the dedicated
        ``except AuthenticationError: raise`` clause is in place.
        """
        from traigent.cloud.auth import AuthenticationError

        request = SessionCreationRequest(
            function_name="test_func",
            configuration_space={"param": [1, 2, 3]},
            objectives=["maximize"],
            max_trials=10,
        )
        # Simulate the 403 path: _post_session_creation -> _handle_session_error
        # raises AuthenticationError carrying the structured detail.
        with (
            patch(
                "traigent.cloud.api_operations.is_backend_offline",
                return_value=False,
            ),
            patch("traigent.cloud.api_operations.AIOHTTP_AVAILABLE", True),
            patch.object(
                self.ops,
                "_post_session_creation",
                new=AsyncMock(
                    side_effect=AuthenticationError(
                        "Authentication failed: missing permission"
                    )
                ),
            ),
        ):
            with pytest.raises(AuthenticationError):
                await self.ops.create_traigent_session_via_api(request)


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

    def test_basic_payload_is_typed_since_phase8(self):
        """The default contract is the TYPED shape (Phase 8) — the legacy
        problem_statement/search_space shape survives only behind
        TRAIGENT_SESSION_CONTRACT=legacy (see the legacy test below)."""
        request = Mock()
        request.function_name = "test_func"
        request.metadata = None
        request.dataset_metadata = {"size": 100}
        request.configuration_space = {"param": [1, 2, 3]}
        request.objectives = ["accuracy"]
        request.promotion_policy = None
        request.tvl_governance = None

        payload = self.ops._build_session_payload(request, 10)

        assert payload["function_name"] == "test_func"
        # DELIBERATELY EVOLVED (live composites-E2E finding): the shorthand
        # list form now normalizes to the typed contract the backend's typed
        # path actually accepts — the verbatim pass-through 400'd on every
        # decorator call with bare-list spaces.
        assert payload["configuration_space"] == {
            "param": {"type": "categorical", "choices": [1, 2, 3]}
        }
        assert payload["objectives"] == ["accuracy"]
        assert payload["max_trials"] == 10
        assert "problem_statement" not in payload

    def test_legacy_payload_behind_contract_flag(self, monkeypatch):
        monkeypatch.setenv("TRAIGENT_SESSION_CONTRACT", "legacy")
        request = Mock()
        request.function_name = "test_func"
        request.metadata = None
        request.dataset_metadata = {"size": 100}
        request.configuration_space = {"param": [1, 2, 3]}
        request.objectives = ["maximize"]
        request.promotion_policy = None
        request.tvl_governance = None

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
        # bare Mock auto-attributes must not reach the policy serializer
        request.promotion_policy = None
        request.tvl_governance = None

        payload = self.ops._build_session_payload(request, 20)

        assert payload["metadata"]["evaluation_set"] == "train"
        assert payload["metadata"]["custom_key"] == "value"
        assert payload["metadata"]["function_name"] == "test_func"

    def test_payload_empty_objectives(self, monkeypatch):
        """Legacy contract: empty objectives fall back to the maximize goal."""
        monkeypatch.setenv("TRAIGENT_SESSION_CONTRACT", "legacy")
        request = Mock()
        request.function_name = "test_func"
        request.metadata = None
        request.dataset_metadata = {}
        request.configuration_space = {}
        request.objectives = []
        request.promotion_policy = None
        request.tvl_governance = None

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

    def test_401_raises_authentication_error(self):
        """Test 401 error raises AuthenticationError."""
        from traigent.cloud.auth import AuthenticationError

        with pytest.raises(AuthenticationError, match="Authentication failed"):
            self.ops._handle_session_error(401, "Unauthorized")

    def test_403_raises_authentication_error(self):
        """Test 403 error raises AuthenticationError."""
        from traigent.cloud.auth import AuthenticationError

        with pytest.raises(AuthenticationError, match="Authentication failed") as exc:
            self.ops._handle_session_error(
                403,
                '{"success":false,"error_code":"INSUFFICIENT_PERMISSIONS",'
                '"message":"Missing required permissions: experiment.write",'
                '"details":{"missing_permissions":["experiment.write"]}}',
            )

        detail = exc.value.session_creation_failure
        assert detail.error_code == "INSUFFICIENT_PERMISSIONS"
        assert detail.message == "Missing required permissions: experiment.write"
        assert detail.missing_permissions == ("experiment.write",)

    def test_500_error_raises_cloud_service_error(self):
        """Test 500 error raises CloudServiceError."""
        with pytest.raises(CloudServiceError, match="Backend HTTP 500"):
            self.ops._handle_session_error(500, "Internal Server Error")

    def test_503_error_raises_cloud_service_error(self):
        """Test 503 error raises CloudServiceError."""
        with pytest.raises(CloudServiceError, match="Backend HTTP 503"):
            self.ops._handle_session_error(503, "Service Unavailable")

    def test_502_error_raises_cloud_service_error(self):
        """Test 502 error raises CloudServiceError."""
        with pytest.raises(CloudServiceError, match="Backend HTTP 502"):
            self.ops._handle_session_error(502, "Bad Gateway")

    def test_504_error_raises_cloud_service_error(self):
        """Test 504 error raises CloudServiceError."""
        with pytest.raises(CloudServiceError, match="Backend HTTP 504"):
            self.ops._handle_session_error(504, "Gateway Timeout")

    def test_other_error_raises_cloud_service_error(self):
        """Test other errors raise CloudServiceError."""
        with pytest.raises(CloudServiceError, match="Session creation failed") as exc:
            self.ops._handle_session_error(400, "Bad Request")

        detail = exc.value.session_creation_failure
        assert detail.status_code == 400
        assert detail.raw_body == "Bad Request"


class TestHandleClientError:
    """Test _handle_client_error method."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_client = Mock()
        self.ops = ApiOperations(mock_client)

    def test_raises_cloud_service_error(self):
        """Test raises CloudServiceError with network error message."""
        error = OSError("Connection failed")
        with pytest.raises(CloudServiceError, match="Network error"):
            self.ops._handle_client_error(error)


class TestHandleGenericSessionException:
    """Test _handle_generic_session_exception method."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_client = Mock()
        self.ops = ApiOperations(mock_client)

    def test_any_error_raises_cloud_service_error(self):
        """Test any error raises CloudServiceError (no string matching)."""
        error = Exception("HTTP 500 Internal Server Error")
        with pytest.raises(CloudServiceError, match="Session creation failed"):
            self.ops._handle_generic_session_exception(error)

    def test_connection_error_raises_cloud_service_error(self):
        """Test connection errors raise CloudServiceError."""
        error = Exception("Connection refused by server")
        with pytest.raises(CloudServiceError, match="Session creation failed"):
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
    async def test_raises_not_implemented(self):
        """Test remote cloud session creation fails closed."""
        request = SessionCreationRequest(
            function_name="test_func",
            configuration_space={"param": [1, 2, 3]},
            objectives=["maximize"],
            max_trials=10,
            billing_tier="standard",
        )
        with pytest.raises(CloudRemoteExecutionUnavailableError, match="use hybrid"):
            await self.ops.create_cloud_session(request)


class TestGetCloudTrialSuggestion:
    """Test get_cloud_trial_suggestion method."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_client = Mock()
        self.ops = ApiOperations(mock_client)

    @pytest.mark.asyncio
    async def test_raises_not_implemented(self):
        """Test remote cloud trial suggestions fail closed."""
        request = NextTrialRequest(
            session_id="session_123",
            previous_results=None,
        )
        with pytest.raises(CloudServiceError, match="use hybrid"):
            await self.ops.get_cloud_trial_suggestion(request)


class TestSubmitCloudTrialResults:
    """Test submit_cloud_trial_results method."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_client = Mock()
        self.ops = ApiOperations(mock_client)

    @pytest.mark.asyncio
    async def test_raises_not_implemented(self):
        """Test remote cloud trial submission fails closed."""
        from traigent.cloud.models import TrialStatus

        submission = TrialResultSubmission(
            session_id="session_123",
            trial_id="trial_123",
            metrics={"accuracy": 0.95},
            duration=1.5,
            status=TrialStatus.COMPLETED,
        )
        with pytest.raises(CloudServiceError, match="use hybrid"):
            await self.ops.submit_cloud_trial_results(submission)


class TestSubmitAgentOptimization:
    """Test submit_agent_optimization method."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_client = Mock()
        self.ops = ApiOperations(mock_client)

    @pytest.mark.asyncio
    async def test_raises_not_implemented(self):
        """Test remote agent optimization fails closed."""
        agent_spec = Mock()
        agent_spec.name = "test_agent"
        request = AgentOptimizationRequest(
            agent_spec=agent_spec,
            dataset=Mock(),
            configuration_space={"param": [1, 2, 3]},
            objectives=["maximize"],
            max_trials=10,
        )
        with pytest.raises(CloudServiceError, match="use hybrid"):
            await self.ops.submit_agent_optimization(request)


class TestExecuteCloudAgent:
    """Test execute_cloud_agent method."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_client = Mock()
        self.ops = ApiOperations(mock_client)

    @pytest.mark.asyncio
    async def test_raises_not_implemented(self):
        """Test remote agent execution fails closed."""
        agent_spec = Mock()
        agent_spec.name = "test_agent"
        agent_spec.id = "agent_123"
        request = AgentExecutionRequest(
            agent_spec=agent_spec,
            input_data={"query": "test"},
        )
        with pytest.raises(CloudServiceError, match="use hybrid"):
            await self.ops.execute_cloud_agent(request)


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
    async def test_create_session_without_aiohttp_fails_closed(self):
        """Test session creation fails closed when aiohttp is unavailable."""
        with (
            patch("traigent.cloud.api_operations.AIOHTTP_AVAILABLE", False),
            patch(
                "traigent.cloud.api_operations.is_backend_offline", return_value=False
            ),
        ):
            request = SessionCreationRequest(
                function_name="test_func",
                configuration_space={"param": [1, 2, 3]},
                objectives=["maximize"],
                max_trials=10,
            )
            with pytest.raises(CloudServiceError, match="aiohttp is required"):
                await self.ops.create_traigent_session_via_api(request)


class TestHandleConnectorError:
    """Test _handle_connector_error method."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_client = Mock()
        self.ops = ApiOperations(mock_client)

    def test_raises_cloud_service_error(self):
        """Test connector error raises CloudServiceError."""
        error = OSError("Connection failed")
        with pytest.raises(CloudServiceError, match="Backend unavailable"):
            self.ops._handle_connector_error(error)

    def test_logs_at_debug_only(self, caplog):
        """Test connector error logs at DEBUG, not WARNING."""
        import logging

        error = OSError("Connection failed to api.example.com")
        with (
            caplog.at_level(logging.DEBUG),
            pytest.raises(CloudServiceError),
        ):
            self.ops._handle_connector_error(error)

        # Should only have DEBUG records, no WARNING
        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_records) == 0


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
            assert (
                mock_session.post.call_args.args[0]
                == "https://api.example.com/sessions"
            )


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
        """Test measures update with all standard Traigent metrics."""
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
        """Test measures update rejects invalid non-numeric values.

        The implementation correctly rejects metrics with non-numeric values
        (like strings) rather than silently converting them. This ensures
        data integrity - users are notified of invalid metrics in their code.
        """
        # No mock needed - validation happens before HTTP call
        # Test with non-numeric value - should be rejected
        result = await self.ops.update_config_run_measures(
            "config_123", {"accuracy": 0.95, "custom": "invalid"}
        )
        # Invalid metrics are rejected (returns False) - this is correct behavior
        # Silently converting "invalid" to 0.0 would mask bugs in user code
        assert result is False

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
    async def test_invalid_status_uses_unknown_default(self):
        """Invalid status -> neutral UNKNOWN, NOT FAILED (issue #1302, AC3).

        Pre-#1302 the gate defaulted unexpected statuses to FAILED — asserting
        failure on a status we did not understand. The corrected behavior sends
        the neutral UNKNOWN member (valid on both backend run enums).
        """
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
            sent_status = mock_session.put.call_args.kwargs["json"]["status"]
            assert sent_status == "UNKNOWN"
            assert sent_status != "FAILED"

    @pytest.mark.asyncio
    async def test_running_status_persists_as_running_not_active(self):
        """In-flight 'running' must persist as RUNNING, not session ACTIVE.

        Core #1302 regression witness on the experiment-run PUT body: before the
        fix this sent ACTIVE which the backend 400-rejects, so an SDK run could
        never persist its in-flight state.
        """
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
                "run_123", "running"
            )
            sent_status = mock_session.put.call_args.kwargs["json"]["status"]
            assert sent_status == "RUNNING"
            assert sent_status not in {"ACTIVE", "CREATED"}

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


class TestTypedConfigurationSpace:
    """Tests for _typed_configuration_space normalization."""

    def test_list_becomes_categorical(self):
        result = _typed_configuration_space({"model": ["gpt-4o-mini", "gpt-4o"]})
        assert result == {"model": {"type": "categorical", "choices": ["gpt-4o-mini", "gpt-4o"]}}

    def test_tuple_becomes_categorical(self):
        result = _typed_configuration_space({"model": ("a", "b")})
        assert result == {"model": {"type": "categorical", "choices": ["a", "b"]}}

    def test_scalar_float_becomes_single_choice_categorical(self):
        # temperature=0.0 (fixed value) must be wrapped, not passed raw to BE
        result = _typed_configuration_space({"temperature": 0.0})
        assert result == {"temperature": {"type": "categorical", "choices": [0.0]}}

    def test_scalar_int_becomes_single_choice_categorical(self):
        result = _typed_configuration_space({"top_k": 5})
        assert result == {"top_k": {"type": "categorical", "choices": [5]}}

    def test_scalar_str_becomes_single_choice_categorical(self):
        result = _typed_configuration_space({"model": "gpt-4o"})
        assert result == {"model": {"type": "categorical", "choices": ["gpt-4o"]}}

    def test_dict_with_low_high_floats_infers_float_type(self):
        result = _typed_configuration_space({"temperature": {"low": 0.0, "high": 1.0}})
        assert result == {"temperature": {"type": "float", "low": 0.0, "high": 1.0}}

    def test_dict_with_low_high_ints_infers_int_type(self):
        result = _typed_configuration_space({"top_k": {"low": 1, "high": 10}})
        assert result == {"top_k": {"type": "int", "low": 1, "high": 10}}

    def test_dict_with_type_already_set_passes_through(self):
        entry = {"type": "float", "low": 0.0, "high": 1.0}
        result = _typed_configuration_space({"temperature": entry})
        assert result["temperature"] is entry

    def test_mixed_space_normalizes_all_entries(self):
        # Realistic user space: model categorical list, temperature scalar fixed value
        result = _typed_configuration_space({
            "model": ["gpt-4o-mini", "gpt-4o"],
            "temperature": 0.0,
        })
        assert result == {
            "model": {"type": "categorical", "choices": ["gpt-4o-mini", "gpt-4o"]},
            "temperature": {"type": "categorical", "choices": [0.0]},
        }

    def test_non_dict_space_passes_through(self):
        assert _typed_configuration_space(None) is None
        assert _typed_configuration_space("bad") == "bad"

    def test_bool_scalar_wrapped_not_confused_with_int(self):
        # bool is a subclass of int; verify it still gets wrapped as categorical
        result = _typed_configuration_space({"flag": True})
        assert result == {"flag": {"type": "categorical", "choices": [True]}}

    def test_dict_with_only_high_infers_float(self):
        # partial range dict — type can't be int if one bound is missing/float
        result = _typed_configuration_space({"x": {"high": 1.5}})
        assert result["x"]["type"] == "float"
