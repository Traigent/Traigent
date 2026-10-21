"""#2550 review round 3: quota refusals are typed and sanitized on every path."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, Mock, patch

import pytest

from traigent.cloud.api_operations import ApiOperations
from traigent.cloud.client import (
    CloudServiceError,
    SessionBackendRefusedError,
    SessionQuotaRefusedError,
    TraigentCloudClient,
)
from traigent.cloud.session_operations import SessionOperations
from traigent.cloud.session_types import SessionCreationFailureDetail
from tests.unit.cloud.test_session_operations_validation import FakeClient

QUOTA_BODY = '{"error_code": "quota_exceeded", "message": "customer text here"}'


def _assert_clean(text: str) -> None:
    assert "customer text" not in text.lower()


class TestApiHandlerLogging:
    def test_429_debug_log_has_no_backend_text(self, caplog):
        ops = ApiOperations(Mock())
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(CloudServiceError):
                ops._handle_session_error(429, QUOTA_BODY)
        blob = " ".join(r.getMessage() for r in caplog.records)
        _assert_clean(blob)
        assert "429" in blob

    def test_non_quota_400_debug_log_has_no_backend_text(self, caplog):
        ops = ApiOperations(Mock())
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(CloudServiceError):
                ops._handle_session_error(400, "customer text here")
        _assert_clean(" ".join(r.getMessage() for r in caplog.records))


class _FakeResponse:
    def __init__(self, status: int, text: str) -> None:
        self.status = status
        self._text = text

    async def text(self) -> str:
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _hybrid_ops(status: int, body: str) -> SessionOperations:
    client = FakeClient()
    session = Mock()
    session.post = Mock(return_value=_FakeResponse(status, body))
    client._ensure_session = AsyncMock(return_value=session)
    client.auth_manager.auth = Mock()
    client.auth_manager.auth.get_headers = AsyncMock(return_value={})
    ops = SessionOperations(client)
    ops._reset_client_session = AsyncMock()
    return ops


@pytest.mark.backend_online
class TestHybridPath:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [429, 402])
    async def test_quota_is_typed_and_sanitized(self, status):
        ops = _hybrid_ops(status, QUOTA_BODY)
        with patch("traigent.cloud.session_operations.AIOHTTP_AVAILABLE", True):
            with pytest.raises(SessionQuotaRefusedError) as info:
                await ops.create_hybrid_session("p", {"a": [1]}, {"max_trials": 1})
        _assert_clean(str(info.value))
        assert info.value.error_code == "quota_exceeded"
        assert info.value.session_creation_failure.message is None
        assert info.value.session_creation_failure.raw_body is None

    @pytest.mark.asyncio
    async def test_other_error_does_not_embed_body(self):
        ops = _hybrid_ops(503, "customer text here")
        with patch("traigent.cloud.session_operations.AIOHTTP_AVAILABLE", True):
            with pytest.raises(CloudServiceError) as info:
                await ops.create_hybrid_session("p", {"a": [1]}, {"max_trials": 1})
        assert not isinstance(info.value, SessionQuotaRefusedError)
        _assert_clean(str(info.value))
        assert "503" in str(info.value)


@pytest.mark.backend_online
class TestGovernedSession:
    @staticmethod
    def _ops(exc: Exception) -> SessionOperations:
        client = FakeClient()

        async def create_via_api(_request):
            raise exc

        client._create_traigent_session_via_api = create_via_api
        return SessionOperations(client)

    def test_governed_quota_refusal_is_typed_and_sanitized(self):
        exc = CloudServiceError("boom customer text here")
        exc.session_creation_failure = SessionCreationFailureDetail.from_http_response(
            429, QUOTA_BODY
        )
        ops = self._ops(exc)
        with patch(
            "traigent.cloud.session_operations.is_backend_offline", return_value=False
        ):
            with pytest.raises(SessionQuotaRefusedError) as info:
                ops.create_session(
                    "fn", {"a": [1, 2]}, promotion_policy={"min_trials": 1}
                )
        _assert_clean(str(info.value))
        detail = info.value.session_creation_failure
        assert detail.raw_body is None and detail.message is None
        assert info.value.__cause__ is None

    def test_governed_503_still_fails_loud_unchanged(self):
        exc = CloudServiceError("503 Service Unavailable")
        ops = self._ops(exc)
        with patch(
            "traigent.cloud.session_operations.is_backend_offline", return_value=False
        ):
            with pytest.raises(CloudServiceError) as info:
                ops.create_session(
                    "fn", {"a": [1, 2]}, promotion_policy={"min_trials": 1}
                )
        assert info.value is exc


ALLOWLISTED_CODES = [
    "quota_exceeded",
    "quota_limit_exceeded",
    "plan_limit_exceeded",
    "usage_limit_exceeded",
]


def _body(code: str) -> str:
    return f'{{"error_code": "{code}", "message": "customer text here"}}'


class TestClassifierCoversAllowlist:
    @pytest.mark.parametrize("code", ALLOWLISTED_CODES)
    @pytest.mark.parametrize("status", [429, 402])
    def test_every_allowlisted_code_is_quota(self, code, status):
        detail = SessionCreationFailureDetail.from_http_response(status, _body(code))
        assert detail.is_quota_refusal
        assert detail.safe_quota_error_code == code
        _assert_clean(detail.one_line_summary())

    def test_unlisted_quota_shaped_code_is_quota_rendered_unknown(self):
        detail = SessionCreationFailureDetail.from_http_response(
            429, _body("some_quota_thing")
        )
        assert detail.is_quota_refusal
        assert detail.safe_quota_error_code == "unknown"

    def test_unrelated_code_is_not_quota(self):
        detail = SessionCreationFailureDetail.from_http_response(
            429, _body("rate_limited")
        )
        assert not detail.is_quota_refusal

    def test_quota_substring_on_503_is_not_quota(self):
        detail = SessionCreationFailureDetail.from_http_response(
            503, _body("quota_service_unavailable")
        )
        assert not detail.is_quota_refusal

    def test_quota_substring_on_429_is_quota_rendered_unknown(self):
        detail = SessionCreationFailureDetail.from_http_response(
            429, _body("quota_whatever")
        )
        assert detail.is_quota_refusal
        assert detail.safe_quota_error_code == "unknown"


@pytest.mark.backend_online
class TestNormalPathNewCodes:
    @pytest.mark.parametrize("code", ["plan_limit_exceeded", "usage_limit_exceeded"])
    def test_typed_error_no_fallback_no_text(self, code, caplog):
        exc = CloudServiceError("boom customer text here")
        exc.session_creation_failure = SessionCreationFailureDetail.from_http_response(
            429, _body(code)
        )
        ops = TestGovernedSession._ops(exc)
        with caplog.at_level(logging.DEBUG):
            with patch(
                "traigent.cloud.session_operations.is_backend_offline",
                return_value=False,
            ):
                with pytest.raises(SessionQuotaRefusedError) as info:
                    ops.create_session("fn", {"a": [1, 2]})
        assert info.value.error_code == code
        _assert_clean(str(info.value))
        _assert_clean(" ".join(r.getMessage() for r in caplog.records))

    @pytest.mark.parametrize("code", ["plan_limit_exceeded", "usage_limit_exceeded"])
    def test_api_handler_no_text(self, code, caplog):
        ops = ApiOperations(Mock())
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(CloudServiceError) as info:
                ops._handle_session_error(429, _body(code))
        _assert_clean(str(info.value))
        _assert_clean(" ".join(r.getMessage() for r in caplog.records))
        assert code in " ".join(r.getMessage() for r in caplog.records)


class _Resp:
    def __init__(self, status: int, text: str) -> None:
        self.status = status
        self._text = text

    async def text(self) -> str:
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@pytest.mark.backend_online
class TestInteractivePath:
    @staticmethod
    def _client(status: int, body: str):
        from traigent.cloud.client import TraigentCloudClient

        client = TraigentCloudClient.__new__(TraigentCloudClient)
        session = Mock()
        session.post = Mock(return_value=_Resp(status, body))
        client._aio_session = session
        client.no_egress = False
        client.api_base_url = "http://x"
        client._get_headers = AsyncMock(return_value={})
        client._serialize_session_request = Mock(return_value={})
        return client

    @staticmethod
    def _request():
        from traigent.cloud.models import SessionCreationRequest

        return SessionCreationRequest(
            function_name="fn",
            configuration_space={"a": [1]},
            objectives=["accuracy"],
            dataset_metadata={"size": 1},
            max_trials=1,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("code", ALLOWLISTED_CODES)
    async def test_client_429_typed_and_clean(self, code):
        client = self._client(429, _body(code))
        with pytest.raises(SessionQuotaRefusedError) as info:
            await client.create_optimization_session(self._request())
        assert info.value.error_code == code
        _assert_clean(str(info.value))

    @pytest.mark.asyncio
    async def test_client_non_quota_status_only(self):
        client = self._client(500, "customer text here")
        with pytest.raises(CloudServiceError) as info:
            await client.create_optimization_session(self._request())
        _assert_clean(str(info.value))
        assert "500" in str(info.value)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("code", ALLOWLISTED_CODES)
    async def test_interactive_optimizer_logs_and_raises_clean(self, code, caplog):
        from traigent.optimizers.interactive_optimizer import (
            InteractiveOptimizer,
            RemoteGuidanceService,
        )

        client = self._client(429, _body(code))
        service = Mock(spec=RemoteGuidanceService)

        async def _create(req):
            return await client.create_optimization_session(req)

        service.create_session = _create
        optimizer = InteractiveOptimizer(
            config_space={"a": [1]},
            objectives=["accuracy"],
            remote_service=service,
            dataset_metadata={"size": 1},
        )
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(SessionQuotaRefusedError) as info:
                await optimizer.initialize_session("fn", 1)
        assert info.value.error_code == code
        _assert_clean(str(info.value))
        _assert_clean(" ".join(r.getMessage() for r in caplog.records))


CLOSED_CODES = [
    "typed_storage_mapping_unavailable",
    "director_tokens_required",
    "director_opaque_ids_required",
    "content_identity_keys_unavailable",
]


class TestClosedRefusalClassification:
    """SDK-52: closed backend refusals are typed, never connectivity."""

    @pytest.mark.parametrize("code", CLOSED_CODES)
    def test_closed_code_is_closed_refusal_and_rendered(self, code):
        detail = SessionCreationFailureDetail.from_http_response(503, _body(code))
        assert detail.is_closed_refusal
        assert not detail.is_quota_refusal
        assert detail.safe_error_code == code
        assert code in detail.one_line_summary()
        _assert_clean(detail.one_line_summary())

    def test_unknown_503_code_is_not_closed_refusal(self):
        detail = SessionCreationFailureDetail.from_http_response(
            503, _body("some_new_backend_code")
        )
        assert not detail.is_closed_refusal
        assert detail.safe_error_code == "unknown"

    def test_closed_code_in_message_only_is_not_closed_refusal(self):
        detail = SessionCreationFailureDetail.from_http_response(
            503, '{"message": "director_tokens_required"}'
        )
        assert not detail.is_closed_refusal

    @pytest.mark.parametrize("code", CLOSED_CODES)
    def test_closed_refusal_is_not_connectivity_and_names_code(self, code):
        from traigent.core.execution_policy_runtime import (
            fallback_reason_from_session_result,
            session_failure_is_connectivity,
        )
        from traigent.core.session_types import (
            SessionCreationFailureReason,
            SessionCreationResult,
        )

        result = SessionCreationResult(
            session_id="",
            backend_connected=False,
            failure_reason=SessionCreationFailureReason.SESSION_FAILED,
            failure_response=SessionCreationFailureDetail.from_http_response(
                503, _body(code)
            ),
        )
        assert not session_failure_is_connectivity(result)
        reason = fallback_reason_from_session_result(result)
        assert reason == f"backend refused the session: {code}"
        assert "unreachable" not in reason.lower()

    def test_unknown_503_still_connectivity(self):
        from traigent.core.execution_policy_runtime import (
            session_failure_is_connectivity,
        )
        from traigent.core.session_types import (
            SessionCreationFailureReason,
            SessionCreationResult,
        )

        result = SessionCreationResult(
            session_id="",
            backend_connected=False,
            failure_reason=SessionCreationFailureReason.SESSION_FAILED,
            failure_response=SessionCreationFailureDetail.from_http_response(
                503, _body("some_new_backend_code")
            ),
        )
        assert session_failure_is_connectivity(result)


def _nested_body(code: str) -> str:
    """Director/content-identity envelope: the closed code is under error.code."""
    return f'{{"error": {{"code": "{code}", "message": "customer text here"}}}}'


class TestClosedRefusalEnvelopes:
    """SDK-52: both real backend envelopes carry the closed code structurally."""

    @pytest.mark.parametrize(
        "code",
        ["director_tokens_required", "director_opaque_ids_required"],
    )
    def test_nested_error_code_is_parsed(self, code):
        detail = SessionCreationFailureDetail.from_http_response(
            503, _nested_body(code)
        )
        assert detail.is_closed_refusal
        assert detail.safe_error_code == code
        _assert_clean(detail.one_line_summary())

    def test_nested_code_in_prose_only_is_not_parsed(self):
        detail = SessionCreationFailureDetail.from_http_response(
            503, '{"error": {"message": "director_tokens_required"}}'
        )
        assert not detail.is_closed_refusal

    def test_503_detail_survives_handle_session_error(self):
        ops = ApiOperations(Mock())
        with pytest.raises(CloudServiceError) as info:
            ops._handle_session_error(503, _body("typed_storage_mapping_unavailable"))
        detail = info.value.session_creation_failure
        assert detail.error_code == "typed_storage_mapping_unavailable"
        _assert_clean(str(info.value))


@pytest.mark.backend_online
class TestClosedRefusalNormalPath:
    """SDK-52: a closed refusal stops the run and never creates a local session."""

    @pytest.mark.parametrize("code", CLOSED_CODES)
    def test_no_local_session_and_code_named(self, code):
        exc = CloudServiceError("boom customer text here")
        exc.session_creation_failure = SessionCreationFailureDetail.from_http_response(
            503, _body(code)
        )
        ops = TestGovernedSession._ops(exc)
        with (
            patch(
                "traigent.cloud.session_operations.is_backend_offline",
                return_value=False,
            ),
            patch.object(ops, "_create_local_fallback_session", autospec=True) as local,
        ):
            with pytest.raises(SessionBackendRefusedError) as info:
                ops.create_session("fn", {"a": [1, 2]})
        local.assert_not_called()
        assert info.value.error_code == code
        assert code in str(info.value)
        _assert_clean(str(info.value))
        assert info.value.session_creation_failure.raw_body is None
        assert info.value.session_creation_failure.message is None


@pytest.mark.backend_online
class TestClosedRefusalHybrid:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "body",
        [
            _body("typed_storage_mapping_unavailable"),
            _nested_body("director_tokens_required"),
        ],
    )
    async def test_code_preserved_and_sanitized(self, body):
        ops = _hybrid_ops(503, body)
        with patch("traigent.cloud.session_operations.AIOHTTP_AVAILABLE", True):
            with pytest.raises(SessionBackendRefusedError) as info:
                await ops.create_hybrid_session("p", {"a": [1]}, {"max_trials": 1})
        assert not isinstance(info.value, SessionQuotaRefusedError)
        assert info.value.error_code in CLOSED_CODES
        assert info.value.session_creation_failure.error_code == info.value.error_code
        _assert_clean(str(info.value))


@pytest.mark.backend_online
class TestClosedRefusalInteractive:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "body",
        [
            _body("content_identity_keys_unavailable"),
            _nested_body("director_opaque_ids_required"),
        ],
    )
    async def test_code_preserved_and_sanitized(self, body):
        from traigent.cloud.client import SessionBackendRefusedError as Refused
        from traigent.cloud.models import SessionCreationRequest

        with patch("traigent.cloud.client.AuthManager") as auth_cls:
            auth = AsyncMock()
            auth.get_headers = AsyncMock(
                return_value={"Authorization": "Bearer test-token"}
            )
            auth_cls.return_value = auth
            client = TraigentCloudClient(api_key="test-key")
        client._session = Mock(closed=False)
        client._session.post = Mock(return_value=_FakeResponse(503, body))
        request = SessionCreationRequest(
            problem_type="optimization",
            objectives=["accuracy"],
            configuration_space={"a": [1, 2]},
        )
        with pytest.raises(Refused) as info:
            await client.create_optimization_session(request)
        assert info.value.error_code in CLOSED_CODES
        assert info.value.session_creation_failure.error_code == info.value.error_code
        _assert_clean(str(info.value))
