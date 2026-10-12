"""#2550 review round 3: quota refusals are typed and sanitized on every path."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, Mock, patch

import pytest

from traigent.cloud.api_operations import ApiOperations
from traigent.cloud.client import CloudServiceError, SessionQuotaRefusedError
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
