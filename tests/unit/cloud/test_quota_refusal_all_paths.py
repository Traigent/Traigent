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
