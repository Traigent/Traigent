"""Strict-bool regression test for ``_validate_api_key_with_backend``.

The backend response body must be parsed strictly: only the literal boolean
``True`` for the ``valid`` key indicates a successful validation. Truthy
non-boolean values (e.g. the string ``"false"``, the int ``1``) MUST be
treated as a rejection so a misconfigured or hostile backend cannot smuggle
a fake-pass through the SDK.

Repro for the prior weakness:

    if isinstance(data, dict) and data.get("valid"):  # truthy check
        return None

A response of ``{"valid": "false"}`` would silently authenticate because the
non-empty string ``"false"`` is truthy in Python. The fix tightens this to
``data.get("valid") is True``.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from traigent.cloud.auth import AuthManager
from traigent.core.backend_session_manager import _classify_session_creation_failure
from traigent.core.session_types import (
    SessionCreationFailureClassification,
    SessionCreationFailureReason,
)


@pytest.fixture(autouse=True)
def _enable_backend_validation(monkeypatch):
    """These tests mock backend validation and must override CI offline mode."""
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    monkeypatch.setenv("ENVIRONMENT", "production")

    def _resolve_public_backend(_host, _port, *_args, **_kwargs):
        return [(0, 0, 0, "", ("93.184.216.34", 0))]

    monkeypatch.setattr(
        "traigent.cloud.url_security.socket.getaddrinfo",
        _resolve_public_backend,
    )


class _FakeResponse:
    """Minimal aiohttp response stub for the success-status path."""

    def __init__(
        self,
        payload,
        *,
        status: int = 200,
        raw_body: str | None = None,
        headers: dict[str, str] | None = None,
    ):
        self.status = status
        self._payload = payload
        self._raw_body = raw_body
        self.headers = headers or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self, content_type=None):  # noqa: ARG002
        return self._payload

    async def text(self):
        if self._raw_body is not None:
            return self._raw_body
        return json.dumps(self._payload)


class _FakeSession:
    """Stub aiohttp.ClientSession returning a configurable JSON payload."""

    last_post_kwargs = None

    def __init__(
        self,
        payload,
        *,
        status: int = 200,
        raw_body: str | None = None,
        response_headers: dict[str, str] | None = None,
    ):
        self._payload = payload
        self._status = status
        self._raw_body = raw_body
        self._response_headers = response_headers

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def post(self, url, headers=None, **kwargs):
        type(self).last_post_kwargs = {
            "url": url,
            "headers": headers,
            **kwargs,
        }
        return _FakeResponse(
            self._payload,
            status=self._status,
            raw_body=self._raw_body,
            headers=self._response_headers,
        )


def _patch_aiohttp(
    payload,
    *,
    status: int = 200,
    raw_body: str | None = None,
    response_headers: dict[str, str] | None = None,
):
    """Patch the aiohttp surface used by ``_validate_api_key_with_backend``."""
    _FakeSession.last_post_kwargs = None
    fake_aiohttp = MagicMock()
    fake_aiohttp.ClientSession = MagicMock(
        return_value=_FakeSession(
            payload,
            status=status,
            raw_body=raw_body,
            response_headers=response_headers,
        )
    )
    fake_aiohttp.ClientTimeout = MagicMock()
    return patch("traigent.cloud.auth.aiohttp", fake_aiohttp)


class _RequestsResponse:
    """Minimal requests response stub for the no-aiohttp fallback path."""

    def __init__(self, status: int, body: bytes, headers: dict[str, str] | None = None):
        self.status_code = status
        self._body = body
        self.headers = headers or {}
        self.text = body.decode("utf-8")

    def json(self):
        return json.loads(self._body.decode("utf-8"))


def _auth_manager_with_public_backend() -> AuthManager:
    """Create an auth manager whose validation URL is not localhost-derived."""
    manager = AuthManager()
    manager.config.backend_base_url = "https://backend.example.test"
    return manager


def _no_egress_auth_manager_with_public_backend() -> AuthManager:
    """Create a no-egress auth manager whose validation URL is public."""
    manager = AuthManager(no_egress=True)
    manager.config.backend_base_url = "https://backend.example.test"
    return manager


def _clear_backend_validation_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove policy env knobs that affect loopback backend validation."""
    for key in (
        "ENVIRONMENT",
        "TRAIGENT_ENV",
        "TRAIGENT_ENVIRONMENT",
        "APP_ENV",
        "FLASK_ENV",
        "TRAIGENT_ALLOW_INSECURE_BACKEND",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.asyncio
async def test_validate_rejects_string_false_payload():
    """A backend returning ``{"valid": "false"}`` must NOT authenticate."""
    manager = _auth_manager_with_public_backend()

    with _patch_aiohttp({"valid": "false"}):
        reason = await manager._validate_api_key_with_backend("tg_" + "x" * 61)

    assert reason is not None, (
        "String 'false' was treated as truthy — strict ``is True`` check missing"
    )
    assert "invalid" in reason.lower() or "reported" in reason.lower()


@pytest.mark.asyncio
async def test_validate_rejects_truthy_int_payload():
    """A backend returning ``{"valid": 1}`` must NOT authenticate (strict bool)."""
    manager = _auth_manager_with_public_backend()

    with _patch_aiohttp({"valid": 1}):
        reason = await manager._validate_api_key_with_backend("tg_" + "x" * 61)

    assert reason is not None
    assert "invalid" in reason.lower() or "reported" in reason.lower()


@pytest.mark.asyncio
async def test_validate_accepts_strict_true_payload():
    """A backend returning the literal ``{"valid": True}`` is the only success."""
    manager = _auth_manager_with_public_backend()

    with _patch_aiohttp({"valid": True}):
        reason = await manager._validate_api_key_with_backend("tg_" + "x" * 61)

    assert reason is None


@pytest.mark.asyncio
async def test_validate_posts_json_payload_to_backend():
    """Backend validation must send a JSON body, not an empty POST."""
    manager = _auth_manager_with_public_backend()
    api_key = "tg_" + "x" * 61  # pragma: allowlist secret

    with _patch_aiohttp({"valid": True}):
        reason = await manager._validate_api_key_with_backend(api_key)

    assert reason is None
    assert _FakeSession.last_post_kwargs is not None
    assert _FakeSession.last_post_kwargs["json"] == {"api_key": api_key}
    assert (
        _FakeSession.last_post_kwargs["headers"]["Content-Type"] == "application/json"
    )


@pytest.mark.asyncio
async def test_validate_respects_no_egress_policy():
    """Runtime no_egress policy must block backend API-key validation."""
    manager = _no_egress_auth_manager_with_public_backend()

    with _patch_aiohttp({"valid": True}) as aiohttp:
        reason = await manager._validate_api_key_with_backend("tg_" + "x" * 61)

    assert reason == "backend egress disabled"
    aiohttp.ClientSession.assert_not_called()


@pytest.mark.asyncio
async def test_validate_uses_stdlib_fallback_when_aiohttp_unavailable():
    """Missing aiohttp must not block API-key validation when the stdlib can call backend."""
    manager = _auth_manager_with_public_backend()
    api_key = "tg_" + "x" * 61  # pragma: allowlist secret

    with (
        patch("traigent.cloud.auth.AIOHTTP_AVAILABLE", False),
        patch(
            "requests.post",
            return_value=_RequestsResponse(200, b'{"valid": true}'),
        ) as post,
    ):
        reason = await manager._validate_api_key_with_backend(api_key)

    assert reason is None
    post.assert_called_once()
    assert post.call_args.kwargs["json"] == {"api_key": api_key}
    assert post.call_args.kwargs["headers"]["Content-Type"] == "application/json"
    assert post.call_args.kwargs["allow_redirects"] is False


@pytest.mark.parametrize(
    ("status", "body", "expected_fragment"),
    [
        (401, '{"error":"Invalid API key"}', "invalid API key"),
        (403, '{"error":"Insufficient scope: experiment.write"}', "insufficient scope"),
    ],
)
def test_backend_validation_http_auth_failures_keep_status_and_body(
    status: int, body: str, expected_fragment: str
):
    """401 and 403 validation failures must stay distinct and keep diagnostics."""
    reason = AuthManager._interpret_backend_key_validation_response(
        status,
        None,
        raw_body=body,
        url="https://backend.example.test/api/v1/keys/validate",
    )

    assert reason is not None
    assert "unauthorized" != reason
    assert f"HTTP {status}" in reason
    assert "https://backend.example.test/api/v1/keys/validate" in reason
    assert body in reason
    assert expected_fragment.lower() in reason.lower()
    assert len(body) <= 200


def test_backend_validation_body_excerpt_is_capped():
    """Non-200 validation reasons include at most a 200-char body excerpt."""
    body = "x" * 250

    reason = AuthManager._interpret_backend_key_validation_response(
        401,
        None,
        raw_body=body,
        url="https://backend.example.test/api/v1/keys/validate",
    )

    assert reason is not None
    assert "x" * 200 in reason
    assert "x" * 201 not in reason


def test_cloudflare_1010_validation_failure_classifies_as_edge_blocked():
    """Cloudflare/WAF markers must not be reported as a bad credential."""
    reason = AuthManager._interpret_backend_key_validation_response(
        403,
        None,
        raw_body="<html>Error code: 1010 browser_signature_banned Cloudflare</html>",
        headers={"cf-ray": "abc123"},
        url="https://backend.example.test/api/v1/keys/validate",
    )

    assert reason is not None
    detail = reason.session_creation_failure
    classification = _classify_session_creation_failure(
        SessionCreationFailureReason.AUTH,
        detail,
        failure_detail=f"API key validation failed: {reason}",
    )

    assert classification == SessionCreationFailureClassification.EDGE_BLOCKED
    assert "edge blocked" in reason.lower()


@pytest.mark.parametrize(
    ("status", "body", "expected_classification"),
    [
        (
            401,
            '{"error":"API key expired"}',
            SessionCreationFailureClassification.EXPIRED_KEY,
        ),
        (
            403,
            '{"error":"Missing required scope experiment.write"}',
            SessionCreationFailureClassification.INSUFFICIENT_SCOPE,
        ),
        (
            401,
            '{"error":"Invalid API key"}',
            SessionCreationFailureClassification.INVALID_OR_REVOKED_KEY,
        ),
    ],
)
def test_backend_validation_failure_classes_map_to_banner_classification(
    status: int,
    body: str,
    expected_classification: SessionCreationFailureClassification,
):
    reason = AuthManager._interpret_backend_key_validation_response(
        status,
        None,
        raw_body=body,
        url="https://backend.example.test/api/v1/keys/validate",
    )

    assert reason is not None
    assert (
        _classify_session_creation_failure(
            SessionCreationFailureReason.AUTH,
            reason.session_creation_failure,
            failure_detail=f"API key validation failed: {reason}",
        )
        == expected_classification
    )


@pytest.mark.asyncio
async def test_validate_uses_sdk_user_agent_and_debug_logs_without_key(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """TRAIGENT_LOG_LEVEL=DEBUG should expose method, URL, and status only."""
    manager = _auth_manager_with_public_backend()
    api_key = "tg_" + "x" * 61  # pragma: allowlist secret
    url = "https://backend.example.test/api/v1/keys/validate"
    monkeypatch.setenv("TRAIGENT_LOG_LEVEL", "DEBUG")

    with (
        _patch_aiohttp(
            None,
            status=401,
            raw_body='{"error":"Invalid API key"}',
        ),
        caplog.at_level(logging.DEBUG, logger="traigent.cloud.auth"),
    ):
        reason = await manager._validate_api_key_with_backend(api_key)

    assert reason is not None
    assert _FakeSession.last_post_kwargs is not None
    assert _FakeSession.last_post_kwargs["headers"]["User-Agent"].startswith(
        "traigent-sdk/"
    )

    debug_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.DEBUG
    ]
    assert any(
        "method=POST" in message and url in message and "status=401" in message
        for message in debug_messages
    )
    assert all(api_key not in message for message in debug_messages)


@pytest.mark.asyncio
async def test_validate_fails_closed_when_requests_fallback_missing():
    """Missing aiohttp and requests must return a validation failure reason."""
    manager = _auth_manager_with_public_backend()

    with (
        patch("traigent.cloud.auth.AIOHTTP_AVAILABLE", False),
        patch.object(
            manager,
            "_validate_api_key_with_backend_sync",
            side_effect=ImportError("requests missing"),
        ),
    ):
        reason = await manager._validate_api_key_with_backend("tg_" + "x" * 61)

    assert reason == "requests library not available for backend validation"


@pytest.mark.asyncio
async def test_validate_rejects_invalid_backend_validation_url():
    """Backend key validation must not send credentials to unsupported URL schemes."""
    manager = AuthManager()
    manager.config.backend_base_url = "ftp://backend.example"

    with (
        patch("traigent.cloud.auth.aiohttp.ClientSession") as session,
        pytest.raises(ValueError, match="must be http\\(s\\) with a host"),
    ):
        await manager._validate_api_key_with_backend("tg_" + "x" * 61)

    session.assert_not_called()


@pytest.mark.asyncio
async def test_validate_rejects_plaintext_backend_validation_url():
    """Backend key validation must not send API keys over plaintext HTTP."""
    manager = AuthManager()
    manager.config.backend_base_url = "http://backend.example.test"

    with (
        patch("traigent.cloud.auth.aiohttp.ClientSession") as session,
        pytest.raises(ValueError, match="must use https in production"),
    ):
        await manager._validate_api_key_with_backend("tg_" + "x" * 61)

    session.assert_not_called()


@pytest.mark.asyncio
async def test_validate_rejects_http_localhost_backend_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loopback HTTP remains denied by default."""
    _clear_backend_validation_env(monkeypatch)
    monkeypatch.setenv("ENVIRONMENT", "production")
    manager = AuthManager()
    manager.config.backend_base_url = "http://localhost:8006"

    with (
        _patch_aiohttp({"valid": True}),
        pytest.raises(ValueError, match="must use https in production"),
    ):
        await manager._validate_api_key_with_backend("tg_" + "x" * 61)

    assert _FakeSession.last_post_kwargs is None


@pytest.mark.asyncio
async def test_validate_rejects_https_localhost_backend_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production backend validation must reject localhost even over HTTPS."""
    _clear_backend_validation_env(monkeypatch)
    monkeypatch.setenv("ENVIRONMENT", "production")
    manager = AuthManager()
    manager.config.backend_base_url = "https://localhost:8006"

    with (
        _patch_aiohttp({"valid": True}),
        pytest.raises(ValueError, match="not allowed in production"),
    ):
        await manager._validate_api_key_with_backend("tg_" + "x" * 61)

    assert _FakeSession.last_post_kwargs is None


@pytest.mark.asyncio
async def test_validate_allows_localhost_backend_in_development(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit development mode may validate API keys against local backends."""
    _clear_backend_validation_env(monkeypatch)
    monkeypatch.setenv("ENVIRONMENT", "development")
    manager = AuthManager()
    manager.config.backend_base_url = "http://localhost:8006"
    api_key = "tg_" + "x" * 61  # pragma: allowlist secret

    with _patch_aiohttp({"valid": True}):
        reason = await manager._validate_api_key_with_backend(api_key)

    assert reason is None
    assert _FakeSession.last_post_kwargs is not None
    assert (
        _FakeSession.last_post_kwargs["url"]
        == "http://localhost:8006/api/v1/keys/validate"
    )
    assert _FakeSession.last_post_kwargs["headers"]["X-API-Key"] == api_key


@pytest.mark.asyncio
async def test_validate_rejects_dns_rebind_hostname_before_sending_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTPS hostnames resolving to loopback must fail before X-API-Key is sent."""
    _clear_backend_validation_env(monkeypatch)
    monkeypatch.setenv("ENVIRONMENT", "production")

    def _resolve_to_loopback(host, _port, *_args, **_kwargs):
        assert host == "rebind.example.test"
        return [(0, 0, 0, "", ("127.0.0.1", 0))]

    monkeypatch.setattr(
        "traigent.cloud.url_security.socket.getaddrinfo",
        _resolve_to_loopback,
    )
    manager = AuthManager()
    manager.config.backend_base_url = "https://rebind.example.test"

    with (
        _patch_aiohttp({"valid": True}),
        pytest.raises(ValueError, match="must not resolve to private"),
    ):
        await manager._validate_api_key_with_backend("tg_" + "x" * 61)

    assert _FakeSession.last_post_kwargs is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "backend_url",
    [
        "https://169.254.169.254/latest/meta-data",
        "https://10.0.0.1",
        "https://127.0.0.1:5000",
        "https://metadata.google.internal",
    ],
)
async def test_validate_rejects_non_global_backend_validation_hosts(backend_url):
    """Backend key validation must not POST credentials to internal hosts."""
    manager = AuthManager()
    manager.config.backend_base_url = backend_url

    with (
        _patch_aiohttp({"valid": True}),
        pytest.raises(ValueError),
    ):
        await manager._validate_api_key_with_backend("tg_" + "x" * 61)

    assert _FakeSession.last_post_kwargs is None
