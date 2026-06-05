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

_ENVIRONMENT_KEYS = ("ENVIRONMENT", "TRAIGENT_ENV", "TRAIGENT_ENVIRONMENT")
_INSECURE_BACKEND_ENV = "TRAIGENT_ALLOW_INSECURE_BACKEND"


@pytest.fixture(autouse=True)
def _enable_backend_validation(monkeypatch):
    """These tests mock backend validation and must override CI offline mode."""
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")


class _FakeResponse:
    """Minimal aiohttp response stub for the success-status path."""

    def __init__(self, payload):
        self.status = 200
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self, content_type=None):  # noqa: ARG002
        return self._payload


class _FakeSession:
    """Stub aiohttp.ClientSession returning a configurable JSON payload."""

    last_post_kwargs = None

    def __init__(self, payload):
        self._payload = payload

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
        return _FakeResponse(self._payload)


def _patch_aiohttp(payload):
    """Patch the aiohttp surface used by ``_validate_api_key_with_backend``."""
    _FakeSession.last_post_kwargs = None
    fake_aiohttp = MagicMock()
    fake_aiohttp.ClientSession = MagicMock(return_value=_FakeSession(payload))
    fake_aiohttp.ClientTimeout = MagicMock()
    return patch("traigent.cloud.auth.aiohttp", fake_aiohttp)


class _RequestsResponse:
    """Minimal requests response stub for the no-aiohttp fallback path."""

    def __init__(self, status: int, body: bytes):
        self.status_code = status
        self._body = body

    def json(self):
        return json.loads(self._body.decode("utf-8"))


def _auth_manager_with_public_backend() -> AuthManager:
    """Create an auth manager whose validation URL is not localhost-derived."""
    manager = AuthManager()
    manager.config.backend_base_url = "https://backend.example.test"
    return manager


def _clear_environment_name(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.mark.asyncio
async def test_validate_rejects_string_false_payload():
    """A backend returning ``{"valid": "false"}`` must NOT authenticate."""
    manager = _auth_manager_with_public_backend()

    with _patch_aiohttp({"valid": "false"}):
        reason = await manager._validate_api_key_with_backend("tg_" + "x" * 61)

    assert (
        reason is not None
    ), "String 'false' was treated as truthy — strict ``is True`` check missing"
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
        _FakeSession.last_post_kwargs["headers"]["Content-Type"]
        == "application/json"
    )


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

    with patch("requests.post") as post:
        reason = await manager._validate_api_key_with_backend("tg_" + "x" * 61)

    assert reason == "backend validation URL is invalid"
    post.assert_not_called()


@pytest.mark.asyncio
async def test_validate_rejects_plaintext_backend_validation_url():
    """Backend key validation must not send API keys over plaintext HTTP."""
    manager = AuthManager()
    manager.config.backend_base_url = "http://backend.example.test"

    with (
        patch("traigent.cloud.auth.AIOHTTP_AVAILABLE", False),
        patch("requests.post") as post,
    ):
        reason = await manager._validate_api_key_with_backend("tg_" + "x" * 61)

    assert reason == "backend validation URL must use HTTPS"
    post.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backend_url", "expected_reason"),
    [
        ("http://169.254.169.254/latest/meta-data", "backend validation URL host is not allowed"),
        ("https://10.0.0.1", "backend validation URL host is not allowed"),
        ("http://127.0.0.1:8006", "backend validation URL host is not allowed"),
        ("http://localhost:8006", "backend validation URL host is not allowed"),
    ],
)
async def test_validate_rejects_non_global_backend_validation_hosts_by_default(
    backend_url,
    expected_reason,
    monkeypatch,
):
    """Backend key validation must not POST credentials to internal hosts."""
    _clear_environment_name(monkeypatch)
    monkeypatch.delenv(_INSECURE_BACKEND_ENV, raising=False)
    manager = AuthManager()
    manager.config.backend_base_url = backend_url

    with (
        patch("traigent.cloud.auth.AIOHTTP_AVAILABLE", False),
        patch("requests.post") as post,
    ):
        reason = await manager._validate_api_key_with_backend("tg_" + "x" * 61)

    assert reason == expected_reason
    post.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "backend_url",
    [
        "http://localhost:8006",
        "http://127.0.0.1:8006",
        "http://10.0.0.1:8006",
    ],
)
async def test_validate_allows_insecure_local_backend_with_non_prod_override(
    backend_url,
    monkeypatch,
    caplog,
):
    """Explicit non-prod override permits local/private dev backends only."""
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv(_INSECURE_BACKEND_ENV, "true")
    manager = AuthManager()
    manager.config.backend_base_url = backend_url

    with (
        caplog.at_level(logging.WARNING, logger="traigent.cloud.auth"),
        patch("traigent.cloud.auth.AIOHTTP_AVAILABLE", False),
        patch(
            "requests.post",
            return_value=_RequestsResponse(200, b'{"valid": true}'),
        ) as post,
    ):
        reason = await manager._validate_api_key_with_backend("tg_" + "x" * 61)

    assert reason is None
    post.assert_called_once()
    assert any(
        "insecure loopback/private backend allowed via "
        "TRAIGENT_ALLOW_INSECURE_BACKEND - dev only" in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backend_url", "expected_reason"),
    [
        ("http://backend.example.test", "backend validation URL must use HTTPS"),
        (
            "http://169.254.169.254/latest/meta-data",
            "backend validation URL host is not allowed",
        ),
    ],
)
async def test_validate_rejects_disallowed_hosts_with_insecure_backend_override(
    backend_url,
    expected_reason,
    monkeypatch,
):
    """The dev override must not become a blanket public-HTTP or SSRF bypass."""
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv(_INSECURE_BACKEND_ENV, "true")
    manager = AuthManager()
    manager.config.backend_base_url = backend_url

    with (
        patch("traigent.cloud.auth.AIOHTTP_AVAILABLE", False),
        patch("requests.post") as post,
    ):
        reason = await manager._validate_api_key_with_backend("tg_" + "x" * 61)

    assert reason == expected_reason
    post.assert_not_called()


@pytest.mark.asyncio
async def test_validate_rejects_insecure_backend_override_in_production(
    monkeypatch,
    caplog,
):
    """Production detection wins even when the dev override env var is set."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv(_INSECURE_BACKEND_ENV, "true")
    manager = AuthManager()
    manager.config.backend_base_url = "http://localhost:8006"

    with (
        caplog.at_level(logging.WARNING, logger="traigent.cloud.auth"),
        patch("traigent.cloud.auth.AIOHTTP_AVAILABLE", False),
        patch("requests.post") as post,
    ):
        reason = await manager._validate_api_key_with_backend("tg_" + "x" * 61)

    assert reason == "backend validation URL host is not allowed"
    post.assert_not_called()
    assert _INSECURE_BACKEND_ENV not in caplog.text


@pytest.mark.asyncio
async def test_validate_allows_non_prod_loopback_without_override(monkeypatch):
    """Explicit non-production loopback remains usable for the documented dev flow."""
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.delenv(_INSECURE_BACKEND_ENV, raising=False)
    manager = AuthManager()
    manager.config.backend_base_url = "http://localhost:8006"

    with (
        patch("traigent.cloud.auth.AIOHTTP_AVAILABLE", False),
        patch(
            "requests.post",
            return_value=_RequestsResponse(200, b'{"valid": true}'),
        ) as post,
    ):
        reason = await manager._validate_api_key_with_backend("tg_" + "x" * 61)

    assert reason is None
    post.assert_called_once()
