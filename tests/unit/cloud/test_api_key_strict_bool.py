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
from unittest.mock import MagicMock, patch

import pytest

from traigent.cloud.auth import AuthManager


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


def test_backend_validation_url_default_denies_http_localhost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loopback HTTP remains denied by default."""
    _clear_backend_validation_env(monkeypatch)

    reason = AuthManager._validate_backend_validation_url(
        "http://localhost:8006/api/v1/keys/validate"
    )

    assert reason is not None
    assert "ENVIRONMENT=development" in reason
    assert "TRAIGENT_ALLOW_INSECURE_BACKEND=true" in reason


def test_backend_validation_url_override_without_environment_still_denies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Policy surface fails closed when the insecure override is set outside explicit dev."""
    _clear_backend_validation_env(monkeypatch)
    monkeypatch.setenv("TRAIGENT_ALLOW_INSECURE_BACKEND", "true")

    reason = AuthManager._validate_backend_validation_url(
        "http://localhost:8006/api/v1/keys/validate"
    )

    assert reason is not None
    assert "TRAIGENT_ALLOW_INSECURE_BACKEND=true" in reason


def test_backend_validation_url_dev_override_allows_loopback_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit non-production plus override allows only loopback HTTP."""
    _clear_backend_validation_env(monkeypatch)
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("TRAIGENT_ALLOW_INSECURE_BACKEND", "true")

    assert (
        AuthManager._validate_backend_validation_url(
            "http://localhost:8006/api/v1/keys/validate"
        )
        is None
    )
    assert (
        AuthManager._validate_backend_validation_url(
            "http://127.42.0.1:8006/api/v1/keys/validate"
        )
        is None
    )


def test_backend_validation_url_dev_override_does_not_allow_non_loopback_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The insecure dev override does not widen private-range or public HTTP URLs."""
    _clear_backend_validation_env(monkeypatch)
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("TRAIGENT_ALLOW_INSECURE_BACKEND", "true")

    assert (
        AuthManager._validate_backend_validation_url(
            "http://192.168.1.10:8006/api/v1/keys/validate"
        )
        == "backend validation URL host is not allowed"
    )
    assert (
        AuthManager._validate_backend_validation_url(
            "http://example.com/api/v1/keys/validate"
        )
        == "backend validation URL must use HTTPS"
    )


def test_backend_validation_url_https_global_hosts_unaffected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTPS global hosts remain valid without the loopback override."""
    _clear_backend_validation_env(monkeypatch)

    assert (
        AuthManager._validate_backend_validation_url(
            "https://example.com/api/v1/keys/validate"
        )
        is None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "backend_url",
    [
        "http://169.254.169.254/latest/meta-data",
        "https://10.0.0.1",
        "http://127.0.0.1:5000",
        "http://localhost:5000",
    ],
)
async def test_validate_rejects_non_global_backend_validation_hosts(backend_url):
    """Backend key validation must not POST credentials to internal hosts."""
    manager = AuthManager()
    manager.config.backend_base_url = backend_url

    with (
        patch("traigent.cloud.auth.AIOHTTP_AVAILABLE", False),
        patch("requests.post") as post,
    ):
        reason = await manager._validate_api_key_with_backend("tg_" + "x" * 61)

    if "127.0.0.1" in backend_url or "localhost" in backend_url:
        assert reason is not None
        assert "TRAIGENT_ALLOW_INSECURE_BACKEND=true" in reason
    else:
        assert reason == "backend validation URL host is not allowed"
    post.assert_not_called()
