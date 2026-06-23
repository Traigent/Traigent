"""Regression tests for #1439: API-key auth must emit X-API-Key only.

Doctrine (auth.py:878-888, _get_api_key_headers): API-key auth sends the
credential solely via ``X-API-Key``.  ``Authorization: Bearer`` is reserved
for JWT auth and MUST NOT be set when using an API key.

Two former offending sites are covered here:

1. ``traigent/cloud/auth.py`` – the dead dual-header dict in
   ``_authenticate_api_key`` (was lines 1209-1213; removed in #1439).
2. ``traigent/cloud/sync_manager.py`` – the live ``SyncManager.__init__``
   path that set both headers (was lines 111-116; fixed in #1439).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from traigent.cloud.auth import AuthManager, _build_api_key_auth_headers
from traigent.cloud.sync_manager import SyncManager
from traigent.config.types import TraigentConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_API_KEY = "tg_" + "a" * 61  # 64 chars, passes SDK prefix validation  # pragma: allowlist secret


async def _stub_validate_ok(self, api_key):  # noqa: ARG001
    """Stub out network backend-validation so tests stay offline."""
    return None


def _patch_backend_validate_ok():
    return patch.object(
        AuthManager, "_validate_api_key_with_backend", new=_stub_validate_ok
    )


# ---------------------------------------------------------------------------
# _build_api_key_auth_headers unit tests
# ---------------------------------------------------------------------------


def test_build_api_key_auth_headers_sets_x_api_key() -> None:
    """Helper must include X-API-Key."""
    headers = _build_api_key_auth_headers(_VALID_API_KEY)
    assert headers.get("X-API-Key") == _VALID_API_KEY


def test_build_api_key_auth_headers_no_authorization() -> None:
    """Helper must NOT set Authorization for an API key (regression #1439)."""
    headers = _build_api_key_auth_headers(_VALID_API_KEY)
    assert "Authorization" not in headers


def test_build_api_key_auth_headers_empty_on_none() -> None:
    """Empty dict is returned when no key is supplied (fail-safe)."""
    assert _build_api_key_auth_headers(None) == {}


# ---------------------------------------------------------------------------
# AuthManager._authenticate_api_key regression (#1439 site 1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_authenticate_api_key_result_headers_no_authorization() -> None:
    """AuthResult.headers for API-key auth must not contain Authorization.

    Regression for auth.py:1209-1213 where a dual-header dict was built
    with both X-API-Key and Authorization: Bearer <api_key_value>.
    """
    manager = AuthManager(api_key=_VALID_API_KEY)

    with _patch_backend_validate_ok():
        result = await manager.authenticate()

    assert result.success, f"Authentication unexpectedly failed: {result.error_message}"
    assert "Authorization" not in result.headers, (
        "AuthResult.headers must not include Authorization for API-key auth"
    )
    assert result.headers.get("X-API-Key") == _VALID_API_KEY


@pytest.mark.asyncio
async def test_authenticate_api_key_headers_x_api_key_only() -> None:
    """After successful API-key auth the returned headers carry exactly X-API-Key."""
    manager = AuthManager(api_key=_VALID_API_KEY)

    with _patch_backend_validate_ok():
        result = await manager.authenticate()

    auth_keys = [
        k for k in result.headers if k.lower() in ("authorization", "x-api-key")
    ]
    assert auth_keys == ["X-API-Key"], (
        f"Expected only X-API-Key in auth-relevant headers; got {auth_keys}"
    )


# ---------------------------------------------------------------------------
# SyncManager regression (#1439 site 2)
# ---------------------------------------------------------------------------


def _make_sync_manager(api_key: str | None, tmp_path: Path) -> SyncManager:
    config = MagicMock(spec=TraigentConfig)
    config.get_local_storage_path.return_value = str(tmp_path / "storage")
    config.custom_params = {}

    with patch(
        "traigent.cloud.sync_manager.BackendConfig.get_cloud_api_url",
        return_value="https://api.traigent.ai/",
    ):
        return SyncManager(config=config, api_key=api_key)


def test_sync_manager_headers_no_authorization(tmp_path: Path) -> None:
    """SyncManager must NOT set Authorization for an API key (regression #1439).

    Regression for sync_manager.py:111-116 where __init__ built
    {'X-API-Key': api_key, 'Authorization': f'Bearer {api_key}'}.
    """
    sm = _make_sync_manager(_VALID_API_KEY, tmp_path)
    assert "Authorization" not in sm.headers, (
        "SyncManager.headers must not include Authorization for API-key auth"
    )


def test_sync_manager_headers_x_api_key_set(tmp_path: Path) -> None:
    """SyncManager must still include X-API-Key when an API key is given."""
    sm = _make_sync_manager(_VALID_API_KEY, tmp_path)
    assert sm.headers.get("X-API-Key") == _VALID_API_KEY


def test_sync_manager_session_no_authorization_header(tmp_path: Path) -> None:
    """The underlying requests.Session must not carry an Authorization header."""
    sm = _make_sync_manager(_VALID_API_KEY, tmp_path)
    session_headers = dict(sm.session.headers)
    assert "Authorization" not in session_headers, (
        "requests.Session must not carry Authorization for API-key auth"
    )


def test_sync_manager_empty_headers_when_no_api_key(tmp_path: Path) -> None:
    """No headers are set when no API key is supplied."""
    sm = _make_sync_manager(None, tmp_path)
    assert sm.headers == {}
