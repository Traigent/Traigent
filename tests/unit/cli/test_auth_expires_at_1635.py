"""SDK #1635: persist a backend-provided key expiry, never an invented one."""

from __future__ import annotations

import io
import json
import types
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest
from rich.console import Console

from traigent.cli import auth_commands
from traigent.cli.auth_commands import STORAGE_SECURE, TraigentAuthCLI
from traigent.cloud.auth import AuthManager
from traigent.config.tenant import TENANT_ENV_VAR

API_KEY = "sk_" + "a" * 43  # pragma: allowlist secret


def _future_iso(days: int = 200) -> str:
    return (datetime.now(UTC) + timedelta(days=days)).isoformat()


def _device_payload(**extra: Any) -> dict[str, Any]:
    return {
        "success": True,
        "message": "Device token issued",
        "data": {
            "api_key": API_KEY,
            "tenant_id": "tenant_123",
            "project_id": "project_456",
            "user": {"id": "user_1", "email": "dev@example.test"},
            "subscription_tier": "free",
            "quota": {"trial_limit": 25, "api_call_limit": 1000},
            **extra,
        },
    }


def _bare_cli() -> TraigentAuthCLI:
    cli = TraigentAuthCLI.__new__(TraigentAuthCLI)
    cli.backend_url = "https://backend.example.test"
    return cli


def _render_status(stored: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.delenv("TRAIGENT_API_KEY", raising=False)
    cli = _bare_cli()
    cli.auth_manager = AuthManager()
    buffer = io.StringIO()
    console = Console(file=buffer, width=200, color_system=None, force_terminal=False)
    with (
        patch.object(cli, "_load_stored_credentials", return_value=stored),
        patch("traigent.cli.auth_commands.console", console),
    ):
        assert cli.status() is True
    return buffer.getvalue()


def test_device_login_stores_expires_at_and_status_shows_it(monkeypatch) -> None:
    expires = _future_iso()
    stored = _bare_cli()._validate_device_token_success(
        _device_payload(expires_at=expires)
    )
    assert stored["expires_at"] == expires
    assert "API Key expiry" in _render_status(stored, monkeypatch)


@pytest.mark.parametrize(
    "extra",
    [{}, {"expires_at": None}, {"expires_at": "not-a-date"}, {"expires_at": 12345}],
)
def test_device_login_without_valid_expires_at_stores_nothing(
    extra, monkeypatch
) -> None:
    stored = _bare_cli()._validate_device_token_success(_device_payload(**extra))
    assert "expires_at" not in stored
    assert "API Key expiry" not in _render_status(stored, monkeypatch)


class _Resp:
    def __init__(self, status: int, payload: dict[str, Any]) -> None:
        self.status = status
        self._payload = payload
        self.headers: dict[str, str] = {}

    async def text(self) -> str:
        return json.dumps(self._payload)


class _Ctx:
    def __init__(self, resp: _Resp) -> None:
        self._resp = resp

    async def __aenter__(self) -> _Resp:
        return self._resp

    async def __aexit__(self, *args: Any) -> None:
        return None


class _Session:
    def __init__(self, responses: list[_Resp]) -> None:
        self._responses = responses

    def post(self, url: str, **kwargs: Any) -> _Ctx:
        return _Ctx(self._responses.pop(0))

    async def __aenter__(self) -> _Session:
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None


async def _password_login(monkeypatch, tmp_path, key_data: dict[str, Any]):
    monkeypatch.setattr(auth_commands, "TRAIGENT_CONFIG_DIR", tmp_path / ".traigent")
    monkeypatch.setattr(
        auth_commands, "CREDENTIALS_FILE", tmp_path / ".traigent" / "credentials.json"
    )
    monkeypatch.setenv("TRAIGENT_AUTH_EMAIL", "user@example.test")
    monkeypatch.setenv("TRAIGENT_AUTH_PASSWORD", "correct-password")
    monkeypatch.setenv(TENANT_ENV_VAR, "tenant_acme")
    responses = [
        _Resp(200, {"success": True, "data": {"access_token": "jwt_value"}}),
        _Resp(
            201, {"data": {"key": "tg_" + "s" * 61, **key_data}}
        ),  # pragma: allowlist secret
    ]
    monkeypatch.setattr(aiohttp, "ClientSession", lambda *a, **k: _Session(responses))
    monkeypatch.setattr(
        aiohttp,
        "ClientTimeout",
        lambda total=None, **kw: types.SimpleNamespace(total=total, **kw),
    )
    saved: list[dict[str, Any]] = []
    cli = TraigentAuthCLI(backend_url_override="https://backend.example.test")
    monkeypatch.setattr(cli, "_check_stored_api_key", AsyncMock(return_value=False))
    monkeypatch.setattr(cli, "_check_env_api_key", AsyncMock(return_value=False))
    monkeypatch.setattr(
        cli, "_save_credentials", lambda c: saved.append(c) or STORAGE_SECURE
    )
    assert await cli.login(non_interactive=True) is True
    return saved[0]


@pytest.mark.asyncio
async def test_password_login_stores_expires_at_and_status_shows_it(
    monkeypatch, tmp_path
) -> None:
    expires = _future_iso()
    saved = await _password_login(monkeypatch, tmp_path, {"expires_at": expires})
    assert saved["expires_at"] == expires
    assert "API Key expiry" in _render_status(saved, monkeypatch)


@pytest.mark.asyncio
@pytest.mark.parametrize("key_data", [{}, {"expires_at": None}, {"expires_at": "x"}])
async def test_password_login_without_valid_expires_at_stores_nothing(
    key_data, monkeypatch, tmp_path
) -> None:
    saved = await _password_login(monkeypatch, tmp_path, key_data)
    assert "expires_at" not in saved
    assert "API Key expiry" not in _render_status(saved, monkeypatch)
