"""Tests for CLI password-login rate limiting."""

from __future__ import annotations

import json
import types
from typing import Any
from unittest.mock import AsyncMock

import aiohttp
import pytest

from traigent.cli import auth_commands
from traigent.cli.auth_commands import STORAGE_SECURE, TraigentAuthCLI
from traigent.config.tenant import TENANT_ENV_VAR, TENANT_HEADER_NAME


class _FakeResponse:
    def __init__(self, status: int, payload: dict[str, Any]) -> None:
        self.status = status
        self._payload = payload
        self.headers: dict[str, str] = {}

    async def text(self) -> str:
        return json.dumps(self._payload)


class _FakePostContext:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeResponse:
        return self._response

    async def __aexit__(self, *args: Any) -> None:
        return None


class _FakeSession:
    def __init__(
        self, responses: list[_FakeResponse], post_calls: list[dict[str, Any]]
    ) -> None:
        self._responses = responses
        self._post_calls = post_calls

    def post(self, url: str, **kwargs: Any) -> _FakePostContext:
        self._post_calls.append({"url": url, **kwargs})
        return _FakePostContext(self._responses.pop(0))

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None


@pytest.mark.asyncio
async def test_cli_password_login_uses_rate_limiter_before_backend_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Repeated CLI password-login failures should trigger the shared limiter."""
    monkeypatch.setattr(auth_commands, "TRAIGENT_CONFIG_DIR", tmp_path / ".traigent")
    monkeypatch.setattr(
        auth_commands, "CREDENTIALS_FILE", tmp_path / ".traigent" / "credentials.json"
    )
    monkeypatch.setenv("TRAIGENT_AUTH_EMAIL", "user@example.test")
    monkeypatch.setenv("TRAIGENT_AUTH_PASSWORD", "correct-password")
    monkeypatch.setenv(TENANT_ENV_VAR, "tenant_acme")

    post_calls: list[dict[str, Any]] = []
    responses = [
        _FakeResponse(401, {"success": False, "error": "bad credentials"}),
        _FakeResponse(401, {"success": False, "error": "bad credentials"}),
        _FakeResponse(401, {"success": False, "error": "bad credentials"}),
        _FakeResponse(
            200,
            {
                "success": True,
                "data": {
                    "access_token": "jwt_token_value",  # pragma: allowlist secret
                    "refresh_token": "refresh_token_value",  # pragma: allowlist secret
                },
            },
        ),
        _FakeResponse(201, {"data": {"key": "created_api_key"}}),
    ]

    monkeypatch.setattr(
        aiohttp,
        "ClientSession",
        lambda *args, **kwargs: _FakeSession(responses, post_calls),
    )
    monkeypatch.setattr(
        aiohttp,
        "ClientTimeout",
        lambda total=None, **kwargs: types.SimpleNamespace(total=total, **kwargs),
    )

    sleep_calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr(auth_commands.asyncio, "sleep", fake_sleep)

    cli = TraigentAuthCLI(backend_url_override="https://backend.example.test")
    monkeypatch.setattr(cli, "_check_stored_api_key", AsyncMock(return_value=False))
    monkeypatch.setattr(cli, "_check_env_api_key", AsyncMock(return_value=False))
    monkeypatch.setattr(cli, "_save_credentials", lambda credentials: STORAGE_SECURE)

    assert await cli.login(non_interactive=True) is False
    assert await cli.login(non_interactive=True) is False
    assert await cli.login(non_interactive=True) is False
    assert await cli.login(non_interactive=True) is True

    assert len(sleep_calls) == 1
    assert sleep_calls[0] >= 8.0
    assert cli.auth_manager._password_auth_handler._failed_attempts == 0
    assert [call["url"] for call in post_calls] == [
        "https://backend.example.test/api/v1/auth/login",
        "https://backend.example.test/api/v1/auth/login",
        "https://backend.example.test/api/v1/auth/login",
        "https://backend.example.test/api/v1/auth/login",
        "https://backend.example.test/api/v1/keys",
    ]
    assert all(
        call["headers"][TENANT_HEADER_NAME] == "tenant_acme" for call in post_calls
    )
