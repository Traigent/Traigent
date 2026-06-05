"""Tests for Traigent device-code authentication."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from traigent.cli import auth_commands
from traigent.cli.auth_commands import SECURE_CLI_CREDENTIAL_NAME, TraigentAuthCLI
from traigent.security.credentials import CredentialType

API_KEY = "sk_" + "a" * 43  # pragma: allowlist secret
DEVICE_CODE = "A" * 43


class _FakeSecureStore:
    def __init__(self) -> None:
        self.saved: tuple[str, str, CredentialType, dict[str, str] | None] | None = None

    def get(self, name: str, check_env: bool = True) -> str | None:
        assert name == SECURE_CLI_CREDENTIAL_NAME
        assert check_env is False
        return None

    def set(
        self,
        name: str,
        value: str,
        credential_type: CredentialType,
        metadata: dict[str, str] | None = None,
    ) -> None:
        self.saved = (name, value, credential_type, metadata)

    def delete_secure(self, name: str) -> bool:
        assert name == SECURE_CLI_CREDENTIAL_NAME
        return True


class _FakeResponse:
    def __init__(self, status: int, payload: dict[str, Any]) -> None:
        self.status = status
        self.payload = payload
        self.headers: dict[str, str] = {}

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def text(self) -> str:
        return json.dumps(self.payload)


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self.responses = responses
        self.post_calls: list[dict[str, Any]] = []

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.post_calls.append({"url": url, **kwargs})
        if not self.responses:
            raise AssertionError(f"Unexpected POST: {url}")
        return self.responses.pop(0)


def _authorize_payload(interval: int = 1) -> dict[str, Any]:
    return {
        "device_code": DEVICE_CODE,
        "user_code": "BCDF-2345",
        "verification_uri": "https://portal.example.test/device",
        "verification_uri_complete": "https://portal.example.test/device?user_code=BCDF-2345",
        "expires_in": 600,
        "interval": interval,
    }


def _success_payload() -> dict[str, Any]:
    return {
        "success": True,
        "data": {
            "api_key": API_KEY,
            "tenant_id": "tenant_123",
            "project_id": "project_456",
            "user": {"id": "user_1", "email": "dev@example.test"},
            "subscription_tier": "trial",
            "quota": {"trial_limit": 25, "api_call_limit": 1000},
        },
    }


def _error_payload(error: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "success": False,
        "error": error,
        "error_code": error,
    }
    if details is not None:
        payload["details"] = details
    return payload


def _install_device_test_fakes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    responses: list[_FakeResponse],
) -> tuple[_FakeSecureStore, _FakeSession]:
    store = _FakeSecureStore()
    session = _FakeSession(responses)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(auth_commands, "TRAIGENT_CONFIG_DIR", tmp_path / ".traigent")
    monkeypatch.setattr(
        auth_commands, "CREDENTIALS_FILE", tmp_path / ".traigent" / "credentials.json"
    )
    monkeypatch.setattr(auth_commands, "get_secure_credential_store", lambda: store)
    assert auth_commands.aiohttp is not None
    monkeypatch.setattr(
        auth_commands.aiohttp,
        "ClientSession",
        lambda *args, **kwargs: session,
    )
    return store, session


def _make_cli() -> TraigentAuthCLI:
    return TraigentAuthCLI(backend_url_override="https://backend.example.test")


def test_device_flow_happy_path_persists_credentials_and_env_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store, session = _install_device_test_fakes(
        monkeypatch,
        tmp_path,
        [
            _FakeResponse(200, _authorize_payload()),
            _FakeResponse(200, _success_payload()),
        ],
    )

    result = asyncio.run(_make_cli().device_login(sleep=lambda _seconds: _noop_sleep()))

    assert result is True
    assert store.saved is not None
    name, serialized, credential_type, metadata = store.saved
    assert name == SECURE_CLI_CREDENTIAL_NAME
    assert credential_type is CredentialType.TOKEN
    assert metadata == {"source": "traigent-cli"}
    saved = json.loads(serialized)
    assert saved["api_key"] == API_KEY
    assert saved["tenant_id"] == "tenant_123"
    assert saved["project_id"] == "project_456"
    assert saved["backend_url"] == "https://backend.example.test"

    env_file = tmp_path / ".env"
    assert env_file.exists()
    assert env_file.stat().st_mode & 0o777 == 0o600
    env_text = env_file.read_text(encoding="utf-8")
    assert f"TRAIGENT_API_KEY={API_KEY}" in env_text
    assert "TRAIGENT_TENANT_ID=tenant_123" in env_text
    assert "TRAIGENT_PROJECT_ID=project_456" in env_text
    assert "TRAIGENT_BACKEND_URL=https://backend.example.test" in env_text

    assert session.post_calls[0]["url"].endswith("/auth/device/authorize")
    assert session.post_calls[1]["url"].endswith("/auth/device/token")
    assert session.post_calls[1]["json"] == {"device_code": DEVICE_CODE}
    captured = capsys.readouterr()
    assert API_KEY not in captured.out
    assert API_KEY not in captured.err


def test_device_flow_pending_then_slow_down_honors_authoritative_interval(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _store, _session = _install_device_test_fakes(
        monkeypatch,
        tmp_path,
        [
            _FakeResponse(200, _authorize_payload(interval=1)),
            _FakeResponse(400, _error_payload("authorization_pending")),
            _FakeResponse(400, _error_payload("slow_down", {"interval": 7})),
            _FakeResponse(200, _success_payload()),
        ],
    )
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    result = asyncio.run(_make_cli().device_login(sleep=fake_sleep))

    assert result is True
    assert sleeps == [1, 7]


@pytest.mark.parametrize(
    ("error_name", "message"),
    [
        ("access_denied", "denied"),
        ("expired_token", "expired"),
    ],
)
def test_device_flow_terminal_errors_exit_clearly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    error_name: str,
    message: str,
) -> None:
    store, _session = _install_device_test_fakes(
        monkeypatch,
        tmp_path,
        [
            _FakeResponse(200, _authorize_payload()),
            _FakeResponse(400, _error_payload(error_name)),
        ],
    )

    result = asyncio.run(_make_cli().device_login(sleep=lambda _seconds: _noop_sleep()))

    assert result is False
    assert store.saved is None
    assert message in capsys.readouterr().out


def test_device_flow_malformed_success_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store, _session = _install_device_test_fakes(
        monkeypatch,
        tmp_path,
        [
            _FakeResponse(200, _authorize_payload()),
            _FakeResponse(
                200,
                {
                    "success": True,
                    "data": {"api_key": API_KEY, "tenant_id": "tenant_123"},
                },
            ),
        ],
    )

    result = asyncio.run(_make_cli().device_login(sleep=lambda _seconds: _noop_sleep()))

    assert result is False
    assert store.saved is None


async def _noop_sleep() -> None:
    return None
