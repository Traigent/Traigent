"""#2832 (login must not delete credentials on an unanswered validation) and
#2833 (logout/status/clear_credentials must not treat a locked store as empty)."""

from __future__ import annotations

import json
from typing import Any

import aiohttp
import pytest

from traigent.cli.auth_commands import (
    SECURE_CLI_CREDENTIAL_NAME,
    TraigentAuthCLI,
)

FAKE_KEY = "tg_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6"  # pragma: allowlist secret


class _FakeStore:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload: str | None = json.dumps(payload)
        self.deleted: list[str] = []

    def get(self, name: str, check_env: bool = True) -> str | None:
        return self.payload

    def delete_secure(self, name: str) -> bool:
        self.deleted.append(name)
        self.payload = None
        return True


class _Response:
    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self._body = body
        self.headers: dict[str, str] = {}

    async def text(self) -> str:
        return self._body

    async def __aenter__(self) -> _Response:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


def _fake_session_class(status: int | None, body: str = "", error: Exception | None = None):
    class _Session:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> _Session:
            return self

        async def __aexit__(self, *exc: object) -> None:
            return None

        def post(self, *args: Any, **kwargs: Any) -> _Response:
            if error is not None:
                raise error
            assert status is not None
            return _Response(status, body)

    return _Session


def _cli(tmp_path, monkeypatch, store: _FakeStore) -> TraigentAuthCLI:
    monkeypatch.setattr(
        "traigent.cli.auth_commands.get_secure_credential_store", lambda: store
    )
    cli = TraigentAuthCLI.__new__(TraigentAuthCLI)
    cli.config_dir = tmp_path
    cli.credentials_file = tmp_path / "credentials.json"
    cli.backend_api_url = "https://api.example.test"
    cli.backend_url = "https://api.example.test"
    cli._backend_url = "https://api.example.test"
    return cli


@pytest.mark.parametrize(
    ("status", "body", "error"),
    [
        (503, "unavailable", None),
        (429, "slow down", None),
        (200, "<html>not json</html>", None),
        (None, "", TimeoutError()),
        (None, "", aiohttp.ClientConnectionError("down")),
    ],
)
def test_unanswered_validation_keeps_stored_credentials(
    tmp_path, monkeypatch, capsys, status, body, error
) -> None:
    import asyncio

    store = _FakeStore({"api_key": FAKE_KEY})
    cli = _cli(tmp_path, monkeypatch, store)
    monkeypatch.setattr("aiohttp.ClientSession", _fake_session_class(status, body, error))

    assert asyncio.run(cli._check_stored_api_key()) is False

    assert store.deleted == []
    assert store.payload is not None
    out = capsys.readouterr().out
    assert "could not validate" in out.lower()
    assert "no longer valid" not in out


@pytest.mark.parametrize(
    ("status", "body"),
    [(401, '{"error":"invalid"}'), (200, '{"valid": false}')],
)
def test_rejected_key_still_clears_credentials(
    tmp_path, monkeypatch, capsys, status, body
) -> None:
    import asyncio

    store = _FakeStore({"api_key": FAKE_KEY})
    cli = _cli(tmp_path, monkeypatch, store)
    monkeypatch.setattr("aiohttp.ClientSession", _fake_session_class(status, body))

    assert asyncio.run(cli._check_stored_api_key()) is False

    assert store.deleted == [SECURE_CLI_CREDENTIAL_NAME]
    assert "no longer valid" in capsys.readouterr().out


def test_env_key_unanswered_says_could_not_validate(
    tmp_path, monkeypatch, capsys
) -> None:
    import asyncio

    cli = _cli(tmp_path, monkeypatch, _FakeStore({}))
    monkeypatch.setenv("TRAIGENT_API_KEY", FAKE_KEY)
    monkeypatch.setattr("aiohttp.ClientSession", _fake_session_class(503, "down"))

    assert asyncio.run(cli._check_env_api_key()) is False
    out = capsys.readouterr().out.lower()
    assert "could not validate" in out
    assert "is invalid" not in out


# ---------------------------------------------------------------- #2833


@pytest.fixture()
def scratch_home(tmp_path, monkeypatch):
    from traigent.security.credentials import _MASTER_SECRET_ENV

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("TRAIGENT_ENVIRONMENT", "development")
    monkeypatch.setenv(_MASTER_SECRET_ENV, "scratch-master-value-0001")
    monkeypatch.delenv("TRAIGENT_API_KEY", raising=False)
    monkeypatch.setattr(
        "traigent.cloud.credential_manager.CREDENTIALS_FILE",
        tmp_path / ".traigent" / "credentials.json",
    )
    cli = TraigentAuthCLI.__new__(TraigentAuthCLI)
    cli.config_dir = tmp_path / ".traigent"
    cli.config_dir.mkdir(parents=True, exist_ok=True)
    cli.credentials_file = cli.config_dir / "credentials.json"
    cli.backend_url = "https://api.example.test"
    cli._backend_url = "https://api.example.test"
    assert cli._save_credentials({"api_key": FAKE_KEY}) is not None
    enc = tmp_path / ".traigent" / "secure_credentials.enc"
    assert enc.exists()
    return cli, enc, _MASTER_SECRET_ENV


def test_logout_with_locked_store_does_not_claim_nothing_stored(
    scratch_home, monkeypatch, capsys
) -> None:
    import asyncio

    cli, enc, master_env = scratch_home
    monkeypatch.delenv(master_env)
    capsys.readouterr()

    ok = asyncio.run(cli.logout())

    out = capsys.readouterr().out
    assert "No stored credentials found" not in out
    assert ok is False
    assert enc.exists()
    assert "TRAIGENT_MASTER_PASSWORD" in out


def test_status_reports_locked_store(scratch_home, monkeypatch, capsys) -> None:
    cli, _enc, master_env = scratch_home
    monkeypatch.delenv(master_env)
    capsys.readouterr()

    assert cli.status() is False
    assert "locked" in capsys.readouterr().out.lower()


def test_no_encrypted_file_is_still_plain_not_authenticated(
    tmp_path, monkeypatch, capsys
) -> None:
    import asyncio

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("TRAIGENT_API_KEY", raising=False)
    cli = TraigentAuthCLI.__new__(TraigentAuthCLI)
    cli.config_dir = tmp_path / ".traigent"
    cli.credentials_file = cli.config_dir / "credentials.json"
    cli._backend_url = "https://api.example.test"

    assert cli.status() is False
    out = capsys.readouterr().out
    assert "Not authenticated" in out
    assert "locked" not in out.lower()
    assert asyncio.run(cli.logout()) is True
    assert "No stored credentials found" in capsys.readouterr().out


def test_credential_manager_clear_removes_encrypted_entry(scratch_home) -> None:
    from traigent.cloud.credential_manager import CredentialManager

    _cli_obj, enc, _ = scratch_home
    assert CredentialManager.is_authenticated() is True

    assert CredentialManager.clear_credentials() is True

    assert CredentialManager.is_authenticated() is False


def test_credential_manager_clear_fails_when_locked_file_remains(
    scratch_home, monkeypatch
) -> None:
    from traigent.cloud.credential_manager import CredentialManager

    _cli_obj, enc, master_env = scratch_home
    monkeypatch.delenv(master_env)

    assert CredentialManager.clear_credentials() is False
    assert enc.exists()
