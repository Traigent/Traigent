"""Tests for traigent auth status — env-var credential recognition.

Regression for #1322: auth status reported "Not authenticated" even when
TRAIGENT_API_KEY was set in the environment.
"""

from __future__ import annotations

import hashlib
import io
from unittest.mock import patch

import pytest
from rich.console import Console

from traigent.cli.auth_commands import TraigentAuthCLI


@pytest.fixture()
def cli() -> TraigentAuthCLI:
    obj = TraigentAuthCLI.__new__(TraigentAuthCLI)
    obj.backend_url = "https://api.traigent.com"
    return obj


def test_status_env_api_key_no_stored_creds(cli, monkeypatch):
    """status() returns True when TRAIGENT_API_KEY is set and no stored creds exist."""
    monkeypatch.setenv(
        "TRAIGENT_API_KEY", "trgnt-sk-test12345678-abcd"
    )  # pragma: allowlist secret
    with (
        patch.object(cli, "_load_stored_credentials", return_value=None),
        patch("traigent.cli.auth_commands.console"),
    ):
        result = cli.status()
    assert result is True


def test_status_no_creds_no_env_var(cli, monkeypatch):
    """status() returns False when neither stored creds nor env var exist."""
    monkeypatch.delenv("TRAIGENT_API_KEY", raising=False)
    with (
        patch.object(cli, "_load_stored_credentials", return_value=None),
        patch("traigent.cli.auth_commands.console"),
    ):
        result = cli.status()
    assert result is False


def test_status_stored_creds_take_precedence(cli, monkeypatch):
    """status() returns True from stored creds (env var also present, both OK)."""
    monkeypatch.setenv(
        "TRAIGENT_API_KEY", "trgnt-sk-test12345678-abcd"
    )  # pragma: allowlist secret
    stored = {
        "user": {"email": "u@example.com", "id": 1},
        "api_key": "stored-key",  # pragma: allowlist secret
        "backend_url": "http://localhost",
    }
    with (
        patch.object(cli, "_load_stored_credentials", return_value=stored),
        patch("traigent.cli.auth_commands.console"),
    ):
        result = cli.status()
    assert result is True


def _render_status(cli: TraigentAuthCLI) -> tuple[bool, str]:
    """Run status() against a real Rich console and return what it printed."""
    buffer = io.StringIO()
    console = Console(file=buffer, width=200, color_system=None, force_terminal=False)
    with (
        patch.object(cli, "_load_stored_credentials", return_value=None),
        patch("traigent.cli.auth_commands.console", console),
    ):
        result = cli.status()
    return result, buffer.getvalue()


_RARE_LETTERS = "qzxjkv"


def _rare_key_body(length: int) -> str:
    """Key body from letters rare in English, with every third character a digit.

    Each 4-character window then holds at least two of q/z/x/j/k/v, so it cannot
    match the table's own words, URLs or numbers by accident.
    """
    digest = hashlib.sha512(str(length).encode()).digest()
    return "".join(
        str(digest[i] % 10) if i % 3 == 0 else _RARE_LETTERS[digest[i] % 6]
        for i in range(length)
    )


@pytest.mark.parametrize("length", [8, 14, 15, 24, 35, 46])
def test_status_env_api_key_shows_no_key_characters(cli, monkeypatch, length):
    """The environment branch reports that a key is set, never any part of it.

    Any 4-character window of the body fails, so a ``key[:4]...key[-4:]`` mask
    would too.
    """
    prefix = "tg_"
    body = _rare_key_body(length - len(prefix))
    api_key = prefix + body
    monkeypatch.setenv("TRAIGENT_API_KEY", api_key)

    result, output = _render_status(cli)

    assert result is True
    leaked = {
        body[i : i + 4] for i in range(len(body) - 3) if body[i : i + 4] in output
    }
    assert not leaked, f"key fragments in status output: {sorted(leaked)}"
    assert "configured (environment variable)" in output


def test_status_env_key_beats_stored_login(cli, monkeypatch):
    """SDK #2227: with both set, status reports the environment key as active."""
    monkeypatch.setenv("TRAIGENT_API_KEY", "tg_" + "e" * 61)
    stored = {
        "user": {"email": "u@example.com", "id": 1},
        "api_key": "stored-key",  # pragma: allowlist secret
        "backend_url": "http://localhost",
    }
    buffer = io.StringIO()
    console = Console(file=buffer, width=200, color_system=None, force_terminal=False)
    with (
        patch.object(cli, "_load_stored_credentials", return_value=stored),
        patch("traigent.cli.auth_commands.console", console),
    ):
        result = cli.status()
    output = buffer.getvalue()
    assert result is True
    assert "environment variable" in output
    assert "not used" in output
    assert "u@example.com" not in output


def _render_stored(cli: TraigentAuthCLI, stored: dict) -> str:
    buffer = io.StringIO()
    console = Console(file=buffer, width=200, color_system=None, force_terminal=False)
    with (
        patch.object(cli, "_load_stored_credentials", return_value=stored),
        patch("traigent.cli.auth_commands.console", console),
    ):
        assert cli.status() is True
    return buffer.getvalue()


def _stored_with_expiry(delta_days: float | None) -> dict:
    from datetime import UTC, datetime, timedelta

    creds = {
        "user": {"email": "u@example.com", "id": 1},
        "api_key": "tg_" + "s" * 61,  # pragma: allowlist secret
        "backend_url": "http://localhost",
    }
    if delta_days is not None:
        creds["expires_at"] = (
            datetime.now(UTC) + timedelta(days=delta_days)
        ).isoformat()
    return creds


@pytest.fixture()
def cli_with_manager(cli, monkeypatch):
    from traigent.cloud.auth import AuthManager

    monkeypatch.delenv("TRAIGENT_API_KEY", raising=False)
    cli.auth_manager = AuthManager()
    return cli


@pytest.mark.parametrize(
    ("delta_days", "expected"),
    [
        (-2, "expired"),
        (3, "rotate now"),
        (20, "plan a rotation"),
        (200, "expires in"),
    ],
)
def test_status_shows_api_key_rotation_state(cli_with_manager, delta_days, expected):
    """SDK #1635: auth status surfaces the key's expiry/rotation state."""
    output = _render_stored(cli_with_manager, _stored_with_expiry(delta_days))
    assert "API Key expiry" in output
    assert expected in output
    assert "tg_ssss" not in output


def test_status_omits_expiry_when_unknown(cli_with_manager):
    """No recorded expiry: say nothing rather than invent a date."""
    output = _render_stored(cli_with_manager, _stored_with_expiry(None))
    assert "API Key expiry" not in output
