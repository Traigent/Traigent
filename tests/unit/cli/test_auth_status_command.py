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
