"""Tests for traigent auth status — env-var credential recognition.

Regression for #1322: auth status reported "Not authenticated" even when
TRAIGENT_API_KEY was set in the environment.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from traigent.cli.auth_commands import TraigentAuthCLI


@pytest.fixture()
def cli() -> TraigentAuthCLI:
    obj = TraigentAuthCLI.__new__(TraigentAuthCLI)
    obj.backend_url = "https://api.traigent.com"
    return obj


def test_status_env_api_key_no_stored_creds(cli, monkeypatch):
    """status() returns True when TRAIGENT_API_KEY is set and no stored creds exist."""
    monkeypatch.setenv("TRAIGENT_API_KEY", "trgnt-sk-test12345678-abcd")  # pragma: allowlist secret
    with patch.object(cli, "_load_stored_credentials", return_value=None), \
         patch("traigent.cli.auth_commands.console"):
        result = cli.status()
    assert result is True


def test_status_no_creds_no_env_var(cli, monkeypatch):
    """status() returns False when neither stored creds nor env var exist."""
    monkeypatch.delenv("TRAIGENT_API_KEY", raising=False)
    with patch.object(cli, "_load_stored_credentials", return_value=None), \
         patch("traigent.cli.auth_commands.console"):
        result = cli.status()
    assert result is False


def test_status_stored_creds_take_precedence(cli, monkeypatch):
    """status() returns True from stored creds (env var also present, both OK)."""
    monkeypatch.setenv("TRAIGENT_API_KEY", "trgnt-sk-test12345678-abcd")  # pragma: allowlist secret
    stored = {
        "user": {"email": "u@example.com", "id": 1},
        "api_key": "stored-key",  # pragma: allowlist secret
        "backend_url": "http://localhost",
    }
    with patch.object(cli, "_load_stored_credentials", return_value=stored), \
         patch("traigent.cli.auth_commands.console"):
        result = cli.status()
    assert result is True
