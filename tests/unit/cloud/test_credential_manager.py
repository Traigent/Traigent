"""Tests for explicit development credential fallback rules."""

from __future__ import annotations

from unittest.mock import patch

from traigent.cloud.credential_manager import CredentialManager


def _clear_env(monkeypatch) -> None:
    for key in (
        "TRAIGENT_API_KEY",
        "TRAIGENT_DEV_MODE",
        "TRAIGENT_GENERATE_MOCKS",
        "TESTING",
    ):
        monkeypatch.delenv(key, raising=False)


def test_testing_env_does_not_enable_dev_fallback(monkeypatch) -> None:
    """Generic TESTING flags must not unlock development credentials."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("TESTING", "true")

    with patch.object(CredentialManager, "_load_cli_credentials", return_value=None):
        assert CredentialManager.get_api_key() is None
        assert CredentialManager.get_credentials() == {}


def test_explicit_dev_mode_still_enables_dev_fallback(monkeypatch) -> None:
    """Development mode should still expose the dev credential path."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("TRAIGENT_DEV_MODE", "true")

    with patch.object(CredentialManager, "_load_cli_credentials", return_value=None):
        assert CredentialManager.get_api_key() == "test_api_key_for_development"
        creds = CredentialManager.get_credentials()

    assert creds["api_key"] == "test_api_key_for_development"  # pragma: allowlist secret
    assert creds["source"] == "development"
