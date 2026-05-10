"""Tests for explicit development credential fallback rules."""

from __future__ import annotations

from unittest.mock import patch

from traigent.cloud.credential_manager import CredentialManager


def _clear_env(monkeypatch) -> None:
    for key in (
        "TRAIGENT_API_KEY",
        "TRAIGENT_DEV_API_KEY",
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


def test_dev_mode_without_dev_api_key_returns_none(monkeypatch) -> None:
    """Enabling TRAIGENT_DEV_MODE alone must NOT return a sentinel credential.

    The SDK no longer ships a hard-coded sentinel string fallback
    (Sonar python:S6418); the operator must explicitly set
    ``TRAIGENT_DEV_API_KEY`` for the dev-mode credential path to fire.
    """
    _clear_env(monkeypatch)
    monkeypatch.setenv("TRAIGENT_DEV_MODE", "true")

    with patch.object(CredentialManager, "_load_cli_credentials", return_value=None):
        assert CredentialManager.get_api_key() is None
        assert CredentialManager.get_credentials() == {}


def test_dev_mode_with_dev_api_key_returns_value(monkeypatch) -> None:
    """When TRAIGENT_DEV_API_KEY is set alongside dev mode, return that value."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("TRAIGENT_DEV_MODE", "true")
    monkeypatch.setenv("TRAIGENT_DEV_API_KEY", "dev-key-xyz")  # pragma: allowlist secret

    with patch.object(CredentialManager, "_load_cli_credentials", return_value=None):
        assert CredentialManager.get_api_key() == "dev-key-xyz"  # pragma: allowlist secret
        creds = CredentialManager.get_credentials()

    assert creds["api_key"] == "dev-key-xyz"  # pragma: allowlist secret
    assert creds["source"] == "development"


def test_dev_mode_with_blank_dev_api_key_returns_none(monkeypatch) -> None:
    """Whitespace-only TRAIGENT_DEV_API_KEY is treated as unset."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("TRAIGENT_DEV_MODE", "true")
    monkeypatch.setenv("TRAIGENT_DEV_API_KEY", "   ")

    with patch.object(CredentialManager, "_load_cli_credentials", return_value=None):
        assert CredentialManager.get_api_key() is None
        assert CredentialManager.get_credentials() == {}
