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


def test_dev_api_key_alone_does_not_activate_dev_fallback(monkeypatch) -> None:
    """Setting TRAIGENT_DEV_API_KEY without TRAIGENT_DEV_MODE / GENERATE_MOCKS
    must NOT activate the dev-mode credential path.

    The dev-mode toggle is the gate; the API-key env var only supplies the
    value once the gate is open. Without the gate, the SDK should behave as
    if no credentials are configured.
    """
    _clear_env(monkeypatch)
    monkeypatch.setenv("TRAIGENT_DEV_API_KEY", "leaked-dev-key")  # pragma: allowlist secret

    with patch.object(CredentialManager, "_load_cli_credentials", return_value=None):
        assert CredentialManager.get_api_key() is None
        assert CredentialManager.get_credentials() == {}


def test_get_auth_headers_jwt_shaped_dev_key_uses_bearer(monkeypatch) -> None:
    """A dot-separated three-segment dev key is treated as a JWT and emitted
    via Authorization: Bearer rather than X-API-Key. Pins the heuristic in
    `get_auth_headers` for the env-driven dev path."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("TRAIGENT_DEV_MODE", "true")
    monkeypatch.setenv("TRAIGENT_DEV_API_KEY", "header.payload.signature")  # pragma: allowlist secret

    with patch.object(CredentialManager, "_load_cli_credentials", return_value=None):
        headers = CredentialManager.get_auth_headers()

    assert headers == {"Authorization": "Bearer header.payload.signature"}


def test_get_auth_headers_opaque_dev_key_uses_x_api_key(monkeypatch) -> None:
    """An opaque (non-JWT-shaped) dev key uses the X-API-Key header."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("TRAIGENT_DEV_MODE", "true")
    monkeypatch.setenv("TRAIGENT_DEV_API_KEY", "tg_opaque_token_value")  # pragma: allowlist secret

    with patch.object(CredentialManager, "_load_cli_credentials", return_value=None):
        headers = CredentialManager.get_auth_headers()

    assert headers == {"X-API-Key": "tg_opaque_token_value"}
