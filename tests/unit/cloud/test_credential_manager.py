"""Tests for explicit development credential fallback rules."""

from __future__ import annotations

import json
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


class _FakeSecureStore:
    def __init__(self, payload: str | None) -> None:
        self.payload = payload

    def get(self, name: str, check_env: bool = True) -> str | None:
        assert name == "cli_credentials"
        assert check_env is False
        return self.payload


def test_load_cli_credentials_prefers_secure_store(monkeypatch) -> None:
    """CLI credentials are read from the encrypted credential store."""
    _clear_env(monkeypatch)
    payload = json.dumps(
        {
            "api_key": "secure-api-key",  # pragma: allowlist secret
            "backend_url": "https://api.example.test",
        }
    )
    monkeypatch.setattr(
        "traigent.cloud.credential_manager.get_secure_credential_store",
        lambda: _FakeSecureStore(payload),
    )

    assert CredentialManager.get_api_key() == "secure-api-key"  # pragma: allowlist secret
    assert CredentialManager.get_stored_backend_url() == "https://api.example.test"


def test_legacy_plaintext_cli_credentials_are_ignored_without_opt_in(
    monkeypatch, tmp_path
) -> None:
    """Plaintext credentials must not be loaded unless migration is explicit."""
    _clear_env(monkeypatch)
    credentials_file = tmp_path / "credentials.json"
    credentials_file.write_text(
        json.dumps({"api_key": "plaintext-api-key"}),  # pragma: allowlist secret
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "traigent.cloud.credential_manager.CREDENTIALS_FILE", credentials_file
    )
    monkeypatch.setattr(
        "traigent.cloud.credential_manager.get_secure_credential_store",
        lambda: _FakeSecureStore(None),
    )

    assert CredentialManager.get_api_key() is None

    monkeypatch.setenv("TRAIGENT_ALLOW_PLAINTEXT_CREDENTIALS", "true")

    assert CredentialManager.get_api_key() == "plaintext-api-key"  # pragma: allowlist secret
