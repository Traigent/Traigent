"""Tests for explicit development credential fallback rules."""

from __future__ import annotations

import json
import logging
from unittest.mock import patch

import pytest

import traigent.cloud.credential_manager as credential_manager_module
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
    monkeypatch.setenv(
        "TRAIGENT_DEV_API_KEY", "dev-key-xyz"
    )  # pragma: allowlist secret

    with patch.object(CredentialManager, "_load_cli_credentials", return_value=None):
        assert (
            CredentialManager.get_api_key() == "dev-key-xyz"
        )  # pragma: allowlist secret
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
    monkeypatch.setenv(
        "TRAIGENT_DEV_API_KEY", "leaked-dev-key"
    )  # pragma: allowlist secret

    with patch.object(CredentialManager, "_load_cli_credentials", return_value=None):
        assert CredentialManager.get_api_key() is None
        assert CredentialManager.get_credentials() == {}


def test_get_auth_headers_jwt_shaped_dev_key_uses_bearer(monkeypatch) -> None:
    """A dot-separated three-segment dev key is treated as a JWT and emitted
    via Authorization: Bearer rather than X-API-Key. Pins the heuristic in
    `get_auth_headers` for the env-driven dev path."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("TRAIGENT_DEV_MODE", "true")
    monkeypatch.setenv(
        "TRAIGENT_DEV_API_KEY", "header.payload.signature"
    )  # pragma: allowlist secret

    with patch.object(CredentialManager, "_load_cli_credentials", return_value=None):
        headers = CredentialManager.get_auth_headers()

    assert headers == {"Authorization": "Bearer header.payload.signature"}


def test_get_auth_headers_opaque_dev_key_uses_x_api_key(monkeypatch) -> None:
    """An opaque (non-JWT-shaped) dev key uses the X-API-Key header."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("TRAIGENT_DEV_MODE", "true")
    monkeypatch.setenv(
        "TRAIGENT_DEV_API_KEY", "tg_opaque_token_value"
    )  # pragma: allowlist secret

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

    assert (
        CredentialManager.get_api_key() == "secure-api-key"
    )  # pragma: allowlist secret
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

    assert (
        CredentialManager.get_api_key() == "plaintext-api-key"
    )  # pragma: allowlist secret


@pytest.mark.parametrize(
    ("url_env_name", "url_env_value", "expected_backend_url"),
    [
        (
            "TRAIGENT_BACKEND_URL",
            "https://env-backend.example.test",
            "https://env-backend.example.test",
        ),
        (
            "TRAIGENT_API_URL",
            "https://env-api.example.test/api/v1",
            "https://env-api.example.test",
        ),
    ],
)
def test_get_credentials_env_url_and_key_override_stale_stored_credentials(
    monkeypatch,
    tmp_path,
    caplog,
    url_env_name,
    url_env_value,
    expected_backend_url,
) -> None:
    """Full CredentialManager path must prefer env over stale CLI credentials."""
    # Held in a variable (not a URL literal in the membership assertion below) so
    # the log-message check does not trip CodeQL py/incomplete-url-substring-sanitization;
    # this is a log assertion, not URL validation.
    stale_stored_url = "http://stale-localhost.example.test"
    _clear_env(monkeypatch)
    monkeypatch.delenv("TRAIGENT_BACKEND_URL", raising=False)
    monkeypatch.delenv("TRAIGENT_API_URL", raising=False)
    monkeypatch.delenv("TRAIGENT_ALLOW_PLAINTEXT_CREDENTIALS", raising=False)

    config_dir = tmp_path / ".traigent"
    config_dir.mkdir()
    credentials_file = config_dir / "credentials.json"
    credentials_file.write_text(
        json.dumps(
            {
                "api_key": "stale-stored-key",  # pragma: allowlist secret
                "backend_url": stale_stored_url,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(credential_manager_module, "TRAIGENT_CONFIG_DIR", config_dir)
    monkeypatch.setattr(credential_manager_module, "CREDENTIALS_FILE", credentials_file)
    monkeypatch.setattr(
        "traigent.cloud.credential_manager.get_secure_credential_store",
        lambda: _FakeSecureStore(None),
    )

    monkeypatch.setenv("TRAIGENT_ALLOW_PLAINTEXT_CREDENTIALS", "true")
    monkeypatch.setenv("TRAIGENT_API_KEY", "env-api-key")  # pragma: allowlist secret
    monkeypatch.setenv(url_env_name, url_env_value)

    with caplog.at_level(logging.INFO, logger="traigent.config.backend_config"):
        credentials = CredentialManager.get_credentials()

    assert credentials == {
        "api_key": "env-api-key",  # pragma: allowlist secret
        "backend_url": expected_backend_url,
        "source": "environment",
    }
    assert any(
        f"{url_env_name} overrides stored CLI backend_url" in record.message
        and stale_stored_url in record.message
        and expected_backend_url in record.message
        for record in caplog.records
    )
