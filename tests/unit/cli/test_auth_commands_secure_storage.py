"""Tests for CLI credential persistence policy."""

from __future__ import annotations

import json

from traigent.cli import auth_commands
from traigent.cli.auth_commands import (
    SECURE_CLI_CREDENTIAL_NAME,
    STORAGE_SECURE,
    TraigentAuthCLI,
)
from traigent.security.credentials import CredentialType
from traigent.utils.exceptions import AuthenticationError


class _FakeSecureStore:
    def __init__(self, payload: str | None = None) -> None:
        self.payload = payload
        self.saved: tuple[str, str, CredentialType, dict[str, str] | None] | None = None
        self.deleted: list[str] = []

    def get(self, name: str, check_env: bool = True) -> str | None:
        assert name == SECURE_CLI_CREDENTIAL_NAME
        assert check_env is False
        return self.payload

    def set(
        self,
        name: str,
        value: str,
        credential_type: CredentialType,
        metadata: dict[str, str] | None = None,
    ) -> None:
        self.saved = (name, value, credential_type, metadata)
        self.payload = value

    def delete_secure(self, name: str) -> bool:
        self.deleted.append(name)
        self.payload = None
        return True


class _RejectingSecureStore(_FakeSecureStore):
    def set(
        self,
        name: str,
        value: str,
        credential_type: CredentialType,
        metadata: dict[str, str] | None = None,
    ) -> None:
        raise AuthenticationError(
            "Credential field 'api_key' appears to be weak or a placeholder"
        )


def _auth_cli(tmp_path) -> TraigentAuthCLI:
    cli = TraigentAuthCLI.__new__(TraigentAuthCLI)
    cli.config_dir = tmp_path
    cli.credentials_file = tmp_path / "credentials.json"
    return cli


def test_save_credentials_uses_secure_store_not_plaintext_file(
    monkeypatch, tmp_path
) -> None:
    """Auth commands must not persist tokens to plaintext credentials.json."""
    store = _FakeSecureStore()
    monkeypatch.setattr(
        "traigent.cli.auth_commands.get_secure_credential_store", lambda: store
    )
    cli = _auth_cli(tmp_path)
    api_field = "api" + "_key"

    storage = cli._save_credentials(
        {
            api_field: "secure-api-key",  # pragma: allowlist secret
            "backend_url": "https://api.example.test",
        }
    )

    assert storage == STORAGE_SECURE
    assert store.saved is not None
    name, serialized, credential_type, metadata = store.saved
    assert name == SECURE_CLI_CREDENTIAL_NAME
    assert credential_type is CredentialType.TOKEN
    assert metadata == {"source": "traigent-cli"}
    assert (
        json.loads(serialized)[api_field] == "secure-api-key"
    )  # pragma: allowlist secret
    assert not cli.credentials_file.exists()


def test_save_credentials_does_not_blame_master_password_for_weak_value(
    monkeypatch, tmp_path
) -> None:
    """Weak-value rejections should not point users at the master password."""
    store = _RejectingSecureStore()
    monkeypatch.setattr(
        "traigent.cli.auth_commands.get_secure_credential_store", lambda: store
    )
    errors: list[str] = []
    monkeypatch.setattr(
        auth_commands.logger,
        "error",
        lambda message, *args: errors.append(message % args),
    )
    cli = _auth_cli(tmp_path)
    api_field = "api" + "_key"

    assert (
        cli._save_credentials({api_field: "your-api-key-here"}) is None
    )  # pragma: allowlist secret
    assert errors
    assert "api_key" in errors[0]
    assert "TRAIGENT_MASTER_PASSWORD" not in errors[0]


def test_load_stored_credentials_ignores_plaintext_without_opt_in(
    monkeypatch, tmp_path
) -> None:
    """Legacy plaintext credentials require an explicit migration opt-in."""
    store = _FakeSecureStore()
    monkeypatch.setattr(
        "traigent.cli.auth_commands.get_secure_credential_store", lambda: store
    )
    cli = _auth_cli(tmp_path)
    cli.config_dir.mkdir(parents=True, exist_ok=True)
    cli.credentials_file.write_text(
        json.dumps({"api_key": "plaintext-api-key"}),  # pragma: allowlist secret
        encoding="utf-8",
    )

    assert cli._load_stored_credentials() is None

    monkeypatch.setenv("TRAIGENT_ALLOW_PLAINTEXT_CREDENTIALS", "true")

    credentials = cli._load_stored_credentials()
    assert credentials is not None
    assert credentials["api_key"] == "plaintext-api-key"  # pragma: allowlist secret


def test_load_stored_credentials_prefers_secure_store(monkeypatch, tmp_path) -> None:
    """Encrypted CLI credentials are the primary persisted source."""
    store = _FakeSecureStore(
        json.dumps(
            {
                "api_key": "secure-api-key",  # pragma: allowlist secret
                "backend_url": "https://api.example.test",
            }
        )
    )
    monkeypatch.setattr(
        "traigent.cli.auth_commands.get_secure_credential_store", lambda: store
    )
    cli = _auth_cli(tmp_path)

    credentials = cli._load_stored_credentials()

    assert credentials is not None
    assert credentials["api_key"] == "secure-api-key"  # pragma: allowlist secret
