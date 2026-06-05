"""Tests for CLI credential persistence policy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from traigent.cli.auth_commands import (
    SECURE_CLI_CREDENTIAL_NAME,
    STORAGE_SECURE,
    TraigentAuthCLI,
)
from traigent.security.credentials import CredentialType, EnhancedCredentialStore

STRONG_API_KEY = "sk_" + "a" * 43  # pragma: allowlist secret


class _FakeSecureStore:
    def __init__(self, payload: str | None = None) -> None:
        self.payload = payload
        self.saved: tuple[str, str, CredentialType, dict[str, str] | None] | None = None
        self.deleted: list[str] = []
        self.secret_fields: dict[str, str] | None = None

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
        secret_fields: dict[str, str] | None = None,
    ) -> None:
        self.saved = (name, value, credential_type, metadata)
        self.secret_fields = secret_fields
        self.payload = value

    def delete_secure(self, name: str) -> bool:
        self.deleted.append(name)
        self.payload = None
        return True


def _auth_cli(tmp_path) -> TraigentAuthCLI:
    cli = TraigentAuthCLI.__new__(TraigentAuthCLI)
    cli.config_dir = tmp_path
    cli.credentials_file = tmp_path / "credentials.json"
    return cli


def _real_store(tmp_path: Path) -> EnhancedCredentialStore:
    return EnhancedCredentialStore(
        master_password="strong-test-passphrase-12345",  # noqa: S106 - test credential  # pragma: allowlist secret
        storage_path=tmp_path / "secure_credentials.enc",
    )


def test_save_credentials_uses_secure_store_not_plaintext_file(
    monkeypatch, tmp_path
) -> None:
    """Auth commands must not persist tokens to plaintext credentials.json."""
    store = _FakeSecureStore()
    monkeypatch.setattr(
        "traigent.cli.auth_commands.get_secure_credential_store", lambda: store
    )
    cli = _auth_cli(tmp_path)

    storage = cli._save_credentials(
        {
            "api_key": "secure-api-key",  # pragma: allowlist secret
            "backend_url": "https://api.example.test",
        }
    )

    assert storage == STORAGE_SECURE
    assert store.saved is not None
    name, serialized, credential_type, metadata = store.saved
    assert name == SECURE_CLI_CREDENTIAL_NAME
    assert credential_type is CredentialType.TOKEN
    assert metadata == {"source": "traigent-cli"}
    assert store.secret_fields == {"api_key": "secure-api-key"}  # pragma: allowlist secret
    assert json.loads(serialized)["api_key"] == "secure-api-key"  # pragma: allowlist secret
    assert not cli.credentials_file.exists()


@pytest.mark.parametrize(
    "email",
    [
        "admin@dev.local",
        "demo@company.com",
        "dev@example.com",
    ],
)
def test_save_credentials_allows_placeholder_words_in_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    email: str,
) -> None:
    store = _real_store(tmp_path)
    monkeypatch.setattr(
        "traigent.cli.auth_commands.get_secure_credential_store", lambda: store
    )
    cli = _auth_cli(tmp_path)

    storage = cli._save_credentials(
        {
            "api_key": STRONG_API_KEY,
            "tenant_id": "tenant_demo",
            "project_id": "project_example",
            "user": {"id": "user_admin", "email": email},
            "backend_url": "https://api.example.com",
        }
    )

    assert storage == STORAGE_SECURE
    serialized = store.get(SECURE_CLI_CREDENTIAL_NAME, check_env=False)
    assert serialized is not None
    saved = json.loads(serialized)
    assert saved["api_key"] == STRONG_API_KEY
    assert saved["user"]["email"] == email


def test_save_credentials_rejects_weak_api_key_without_master_password_hint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = _real_store(tmp_path)
    monkeypatch.setattr(
        "traigent.cli.auth_commands.get_secure_credential_store", lambda: store
    )
    cli = _auth_cli(tmp_path)

    storage = cli._save_credentials(
        {
            "api_key": "sk_demo_placeholder_value_abcdefghi",  # pragma: allowlist secret
            "user": {"email": "user@safe.local"},
            "backend_url": "https://api.safe.local",
        }
    )
    cli._display_storage_location(storage)

    assert storage is None
    output = capsys.readouterr().out
    assert "api_key" in output
    assert "demo" in output
    assert "TRAIGENT_MASTER_PASSWORD" not in output


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
