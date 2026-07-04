"""Regression tests for Traigent#1721: auth commands must not report success
when credential persistence fails.

- ``login`` (password flow) previously returned True even when
  ``_save_credentials`` failed to persist the new credentials.
- ``refresh`` discarded the ``_save_credentials`` return value entirely.
- ``logout`` reported success even when the secure-store deletion could not
  be confirmed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from traigent.cli.auth_commands import STORAGE_SECURE, TraigentAuthCLI
from traigent.security.credentials import SecurityError


class _FakeAuthManager:
    """Minimal auth-manager stub for ``_perform_token_refresh``/``refresh``."""

    async def refresh_authentication(self) -> bool:
        return True

    async def get_auth_headers(self) -> dict[str, str]:
        return {"Authorization": "Bearer new_jwt_token"}


def _bare_cli() -> TraigentAuthCLI:
    cli = TraigentAuthCLI.__new__(TraigentAuthCLI)
    cli.backend_url = "https://api.traigent.test"
    return cli


@pytest.mark.asyncio
async def test_login_returns_false_when_credential_save_fails(monkeypatch):
    """Authentication succeeds but persistence fails -> login() must fail too."""
    cli = _bare_cli()
    monkeypatch.setattr(cli, "_check_stored_api_key", AsyncMock(return_value=False))
    monkeypatch.setattr(cli, "_check_env_api_key", AsyncMock(return_value=False))
    monkeypatch.setattr(
        cli, "_get_user_credentials", lambda *a, **k: ("user@example.test", "pw")
    )
    monkeypatch.setattr(
        cli, "_enforce_password_login_rate_limit", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        cli,
        "_authenticate_with_backend",
        AsyncMock(
            return_value={
                "user": {"email": "user@example.test"},
                "jwt_token": "jwt",  # pragma: allowlist secret
            }
        ),
    )
    monkeypatch.setattr(cli, "_reset_password_login_failures", lambda: None)
    # Simulate secure-storage persistence failure (e.g. no master password).
    monkeypatch.setattr(cli, "_save_credentials", lambda credentials: None)

    result = await cli.login(email="user@example.test", non_interactive=True)

    assert result is False


@pytest.mark.asyncio
async def test_login_returns_true_when_credential_save_succeeds(monkeypatch):
    """Regression guard: the happy path must stay unchanged."""
    cli = _bare_cli()
    monkeypatch.setattr(cli, "_check_stored_api_key", AsyncMock(return_value=False))
    monkeypatch.setattr(cli, "_check_env_api_key", AsyncMock(return_value=False))
    monkeypatch.setattr(
        cli, "_get_user_credentials", lambda *a, **k: ("user@example.test", "pw")
    )
    monkeypatch.setattr(
        cli, "_enforce_password_login_rate_limit", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        cli,
        "_authenticate_with_backend",
        AsyncMock(
            return_value={
                "user": {"email": "user@example.test"},
                "jwt_token": "jwt",  # pragma: allowlist secret
            }
        ),
    )
    monkeypatch.setattr(cli, "_reset_password_login_failures", lambda: None)
    monkeypatch.setattr(cli, "_save_credentials", lambda credentials: STORAGE_SECURE)

    result = await cli.login(email="user@example.test", non_interactive=True)

    assert result is True


@pytest.mark.asyncio
async def test_perform_token_refresh_fails_when_save_fails(monkeypatch):
    """refresh() must not discard the ``_save_credentials`` return value."""
    cli = _bare_cli()
    cli.auth_manager = _FakeAuthManager()
    monkeypatch.setattr(cli, "_save_credentials", lambda credentials: None)

    result = await cli._perform_token_refresh(
        {"jwt_token": "old", "refresh_token": "rt"}  # pragma: allowlist secret
    )

    assert result is False


@pytest.mark.asyncio
async def test_perform_token_refresh_succeeds_when_save_succeeds(monkeypatch):
    """Regression guard: successful persistence still reports success."""
    cli = _bare_cli()
    cli.auth_manager = _FakeAuthManager()
    monkeypatch.setattr(cli, "_save_credentials", lambda credentials: STORAGE_SECURE)

    result = await cli._perform_token_refresh(
        {"jwt_token": "old", "refresh_token": "rt"}  # pragma: allowlist secret
    )

    assert result is True


@pytest.mark.asyncio
async def test_refresh_returns_false_when_persistence_fails(monkeypatch):
    """End-to-end refresh(): a save failure must surface as a failed refresh."""
    cli = _bare_cli()
    cli.auth_manager = _FakeAuthManager()
    monkeypatch.setattr(
        cli,
        "_load_stored_credentials",
        lambda: {"refresh_token": "rt"},  # pragma: allowlist secret
    )
    monkeypatch.setattr(cli, "_save_credentials", lambda credentials: None)

    assert await cli.refresh() is False


def test_clear_credentials_fails_when_existing_secure_credential_cannot_be_deleted(
    monkeypatch, tmp_path
):
    """logout()'s ``_clear_credentials`` must not report success when an
    encrypted credential exists on disk but the secure store cannot be opened
    to delete it (Traigent#1721 g8)."""
    cli = _bare_cli()
    cli.credentials_file = _NoOpPath()

    # Positive evidence a secure credential exists on disk.
    secure_file = tmp_path / "secure_credentials.enc"
    secure_file.write_bytes(b"encrypted-blob")
    monkeypatch.setattr(
        "traigent.cli.auth_commands._secure_credential_store_path",
        lambda: secure_file,
    )

    def _raise_store() -> None:
        raise SecurityError("secure store unavailable")

    monkeypatch.setattr(
        "traigent.cli.auth_commands.get_secure_credential_store", _raise_store
    )

    assert cli._clear_credentials() is False


def test_clear_credentials_succeeds_when_secure_store_unconfigured_and_no_secure_file(
    monkeypatch, tmp_path
):
    """Regression for the false-positive logout failure: plaintext-only creds
    with no master secret. The plaintext file deletes fine, the secure store
    cannot be opened (SecurityError), but no encrypted credential exists on
    disk, so logout must still succeed (Traigent#1721)."""
    cli = _bare_cli()
    cli.credentials_file = _NoOpPath()

    # No secure credential file exists on disk.
    monkeypatch.setattr(
        "traigent.cli.auth_commands._secure_credential_store_path",
        lambda: tmp_path / "secure_credentials.enc",
    )

    def _raise_store() -> None:
        raise SecurityError("A credential-store master secret is required.")

    monkeypatch.setattr(
        "traigent.cli.auth_commands.get_secure_credential_store", _raise_store
    )

    assert cli._clear_credentials() is True


def test_clear_credentials_fails_when_secure_store_delete_raises_oserror(monkeypatch):
    """A filesystem-level failure while deleting an opened store is a real
    failure (the store opened, so master secret was configured)."""
    cli = _bare_cli()
    cli.credentials_file = _NoOpPath()

    class _FailingStore:
        def delete_secure(self, name: str) -> bool:
            raise OSError("disk error")

    monkeypatch.setattr(
        "traigent.cli.auth_commands.get_secure_credential_store",
        lambda: _FailingStore(),
    )

    assert cli._clear_credentials() is False


def test_clear_credentials_succeeds_when_secure_store_delete_confirmed(monkeypatch):
    """Regression guard: a clean delete still reports success."""
    cli = _bare_cli()
    cli.credentials_file = _NoOpPath()

    class _FakeStore:
        def delete_secure(self, name: str) -> bool:
            return True

    monkeypatch.setattr(
        "traigent.cli.auth_commands.get_secure_credential_store", lambda: _FakeStore()
    )

    assert cli._clear_credentials() is True


@pytest.mark.asyncio
async def test_logout_succeeds_for_plaintext_only_creds_without_master_secret(
    monkeypatch, tmp_path
):
    """End-to-end ``logout()`` for legacy plaintext-only credentials with no
    master secret must succeed (Traigent#1721): the plaintext file is removed
    and the absent/unopenable secure store is not treated as a failure."""
    cli = _bare_cli()
    # Legacy plaintext credential present; no api_key means no backend revoke.
    monkeypatch.setattr(
        cli, "_load_stored_credentials", lambda: {"jwt_token": "legacy"}
    )
    plaintext = tmp_path / "credentials.json"
    plaintext.write_text("{}")
    cli.credentials_file = plaintext
    # No secure credential on disk; the secure store cannot be opened.
    monkeypatch.setattr(
        "traigent.cli.auth_commands._secure_credential_store_path",
        lambda: tmp_path / "secure_credentials.enc",
    )

    def _raise_store() -> None:
        raise SecurityError("A credential-store master secret is required.")

    monkeypatch.setattr(
        "traigent.cli.auth_commands.get_secure_credential_store", _raise_store
    )

    assert await cli.logout() is True
    assert not plaintext.exists()


class _NoOpPath:
    """Stand-in for ``self.credentials_file`` that reports "does not exist"."""

    def exists(self) -> bool:
        return False
