"""End-to-end regression test for unified authentication flows."""

import asyncio
import time

import pytest

from traigent.cloud.auth import (
    AuthManager,
    AuthResult,
    AuthStatus,
    UnifiedAuthConfig,
)
from traigent.cloud.password_auth_handler import PasswordAuthHandler


@pytest.mark.asyncio
async def test_auth_flow_end_to_end(monkeypatch):
    """Authenticate, fetch headers, force refresh, and verify resiliency."""
    config = UnifiedAuthConfig(auto_refresh=True, cache_credentials=False)
    manager = AuthManager(config=config)

    login_payload = {
        "access_token": "eyJhbGciOiJIUzI1NiJ9.login.payload.signature",
        "refresh_token": "refresh_token_12345678901234567890",
        "expires_in": 3600,
        "user": {"email": "user@example.com"},
    }

    refreshed_payload = {
        "access_token": "eyJhbGciOiJIUzI1NiJ9.refreshed.payload.signature",
        "refresh_token": "rotated_refresh_token_1234567890",
        "expires_in": 7200,
        "user": {"email": "user@example.com"},
    }

    refresh_calls: list[str] = []

    async def fake_login(self, credentials: dict[str, str]):
        assert credentials["email"] == "user@example.com"
        assert credentials["password"] == "correct-password"
        return login_payload

    async def fake_refresh(self) -> AuthResult:
        # Track that refresh was called with the stored refresh token
        if self._refresh_token_secure:
            refresh_calls.append(self._refresh_token_secure.get_value())
        # Increment the stats counter (normally done by _refresh_token)
        self._stats["token_refreshes"] += 1
        token_data = refreshed_payload
        self._store_secure_tokens(token_data)
        updated_credentials = self._build_credentials_from_token_data(token_data)
        self._credentials = updated_credentials
        self._auth_status = AuthStatus.AUTHENTICATED

        if self.config.auto_refresh and updated_credentials.refresh_token:
            self._schedule_token_refresh(updated_credentials)

        headers = {}
        if updated_credentials.jwt_token:
            headers["Authorization"] = f"Bearer {updated_credentials.jwt_token}"

        return AuthResult(
            success=True,
            status=AuthStatus.AUTHENTICATED,
            credentials=updated_credentials,
            headers=headers,
            expires_in=token_data.get("expires_in"),
        )

    monkeypatch.setattr(
        PasswordAuthHandler,
        "_perform_authentication",
        fake_login,
        raising=False,
    )
    monkeypatch.setattr(
        AuthManager,
        "_refresh_token",
        fake_refresh,
        raising=False,
    )

    auth_result = await manager.authenticate(
        {"email": "user@example.com", "password": "correct-password"}
    )
    assert auth_result.success
    assert manager._current_token is not None
    assert manager._refresh_token_secure is not None

    headers = await manager.get_auth_headers()
    assert headers["Authorization"].startswith("Bearer eyJhbGciOiJIUzI1NiJ9.login")

    # Force token expiry so a refresh is required
    manager._current_token._expires_at = time.time() + 1
    await asyncio.sleep(1.1)

    refreshed_headers = await manager.get_auth_headers()
    assert refreshed_headers["Authorization"].endswith("refreshed.payload.signature")
    assert refresh_calls == ["refresh_token_12345678901234567890"]
    assert manager.get_auth_status() == AuthStatus.AUTHENTICATED
    assert manager._stats["token_refreshes"] == 1
