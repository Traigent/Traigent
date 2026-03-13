"""Focused tests for PasswordAuthHandler dev-mode fallback behavior."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from traigent.cloud.auth import InvalidCredentialsError
from traigent.cloud.password_auth_handler import PasswordAuthHandler


def test_default_backend_no_longer_implies_dev_mode():
    """Cloud auth should not infer dev mode from the generic backend fallback."""
    handler = PasswordAuthHandler()

    with (
        patch.dict("os.environ", {}, clear=True),
        patch(
            "traigent.cloud.credential_manager.CredentialManager.get_stored_backend_url",
            return_value=None,
        ),
    ):
        assert handler._is_dev_mode_enabled() is False


def test_explicit_local_backend_still_enables_dev_mode():
    """Explicit localhost backend configuration should still enable dev mode."""
    handler = PasswordAuthHandler()

    with patch.dict(
        "os.environ",
        {"TRAIGENT_BACKEND_URL": "http://localhost:5000"},
        clear=True,
    ):
        assert handler._is_dev_mode_enabled() is True


@pytest.mark.asyncio
async def test_invalid_credentials_propagate_even_in_dev_mode():
    """Wrong credentials should fail loudly instead of returning mock tokens."""
    handler = PasswordAuthHandler()
    credentials = {
        "email": "dev@example.com",
        "password": "password123",  # pragma: allowlist secret
    }

    with (
        patch.object(handler, "_is_dev_mode_enabled", return_value=True),
        patch(
            "traigent.cloud.resilient_client.ResilientClient.execute_with_retry",
            new=AsyncMock(side_effect=InvalidCredentialsError("Invalid credentials")),
        ),
    ):
        with pytest.raises(InvalidCredentialsError):
            await handler._perform_authentication(credentials)


@pytest.mark.asyncio
async def test_dev_mode_still_falls_back_on_backend_outage():
    """Explicit dev mode may still use mock tokens for non-auth backend failures."""
    handler = PasswordAuthHandler()
    credentials = {
        "email": "dev@example.com",
        "password": "password123",  # pragma: allowlist secret
    }

    with (
        patch.object(handler, "_is_dev_mode_enabled", return_value=True),
        patch(
            "traigent.cloud.resilient_client.ResilientClient.execute_with_retry",
            new=AsyncMock(side_effect=RuntimeError("backend down")),
        ),
    ):
        token_data = await handler._perform_authentication(credentials)

    assert token_data["dev_mode"] is True
    assert token_data["user"]["email"] == credentials["email"]
