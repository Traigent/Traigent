"""Focused tests for PasswordAuthHandler dev-mode fallback behavior."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from traigent.cloud.auth import InvalidCredentialsError
from traigent.cloud.password_auth_handler import PasswordAuthHandler


@pytest.mark.asyncio
async def test_invalid_credentials_propagate_even_in_dev_mode():
    """Wrong credentials should fail loudly instead of returning mock tokens."""
    handler = PasswordAuthHandler()
    credentials = {"email": "dev@example.com", "password": "password123"}  # pragma: allowlist secret

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
    credentials = {"email": "dev@example.com", "password": "password123"}  # pragma: allowlist secret

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
