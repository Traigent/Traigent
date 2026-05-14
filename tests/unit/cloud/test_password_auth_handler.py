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


def test_dev_mode_still_enforces_credential_format():
    """Dev mode must not allow malformed email/password dictionaries."""
    handler = PasswordAuthHandler()

    with patch.object(handler, "_is_dev_mode_enabled", return_value=True):
        assert handler._validate_credentials({"email": "bad", "password": "x"}) is False


def test_validated_backend_api_url_rejects_private_host_outside_dev_mode():
    """Password login must not send credentials to private hosts in production mode."""
    handler = PasswordAuthHandler()

    with patch.object(handler, "_is_dev_mode_enabled", return_value=False):
        with pytest.raises(ValueError, match="host is not allowed"):
            handler._validated_backend_api_url("http://127.0.0.1:5000/api/v1")


def test_validated_backend_api_url_allows_local_host_in_dev_mode():
    """Explicit dev mode may target a local backend."""
    handler = PasswordAuthHandler()

    with patch.object(handler, "_is_dev_mode_enabled", return_value=True):
        assert (
            handler._validated_backend_api_url("http://localhost:5000/api/v1/")
            == "http://localhost:5000/api/v1"
        )


def test_validated_backend_api_url_rejects_embedded_credentials():
    """Backend URL validation should fail before credentials are posted."""
    handler = PasswordAuthHandler()

    with pytest.raises(ValueError, match="must not include credentials"):
        handler._validated_backend_api_url(
            "https://user:pass@portal.traigent.ai/api/v1"  # pragma: allowlist secret
        )


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
async def test_dev_mode_backend_outage_fails_without_mock_opt_in():
    """Dev mode alone must not turn backend outages into authenticated tokens."""
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

    assert token_data is None


@pytest.mark.asyncio
async def test_dev_mode_mock_auth_requires_explicit_opt_in():
    """Mock password auth is allowed only after explicit local opt-in."""
    handler = PasswordAuthHandler()
    credentials = {
        "email": "dev@example.com",
        "password": "password123",  # pragma: allowlist secret
    }

    with (
        patch.dict(
            "os.environ",
            {"TRAIGENT_ALLOW_MOCK_PASSWORD_AUTH": "true"},  # pragma: allowlist secret
            clear=True,
        ),
        patch.object(handler, "_is_dev_mode_enabled", return_value=True),
        patch(
            "traigent.cloud.resilient_client.ResilientClient.execute_with_retry",
            new=AsyncMock(side_effect=RuntimeError("backend down")),
        ),
    ):
        token_data = await handler._perform_authentication(credentials)

    assert token_data is not None
    assert token_data["dev_mode"] is True
    assert token_data["user"]["email"] == credentials["email"]


@pytest.mark.asyncio
async def test_mock_auth_opt_in_is_ignored_outside_dev_mode():
    """The mock-auth escape hatch must still require dev/local mode."""
    handler = PasswordAuthHandler()
    credentials = {
        "email": "dev@example.com",
        "password": "password123",  # pragma: allowlist secret
    }

    with (
        patch.dict(
            "os.environ",
            {"TRAIGENT_ALLOW_MOCK_PASSWORD_AUTH": "true"},  # pragma: allowlist secret
            clear=True,
        ),
        patch.object(handler, "_is_dev_mode_enabled", return_value=False),
        patch(
            "traigent.cloud.resilient_client.ResilientClient.execute_with_retry",
            new=AsyncMock(side_effect=RuntimeError("backend down")),
        ),
    ):
        token_data = await handler._perform_authentication(credentials)

    assert token_data is None
