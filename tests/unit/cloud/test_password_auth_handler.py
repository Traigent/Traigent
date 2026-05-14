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


def test_mock_auth_fallback_still_requires_dev_mode():
    """Mock auth opt-in must not bypass the non-production guard."""
    handler = PasswordAuthHandler()

    with (
        patch.dict("os.environ", {"TRAIGENT_ALLOW_MOCK_PASSWORD_AUTH": "1"}),
        patch.object(handler, "_is_dev_mode_enabled", return_value=False),
    ):
        assert handler._is_mock_auth_fallback_enabled() is False


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
async def test_dev_mode_does_not_fall_back_on_backend_outage_without_opt_in():
    """Dev mode alone should not produce mock tokens for backend failures."""
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
async def test_explicit_mock_auth_opt_in_falls_back_on_backend_outage():
    """Explicit dev mock-auth opt-in may use mock tokens for backend failures."""
    handler = PasswordAuthHandler()
    credentials = {
        "email": "dev@example.com",
        "password": "password123",  # pragma: allowlist secret
    }

    with (
        patch.dict("os.environ", {"TRAIGENT_ALLOW_MOCK_PASSWORD_AUTH": "1"}),
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
async def test_password_auth_rejects_private_backend_url_outside_dev():
    """Production password auth must reject private backend URLs before POSTing."""
    handler = PasswordAuthHandler()
    credentials = {
        "email": "dev@example.com",
        "password": "password123",  # pragma: allowlist secret
    }

    with (
        patch.object(handler, "_is_dev_mode_enabled", return_value=False),
        patch(
            "traigent.config.backend_config.BackendConfig.get_cloud_api_url",
            return_value="http://127.0.0.1:5000/api/v1",
        ),
    ):
        token_data = await handler._perform_authentication(credentials)

    assert token_data is None


@pytest.mark.asyncio
async def test_password_auth_success_uses_validated_backend_url():
    """Successful backend auth should still execute through the validated URL."""
    handler = PasswordAuthHandler()
    credentials = {
        "email": "dev@example.com",
        "password": "password123",  # pragma: allowlist secret
    }
    captured: dict[str, object] = {}

    class _Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def json(self):
            return {
                "success": True,
                "data": {
                    "access_token": "access-token",
                    "refresh_token": "refresh-token",
                    "expires_in": 3600,
                },
            }

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def post(self, url, json, headers, timeout):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            captured["timeout"] = timeout
            return _Response()

    async def _execute(_self, fn, **_kwargs):
        return await fn()

    with (
        patch(
            "traigent.config.backend_config.BackendConfig.get_cloud_api_url",
            return_value="https://api.example.com/api/v1",
        ),
        patch(
            "traigent.cloud.resilient_client.ResilientClient.execute_with_retry",
            new=_execute,
        ),
        patch(
            "traigent.cloud.password_auth_handler.aiohttp.ClientSession", new=_Session
        ),
    ):
        token_data = await handler._perform_authentication(credentials)

    assert token_data is not None
    assert token_data["access_token"] == "access-token"
    assert captured["url"] == "https://api.example.com/api/v1/auth/login"


@pytest.mark.asyncio
async def test_password_auth_backend_error_redacts_response_body():
    """Backend error bodies should not be returned or logged by the handler."""
    handler = PasswordAuthHandler()
    credentials = {
        "email": "dev@example.com",
        "password": "password123",  # pragma: allowlist secret
    }

    class _Response:
        status = 500

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def text(self):
            return "upstream leaked token=secret-value"

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def post(self, *_args, **_kwargs):
            return _Response()

    async def _execute(_self, fn, **_kwargs):
        return await fn()

    with (
        patch(
            "traigent.config.backend_config.BackendConfig.get_cloud_api_url",
            return_value="https://api.example.com/api/v1",
        ),
        patch(
            "traigent.cloud.resilient_client.ResilientClient.execute_with_retry",
            new=_execute,
        ),
        patch(
            "traigent.cloud.password_auth_handler.aiohttp.ClientSession", new=_Session
        ),
    ):
        token_data = await handler._perform_authentication(credentials)

    assert token_data is None
