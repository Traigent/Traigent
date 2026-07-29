"""Focused tests for PasswordAuthHandler dev-mode fallback behavior."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from traigent.cloud.auth import InvalidCredentialsError
from traigent.cloud.password_auth_handler import PasswordAuthHandler

pytestmark = pytest.mark.backend_online

# https so it satisfies the production URL guard, and `.test` (RFC 6761) so the
# name is reserved and can never be registered by anyone.
#
# Non-routability is NOT what keeps these tests offline: the fixture below
# replaces url_security's getaddrinfo with a fixed public address precisely so
# the guard's public-address check passes, which defeats `.test` not resolving.
# What keeps them offline is that every test mocks the transport
# (ResilientClient.execute_with_retry), so no socket is ever opened.
PINNED_BACKEND_URL = "https://backend.example.test"


@pytest.fixture(autouse=True)
def _enable_backend_auth(monkeypatch):
    """Most password-auth tests mock backend calls and must bypass CI offline mode."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    # Pin the backend URL: the last ambient input this fixture did not control.
    # get_cloud_backend_url falls through TRAIGENT_BACKEND_URL -> TRAIGENT_API_URL
    # -> stored `traigent auth login` credentials -> the https default, so
    # unsetting is not isolation. TRAIGENT_API_URL is deleted separately because
    # _build_api_url reads it directly and it does NOT sit behind
    # TRAIGENT_BACKEND_URL.
    monkeypatch.setenv("TRAIGENT_BACKEND_URL", PINNED_BACKEND_URL)
    monkeypatch.delenv("TRAIGENT_API_URL", raising=False)

    def _resolve_public_backend(_host, _port, *_args, **_kwargs):
        return [(0, 0, 0, "", ("93.184.216.34", 0))]

    monkeypatch.setattr(
        "traigent.cloud.url_security.socket.getaddrinfo",
        _resolve_public_backend,
    )


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
    """Explicit localhost backend config should enable dev mode only in non-production."""
    handler = PasswordAuthHandler()

    with patch.dict(
        "os.environ",
        {
            "ENVIRONMENT": "development",
            "TRAIGENT_BACKEND_URL": "http://localhost:5000",
        },
        clear=True,
    ):
        assert handler._is_dev_mode_enabled() is True


def test_local_backend_without_non_prod_env_does_not_enable_dev_mode():
    """A localhost backend URL alone must not opt policy code out of production."""
    handler = PasswordAuthHandler()

    with patch.dict(
        "os.environ",
        {"TRAIGENT_BACKEND_URL": "http://localhost:5000"},
        clear=True,
    ):
        assert handler._is_dev_mode_enabled() is False


def test_dev_mode_flag_without_non_prod_env_does_not_enable_dev_mode():
    """TRAIGENT_DEV_MODE alone should fail closed in unknown deployments."""
    handler = PasswordAuthHandler()

    with patch.dict("os.environ", {"TRAIGENT_DEV_MODE": "1"}, clear=True):
        assert handler._is_dev_mode_enabled() is False


def test_dev_mode_flag_with_non_prod_env_enables_dev_mode():
    """TRAIGENT_DEV_MODE can opt in only after an explicit non-production env."""
    handler = PasswordAuthHandler()

    with patch.dict(
        "os.environ",
        {"ENVIRONMENT": "development", "TRAIGENT_DEV_MODE": "1"},
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
async def test_offline_mode_skips_backend_password_auth(monkeypatch):
    """Offline mode should fail closed without attempting backend login."""
    handler = PasswordAuthHandler()
    credentials = {
        "email": "dev@example.com",
        "password": "password123",  # pragma: allowlist secret
    }
    execute = AsyncMock(return_value={"access_token": "should-not-be-used"})

    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    with patch(
        "traigent.cloud.resilient_client.ResilientClient.execute_with_retry",
        new=execute,
    ):
        token_data = await handler._perform_authentication(credentials)

    assert token_data is None
    execute.assert_not_called()


@pytest.mark.asyncio
async def test_no_egress_skips_backend_password_auth(monkeypatch):
    """Runtime no_egress policy should fail closed without backend login."""
    handler = PasswordAuthHandler(no_egress=True)
    credentials = {
        "email": "dev@example.com",
        "password": "password123",  # pragma: allowlist secret
    }
    execute = AsyncMock(return_value={"access_token": "should-not-be-used"})

    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    with patch(
        "traigent.cloud.resilient_client.ResilientClient.execute_with_retry",
        new=execute,
    ):
        token_data = await handler._perform_authentication(credentials)

    assert token_data is None
    execute.assert_not_called()


@pytest.mark.asyncio
async def test_invalid_credentials_propagate_even_in_dev_mode():
    """Wrong credentials should fail loudly instead of returning mock tokens."""
    handler = PasswordAuthHandler()
    credentials = {
        "email": "dev@example.com",
        "password": "password123",  # pragma: allowlist secret
    }

    execute = AsyncMock(side_effect=InvalidCredentialsError("Invalid credentials"))

    with (
        patch.object(handler, "_is_dev_mode_enabled", return_value=True),
        patch(
            "traigent.cloud.resilient_client.ResilientClient.execute_with_retry",
            new=execute,
        ),
    ):
        with pytest.raises(InvalidCredentialsError):
            await handler._perform_authentication(credentials)

    # Reachability: _perform_authentication has two pre-transport exits that both
    # `return None` -- the offline egress guard (cloud_backend_egress_disabled,
    # the #2033 hazard) and the URL guard. Either turns this into a bare
    # "DID NOT RAISE"; asserting the transport ran names the real cause instead.
    execute.assert_awaited_once()


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

    execute = AsyncMock(side_effect=RuntimeError("backend down"))

    with (
        patch.dict("os.environ", {"TRAIGENT_ALLOW_MOCK_PASSWORD_AUTH": "1"}),
        patch.object(handler, "_is_dev_mode_enabled", return_value=True),
        patch(
            "traigent.cloud.resilient_client.ResilientClient.execute_with_retry",
            new=execute,
        ),
    ):
        token_data = await handler._perform_authentication(credentials)

    # Reachability: the mock-token fallback lives in the `except Exception`
    # around execute_with_retry, so it is reachable ONLY after the transport
    # raised. Both pre-transport exits (the offline egress guard and the URL
    # guard) `return None` instead, which the assertion below would catch as a
    # bare None; asserting the transport ran names the cause instead.
    execute.assert_awaited_once()
    assert token_data is not None
    assert token_data["dev_mode"] is True
    assert token_data["user"]["email"] == credentials["email"]


@pytest.mark.asyncio
async def test_stored_cli_credentials_do_not_steer_login_url():
    """Stored `traigent auth login` state must not reach the login URL.

    Regression for #2034. With no env vars set at all,
    ``_get_configured_backend_origin`` falls through to the backend_url saved by
    ``traigent auth login``, so *unsetting* TRAIGENT_BACKEND_URL is not
    isolation -- only pinning it is. This is the one vector the filed issue
    missed: it needs no environment variable to fire.
    """
    from traigent.config.backend_config import BackendConfig

    handler = PasswordAuthHandler()
    credentials = {
        "email": "dev@example.com",
        "password": "password123",  # pragma: allowlist secret
    }
    execute = AsyncMock(side_effect=InvalidCredentialsError("Invalid credentials"))

    # Patch the accessor, not HOME: CREDENTIALS_FILE is resolved at import time,
    # so this must never read the developer's real ~/.traigent.
    with (
        patch(
            "traigent.cloud.credential_manager.CredentialManager.get_stored_backend_url",
            return_value="http://127.0.0.1:5000",
        ),
        patch.object(handler, "_is_dev_mode_enabled", return_value=True),
        patch(
            "traigent.cloud.resilient_client.ResilientClient.execute_with_retry",
            new=execute,
        ),
    ):
        assert BackendConfig.get_cloud_backend_url() == PINNED_BACKEND_URL
        assert BackendConfig.get_cloud_api_url().startswith(PINNED_BACKEND_URL)

        with pytest.raises(InvalidCredentialsError):
            await handler._perform_authentication(credentials)

    # Reachability: pins that the login POST was actually attempted against the
    # pinned URL. Both pre-transport exits of _perform_authentication (the
    # offline egress guard and the URL guard) `return None` without raising, so
    # without this the stored-credential vector would go unexercised.
    execute.assert_awaited_once()


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
async def test_password_auth_rejects_dns_private_backend_before_post(monkeypatch):
    """Password credentials must not POST to a hostname resolving to private IPs."""
    handler = PasswordAuthHandler()
    credentials = {
        "email": "dev@example.com",
        "password": "password123",  # pragma: allowlist secret
    }

    def _resolve_to_private(host, _port, *_args, **_kwargs):
        assert host == "login-rebind.example.test"
        return [(0, 0, 0, "", ("10.0.0.5", 0))]

    monkeypatch.setattr(
        "traigent.cloud.url_security.socket.getaddrinfo",
        _resolve_to_private,
    )
    execute = AsyncMock(return_value={"access_token": "should-not-be-used"})

    with (
        patch(
            "traigent.config.backend_config.BackendConfig.get_cloud_api_url",
            return_value="https://login-rebind.example.test/api/v1",
        ),
        patch(
            "traigent.cloud.resilient_client.ResilientClient.execute_with_retry",
            new=execute,
        ),
    ):
        token_data = await handler._perform_authentication(credentials)

    assert token_data is None
    execute.assert_not_called()


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
        def __init__(self, **_kwargs):
            pass

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
        def __init__(self, **_kwargs):
            pass

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
