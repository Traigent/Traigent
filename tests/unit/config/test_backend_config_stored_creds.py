"""Tests for BackendConfig stored credential fallback and CLI auth payload.

Validates that BackendConfig.get_api_key() and get_backend_url() correctly
fall through to CLI-stored credentials when environment variables are absent,
and that the CLI login sends the correct permissions and headers.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from traigent.config.backend_config import BackendConfig


class TestBackendConfigStoredApiKey:
    """BackendConfig.get_api_key() should read stored api_key when env missing."""

    def test_returns_stored_api_key_when_env_missing(self):
        """When TRAIGENT_API_KEY is unset, should load api_key from stored creds."""
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "traigent.cloud.credential_manager.CredentialManager.get_stored_api_key_only",
                return_value="tg_stored_test_key",
            ),
        ):
            result = BackendConfig.get_api_key()

        assert result == "tg_stored_test_key"

    def test_env_var_takes_priority_over_stored(self):
        """TRAIGENT_API_KEY env var should take priority over stored creds."""
        with (
            patch.dict(
                "os.environ",
                {"TRAIGENT_API_KEY": "tg_env_key"},  # pragma: allowlist secret
                clear=True,
            ),
            patch(
                "traigent.cloud.credential_manager.CredentialManager.get_stored_api_key_only",
                return_value="tg_stored_key",
            ),
        ):
            result = BackendConfig.get_api_key()

        assert result == "tg_env_key"

    def test_does_not_return_jwt_when_no_api_key(self):
        """When stored creds have jwt_token but no api_key, should return None."""
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "traigent.cloud.credential_manager.CredentialManager.get_stored_api_key_only",
                return_value=None,
            ),
            patch("traigent.utils.env_config.is_backend_offline", return_value=True),
        ):
            result = BackendConfig.get_api_key()

        assert result is None

    def test_handles_import_error_gracefully(self):
        """If credential_manager import fails, should return None gracefully."""
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.dict("sys.modules", {"traigent.cloud.credential_manager": None}),
            patch("traigent.utils.env_config.is_backend_offline", return_value=True),
        ):
            result = BackendConfig.get_api_key()

        assert result is None


class TestBackendConfigStoredBackendUrl:
    """BackendConfig.get_backend_url() should read stored backend_url when env missing."""

    def test_returns_stored_backend_url_when_env_missing(self):
        """When env vars are unset, should load backend_url from stored creds."""
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "traigent.cloud.credential_manager.CredentialManager.get_stored_backend_url",
                return_value="https://api.traigent.ai",
            ),
        ):
            result = BackendConfig.get_backend_url()

        assert result == "https://api.traigent.ai"

    def test_env_var_takes_priority_over_stored(self):
        """TRAIGENT_BACKEND_URL env var should take priority over stored creds."""
        with (
            patch.dict(
                "os.environ",
                {"TRAIGENT_BACKEND_URL": "https://env.example.com"},
                clear=True,
            ),
            patch(
                "traigent.cloud.credential_manager.CredentialManager.get_stored_backend_url",
                return_value="https://stored.example.com",
            ),
        ):
            result = BackendConfig.get_backend_url()

        assert result == "https://env.example.com"

    def test_falls_back_to_default_when_no_stored_url(self):
        """When no env vars and no stored creds, should return default URL."""
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "traigent.cloud.credential_manager.CredentialManager.get_stored_backend_url",
                return_value=None,
            ),
        ):
            result = BackendConfig.get_backend_url()

        # Default is DEFAULT_PROD_URL (cloud portal)
        assert result == BackendConfig.DEFAULT_PROD_URL


class TestBackendConfigDefaultBehavior:
    """Verify default URL behavior for different environment configurations."""

    def test_no_env_no_creds_defaults_to_cloud(self):
        """External SDK user with no env vars should get cloud portal URL."""
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "traigent.cloud.credential_manager.CredentialManager.get_stored_backend_url",
                return_value=None,
            ),
        ):
            result = BackendConfig.get_backend_url()

        assert result == BackendConfig.DEFAULT_PROD_URL

    def test_development_env_defaults_to_local(self):
        """Internal dev with TRAIGENT_ENV=development should get localhost."""
        with (
            patch.dict(
                "os.environ",
                {"TRAIGENT_ENV": "development"},
                clear=True,
            ),
            patch(
                "traigent.cloud.credential_manager.CredentialManager.get_stored_backend_url",
                return_value=None,
            ),
        ):
            result = BackendConfig.get_backend_url()

        assert "localhost" in result or "127.0.0.1" in result

    def test_explicit_env_var_overrides_everything(self):
        """TRAIGENT_BACKEND_URL should override all defaults."""
        with (
            patch.dict(
                "os.environ",
                {"TRAIGENT_BACKEND_URL": "https://custom.example.com"},
                clear=True,
            ),
            patch(
                "traigent.cloud.credential_manager.CredentialManager.get_stored_backend_url",
                return_value="https://stored.example.com",
            ),
        ):
            result = BackendConfig.get_backend_url()

        assert result == "https://custom.example.com"

    def test_cloud_default_warns_without_any_credentials(self):
        """Defaulting to cloud without any credentials should log a warning."""
        from traigent.cloud.backend_components import BackendClientConfig

        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "traigent.cloud.credential_manager.CredentialManager.get_stored_backend_url",
                return_value=None,
            ),
            patch(
                "traigent.config.backend_config.BackendConfig.has_auth_credentials",
                return_value=False,
            ),
            patch("traigent.cloud.backend_components.logger") as mock_logger,
        ):
            BackendClientConfig()

        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "no credentials found" in warning_msg

    def test_cloud_default_no_warning_with_stored_api_key(self):
        """Should NOT warn when stored API key credentials exist."""
        from traigent.cloud.backend_components import BackendClientConfig

        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "traigent.cloud.credential_manager.CredentialManager.get_stored_backend_url",
                return_value=None,
            ),
            patch(
                "traigent.config.backend_config.BackendConfig.has_auth_credentials",
                return_value=True,
            ),
            patch("traigent.cloud.backend_components.logger") as mock_logger,
        ):
            BackendClientConfig()

        mock_logger.warning.assert_not_called()

    def test_cloud_default_no_warning_with_stored_jwt_credentials(self):
        """JWT-authenticated CLI users should not get a missing-credentials warning."""
        from traigent.cloud.backend_components import BackendClientConfig

        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "traigent.cloud.credential_manager.CredentialManager.get_stored_backend_url",
                return_value=None,
            ),
            patch(
                "traigent.cloud.credential_manager.CredentialManager.get_credentials",
                return_value={"jwt_token": "header.payload.signature"},
            ),
            patch("traigent.cloud.backend_components.logger") as mock_logger,
        ):
            BackendClientConfig()

        mock_logger.warning.assert_not_called()


class TestCliAuthPayload:
    """CLI _authenticate_with_backend() should send write permissions and User-Agent."""

    @pytest.mark.asyncio
    async def test_key_creation_includes_write_permissions_and_user_agent(self):
        """POST /keys payload must include write scopes; both requests need User-Agent."""
        from traigent.cli.auth_commands import TraigentAuthCLI

        # Mock responses for login (200) and key creation (201)
        login_response = AsyncMock()
        login_response.status = 200
        login_response.text = AsyncMock(
            return_value=json.dumps(
                {"success": True, "data": {"access_token": "fake_jwt"}}
            )
        )
        login_response.headers = {}

        key_response = AsyncMock()
        key_response.status = 201
        key_response.text = AsyncMock(
            return_value=json.dumps({"data": {"key": "tg_created_key"}})
        )
        key_response.headers = {}

        # Track all POST calls
        post_calls: list[dict] = []

        class FakeCtxManager:
            def __init__(self, resp):
                self.resp = resp

            async def __aenter__(self):
                return self.resp

            async def __aexit__(self, *args):
                pass

        class FakeSession:
            def __init__(self):
                self._call_idx = 0

            def post(self, url, **kwargs):
                post_calls.append({"url": url, **kwargs})
                resp = login_response if self._call_idx == 0 else key_response
                self._call_idx += 1
                return FakeCtxManager(resp)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        with (
            patch("aiohttp.ClientSession", return_value=FakeSession()),
            patch.object(
                TraigentAuthCLI,
                "__init__",
                lambda self: setattr(self, "backend_api_url", "http://test/api/v1")
                or setattr(self, "backend_url", "http://test"),
            ),
        ):
            cli = object.__new__(TraigentAuthCLI)
            cli.backend_api_url = "http://test/api/v1"
            cli.backend_url = "http://test"
            result = await cli._authenticate_with_backend("a@b.com", "pass123")

        # Verify login request (first POST) has User-Agent
        login_call = post_calls[0]
        assert login_call["headers"]["User-Agent"] == "Traigent-SDK-CLI/1.0"

        # Verify key creation request (second POST) has User-Agent + write permissions
        key_call = post_calls[1]
        assert key_call["headers"]["User-Agent"] == "Traigent-SDK-CLI/1.0"

        payload = key_call["json"]
        assert "permissions" in payload
        perms = payload["permissions"]
        assert "experiment.write" in perms
        assert "session.write" in perms
        assert "write" in perms

        # Verify result
        assert result["api_key"] == "tg_created_key"  # pragma: allowlist secret
