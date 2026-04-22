"""Tests for BackendConfig URL/credential resolution and CLI auth payload."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from traigent.config.backend_config import (
    DEFAULT_CLOUD_URL,
    SIGNUP_URL,
    BackendConfig,
    get_no_credentials_hint,
)


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

        assert result == BackendConfig.DEFAULT_PROD_URL


class TestBackendConfigDefaultBehavior:
    """Verify default URL behavior for different environment configurations."""

    def test_no_env_no_creds_defaults_to_cloud(self):
        """Generic backend resolution should use the cloud URL when nothing is configured."""
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "traigent.cloud.credential_manager.CredentialManager.get_stored_backend_url",
                return_value=None,
            ),
        ):
            result = BackendConfig.get_backend_url()

        assert result == BackendConfig.DEFAULT_PROD_URL

    def test_cloud_helpers_default_to_cloud(self):
        """Cloud-facing entry points should default to the portal URL."""
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "traigent.cloud.credential_manager.CredentialManager.get_stored_backend_url",
                return_value=None,
            ),
        ):
            backend_result = BackendConfig.get_cloud_backend_url()
            api_result = BackendConfig.get_cloud_api_url()

        assert backend_result == BackendConfig.DEFAULT_PROD_URL
        assert api_result == f"{BackendConfig.DEFAULT_PROD_URL}/api/v1"

    def test_development_env_still_defaults_to_cloud(self):
        """Development env should not silently override the cloud-first default."""
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

        assert result == BackendConfig.DEFAULT_PROD_URL

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

    def test_cloud_env_warns_without_any_credentials(self):
        """Defaulting backend client config to cloud via env should log a warning."""
        from traigent.cloud.backend_components import BackendClientConfig

        with (
            patch.dict(
                "os.environ",
                {"TRAIGENT_BACKEND_URL": BackendConfig.DEFAULT_PROD_URL},
                clear=True,
            ),
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

    def test_cloud_env_no_warning_with_stored_api_key(self):
        """Should NOT warn when cloud env config is paired with stored API keys."""
        from traigent.cloud.backend_components import BackendClientConfig

        with (
            patch.dict(
                "os.environ",
                {"TRAIGENT_BACKEND_URL": BackendConfig.DEFAULT_PROD_URL},
                clear=True,
            ),
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

    def test_cloud_env_no_warning_with_stored_jwt_credentials(self):
        """JWT-authenticated CLI users should not get a missing-credentials warning."""
        from traigent.cloud.backend_components import BackendClientConfig

        with (
            patch.dict(
                "os.environ",
                {"TRAIGENT_BACKEND_URL": BackendConfig.DEFAULT_PROD_URL},
                clear=True,
            ),
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
    async def test_key_creation_requires_explicit_tenant_id(self):
        """CLI auth must not silently create API keys against a server-default tenant."""
        from traigent.cli.auth_commands import TraigentAuthCLI
        from traigent.cloud.auth import AuthenticationError

        with (
            patch.dict(
                "os.environ", {"TRAIGENT_PROJECT_ID": "project_alpha"}, clear=True
            ),
            patch("aiohttp.ClientSession") as mock_session,
        ):
            cli = object.__new__(TraigentAuthCLI)
            cli.backend_api_url = "http://test/api/v1"
            cli.backend_url = "http://test"

            with pytest.raises(AuthenticationError, match="TRAIGENT_TENANT_ID"):
                await cli._authenticate_with_backend("a@b.com", "pass123")

        mock_session.assert_not_called()

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
            patch.dict(
                "os.environ",
                {
                    "TRAIGENT_TENANT_ID": "tenant_acme",
                    "TRAIGENT_PROJECT_ID": "project_alpha",
                },
                clear=True,
            ),
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
        assert login_call["headers"]["X-Tenant-Id"] == "tenant_acme"

        # Verify key creation request has User-Agent, tenant context, and write permissions.
        key_call = post_calls[1]
        assert key_call["headers"]["User-Agent"] == "Traigent-SDK-CLI/1.0"
        assert key_call["headers"]["X-Tenant-Id"] == "tenant_acme"

        payload = key_call["json"]
        assert payload["project_id"] == "project_alpha"
        assert "permissions" in payload
        perms = payload["permissions"]
        assert "experiment.write" in perms
        assert "session.write" in perms
        assert "write" in perms

        # Verify result
        assert result["api_key"] == "tg_created_key"  # pragma: allowlist secret


class TestCliAuthEnvFileGuard:
    """CLI should only write API keys to a local .env file."""

    def test_resolve_env_file_path_requires_dotenv_name(self, tmp_path):
        from traigent.cli.auth_commands import TraigentAuthCLI

        with patch("pathlib.Path.cwd", return_value=tmp_path):
            with pytest.raises(ValueError, match="must point to a .env file"):
                TraigentAuthCLI._resolve_env_file_path(tmp_path / "secrets.txt")

    def test_resolve_env_file_path_stays_within_cwd(self, tmp_path):
        from traigent.cli.auth_commands import TraigentAuthCLI

        outside = tmp_path.parent / ".env"
        with patch("pathlib.Path.cwd", return_value=tmp_path):
            with pytest.raises(ValueError, match="must remain within the current"):
                TraigentAuthCLI._resolve_env_file_path(outside)


class TestCentralizedCredentialHints:
    """Verify SIGNUP_URL is derived from DEFAULT_CLOUD_URL and appears in warnings."""

    def test_signup_url_is_portal_root(self):
        """SIGNUP_URL must equal the portal base (users register or login there)."""
        assert SIGNUP_URL == DEFAULT_CLOUD_URL

    def test_hint_contains_signup_url(self):
        """get_no_credentials_hint() must include the signup URL."""
        hint = get_no_credentials_hint()
        assert SIGNUP_URL in hint

    def test_class_constant_matches_module_constant(self):
        """BackendConfig.DEFAULT_PROD_URL must equal module-level DEFAULT_CLOUD_URL."""
        assert BackendConfig.DEFAULT_PROD_URL == DEFAULT_CLOUD_URL

    def test_get_api_key_warning_includes_signup_url(self, caplog):
        """get_api_key() warning must contain the signup URL."""
        import logging

        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "traigent.cloud.credential_manager.CredentialManager.get_stored_api_key_only",
                return_value=None,
            ),
            patch("traigent.utils.env_config.is_backend_offline", return_value=False),
            caplog.at_level(logging.WARNING, logger="traigent.config.backend_config"),
        ):
            BackendConfig.get_api_key()

        assert any(SIGNUP_URL in msg for msg in caplog.messages), (
            f"Expected {SIGNUP_URL!r} in warning messages: {caplog.messages}"
        )
