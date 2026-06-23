"""Tests for BackendConfig.get_cloud_web_url() portal/web URL resolution.

Covers the dev split-host fix: experiment view URLs must target the portal
frontend origin, not the API backend origin.
"""

# Traceability: CONC-CloudService FUNC-CLOUD-HYBRID REQ-CLOUD-009

from __future__ import annotations

from unittest.mock import patch

from traigent.config.backend_config import BackendConfig


class TestGetCloudWebUrl:
    """BackendConfig.get_cloud_web_url() returns the portal/web frontend origin."""

    def test_no_env_returns_default_prod_url(self) -> None:
        """With no env vars or stored creds the default prod portal URL is returned."""
        with (
            patch.dict(
                "os.environ",
                {},
                clear=True,
            ),
            patch(
                "traigent.cloud.credential_manager.CredentialManager.get_stored_backend_url",
                return_value=None,
            ),
        ):
            result = BackendConfig.get_cloud_web_url()
        assert result == "https://portal.traigent.ai"

    def test_traigent_web_url_env_takes_priority(self) -> None:
        """TRAIGENT_WEB_URL overrides everything else."""
        with patch.dict(
            "os.environ",
            {
                "TRAIGENT_WEB_URL": "https://my-portal.example.com",
                "TRAIGENT_BACKEND_URL": "https://api.traigent.ai",
            },
        ):
            result = BackendConfig.get_cloud_web_url()
        assert result == "https://my-portal.example.com"

    def test_traigent_portal_url_env_used_when_web_url_absent(self) -> None:
        """TRAIGENT_PORTAL_URL is used when TRAIGENT_WEB_URL is not set."""
        with patch.dict(
            "os.environ",
            {
                "TRAIGENT_PORTAL_URL": "https://portal-staging.example.com",
                "TRAIGENT_BACKEND_URL": "https://api-staging.example.com",
            },
        ):
            result = BackendConfig.get_cloud_web_url()
        assert result == "https://portal-staging.example.com"

    def test_api_dev_host_derives_to_portal_dev(self) -> None:
        """api-dev.traigent.ai -> portal-dev.traigent.ai (split-host dev env)."""
        with patch.dict(
            "os.environ",
            {"TRAIGENT_BACKEND_URL": "https://api-dev.traigent.ai"},
        ):
            result = BackendConfig.get_cloud_web_url()
        assert result == "https://portal-dev.traigent.ai"

    def test_api_dot_host_derives_to_portal_dot(self) -> None:
        """api.traigent.ai -> portal.traigent.ai (dot-separated api subdomain)."""
        with patch.dict(
            "os.environ",
            {"TRAIGENT_BACKEND_URL": "https://api.traigent.ai"},
        ):
            result = BackendConfig.get_cloud_web_url()
        assert result == "https://portal.traigent.ai"

    def test_prod_portal_origin_unchanged(self) -> None:
        """https://portal.traigent.ai does not start with 'api' — returned as-is."""
        with patch.dict(
            "os.environ",
            {"TRAIGENT_BACKEND_URL": "https://portal.traigent.ai"},
        ):
            result = BackendConfig.get_cloud_web_url()
        assert result == "https://portal.traigent.ai"

    def test_non_api_custom_origin_returned_unchanged(self) -> None:
        """A custom origin without an 'api' prefix is returned without rewriting."""
        with patch.dict(
            "os.environ",
            {"TRAIGENT_BACKEND_URL": "https://backend.internal.example.com"},
        ):
            result = BackendConfig.get_cloud_web_url()
        assert result == "https://backend.internal.example.com"

    def test_derive_web_origin_bare_api_host(self) -> None:
        """Host exactly equal to 'api' is rewritten to 'portal'."""
        result = BackendConfig._derive_web_origin_from_api_origin("https://api")
        # urlunparse will not include a trailing slash for empty path
        assert result is not None
        assert "portal" in result
        assert "api" not in result.split("://", 1)[1]

    def test_derive_web_origin_none_returns_none(self) -> None:
        """None input returns None — no crash."""
        assert BackendConfig._derive_web_origin_from_api_origin(None) is None

    def test_uppercase_api_dot_host_derives_to_portal(self) -> None:
        """API.traigent.ai (uppercase) -> portal.traigent.ai (case-insensitive host)."""
        result = BackendConfig._derive_web_origin_from_api_origin(
            "https://API.traigent.ai"
        )
        assert result == "https://portal.traigent.ai"

    def test_uppercase_api_dev_host_derives_to_portal_dev(self) -> None:
        """API-DEV.traigent.ai (uppercase) -> portal-dev.traigent.ai."""
        result = BackendConfig._derive_web_origin_from_api_origin(
            "https://API-DEV.traigent.ai"
        )
        assert result == "https://portal-dev.traigent.ai"

    def test_api_host_with_port_preserves_port(self) -> None:
        """A port on the api host is preserved on the derived portal host."""
        result = BackendConfig._derive_web_origin_from_api_origin(
            "https://api-dev.traigent.ai:8443"
        )
        assert result == "https://portal-dev.traigent.ai:8443"

    def test_traigent_api_url_origin_also_derives_portal(self) -> None:
        """TRAIGENT_API_URL host origin is used for derivation when BACKEND_URL absent."""
        with (
            patch.dict(
                "os.environ",
                {"TRAIGENT_API_URL": "https://api-staging.traigent.ai/api/v1"},
                clear=True,
            ),
            patch(
                "traigent.cloud.credential_manager.CredentialManager.get_stored_backend_url",
                return_value=None,
            ),
        ):
            result = BackendConfig.get_cloud_web_url()
        assert result == "https://portal-staging.traigent.ai"
