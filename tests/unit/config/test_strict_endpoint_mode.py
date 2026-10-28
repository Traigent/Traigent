"""Strict-endpoint (self-hosted) mode must never fall back to a vendor URL.

Covers issue Traigent/Traigent#2770: with TRAIGENT_REQUIRE_EXPLICIT_ENDPOINT=1
or TRAIGENT_DEPLOYMENT_PROFILE=self_hosted, an unset endpoint is a
configuration error and no request is ever made to a traigent.ai host.
"""

# Traceability: CONC-Security FUNC-SECURITY REQ-CLOUD-009

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlparse

import aiohttp
import pytest

from traigent.config.backend_config import (
    DEFAULT_CLOUD_URL,
    BackendConfig,
    EndpointNotConfiguredError,
    get_no_credentials_hint,
    get_signup_url,
    is_strict_endpoint_mode,
)
from traigent.utils.exceptions import ConfigurationError

STORED_URL = (
    "traigent.cloud.credential_manager.CredentialManager.get_stored_backend_url"
)


@pytest.fixture
def strict_env() -> Iterator[None]:
    env = {"TRAIGENT_REQUIRE_EXPLICIT_ENDPOINT": "1"}
    with (
        patch.dict("os.environ", env, clear=True),
        patch(STORED_URL, return_value=None),
    ):
        yield


@pytest.fixture
def default_env() -> Iterator[None]:
    with patch.dict("os.environ", {}, clear=True), patch(STORED_URL, return_value=None):
        yield


class TestEnvParsing:
    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", " Yes "])
    def test_require_explicit_truthy(self, value: str) -> None:
        with patch.dict(
            "os.environ", {"TRAIGENT_REQUIRE_EXPLICIT_ENDPOINT": value}, clear=True
        ):
            assert is_strict_endpoint_mode() is True

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off"])
    def test_require_explicit_falsy(self, value: str) -> None:
        with patch.dict(
            "os.environ", {"TRAIGENT_REQUIRE_EXPLICIT_ENDPOINT": value}, clear=True
        ):
            assert is_strict_endpoint_mode() is False

    @pytest.mark.parametrize("value", ["self_hosted", "SELF_HOSTED", "self-hosted"])
    def test_self_hosted_profile(self, value: str) -> None:
        with patch.dict(
            "os.environ", {"TRAIGENT_DEPLOYMENT_PROFILE": value}, clear=True
        ):
            assert is_strict_endpoint_mode() is True

    def test_other_profile_is_not_strict(self) -> None:
        with patch.dict(
            "os.environ", {"TRAIGENT_DEPLOYMENT_PROFILE": "saas"}, clear=True
        ):
            assert is_strict_endpoint_mode() is False

    def test_unset_is_not_strict(self, default_env: None) -> None:
        assert is_strict_endpoint_mode() is False


class TestUnsetEndpoint:
    def test_error_type_is_configuration_error(self) -> None:
        assert issubclass(EndpointNotConfiguredError, ConfigurationError)

    def test_get_backend_url_raises(self, strict_env: None) -> None:
        with pytest.raises(EndpointNotConfiguredError, match="TRAIGENT_BACKEND_URL"):
            BackendConfig.get_backend_url()

    def test_get_cloud_backend_url_raises(self, strict_env: None) -> None:
        with pytest.raises(EndpointNotConfiguredError):
            BackendConfig.get_cloud_backend_url()

    def test_api_urls_raise(self, strict_env: None) -> None:
        with pytest.raises(EndpointNotConfiguredError):
            BackendConfig.get_backend_api_url()
        with pytest.raises(EndpointNotConfiguredError):
            BackendConfig.get_cloud_api_url()

    def test_profile_also_raises(self) -> None:
        with (
            patch.dict(
                "os.environ", {"TRAIGENT_DEPLOYMENT_PROFILE": "self_hosted"}, clear=True
            ),
            patch(STORED_URL, return_value=None),
            pytest.raises(EndpointNotConfiguredError),
        ):
            BackendConfig.get_backend_url()

    def test_default_mode_still_uses_cloud_default(self, default_env: None) -> None:
        assert BackendConfig.get_backend_url() == DEFAULT_CLOUD_URL
        assert BackendConfig.get_cloud_backend_url() == DEFAULT_CLOUD_URL
        assert BackendConfig.get_cloud_web_url() == DEFAULT_CLOUD_URL


class TestExplicitEndpoint:
    def test_backend_url_env_is_used(self, strict_env: None) -> None:
        with patch.dict(
            "os.environ", {"TRAIGENT_BACKEND_URL": "https://tg.corp.internal"}
        ):
            assert BackendConfig.get_backend_url() == "https://tg.corp.internal"
            assert BackendConfig.get_cloud_backend_url() == "https://tg.corp.internal"
            assert (
                BackendConfig.get_backend_api_url() == "https://tg.corp.internal/api/v1"
            )

    def test_api_url_env_is_used(self, strict_env: None) -> None:
        with patch.dict(
            "os.environ", {"TRAIGENT_API_URL": "https://tg.corp.internal/api/v1"}
        ):
            assert BackendConfig.get_backend_url() == "https://tg.corp.internal"

    def test_stored_credentials_url_counts_as_explicit(self) -> None:
        with (
            patch.dict(
                "os.environ", {"TRAIGENT_REQUIRE_EXPLICIT_ENDPOINT": "1"}, clear=True
            ),
            patch(STORED_URL, return_value="https://stored.corp.internal"),
        ):
            assert BackendConfig.get_backend_url() == "https://stored.corp.internal"

    def test_web_url_derived_from_explicit_backend(self, strict_env: None) -> None:
        with patch.dict(
            "os.environ", {"TRAIGENT_BACKEND_URL": "https://api-x.corp.internal"}
        ):
            assert BackendConfig.get_cloud_web_url() == "https://portal-x.corp.internal"


class TestSecondaryEndpointsDisabled:
    def test_web_url_raises_without_endpoint(self, strict_env: None) -> None:
        with pytest.raises(EndpointNotConfiguredError):
            BackendConfig.get_cloud_web_url()

    def test_explicit_web_url_allowed(self, strict_env: None) -> None:
        with patch.dict(
            "os.environ", {"TRAIGENT_WEB_URL": "https://portal.corp.internal"}
        ):
            assert BackendConfig.get_cloud_web_url() == "https://portal.corp.internal"

    def test_signup_url_and_hint_hide_vendor(self, strict_env: None) -> None:
        assert get_signup_url() is None
        assert "traigent.ai" not in get_no_credentials_hint()

    def test_signup_hint_in_default_mode(self, default_env: None) -> None:
        assert get_signup_url() == "https://portal.traigent.ai"
        assert "portal.traigent.ai" in get_no_credentials_hint()

    def test_analytics_endpoint_has_no_vendor_default(self, strict_env: None) -> None:
        from traigent.utils.local_analytics import LocalAnalytics

        cfg = SimpleNamespace(
            analytics_endpoint=None,
            enable_usage_analytics=True,
            execution_mode="local",
            get_local_storage_path=lambda: None,
        )
        with (
            patch("traigent.utils.local_analytics.LocalStorageManager"),
            patch.object(LocalAnalytics, "_get_or_create_user_id", return_value="u"),
        ):
            assert LocalAnalytics(cfg).analytics_endpoint is None  # type: ignore[arg-type]

    def test_dataset_converter_requires_endpoint(self, strict_env: None) -> None:
        from traigent.cloud.dataset_converter import DatasetConverter

        # Construction (the module-level ``converter``) must not raise ...
        unset = DatasetConverter()
        # ... but using the vendor default must.
        with pytest.raises(EndpointNotConfiguredError):
            _ = unset.backend_base_url
        converted = urlparse(
            DatasetConverter(backend_base_url="https://tg.corp.internal").backend_base_url
        )
        assert (converted.scheme, converted.hostname) == ("https", "tg.corp.internal")

    def test_csp_origin_has_no_vendor(self, strict_env: None) -> None:
        from traigent.security.headers import _default_backend_origin

        assert _default_backend_origin() == ""


class TestUnreachableEndpointNoVendorRequest:
    def test_unreachable_explicit_endpoint_errors_without_vendor_request(self) -> None:
        import asyncio

        from traigent.cloud.backend_client import BackendIntegratedClient

        seen: list[str] = []

        async def fake_request(
            self: object, method: str, str_or_url: object, **kw: object
        ) -> None:
            seen.append(str(str_or_url))
            raise aiohttp.ClientConnectionError("unreachable")

        env = {
            "TRAIGENT_REQUIRE_EXPLICIT_ENDPOINT": "1",
            "TRAIGENT_BACKEND_URL": "https://tg.unreachable.invalid",
        }
        with (
            patch.dict("os.environ", env, clear=True),
            patch(STORED_URL, return_value=None),
            patch.object(aiohttp.ClientSession, "_request", fake_request),
        ):
            client = BackendIntegratedClient(
                api_key="tg_" + "x" * 40,  # pragma: allowlist secret
                max_retries=0,
                timeout=2.0,
            )

            async def run() -> None:
                async with client:
                    await client.create_hybrid_session(
                        problem_statement="p",
                        search_space={"type": "t"},
                        optimization_config={"max_trials": 1},
                    )

            try:
                asyncio.run(run())
                raised = False
            except Exception:
                raised = True
        assert raised, "unreachable explicit endpoint must surface an error"
        assert all("traigent.ai" not in u for u in seen), seen
        assert all("tg.unreachable.invalid" in u for u in seen), seen
