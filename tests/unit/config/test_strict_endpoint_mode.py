"""Strict-endpoint (self-hosted) mode must never fall back to a vendor URL.

Covers issue Traigent/Traigent#2770: with TRAIGENT_REQUIRE_EXPLICIT_ENDPOINT=1
or TRAIGENT_DEPLOYMENT_PROFILE=self_hosted, an unset endpoint is a
configuration error and no request is ever made to a traigent.ai host.
"""

# Traceability: CONC-Security FUNC-SECURITY REQ-CLOUD-009

from __future__ import annotations

import contextlib
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
            DatasetConverter(
                backend_base_url="https://tg.corp.internal"
            ).backend_base_url
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


class _TransportTripwire:
    """Records every attempt to reach a network transport.

    Patches the aiohttp request entry point and DNS resolution so a test fails
    loudly if the SDK tries to talk to any host (the vendor cloud included).
    """

    def __init__(self) -> None:
        self.http_calls: list[str] = []
        self.dns_lookups: list[str] = []

    def install(self, stack: contextlib.ExitStack) -> None:
        async def fake_request(
            _self: object, method: str, str_or_url: object, **kw: object
        ) -> None:
            self.http_calls.append(f"{method} {str_or_url}")
            raise AssertionError(f"HTTP transport reached: {method} {str_or_url}")

        def fake_getaddrinfo(host: object, *args: object, **kw: object) -> None:
            self.dns_lookups.append(str(host))
            raise AssertionError(f"DNS lookup attempted: {host}")

        stack.enter_context(
            patch.object(aiohttp.ClientSession, "_request", fake_request)
        )
        stack.enter_context(patch("socket.getaddrinfo", fake_getaddrinfo))
        stack.enter_context(
            patch("asyncio.base_events.BaseEventLoop.getaddrinfo", fake_getaddrinfo)
        )


def _build_cloud_client_and_create_session() -> None:
    import asyncio

    from traigent.cloud.client import TraigentCloudClient

    async def run() -> None:
        client = TraigentCloudClient(
            api_key="tg_" + "x" * 40,  # pragma: allowlist secret
            max_retries=0,
            timeout=2.0,
        )
        async with client:
            await client.create_session(
                function_name="fn",
                configuration_space={"temperature": [0.0, 1.0]},
                objectives=["accuracy"],
                max_trials=1,
            )

    asyncio.run(run())


def _build_backend_integrated_client() -> None:
    import asyncio

    from traigent.cloud.backend_client import BackendIntegratedClient

    async def run() -> None:
        client = BackendIntegratedClient(
            api_key="tg_" + "x" * 40,  # pragma: allowlist secret
            max_retries=0,
            timeout=2.0,
        )
        async with client:
            await client.create_hybrid_session(
                problem_statement="p",
                search_space={"type": "t"},
                optimization_config={"max_trials": 1},
            )

    asyncio.run(run())


class TestBankInstallNoVendorFallback:
    """Bank install: strict mode and no endpoint => refuse before any transport."""

    def test_bank_install_missing_endpoint_refuses_before_transport(
        self, strict_env: None
    ) -> None:

        tripwire = _TransportTripwire()
        with contextlib.ExitStack() as stack:
            tripwire.install(stack)
            with pytest.raises(ConfigurationError):
                _build_cloud_client_and_create_session()
        assert tripwire.http_calls == []
        assert tripwire.dns_lookups == []

    def test_bank_install_backend_integrated_client_refuses_before_transport(
        self, strict_env: None
    ) -> None:

        tripwire = _TransportTripwire()
        with contextlib.ExitStack() as stack:
            tripwire.install(stack)
            # Either construction or the first call must refuse; none may transport.
            with pytest.raises(ConfigurationError):
                _build_backend_integrated_client()
        assert tripwire.http_calls == []
        assert tripwire.dns_lookups == []

    @pytest.mark.parametrize(
        "resolver",
        [
            pytest.param(BackendConfig.get_backend_url, id="get_backend_url"),
            pytest.param(
                BackendConfig.get_cloud_backend_url, id="get_cloud_backend_url"
            ),
            pytest.param(BackendConfig.get_backend_api_url, id="get_backend_api_url"),
            pytest.param(BackendConfig.get_cloud_api_url, id="get_cloud_api_url"),
            pytest.param(BackendConfig.get_cloud_web_url, id="get_cloud_web_url"),
            pytest.param(
                BackendConfig.get_default_cloud_url_or_raise,
                id="get_default_cloud_url_or_raise",
            ),
            pytest.param(BackendConfig.build_api_base, id="build_api_base"),
        ],
    )
    def test_bank_install_every_backend_resolver_refuses(
        self, strict_env: None, resolver: object
    ) -> None:
        with pytest.raises(EndpointNotConfiguredError):
            resolver()  # type: ignore[operator]
