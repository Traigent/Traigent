"""Credentialed clients must reject unsafe backend origins before sending auth."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from traigent.admin.config import EnterpriseAdminConfig
from traigent.analytics.example_insights import ExampleInsightsClient
from traigent.analytics.next_steps import NextStepsClient
from traigent.cloud.backend_client import BackendIntegratedClient
from traigent.cloud.benchmark_client import BenchmarkClientConfig
from traigent.cloud.client import TraigentCloudClient

ClientFactory = Callable[[str], Any]


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(
            lambda url: ExampleInsightsClient(backend_url=url, api_key="test-key"),
            id="ExampleInsightsClient",
        ),
        pytest.param(
            lambda url: NextStepsClient(backend_url=url, api_key="test-key"),
            id="NextStepsClient",
        ),
        pytest.param(
            lambda url: BenchmarkClientConfig(
                backend_origin=url,
                api_key="test-key",
                tenant_id=None,
                project_id=None,
            ),
            id="BenchmarkClientConfig",
        ),
        pytest.param(
            lambda url: TraigentCloudClient(api_key="test-key", base_url=url),
            id="TraigentCloudClient",
        ),
        pytest.param(
            lambda url: BackendIntegratedClient(api_key="test-key", base_url=url),
            id="BackendIntegratedClient",
        ),
        pytest.param(
            lambda url: EnterpriseAdminConfig(
                backend_origin=url,
                api_key="test-key",
            ),
            id="EnterpriseAdminConfig",
        ),
    ],
)
@pytest.mark.parametrize(
    "unsafe_url",
    [
        "http://169.254.169.254",
        "http://127.0.0.1:5000",
    ],
)
def test_credentialed_clients_reject_unsafe_origins_in_production(
    monkeypatch: pytest.MonkeyPatch,
    factory: ClientFactory,
    unsafe_url: str,
) -> None:
    monkeypatch.setenv("TRAIGENT_ENV", "production")

    with pytest.raises(ValueError):
        factory(unsafe_url)


@pytest.mark.parametrize(
    "factory",
    [
        lambda url: ExampleInsightsClient(backend_url=url, api_key="test-key"),
        lambda url: NextStepsClient(backend_url=url, api_key="test-key"),
        lambda url: BenchmarkClientConfig(
            backend_origin=url,
            api_key="test-key",
            tenant_id=None,
            project_id=None,
        ),
        lambda url: TraigentCloudClient(api_key="test-key", base_url=url),
        lambda url: BackendIntegratedClient(api_key="test-key", base_url=url),
        lambda url: EnterpriseAdminConfig(backend_origin=url, api_key="test-key"),
    ],
)
def test_credentialed_clients_allow_localhost_in_development(
    monkeypatch: pytest.MonkeyPatch,
    factory: ClientFactory,
) -> None:
    monkeypatch.setenv("TRAIGENT_ENV", "development")

    factory("http://localhost:5000/")
