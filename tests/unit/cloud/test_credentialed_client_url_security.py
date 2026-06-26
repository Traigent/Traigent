"""Credentialed clients must reject unsafe backend origins before sending auth."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest

from traigent.admin.config import EnterpriseAdminConfig
from traigent.analytics.example_insights import ExampleInsightsClient
from traigent.analytics.next_steps import NextStepsClient
from traigent.analytics.optimization_plan import OptimizationPlanClient
from traigent.cli import auth_commands
from traigent.cli.auth_commands import TraigentAuthCLI
from traigent.cloud.analytics_client import BackendAnalyticsClient
from traigent.cloud.backend_client import BackendIntegratedClient
from traigent.cloud.benchmark_client import BenchmarkClientConfig
from traigent.cloud.client import TraigentCloudClient
from traigent.cloud.sync_manager import SyncManager
from traigent.config.types import TraigentConfig
from traigent.core_metrics import CoreMetricsClient, CoreMetricsConfig
from traigent.evaluation import EvaluationClient
from traigent.evaluation.config import EvaluationConfig
from traigent.integrations.observability.workflow_traces import WorkflowTracesClient
from traigent.projects import ProjectManagementClient
from traigent.projects.config import ProjectManagementConfig
from traigent.prompts import PromptManagementClient
from traigent.prompts.config import PromptManagementConfig

ClientFactory = Callable[[str], Any]


CLIENT_FACTORY_PARAMS = [
    pytest.param(
        lambda url: ExampleInsightsClient(backend_url=url, api_key="test-key"),
        id="ExampleInsightsClient",
    ),
    pytest.param(
        lambda url: NextStepsClient(backend_url=url, api_key="test-key"),
        id="NextStepsClient",
    ),
    pytest.param(
        lambda url: OptimizationPlanClient(backend_url=url, api_key="test-key"),
        id="OptimizationPlanClient",
    ),
    pytest.param(
        lambda url: BackendAnalyticsClient(backend_url=url, api_key="test-key"),
        id="BackendAnalyticsClient",
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
        lambda url: ProjectManagementConfig(
            backend_origin=url,
            api_key="test-key",
            tenant_id=None,
        ),
        id="ProjectManagementConfig",
    ),
    pytest.param(
        lambda url: PromptManagementConfig(
            backend_origin=url,
            api_key="test-key",
            tenant_id=None,
            project_id=None,
        ),
        id="PromptManagementConfig",
    ),
    pytest.param(
        lambda url: EvaluationConfig(
            backend_origin=url,
            api_key="test-key",
            tenant_id=None,
            project_id=None,
        ),
        id="EvaluationConfig",
    ),
    pytest.param(
        lambda url: CoreMetricsConfig(
            backend_origin=url,
            api_key="test-key",
            tenant_id=None,
            project_id=None,
        ),
        id="CoreMetricsConfig",
    ),
    pytest.param(
        lambda url: WorkflowTracesClient(backend_url=url, auth_token="test-key"),
        id="WorkflowTracesClient",
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
]


def _sync_manager_factory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    url: str,
) -> SyncManager:
    monkeypatch.setenv("TRAIGENT_BACKEND_URL", url)
    monkeypatch.setenv("TRAIGENT_API_URL", url)
    config = MagicMock(spec=TraigentConfig)
    config.get_local_storage_path.return_value = str(tmp_path / "storage")
    config.custom_params = {}
    return SyncManager(config=config, api_key="test-key")


@pytest.mark.parametrize(
    "factory",
    CLIENT_FACTORY_PARAMS,
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
    "unsafe_url",
    [
        "http://169.254.169.254",
        "http://127.0.0.1:5000",
    ],
)
def test_sync_manager_rejects_unsafe_origin_before_session_headers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    unsafe_url: str,
) -> None:
    monkeypatch.setenv("TRAIGENT_ENV", "production")

    with pytest.raises(ValueError):
        _sync_manager_factory(monkeypatch, tmp_path, unsafe_url)


@pytest.mark.parametrize(
    "unsafe_url",
    [
        "http://169.254.169.254",
        "http://127.0.0.1:5000",
    ],
)
def test_auth_cli_rejects_unsafe_origin_before_api_key_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    unsafe_url: str,
) -> None:
    monkeypatch.setenv("TRAIGENT_ENV", "production")
    config_dir = tmp_path / ".traigent"
    monkeypatch.setattr(auth_commands, "TRAIGENT_CONFIG_DIR", config_dir)
    monkeypatch.setattr(
        auth_commands, "CREDENTIALS_FILE", config_dir / "credentials.json"
    )

    with pytest.raises(ValueError):
        TraigentAuthCLI(backend_url_override=unsafe_url)


@pytest.mark.asyncio
async def test_check_api_key_rejects_unsafe_origin_before_aiohttp_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("aiohttp")
    monkeypatch.setenv("TRAIGENT_ENV", "production")
    opened = False

    def fail_if_called(*_args, **_kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("credentialed API-key validation reached aiohttp")

    monkeypatch.setattr(auth_commands.aiohttp, "ClientSession", fail_if_called)

    with pytest.raises(ValueError):
        await auth_commands._check_api_key(
            "http://127.0.0.1:5000/api/v1",
            "test-key",
        )

    assert opened is False


@pytest.mark.parametrize(
    "call_client",
    [
        pytest.param(
            lambda url: ProjectManagementClient(
                ProjectManagementConfig(backend_origin=url, api_key="test-key")
            ).list_projects(),
            id="ProjectManagementClient",
        ),
        pytest.param(
            lambda url: PromptManagementClient(
                PromptManagementConfig(backend_origin=url, api_key="test-key")
            ).list_prompts(),
            id="PromptManagementClient",
        ),
        pytest.param(
            lambda url: EvaluationClient(
                EvaluationConfig(backend_origin=url, api_key="test-key")
            ).list_evaluators(),
            id="EvaluationClient",
        ),
        pytest.param(
            lambda url: CoreMetricsClient(
                CoreMetricsConfig(backend_origin=url, api_key="test-key")
            ).get_core_metrics_overview(),
            id="CoreMetricsClient",
        ),
    ],
)
def test_management_clients_reject_unsafe_origin_before_urlopen(
    monkeypatch: pytest.MonkeyPatch,
    call_client: ClientFactory,
) -> None:
    monkeypatch.setenv("TRAIGENT_ENV", "production")
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    opened = False

    def fail_if_called(*_args, **_kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("credentialed request reached urlopen")

    monkeypatch.setattr("urllib.request.urlopen", fail_if_called)

    with pytest.raises(ValueError):
        call_client("http://127.0.0.1:5000")

    assert opened is False


@pytest.mark.parametrize(
    "factory",
    CLIENT_FACTORY_PARAMS,
)
def test_credentialed_clients_allow_localhost_in_development(
    monkeypatch: pytest.MonkeyPatch,
    factory: ClientFactory,
) -> None:
    monkeypatch.setenv("TRAIGENT_ENV", "development")

    factory("http://localhost:5000/")


def test_sync_manager_allows_localhost_in_development(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("TRAIGENT_ENV", "development")

    manager = _sync_manager_factory(monkeypatch, tmp_path, "http://localhost:5000/")

    assert manager.base_url == "http://localhost:5000/api/v1"
