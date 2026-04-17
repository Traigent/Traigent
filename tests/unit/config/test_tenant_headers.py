from __future__ import annotations

from traigent.admin.config import EnterpriseAdminConfig
from traigent.config.project import (
    PROJECT_ENV_VAR,
    read_optional_project_env,
    scope_api_path,
)
from traigent.config.tenant import TENANT_ENV_VAR, TENANT_HEADER_NAME
from traigent.core_metrics.config import CoreMetricsConfig
from traigent.evaluation.config import EvaluationConfig
from traigent.observability.config import ObservabilityConfig
from traigent.prompts.config import PromptManagementConfig


def test_client_configs_propagate_tenant_id_header(monkeypatch) -> None:
    monkeypatch.setenv(TENANT_ENV_VAR, "tenant_env")
    monkeypatch.setenv(PROJECT_ENV_VAR, "project_env")

    observability_headers = ObservabilityConfig(
        backend_origin="https://backend.example",
        api_key="sk-test",  # pragma: allowlist secret
    ).build_headers()
    prompt_headers = PromptManagementConfig(
        backend_origin="https://backend.example",
        api_key="sk-test",  # pragma: allowlist secret
    ).build_headers()
    evaluation_headers = EvaluationConfig(
        backend_origin="https://backend.example",
        api_key="sk-test",  # pragma: allowlist secret
    ).build_headers()
    core_metrics_headers = CoreMetricsConfig(
        backend_origin="https://backend.example",
        api_key="sk-test",  # pragma: allowlist secret
    ).build_headers()
    admin_headers = EnterpriseAdminConfig(
        backend_origin="https://backend.example",
        api_key="sk-test",  # pragma: allowlist secret
    ).build_headers()

    assert observability_headers[TENANT_HEADER_NAME] == "tenant_env"
    assert prompt_headers[TENANT_HEADER_NAME] == "tenant_env"
    assert evaluation_headers[TENANT_HEADER_NAME] == "tenant_env"
    assert core_metrics_headers[TENANT_HEADER_NAME] == "tenant_env"
    assert admin_headers[TENANT_HEADER_NAME] == "tenant_env"
    assert ObservabilityConfig(backend_origin="https://backend.example").api_path.endswith(
        "/projects/project_env/observability"
    )
    assert PromptManagementConfig(backend_origin="https://backend.example").api_path.endswith(
        "/projects/project_env/prompts"
    )
    assert EvaluationConfig(backend_origin="https://backend.example").api_path.endswith(
        "/projects/project_env"
    )
    assert CoreMetricsConfig(backend_origin="https://backend.example").api_path.endswith(
        "/projects/project_env"
    )


def test_client_configs_allow_explicit_tenant_override() -> None:
    headers = EvaluationConfig(
        backend_origin="https://backend.example",
        api_key="sk-test",  # pragma: allowlist secret
        tenant_id="tenant_explicit",
    ).build_headers()

    assert headers[TENANT_HEADER_NAME] == "tenant_explicit"


def test_client_configs_skip_tenant_header_without_env_or_override(monkeypatch) -> None:
    monkeypatch.delenv(TENANT_ENV_VAR, raising=False)
    monkeypatch.delenv(PROJECT_ENV_VAR, raising=False)

    headers = PromptManagementConfig(
        backend_origin="https://backend.example",
        api_key="sk-test",  # pragma: allowlist secret
    ).build_headers()

    assert TENANT_HEADER_NAME not in headers


def test_client_configs_treat_blank_env_values_as_missing(monkeypatch) -> None:
    monkeypatch.setenv(TENANT_ENV_VAR, "   ")
    monkeypatch.setenv(PROJECT_ENV_VAR, "")

    observability_config = ObservabilityConfig(
        backend_origin="https://backend.example",
        api_key="sk-test",  # pragma: allowlist secret
    )
    headers = observability_config.build_headers()

    assert observability_config.tenant_id is None
    assert observability_config.project_id is None
    assert TENANT_HEADER_NAME not in headers
    assert observability_config.api_path == "/api/v1beta/observability"


def test_client_configs_normalize_explicit_blank_overrides() -> None:
    config = EvaluationConfig(
        backend_origin="https://backend.example",
        api_key="sk-test",  # pragma: allowlist secret
        tenant_id="   ",
        project_id="  ",
    )

    assert config.tenant_id is None
    assert config.project_id is None
    assert TENANT_HEADER_NAME not in config.build_headers()
    assert config.api_path == "/api/v1beta"


def test_project_scope_helpers_normalize_paths_and_blank_env(monkeypatch) -> None:
    monkeypatch.setenv(PROJECT_ENV_VAR, "   ")

    assert read_optional_project_env() is None
    assert scope_api_path("/api/v1beta/core-metrics/overview", "project_alpha") == (
        "/api/v1beta/projects/project_alpha/core-metrics/overview"
    )
    assert scope_api_path("/api/v1beta/projects/project_alpha/prompts", "project_beta") == (
        "/api/v1beta/projects/project_alpha/prompts"
    )
    assert scope_api_path("/api/v1/measures", "project_alpha") == (
        "/api/v1beta/projects/project_alpha/measures"
    )
    assert scope_api_path("/api/v1/experiments", "project_alpha") == (
        "/api/v1beta/projects/project_alpha/experiments"
    )
    assert scope_api_path("/api/v1/experiments", None) == "/api/v1/experiments"
