from __future__ import annotations

from traigent.admin.config import EnterpriseAdminConfig
from traigent.config.tenant import TENANT_ENV_VAR, TENANT_HEADER_NAME
from traigent.evaluation.config import EvaluationConfig
from traigent.observability.config import ObservabilityConfig
from traigent.prompts.config import PromptManagementConfig


def test_client_configs_propagate_tenant_id_header(monkeypatch) -> None:
    monkeypatch.setenv(TENANT_ENV_VAR, "tenant_env")

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
    admin_headers = EnterpriseAdminConfig(
        backend_origin="https://backend.example",
        api_key="sk-test",  # pragma: allowlist secret
    ).build_headers()

    assert observability_headers[TENANT_HEADER_NAME] == "tenant_env"
    assert prompt_headers[TENANT_HEADER_NAME] == "tenant_env"
    assert evaluation_headers[TENANT_HEADER_NAME] == "tenant_env"
    assert admin_headers[TENANT_HEADER_NAME] == "tenant_env"


def test_client_configs_allow_explicit_tenant_override() -> None:
    headers = EvaluationConfig(
        backend_origin="https://backend.example",
        api_key="sk-test",  # pragma: allowlist secret
        tenant_id="tenant_explicit",
    ).build_headers()

    assert headers[TENANT_HEADER_NAME] == "tenant_explicit"


def test_client_configs_skip_tenant_header_without_env_or_override(monkeypatch) -> None:
    monkeypatch.delenv(TENANT_ENV_VAR, raising=False)

    headers = PromptManagementConfig(
        backend_origin="https://backend.example",
        api_key="sk-test",  # pragma: allowlist secret
    ).build_headers()

    assert TENANT_HEADER_NAME not in headers
