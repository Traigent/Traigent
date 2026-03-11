from __future__ import annotations

from traigent.admin.config import EnterpriseAdminConfig
from traigent.evaluation.config import EvaluationConfig
from traigent.observability.config import ObservabilityConfig
from traigent.prompts.config import PromptManagementConfig


def test_client_configs_propagate_tenant_id_header(monkeypatch) -> None:
    monkeypatch.setenv("TRAIGENT_TENANT_ID", "tenant_env")

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

    assert observability_headers["X-Tenant-Id"] == "tenant_env"
    assert prompt_headers["X-Tenant-Id"] == "tenant_env"
    assert evaluation_headers["X-Tenant-Id"] == "tenant_env"
    assert admin_headers["X-Tenant-Id"] == "tenant_env"


def test_client_configs_allow_explicit_tenant_override() -> None:
    headers = EvaluationConfig(
        backend_origin="https://backend.example",
        api_key="sk-test",  # pragma: allowlist secret
        tenant_id="tenant_explicit",
    ).build_headers()

    assert headers["X-Tenant-Id"] == "tenant_explicit"
