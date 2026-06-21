from types import SimpleNamespace

from traigent.config.backend_config import BackendConfig
from traigent.core.orchestrator import OptimizationOrchestrator


def test_orchestrator_cloud_url_includes_session_context(monkeypatch) -> None:
    """Cloud URL construction passes owning project/tenant context."""
    monkeypatch.setattr(
        BackendConfig,
        "get_cloud_backend_url",
        lambda: "https://portal.traigent.ai/",
    )
    result = SimpleNamespace(
        metadata={
            "experiment_id": "exp/123",
            "project_id": "project/alpha",
            "tenant_id": "tenant acme",
        },
        experiment_id=None,
        cloud_url=None,
    )

    OptimizationOrchestrator._populate_experiment_cloud_url(result)

    assert result.experiment_id == "exp/123"
    assert (
        result.cloud_url
        == "https://portal.traigent.ai/experiments/view/exp%2F123"
        "?project_id=project%2Falpha&tenant_id=tenant%20acme"
    )
