from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from traigent.cloud.api_operations import ApiOperations
from traigent.config.backend_config import BackendConfig
from traigent.core.orchestrator import OptimizationOrchestrator


def test_orchestrator_cloud_url_includes_session_context(monkeypatch) -> None:
    """Cloud URL construction passes owning project/tenant context."""
    monkeypatch.setattr(
        BackendConfig,
        "get_cloud_web_url",
        lambda: "https://portal.traigent.ai/",
    )
    result = SimpleNamespace(
        metadata={
            "experiment_id": "exp/123",
            "experiment_run_id": "run/456",
            "project_id": "project/alpha",
            "tenant_id": "tenant acme",
        },
        experiment_id=None,
        experiment_run_id=None,
        cloud_url=None,
    )

    OptimizationOrchestrator._populate_experiment_cloud_url(result)

    assert result.experiment_id == "exp/123"
    assert result.experiment_run_id == "run/456"
    assert (
        result.cloud_url == "https://portal.traigent.ai/experiments/view/exp%2F123"
        "?run_id=run%2F456&project_id=project%2Falpha&tenant_id=tenant%20acme"
    )


_META = {
    "experiment_id": "e-1",
    "experiment_run_id": "r-1",
    "total_configurations": 4,
    "agent_id": None,
}
_SCOPED = {
    "session_id": "s-1",
    "status": "created",
    "metadata": dict(_META),
    "project_id": "p-1",
    "tenant_id": "t-1",
}
_LEGACY = {
    "success": True,
    "message": "ok",
    "session_id": "s-1",
    "metadata": dict(_META),
    "data": {"session_id": "s-1", "metadata": dict(_META)},
    "project_id": "p-1",
    "tenant_id": "t-1",
}
_SCOPELESS = {"session_id": "s-1", "status": "created", "metadata": dict(_META)}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "expected_query"),
    [
        (_SCOPED, "?run_id=r-1&project_id=p-1&tenant_id=t-1"),
        (_LEGACY, "?run_id=r-1&project_id=p-1&tenant_id=t-1"),
        (_SCOPELESS, "?run_id=r-1"),
        (dict(_SCOPED, project_id="", tenant_id=None), "?run_id=r-1"),
    ],
    ids=["typed", "legacy", "scopeless", "empty-and-null"],
)
async def test_live_create_response_to_orchestrator_cloud_url(
    monkeypatch, payload, expected_query
) -> None:
    """Contract backend #3735 / schema #536: the real create-session response,
    parsed by ApiOperations, yields the scoped (or bare) orchestrator cloud_url.

    The metadata dict mirrors what BackendSessionManager.attach_session_metadata
    builds from the parsed result (owning context + session mapping ids).
    """
    monkeypatch.setattr(
        BackendConfig, "get_cloud_web_url", lambda: "https://portal.traigent.ai/"
    )
    response = AsyncMock()
    response.json = AsyncMock(return_value=payload)
    parsed = await ApiOperations(Mock())._parse_session_response(response)
    metadata = {
        key: value
        for key, value in (
            ("project_id", parsed.project_id),
            ("tenant_id", parsed.tenant_id),
        )
        if value
    }
    _session_id, metadata["experiment_id"], metadata["experiment_run_id"] = parsed
    result = SimpleNamespace(
        metadata=metadata,
        experiment_id=None,
        experiment_run_id=None,
        cloud_url=None,
    )

    OptimizationOrchestrator._populate_experiment_cloud_url(result)

    assert (
        result.cloud_url
        == f"https://portal.traigent.ai/experiments/view/e-1{expected_query}"
    )
