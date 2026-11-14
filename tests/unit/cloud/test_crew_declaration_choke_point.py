"""A caller's raw ``crew_declaration`` never crosses, on any metadata wire path.

The choke point is ``filter_wire_metadata``: it drops the top-level key, and the
sanctioned insert (``crew_declaration_to_wire``) runs after it.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from traigent.utils.wire_text import filter_wire_metadata

CANARY = "canary-crew-choke-9d41c7"
RAW = {"crew_key": CANARY, "role_key": CANARY}


def test_filter_wire_metadata_drops_only_the_declaration_key():
    meta = {"crew_declaration": RAW, "region": "us", "n": 1}
    out = filter_wire_metadata(meta)
    assert out == {"region": "us", "n": 1}
    assert "crew_declaration" in meta  # caller's dict is not mutated


def _trial_serializer(meta):
    from traigent.cloud.client import TraigentCloudClient

    result = MagicMock()
    result.session_id = "s"
    result.trial_id = "t"
    result.metrics = {}
    result.duration = 1.0
    result.status.value = "COMPLETED"
    result.error_message = None
    result.metadata = meta
    return TraigentCloudClient._serialize_trial_result(MagicMock(), result)


def _trial_ops(meta):
    from traigent.cloud.trial_operations import TrialOperations

    return TrialOperations(MagicMock())._build_trial_result_data(
        "t1", {"a": 1}, {"accuracy": 1.0}, "COMPLETED", meta
    )


def _submit_metrics(meta):
    from traigent.cloud.optimizer_client import _wire_submission

    return _wire_submission({"trial_id": "t", "metrics": {}, "metadata": meta})


def _create_score(meta):
    from traigent.evaluation import (
        EvaluationClient,
        EvaluationTargetRefDTO,
        EvaluationTargetType,
    )

    sent: list = []

    def sender(method, path, payload):
        sent.append(payload)
        raise RuntimeError("stop after capture")

    target = EvaluationTargetRefDTO(
        target_type=EvaluationTargetType.OBSERVABILITY_TRACE, target_id="t1"
    )
    with pytest.raises(RuntimeError):
        EvaluationClient(request_sender=sender).create_score(
            measure_id="m", target=target, numeric_value=1.0, metadata=meta
        )
    return sent[0]


def _create_tenant(meta):
    from traigent.admin.client import EnterpriseAdminClient

    sent: list = []

    def sender(method, path, payload):
        sent.append(payload)
        raise RuntimeError("stop after capture")

    with pytest.raises(RuntimeError):
        EnterpriseAdminClient(request_sender=sender).create_tenant(
            name="n", slug="s", metadata=meta
        )
    return sent[0]


def _update_tenant(meta):
    from traigent.admin.client import EnterpriseAdminClient

    sent: list = []

    def sender(method, path, payload):
        sent.append(payload)
        raise RuntimeError("stop after capture")

    with pytest.raises(RuntimeError):
        EnterpriseAdminClient(request_sender=sender).update_tenant(
            "tenant_1", metadata=meta
        )
    return sent[0]


def _create_agent_bridge(meta):
    from traigent.cloud.backend_bridges import SDKBackendBridge
    from traigent.cloud.models import AgentSpecification

    spec = AgentSpecification(
        id="a1",
        name="n",
        agent_type="conversational",
        agent_platform="openai",
        prompt_template="p",
        model_parameters={},
        metadata=meta,
    )
    return SDKBackendBridge().agent_specification_to_backend(spec)


@pytest.mark.parametrize(
    "build",
    [
        _trial_serializer,
        _trial_ops,
        _submit_metrics,
        _create_score,
        _create_tenant,
        _update_tenant,
        _create_agent_bridge,
    ],
)
def test_raw_crew_declaration_never_crosses(build):
    body = build({"crew_declaration": RAW, "keep": "ok"})
    assert CANARY not in json.dumps(body, default=str)
    assert body["metadata"]["keep"] == "ok"
    assert "crew_declaration" not in body["metadata"]


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [CANARY, RAW])
async def test_mcp_create_experiment_drops_raw_crew_declaration(monkeypatch, value):
    from traigent.cloud.models import OptimizationRequest
    from traigent.cloud.production_mcp_client import (
        MCPResponse,
        MCPServerConfig,
        ProductionMCPClient,
    )
    from traigent.evaluators.base import Dataset, EvaluationExample

    request = OptimizationRequest(
        function_name="fn",
        dataset=Dataset(
            name="d",
            examples=[EvaluationExample(input_data={"x": 1}, expected_output="y")],
        ),
        configuration_space={"p": [1]},
        objectives=["maximize"],
        metadata={"crew_declaration": value, "keep": "ok"},
    )
    client = ProductionMCPClient(MCPServerConfig(server_path="python"))
    captured: list[dict] = []

    async def fake_call_tool(tool_name, arguments, operation_id=None):
        captured.append(arguments)
        return MCPResponse(success=True, data={"experiment_id": "exp_1"})

    monkeypatch.setattr(client, "call_tool", fake_call_tool)
    await client.create_experiment(request, agent_id="agent_1", example_set_id="es_1")
    assert captured
    assert CANARY not in json.dumps(captured[0], default=str)
    assert "crew_declaration" not in captured[0]["metadata"]
