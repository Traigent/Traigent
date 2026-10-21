"""Closed codes and key allowlists for text sent to the backend (#2241)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from traigent.api.types import AgentConfiguration, AgentDefinition, AgentMeta
from traigent.cloud.trial_operations import TrialOperations
from traigent.cloud.wire_text import (
    WIRE_ERROR_CODES,
    error_code_for_wire,
    filter_agent_configuration,
    filter_summary_stats_metadata,
    filter_surrogate_evaluator,
)
from traigent.core.metadata_helpers import _add_summary_stats

CANARY = "CANARY_WIRE_TEXT_2241_q7"


def test_error_code_is_always_a_closed_member() -> None:
    for text in (
        f"RuntimeError: prompt was {CANARY}",
        f"model said {CANARY}",
        "something odd",
        "Rate limit exceeded",
        "No examples succeeded",
    ):
        assert error_code_for_wire(text) in WIRE_ERROR_CODES
    assert error_code_for_wire(None) is None
    assert error_code_for_wire("") is None
    assert CANARY not in str(error_code_for_wire(f"x {CANARY}"))


def test_agent_configuration_drops_unlisted_keys_and_free_text() -> None:
    cfg = AgentConfiguration(
        agents={
            "a1": AgentDefinition(
                display_name="Agent",
                parameter_keys=["p"],
                measure_ids=["m"],
                meta=AgentMeta(color="#fff", icon="robot", description=CANARY),
            )
        }
    ).to_dict()
    cfg["agents"]["a1"][f"extra_{CANARY}"] = CANARY
    cfg[f"top_{CANARY}"] = {"nested": CANARY}
    out = filter_agent_configuration(cfg)
    assert CANARY not in json.dumps(out)
    assert out["agents"]["a1"]["display_name"] == "Agent"
    assert out["agents"]["a1"]["meta"] == {"color": "#fff", "icon": "robot"}
    assert out["agents"]["a1"]["parameter_keys"] == ["p"]


def test_surrogate_descriptor_allowlist() -> None:
    desc = {
        "evaluator_id": "my_scorer",
        "metric_name": "surrogate_score",
        "judge_model": None,
        "prompt": CANARY,
        "config": {"fingerprint_source": "fp1:abc", "note": CANARY},
        "free_text": CANARY,
        f"k_{CANARY}": 1,
    }
    out = filter_surrogate_evaluator(desc)
    assert CANARY not in json.dumps(out)
    assert "prompt" not in out
    assert out["config"] == {"fingerprint_source": "fp1:abc"}


def test_serialized_trial_result_carries_only_closed_error_code() -> None:
    from traigent.cloud.client import TraigentCloudClient

    result = MagicMock()
    result.session_id = "s"
    result.trial_id = "t"
    result.metrics = {}
    result.duration = 1.0
    result.status.value = "FAILED"
    result.error_message = f"ValueError: prompt was {CANARY}"
    result.metadata = {}
    payload = TraigentCloudClient._serialize_trial_result(MagicMock(), result)
    assert payload["error_message"] == "trial_failed"
    assert CANARY not in json.dumps(payload)


def test_summary_stats_metadata_allowlist_and_bounded_scalars() -> None:
    meta = {
        "sdk_version": "1.0",
        "examples_consumed": 3,
        "note": CANARY,
        "source": {"nested": CANARY},
        "aggregation_method": "x" * 1000,
    }
    out = filter_summary_stats_metadata(meta)
    assert CANARY not in json.dumps(out)
    assert "note" not in out and "source" not in out
    assert len(out["aggregation_method"]) <= 128


def test_add_summary_stats_filters_metadata() -> None:
    trial = MagicMock()
    trial.summary_stats = {
        "metrics": {},
        "metadata": {"free": CANARY, "sdk_version": "1.0"},
    }
    meta: dict = {}
    from traigent.config.types import ExecutionMode

    _add_summary_stats(meta, trial, ExecutionMode.HYBRID)
    assert CANARY not in json.dumps(meta)
    assert meta["summary_stats"]["metadata"]["aggregation_level"] == "trial"


def test_trial_result_data_filters_metadata_dicts() -> None:
    ops = TrialOperations(MagicMock())
    data = ops._build_trial_result_data(
        "t1",
        {"a": 1},
        {"accuracy": 1.0},
        "COMPLETED",
        metadata={
            "surrogate_evaluator": {"evaluator_id": "e", "leak": CANARY},
            "agent_configuration": {"version": "1.0", "leak": CANARY},
        },
    )
    assert CANARY not in json.dumps(data)
    assert data["metadata"]["surrogate_evaluator"] == {"evaluator_id": "e"}


def test_session_creation_metadata_filtered() -> None:
    # Production path helper used by BackendSessionManager.create_session.
    cfg = AgentConfiguration().to_dict()
    cfg[f"leak_{CANARY}"] = CANARY
    assert CANARY not in json.dumps(filter_agent_configuration(cfg))


@pytest.mark.parametrize("bad", [None, "str", 5, [1]])
def test_filters_fail_closed_on_non_mapping(bad) -> None:
    assert filter_agent_configuration(bad) == {}
    assert filter_surrogate_evaluator(bad) == {}
    assert filter_summary_stats_metadata(bad) == {}


# --- review r78: remaining wire paths -------------------------------------


def test_serialized_trial_result_filters_metadata_maps() -> None:
    """Finding 1: the trial serializer (submit_trial_result, previous_results)."""
    from traigent.cloud.client import TraigentCloudClient

    result = MagicMock()
    result.session_id = "s"
    result.trial_id = "t"
    result.metrics = {}
    result.duration = 1.0
    result.status.value = "COMPLETED"
    result.error_message = None
    result.metadata = {
        "surrogate_evaluator": {"evaluator_id": "e", "prompt": CANARY},
        "agent_configuration": {"version": "1", "agents": {"a": {"bogus": CANARY}}},
        "summary_stats": {"metadata": {"note": CANARY}},
    }
    payload = TraigentCloudClient._serialize_trial_result(MagicMock(), result)
    assert CANARY not in json.dumps(payload)
    assert payload["metadata"]["surrogate_evaluator"] == {"evaluator_id": "e"}


def test_session_creation_payloads_filter_metadata_maps() -> None:
    """Finding 2: every session-create serialization boundary, not only the
    BackendSessionManager path."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    from traigent.cloud.api_operations import ApiOperations
    from traigent.cloud.client import TraigentCloudClient
    from traigent.cloud.models import SessionCreationRequest

    request = SessionCreationRequest(
        function_name="qa_agent",
        configuration_space={"temperature": [0.0, 1.0]},
        objectives=[{"name": "accuracy", "orientation": "maximize", "weight": 1.0}],
        dataset_metadata={"size": 1},
        metadata={
            "agent_configuration": {
                "version": "1",
                "agents": {"a": {"display_name": "A", "meta": {"description": CANARY}}},
            },
            "surrogate_evaluator": {"evaluator_id": "e", "prompt": CANARY},
            "summary_stats": {"metadata": {"note": CANARY}},
        },
    )
    typed = ApiOperations(Mock())._build_typed_session_payload(request, 5)
    legacy = ApiOperations(Mock())._build_legacy_session_payload(request, 5)
    stub = SimpleNamespace(_ensure_owner_metadata=lambda metadata: metadata or {})
    cloud = TraigentCloudClient._serialize_session_request(stub, request)
    for payload in (typed, legacy, cloud):
        assert CANARY not in json.dumps(payload["metadata"])
        assert payload["metadata"]["surrogate_evaluator"] == {"evaluator_id": "e"}


def test_trial_result_data_filters_non_dict_metadata_maps() -> None:
    """Finding 3: the guard must not fail open on a non-dict value."""
    ops = TrialOperations(MagicMock())
    data = ops._build_trial_result_data(
        "t1",
        {"a": 1},
        {"accuracy": 1.0},
        "COMPLETED",
        metadata={
            "surrogate_evaluator": [{"prompt": CANARY}],
            "agent_configuration": f"{CANARY} free text",
        },
    )
    assert CANARY not in json.dumps(data)
    assert data["metadata"]["surrogate_evaluator"] == {}
    assert data["metadata"]["agent_configuration"] == {}


@pytest.mark.asyncio
async def test_submitted_summary_stats_copies_are_both_filtered(monkeypatch) -> None:
    """Finding 4: the top-level summary_stats copy and the metadata copy that
    rides in result_data["metadata"] must both be filtered."""
    from unittest.mock import patch

    import traigent.cloud.trial_operations as trial_ops

    captured: dict = {}

    def _capture(data):
        captured.update(data)
        raise ValueError("stop after capture")

    client = MagicMock()
    client._map_to_backend_status.return_value = "COMPLETED"
    ops = TrialOperations(client)
    monkeypatch.setattr(trial_ops, "AIOHTTP_AVAILABLE", True)
    with (
        patch.object(trial_ops, "is_backend_offline", return_value=False),
        patch.object(TrialOperations, "_raise_if_backend_egress_disabled"),
        patch.object(trial_ops, "ensure_cost_metric_for_budgeted_completed_submission"),
        patch.object(
            trial_ops, "validate_configuration_run_submission", side_effect=_capture
        ),
    ):
        ok = await ops.submit_trial_result_via_session(
            "sess",
            "trial",
            {"a": 1},
            {
                "accuracy": 1.0,
                "summary_stats": {
                    "total_examples": 1,
                    "metadata": {"note": CANARY, "sdk_version": "1.0"},
                },
            },
            "COMPLETED",
            metadata={
                "summary_stats": {"metadata": {"note": CANARY}, "total_examples": 1}
            },
        )
    assert ok is False
    assert captured, "submission was not built"
    assert CANARY not in json.dumps(captured)
    assert "note" not in captured["summary_stats"]["metadata"]
    assert "note" not in captured["metadata"]["summary_stats"]["metadata"]


@pytest.mark.backend_online
@pytest.mark.asyncio
async def test_mcp_configuration_run_update_sends_closed_code() -> None:
    """Finding 5: update_configuration_run_results must not send raw text."""
    from unittest.mock import AsyncMock

    from traigent.cloud.production_mcp_client import (
        MCPServerConfig,
        ProductionMCPClient,
    )

    client = ProductionMCPClient(
        MCPServerConfig(server_path="python"), enable_fallback=False
    )
    client.call_tool = AsyncMock(return_value=MagicMock())
    await client.update_configuration_run_results(
        "cfg-run-1",
        {"accuracy": 0.0},
        "FAILED",
        error_message=f"ValueError: prompt was {CANARY}",
    )
    args = client.call_tool.await_args.args[1]
    assert args["error_message"] in WIRE_ERROR_CODES
    assert CANARY not in json.dumps(args)


@pytest.mark.parametrize(
    "code",
    [
        "no_examples_succeeded",
        "trial_cancelled",
        "trial_pruned",
        "trial_failed",
        "vendor_rate_limit",
        "vendor_service_unavailable",
    ],
)
def test_error_code_is_idempotent_for_closed_members(code: str) -> None:
    """Finding 6: a value already in WIRE_ERROR_CODES passes through unchanged."""
    assert code in WIRE_ERROR_CODES
    assert error_code_for_wire(code) == code
    assert error_code_for_wire(error_code_for_wire(code)) == code


def test_error_code_double_conversion_keeps_classification() -> None:
    once = error_code_for_wire("No examples succeeded for this trial")
    assert once == "no_examples_succeeded"
    assert error_code_for_wire(once) == "no_examples_succeeded"
    cancelled = error_code_for_wire("Trial cancelled by user")
    assert error_code_for_wire(cancelled) == "trial_cancelled"


# ---------------------------------------------------------------------------
# Review r79 (#2241): remaining wire paths that forwarded named metadata maps.
# Each canary must be absent from the captured outbound payload.
# ---------------------------------------------------------------------------

_TENANT_FIXTURE = {
    "id": "tenant_1",
    "name": "Acme Corp",
    "slug": "acme",
    "status": "active",
    "tier": "professional",
    "quotas": {},
    "metadata": {},
    "membership_count": 1,
    "has_sso_config": False,
    "created_at": "2026-03-10T12:00:00+00:00",
    "updated_at": "2026-03-10T12:00:00+00:00",
}


def test_admin_update_tenant_filters_metadata_and_keeps_omission() -> None:
    from traigent.admin.client import EnterpriseAdminClient

    sent: list[dict | None] = []

    def request_sender(method: str, path: str, payload: dict | None):
        sent.append(payload)
        return {"data": dict(_TENANT_FIXTURE)}

    client = EnterpriseAdminClient(request_sender=request_sender)
    client.update_tenant(
        "tenant_1",
        metadata={"summary_stats": {"metadata": {"note": CANARY}}, "region": "us"},
    )
    assert CANARY not in json.dumps(sent[-1])
    assert sent[-1]["metadata"]["summary_stats"]["metadata"] == {}

    client.update_tenant("tenant_1", status="suspended")
    assert "metadata" not in sent[-1]


@pytest.mark.asyncio
async def test_mcp_create_experiment_drops_scalar_surrogate_descriptor(
    monkeypatch,
) -> None:
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
        metadata={"surrogate_evaluator": CANARY},
    )
    client = ProductionMCPClient(MCPServerConfig(server_path="python"))
    captured: list[dict] = []

    async def fake_call_tool(tool_name, arguments, operation_id=None):
        captured.append({"tool": tool_name, "arguments": arguments})
        return MCPResponse(success=True, data={"experiment_id": "exp_1"})

    monkeypatch.setattr(client, "call_tool", fake_call_tool)
    await client.create_experiment(request, agent_id="agent_1", example_set_id="es_1")
    assert captured and captured[0]["tool"] == "create_experiment"
    assert CANARY not in json.dumps(captured[0]["arguments"], default=str)
    assert "surrogate_evaluator" not in captured[0]["arguments"]["metadata"]


@pytest.mark.asyncio
async def test_hybrid_session_payload_filters_metadata(monkeypatch) -> None:
    import threading

    from traigent.cloud.session_operations import SessionOperations

    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    captured: dict = {}

    class _Response:
        status = 201

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def json(self):
            return {"session_id": "hs_1", "token": "tok", "optimizer_endpoint": ""}

        async def text(self):
            return ""

    class _Session:
        def post(self, url, json=None, headers=None):
            captured["url"] = url
            captured["json"] = json
            return _Response()

    client = MagicMock()
    client._url_invalid = False
    client.no_egress = False
    client._ensure_session = AsyncMock(return_value=_Session())
    client.backend_config.api_base_url = "https://api.example.com"
    client.auth_manager.auth.get_headers = AsyncMock(return_value={})
    client._active_sessions = {}
    client._active_sessions_lock = threading.Lock()

    ops = SessionOperations(client)
    await ops.create_hybrid_session(
        "problem",
        {"p": [1]},
        {"objectives": ["maximize"]},
        metadata={
            "surrogate_evaluator": {"prompt": CANARY},
            "summary_stats": {"metadata": {"note": CANARY}},
        },
    )
    assert captured["url"].endswith("/hybrid/sessions")
    assert CANARY not in json.dumps(captured["json"], default=str)
    assert captured["json"]["metadata"]["surrogate_evaluator"] == {}


@pytest.fixture
def _optimizer_dns(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setattr(
        "traigent.cloud.url_security.socket.getaddrinfo",
        lambda *_a, **_k: [(0, 0, 0, "", ("93.184.216.34", 0))],
    )


class _OkResponse:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def raise_for_status(self):
        return None

    async def json(self):
        return {}


class _RecordingSession:
    def __init__(self) -> None:
        self.bodies: list = []

    def post(self, url, json=None, **_kwargs):
        self.bodies.append(json)
        return _OkResponse()


@pytest.mark.asyncio
@pytest.mark.backend_online
async def test_optimizer_http_payloads_filter_metadata(_optimizer_dns) -> None:
    from traigent.cloud.optimizer_client import OptimizerDirectClient

    client = OptimizerDirectClient("https://api.example.com", "secret-token")
    client.session = _RecordingSession()
    dirty = {
        "trial_id": "t1",
        "metrics": {"score": 0.5},
        "execution_time": 1.0,
        "metadata": {"surrogate_evaluator": {"prompt": CANARY}},
    }
    await client._submit_single("s1", dict(dirty))
    await client._submit_batch("s1", [dict(dirty)])
    assert len(client.session.bodies) == 2
    for body in client.session.bodies:
        assert CANARY not in json.dumps(body, default=str)
    # The caller's own dict is not mutated (detached copy).
    assert dirty["metadata"]["surrogate_evaluator"]["prompt"] == CANARY


@pytest.mark.asyncio
@pytest.mark.backend_online
async def test_optimizer_buffered_submission_filters_metadata(_optimizer_dns) -> None:
    from traigent.cloud.optimizer_client import OptimizerDirectClient

    client = OptimizerDirectClient("https://api.example.com", "secret-token")
    client._batch_size = 1
    flushed: list = []

    async def fake_flush_buffer():
        flushed.extend(list(client._metric_buffer))
        client._metric_buffer.clear()
        return {"status": "flushed"}

    client._flush_buffer = fake_flush_buffer
    await client.submit_metrics(
        "s1",
        "t1",
        {"score": 0.5},
        1.0,
        metadata={"agent_configuration": {f"unlisted_{CANARY}": CANARY}},
    )
    assert flushed
    assert CANARY not in json.dumps(flushed, default=str)
