"""Default-deny for the NON-session projectors (results, summary submission, optimizer
metrics, next-trial, parent run).

Session-create got an explicit top-level allowlist, closed specs and a wire-form check. These
tests pin the same treatment for the other phases: an unknown top-level key refuses, a trial's
``config`` is held to the closed search-space check, a NaN metric is still DROPPED (not
refused), and every body the real producers build is still accepted.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from traigent.cloud.client import TraigentCloudClient
from traigent.cloud.models import NextTrialRequest, TrialResultSubmission, TrialStatus
from traigent.cloud.optimizer_client import OptimizerDirectClient
from traigent.cloud.trial_operations import TrialOperations
from traigent.utils import optimization_only_egress as oo
from traigent.utils.trial_costs import COST_FLOOR_WIRE_KEY, lift_cost_floor_to_body

# SDK #2033: opt into the connected/backend code paths (see pyproject markers).
pytestmark = pytest.mark.backend_online

SECRET = "private customer prompt"  # pragma: allowlist secret
API = "https://api.example.test/api/v1"
SID = "sess-1"
URLS = {
    oo.PHASE_RESULTS: f"{API}/sessions/{SID}/results",
    oo.PHASE_SUMMARY_SUBMISSION: f"{API}/sessions/{SID}/results",
    oo.PHASE_NEXT_TRIAL: f"{API}/sessions/{SID}/next-trial",
    oo.PHASE_OPTIMIZER_METRICS: f"{API}/hybrid/optimizer/{SID}",
    oo.PHASE_PARENT_RUN: f"{API}/experiments/exp1/runs",
}


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv(oo.ENV_VAR, raising=False)
    oo.reset_latch()
    yield
    oo.reset_latch()


def _judge(phase: str, body: Any) -> None:
    """The transport backstop on the serialized body: the body must be a fixed point."""
    oo.enforce_outgoing(
        "POST", URLS[phase], {}, json.dumps(body).encode(), explicit=True
    )


def _refused_or_clean(phase: str, body: Any) -> None:
    try:
        projected = oo.project_for_send(phase, body, explicit=True, declared=())
        raw = json.dumps(projected).encode()
        oo.enforce_outgoing("POST", URLS[phase], {}, raw, explicit=True)
    except (oo.MetadataContractViolation, oo.OptimizationOnlyRefusal):
        return
    assert SECRET not in raw.decode()


# ------------------------------------------------------------ the real producers' bodies


def _post_mocks() -> tuple[Mock, Mock]:
    response = Mock()
    response.status = 201
    post_ctx = AsyncMock()
    post_ctx.__aenter__ = AsyncMock(return_value=response)
    post_ctx.__aexit__ = AsyncMock(return_value=False)
    session = AsyncMock()
    session.post = Mock(return_value=post_ctx)
    session_ctx = AsyncMock()
    session_ctx.__aenter__ = AsyncMock(return_value=session)
    session_ctx.__aexit__ = AsyncMock(return_value=False)
    return session, session_ctx


def _trial_ops() -> TrialOperations:
    client = Mock()
    client.optimization_only = True
    client.backend_config = Mock()
    client.backend_config.backend_base_url = "https://api.example.test"
    client.backend_config.api_base_url = API
    client.auth_manager = AsyncMock()
    client.auth_manager.augment_headers = AsyncMock(return_value={})
    client._map_to_backend_status = Mock(return_value="COMPLETED")
    client._sanitize_error_message = Mock(side_effect=lambda m: m)
    ops = TrialOperations(client)
    ops._handle_trial_success_response = AsyncMock(return_value=True)
    return ops


async def _drive(call: Any) -> list[Any]:
    """Run ``call(ops)`` against a capturing HTTP session; return every ``json=`` body sent."""
    session, session_ctx = _post_mocks()
    ops = _trial_ops()
    with (
        patch("traigent.cloud.trial_operations.is_backend_offline", return_value=False),
        patch("traigent.cloud.trial_operations.AIOHTTP_AVAILABLE", True),
        patch("traigent.cloud.trial_operations.validate_configuration_run_submission"),
        patch("traigent.cloud.trial_operations.aiohttp") as aiohttp_mod,
    ):
        aiohttp_mod.ClientSession = Mock(return_value=session_ctx)
        aiohttp_mod.ClientTimeout = Mock()
        ops.client._ensure_session = AsyncMock(return_value=session)
        outcome = await call(ops)
    assert outcome is not False, "the real producer's send was refused"
    return [c.kwargs["json"] for c in session.post.call_args_list]


_CONFIG = {"model": "gpt-4o-mini", "temperature": 0.2, "use_cache": True, "k": 3}
_METRICS = {"accuracy": 0.9, "cost": 0.01, "latency": 1.5}
_MEASURES = [{"example_id": "e1", "metrics": {"accuracy": 1.0}}]


@pytest.mark.asyncio
async def test_register_trial_start_body_is_accepted() -> None:
    bodies = await _drive(
        lambda ops: ops.register_trial_start(SID, "trial_1", dict(_CONFIG))
    )
    assert bodies and set(bodies[0]) == {"trial_id", "config", "status", "metrics"}
    _judge(oo.PHASE_RESULTS, bodies[0])


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_trial_result_body_is_accepted(failed: bool) -> None:
    bodies = await _drive(
        lambda ops: ops.submit_trial_result_via_session(
            SID,
            "trial_1",
            dict(_CONFIG),
            dict(_METRICS),
            "failed" if failed else "completed",
            error_message="prompt 'tell me about Alice' failed" if failed else None,
            metadata={"duration": 1.5, "trial_number": 2, "model": "gpt-4o"},
        )
    )
    body = bodies[-1]
    assert {"trial_id", "config", "metrics", "status", "metadata"} <= set(body)
    assert "Alice" not in json.dumps(body)
    _judge(oo.PHASE_RESULTS, body)


@pytest.mark.asyncio
async def test_privacy_mode_redacted_config_is_accepted() -> None:
    redacted = TrialOperations._redact_privacy_config(
        {"model": "gpt-4o", "system_prompt": SECRET, "stops": [SECRET], "k": 3}
    )
    assert redacted["system_prompt"].startswith("[REDACTED")
    bodies = await _drive(
        lambda ops: ops.register_trial_start(SID, "trial_1", dict(redacted))
    )
    assert bodies[0]["config"] == redacted
    assert SECRET not in json.dumps(bodies[0])


@pytest.mark.asyncio
async def test_summary_submission_body_is_accepted() -> None:
    stats = {
        "metrics": {"accuracy": {"count": 3, "mean": 0.5, "std": 0.1}},
        "execution_time": 1.2,
        "total_examples": 3,
        "metadata": {"source": "pandas"},
    }
    bodies = await _drive(
        lambda ops: ops.submit_summary_stats(SID, "trial_1", dict(_CONFIG), stats)
    )
    body = bodies[-1]
    assert {"trial_id", "config", "metrics", "metadata", "status", "summary_stats"} <= (
        set(body)
    )
    _judge(oo.PHASE_SUMMARY_SUBMISSION, body)


def test_cloud_client_result_and_next_trial_bodies_are_accepted() -> None:
    stub = SimpleNamespace()
    stub._serialize_trial_result = lambda r: (
        TraigentCloudClient._serialize_trial_result(stub, r)
    )
    result = TrialResultSubmission(
        session_id=SID,
        trial_id="trial_1",
        metrics=dict(_METRICS),
        duration=1.5,
        status=TrialStatus.COMPLETED,
        error_message="vendor_rate_limit",
        metadata={"duration": 1.5, "model": "gpt-4o"},
    )
    body = TraigentCloudClient._serialize_trial_result(stub, result)
    body["metrics"] = dict(body["metrics"])
    body["metrics"][COST_FLOOR_WIRE_KEY] = 0.01
    lift_cost_floor_to_body(body)
    assert COST_FLOOR_WIRE_KEY in body
    _judge(oo.PHASE_RESULTS, body)
    request = NextTrialRequest(session_id=SID, previous_results=[result])
    next_body = TraigentCloudClient._serialize_next_trial_request(stub, request)
    assert set(next_body) == {"session_id", "previous_results", "request_metadata"}
    _judge(oo.PHASE_NEXT_TRIAL, next_body)
    empty = TraigentCloudClient._serialize_next_trial_request(
        stub, NextTrialRequest(session_id=SID)
    )
    _judge(oo.PHASE_NEXT_TRIAL, empty)
    _judge(oo.PHASE_NEXT_TRIAL, {"session_id": SID, "previous_results": []})


def test_sync_manager_import_bodies_are_accepted() -> None:
    import requests

    from traigent.cloud.sync_manager import SyncManager

    with patch("traigent.cloud.sync_manager.LocalStorageManager"):
        manager = SyncManager(config=MagicMock(), api_key="tg_" + "a" * 61)
    manager._session = MagicMock(spec=requests.Session)
    sent: list[tuple[str, Any]] = []

    def _post(url: str, **kwargs: Any) -> Mock:
        sent.append((url, kwargs["json"]))
        response = Mock(status_code=201, text="Created")
        response.json.return_value = (
            {"suggestion": {"trial_id": "backend-trial-1"}}
            if url.endswith("/next-trial")
            else {"id": "result-1"}
        )
        return response

    manager._session.post.side_effect = _post
    runs = [
        {
            "experiment_parameters": dict(_CONFIG),
            "measures": dict(_METRICS),
            "example_measures": list(_MEASURES),
            "status": "COMPLETED",
        }
    ]
    with patch.object(
        manager, "_optimization_only_policy", return_value=oo.Policy(True, None)
    ):
        outcome = manager._sync_session_results(SID, runs)
    assert outcome["success"], outcome
    kinds = {url.rsplit("/", 1)[1]: body for url, body in sent}
    _judge(oo.PHASE_NEXT_TRIAL, kinds["next-trial"])
    assert set(kinds["results"]) == {
        "trial_id",
        "config",
        "status",
        "metrics",
        "measures",
    }
    _judge(oo.PHASE_RESULTS, kinds["results"])


def test_optimizer_metric_submission_bodies_are_accepted() -> None:
    client = object.__new__(OptimizerDirectClient)
    client.endpoint = "https://api.example.test/api/v1/hybrid/optimizer"
    client.token = "tok"  # pragma: allowlist secret
    client.optimization_only = True
    body = client._wire_submission(
        {
            "trial_id": "trial_1",
            "metrics": dict(_METRICS),
            "execution_time": 1.5,
            "metadata": {"duration": 1.5, "notes": SECRET},
        },
        SID,
    )
    assert SECRET not in json.dumps(body)
    assert set(body) == {"trial_id", "metrics", "execution_time", "metadata"}
    _judge(oo.PHASE_OPTIMIZER_METRICS, body)
    batch = {"submissions": [body]}
    oo.enforce_outgoing(
        "POST",
        f"{API}/hybrid/optimizer/{SID}/batch",
        {},
        json.dumps(batch).encode(),
        explicit=True,
    )
    # metadata may be absent or empty
    _judge(
        oo.PHASE_OPTIMIZER_METRICS,
        client._wire_submission(
            {"trial_id": "t", "metrics": {}, "metadata": None}, SID
        ),
    )


def test_parent_run_documented_body_is_accepted() -> None:
    body = {
        "name": "Alice's run",
        "description": SECRET,
        "run_id": "run_1",
        "metadata": {"agent_id": "a1", "benchmark_id": "b1", "measures": ["accuracy"]},
        "experiment_data": {
            "id": "exp1",
            "agent_id": "a1",
            "measures": ["accuracy"],
            "metadata": {"agent_id": "a1"},
            "name": "free",
        },
    }
    projected = oo.project_for_send(oo.PHASE_PARENT_RUN, body, explicit=True)
    assert "name" not in projected and "description" not in projected
    assert "name" not in projected["experiment_data"]
    assert SECRET not in json.dumps(projected)
    _judge(oo.PHASE_PARENT_RUN, projected)


# ------------------------------------------------------------ the allowlists, pinned

_RESULT_KEYS = {
    "trial_id",
    "metrics",
    "measures",
    "summary_stats",
    "metadata",
    "error_message",
    "config",
    "session_id",
    "status",
    "duration",
    COST_FLOOR_WIRE_KEY,
}
_OPTIMIZER_KEYS = {"trial_id", "metrics", "metadata", "execution_time"}
_NEXT_TRIAL_KEYS = {"session_id", "previous_results", "request_metadata"}
_PARENT_RUN_KEYS = {
    "name",
    "description",
    "id",
    "run_id",
    "metadata",
    "experiment_data",
}


def test_envelopes_are_exactly_the_producer_keys() -> None:
    assert set(oo._RESULT_ENVELOPE) == _RESULT_KEYS
    assert set(oo._OPTIMIZER_METRICS_ENVELOPE) == _OPTIMIZER_KEYS
    assert set(oo._NEXT_TRIAL_ENVELOPE) == _NEXT_TRIAL_KEYS
    assert set(oo._PARENT_RUN_ENVELOPE) == _PARENT_RUN_KEYS


@pytest.mark.parametrize(
    "phase",
    [
        oo.PHASE_RESULTS,
        oo.PHASE_SUMMARY_SUBMISSION,
        oo.PHASE_OPTIMIZER_METRICS,
        oo.PHASE_NEXT_TRIAL,
        oo.PHASE_PARENT_RUN,
    ],
)
@pytest.mark.parametrize(
    "extra",
    [{"free_text": SECRET}, {"note": 1}, {SECRET: 1}, {"nested": {"prompt": SECRET}}],
)
def test_unknown_top_level_key_is_refused_without_echo(
    phase: str, extra: dict[str, Any]
) -> None:
    base: dict[str, Any] = {"session_id": SID} if phase == oo.PHASE_NEXT_TRIAL else {}
    with pytest.raises(oo.MetadataContractViolation) as caught:
        oo.project_for_send(phase, {**base, **extra}, explicit=True)
    assert SECRET not in str(caught.value)


def test_unknown_key_inside_next_trial_previous_results_and_experiment_data() -> None:
    with pytest.raises(oo.MetadataContractViolation):
        oo.project_for_send(
            oo.PHASE_NEXT_TRIAL,
            {"session_id": SID, "previous_results": [{"trial_id": "t", "x": SECRET}]},
            explicit=True,
        )
    with pytest.raises(oo.MetadataContractViolation):
        oo.project_for_send(
            oo.PHASE_PARENT_RUN,
            {"experiment_data": {"id": "e", "free": SECRET}},
            explicit=True,
        )


# ------------------------------------------------------------ config is a closed value


@pytest.mark.parametrize(
    "config",
    [
        {"prompt": SECRET},
        {"nested": {"prompt": [SECRET]}},
        {SECRET: 1},
        {"a": [1, SECRET]},
        {"x": float("nan")},
        SECRET,
        [SECRET],
        5,
    ],
)
@pytest.mark.parametrize("phase", [oo.PHASE_RESULTS, oo.PHASE_SUMMARY_SUBMISSION])
def test_config_with_free_text_or_wrong_shape_is_refused(
    phase: str, config: Any
) -> None:
    with pytest.raises(oo.MetadataContractViolation):
        oo.project_for_send(phase, {"trial_id": "t", "config": config}, explicit=True)
    with pytest.raises(oo.MetadataContractViolation):
        oo.project_for_send(
            phase, {"trial_id": "t", "metadata": {"config": config}}, explicit=True
        )


def test_config_closed_values_pass_and_privacy_markers_pass() -> None:
    config = {
        "model": "gpt-4o",
        "t": 0.0,
        "flag": True,
        "none": None,
        "stops": ["a", "b"],
        "system_prompt": "[REDACTED:23 chars]",
        "tools": "[REDACTED:list]",
        "x": "[REDACTED]",
    }
    out = oo.project_for_send(
        oo.PHASE_RESULTS, {"trial_id": "t", "config": config}, explicit=True
    )
    assert out["config"] == config
    # a marker lookalike carrying content is not a marker
    for bad in ("[REDACTED: private customer prompt]", "[REDACTED:5 chars] extra"):
        with pytest.raises(oo.MetadataContractViolation):
            oo.project_for_send(
                oo.PHASE_RESULTS, {"trial_id": "t", "config": {"k": bad}}, explicit=True
            )


def test_search_space_check_is_unchanged_by_the_marker_allowance() -> None:
    with pytest.raises(oo.MetadataContractViolation):
        oo.project_session_create(
            {"configuration_space": {"a": ["[REDACTED]"]}, "function_name": "f"}
        )


# ------------------------------------------------------------ NaN keeps being dropped


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize(
    "phase",
    [oo.PHASE_RESULTS, oo.PHASE_SUMMARY_SUBMISSION, oo.PHASE_OPTIMIZER_METRICS],
)
def test_non_finite_metric_is_dropped_not_refused(phase: str, value: float) -> None:
    out = oo.project_for_send(
        phase,
        {"trial_id": "t", "metrics": {"accuracy": value, "cost": 1.0}},
        explicit=True,
    )
    assert out["metrics"] == {"cost": 1.0}
    json.dumps(out, allow_nan=False)


def test_non_finite_metric_dropped_inside_next_trial_previous_results() -> None:
    out = oo.project_for_send(
        oo.PHASE_NEXT_TRIAL,
        {
            "session_id": SID,
            "previous_results": [
                {"trial_id": "t", "metrics": {"accuracy": math.nan, "cost": 2}}
            ],
        },
        explicit=True,
    )
    assert out["previous_results"][0]["metrics"] == {"cost": 2}


@pytest.mark.parametrize("key", ["duration", "cost_floor_usd"])
def test_non_finite_top_level_scalar_is_refused(key: str) -> None:
    with pytest.raises(oo.MetadataContractViolation):
        oo.project_for_send(
            oo.PHASE_RESULTS, {"trial_id": "t", key: math.nan}, explicit=True
        )


def test_wire_form_not_python_form_is_judged() -> None:
    class _Key(str):
        def __hash__(self) -> int:
            return hash("config")

        def __eq__(self, other: object) -> bool:
            return other == "config"

    for phase in URLS:
        body: dict[str, Any] = {_Key(SECRET): {"a": 1}}
        with pytest.raises(oo.MetadataContractViolation):
            oo.project_for_send(phase, body, explicit=True)


# ------------------------------------------------------------ mode off is untouched


def test_mode_off_returns_the_payload_itself() -> None:
    body = {"trial_id": "t", "config": {"prompt": SECRET}, "anything": [SECRET]}
    for phase in URLS:
        assert oo.project_for_send(phase, body, explicit=None) is body
        assert oo.project_for_send(phase, body, explicit=False) is body


# ------------------------------------------------------------ SECRET-nesting matrix

_WRONG_VALUES = [
    pytest.param({"prompt": SECRET}, id="nested"),
    pytest.param(SECRET, id="bare"),
    pytest.param([SECRET], id="list"),
    pytest.param({"a": {"prompt": [SECRET]}}, id="deep"),
]


@pytest.mark.parametrize("value", _WRONG_VALUES)
@pytest.mark.parametrize("key", sorted(_RESULT_KEYS))
@pytest.mark.parametrize("phase", [oo.PHASE_RESULTS, oo.PHASE_SUMMARY_SUBMISSION])
def test_result_keys_never_pass_nested_content(
    phase: str, key: str, value: Any
) -> None:
    _refused_or_clean(phase, {"trial_id": "t", key: value})


@pytest.mark.parametrize("value", _WRONG_VALUES)
@pytest.mark.parametrize("key", sorted(_OPTIMIZER_KEYS))
def test_optimizer_keys_never_pass_nested_content(key: str, value: Any) -> None:
    _refused_or_clean(oo.PHASE_OPTIMIZER_METRICS, {"trial_id": "t", key: value})


@pytest.mark.parametrize("value", _WRONG_VALUES)
@pytest.mark.parametrize("key", sorted(_NEXT_TRIAL_KEYS))
def test_next_trial_keys_never_pass_nested_content(key: str, value: Any) -> None:
    _refused_or_clean(oo.PHASE_NEXT_TRIAL, {"session_id": SID, key: value})


@pytest.mark.parametrize("value", _WRONG_VALUES)
@pytest.mark.parametrize("key", sorted(_RESULT_KEYS))
def test_next_trial_previous_result_keys_never_pass_nested_content(
    key: str, value: Any
) -> None:
    _refused_or_clean(
        oo.PHASE_NEXT_TRIAL,
        {"session_id": SID, "previous_results": [{"trial_id": "t", key: value}]},
    )


@pytest.mark.parametrize("value", _WRONG_VALUES)
@pytest.mark.parametrize("key", sorted(_PARENT_RUN_KEYS))
def test_parent_run_keys_never_pass_nested_content(key: str, value: Any) -> None:
    _refused_or_clean(oo.PHASE_PARENT_RUN, {key: value})


@pytest.mark.parametrize("value", _WRONG_VALUES)
@pytest.mark.parametrize(
    "key", ["id", "run_id", "agent_id", "benchmark_id", "measures", "metadata"]
)
def test_parent_run_experiment_data_keys_never_pass_nested_content(
    key: str, value: Any
) -> None:
    _refused_or_clean(oo.PHASE_PARENT_RUN, {"experiment_data": {key: value}})


# ------------------------------------------------------------ configuration-run phase

_CONFIG_RUN_KEYS = sorted(oo.CONFIG_RUN_FIELDS)
_CONFIG_RUN_URL = f"{API}/configuration-runs/c1/status"


def _config_run_refused_or_clean(body: Any) -> None:
    try:
        projected = oo.project_for_send(
            oo.PHASE_CONFIGURATION_RUN, body, explicit=True, declared=()
        )
        raw = json.dumps(projected).encode()
        oo.enforce_outgoing("PUT", _CONFIG_RUN_URL, {}, raw, explicit=True)
    except (oo.MetadataContractViolation, oo.OptimizationOnlyRefusal):
        return
    assert SECRET not in raw.decode()


@pytest.mark.parametrize("value", _WRONG_VALUES)
@pytest.mark.parametrize("key", _CONFIG_RUN_KEYS)
def test_configuration_run_keys_never_pass_nested_content(key: str, value: Any) -> None:
    _config_run_refused_or_clean({key: value})


@pytest.mark.parametrize("extra", [{"free": SECRET}, {"id": "forged"}, {SECRET: 1}])
def test_configuration_run_unknown_key_is_refused_without_echo(
    extra: dict[str, Any],
) -> None:
    with pytest.raises(oo.MetadataContractViolation) as caught:
        oo.project_for_send(
            oo.PHASE_CONFIGURATION_RUN, {"status": "COMPLETED", **extra}, explicit=True
        )
    assert SECRET not in str(caught.value)


def test_configuration_run_experiment_parameters_use_the_config_check() -> None:
    ok = {"model": "gpt-4o", "k": 3, "p": "[REDACTED:5 chars]"}
    out = oo.project_for_send(
        oo.PHASE_CONFIGURATION_RUN, {"experiment_parameters": ok}, explicit=True
    )
    assert out["experiment_parameters"] == ok
    with pytest.raises(oo.MetadataContractViolation):
        oo.project_for_send(
            oo.PHASE_CONFIGURATION_RUN,
            {"experiment_parameters": {"prompt": SECRET}},
            explicit=True,
        )


def test_configuration_run_mode_off_is_untouched() -> None:
    body = {"experiment_parameters": {"prompt": SECRET}, "x": 1}
    assert oo.project_for_send(oo.PHASE_CONFIGURATION_RUN, body, explicit=None) is body


@pytest.mark.asyncio
async def test_configuration_run_producer_bodies_are_accepted() -> None:
    from traigent.cloud.api_operations import ApiOperations

    session, session_ctx = _post_mocks()
    session.put = session.post
    client = Mock()
    client.optimization_only = True
    client.backend_config.api_base_url = API
    client.auth_manager = AsyncMock()
    client.auth_manager.augment_headers = AsyncMock(return_value={})
    client._map_to_backend_status = Mock(return_value="COMPLETED")
    ops = ApiOperations(client)
    bodies = [
        {"status": "COMPLETED"},
        {"status": "FAILED", "error_message": "trial_failed"},
        {"measures": [dict(_METRICS)]},
        {"measures": None},
        {"summary_stats": {"metrics": {"accuracy": {"mean": 0.5}}}},
    ]
    for body in bodies:
        projected = oo.project_for_send(oo.PHASE_CONFIGURATION_RUN, body, explicit=True)
        oo.enforce_outgoing(
            "PUT", _CONFIG_RUN_URL, {}, json.dumps(projected).encode(), explicit=True
        )
    assert ops is not None
    # intermediate report and finalize producers
    report = {
        "session_id": SID,
        "trial_id": "t1",
        "running_score": 0.5,
        "examples_attempted": 3,
        "objective_name": "accuracy",
        "partial_cost_usd": 0.01,
    }
    oo.enforce_outgoing(
        "POST",
        f"{API}/sessions/{SID}/intermediate-report",
        {},
        json.dumps(report).encode(),
        explicit=True,
    )
    with pytest.raises(oo.OptimizationOnlyRefusal):
        oo.enforce_outgoing(
            "POST",
            f"{API}/sessions/{SID}/intermediate-report",
            {},
            json.dumps({**report, "note": SECRET}).encode(),
            explicit=True,
        )
    for fin in (
        {"reason": "sdk_explicit_finalization", "experiment_run_id": "run_1"},
        {"session_id": SID, "include_full_history": False, "metadata": {}},
    ):
        oo.enforce_outgoing(
            "POST",
            f"{API}/sessions/{SID}/finalize",
            {},
            json.dumps(fin).encode(),
            explicit=True,
        )
