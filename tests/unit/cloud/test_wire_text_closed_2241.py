"""Closed codes and key allowlists for text sent to the backend (#2241)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

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
