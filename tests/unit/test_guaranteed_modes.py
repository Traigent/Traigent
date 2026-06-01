"""Fail-closed guard + request-builder tests for the SDK guaranteed-modes helpers."""

from __future__ import annotations

import pytest

from traigent.guaranteed_modes import (
    build_guaranteed_selection_request,
    deployable_config,
    is_deployable,
)


def _certified(cfg: str = "cfg_cheap") -> dict:
    return {
        "status": "CERTIFIED_SELECTION",
        "deployable": True,
        "selected_config": cfg,
        "fallback_config": "cfg_base",
        "certificate": {},
    }


def test_certified_result_is_deployable() -> None:
    result = _certified()
    assert is_deployable(result)
    assert deployable_config(result) == "cfg_cheap"


def test_no_decision_is_not_deployable() -> None:
    result = {
        "status": "NO_DECISION_BASELINE_FALLBACK",
        "deployable": False,
        "selected_config": None,
        "fallback_config": "cfg_base",
    }
    assert not is_deployable(result)
    assert deployable_config(result) is None


def test_best_effort_is_not_deployable() -> None:
    result = {
        "status": "BEST_EFFORT_UNCERTIFIED",
        "deployable": False,
        "selected_config": None,
        "fallback_config": "cfg_base",
    }
    assert not is_deployable(result)
    assert deployable_config(result) is None


def test_certified_with_false_deployable_flag_is_denied() -> None:
    result = _certified()
    result["deployable"] = False
    assert not is_deployable(result)
    assert deployable_config(result) is None


def test_malformed_results_are_denied() -> None:
    assert not is_deployable(None)
    assert deployable_config(None) is None
    assert not is_deployable(
        {"status": "CERTIFIED_SELECTION", "deployable": True, "selected_config": ""}
    )


def test_request_builder_mode1_requires_baseline() -> None:
    with pytest.raises(ValueError, match="baseline_ref"):
        build_guaranteed_selection_request("keep_accuracy_reduce_cost", delta=0.05)
    request = build_guaranteed_selection_request(
        "keep_accuracy_reduce_cost", delta=0.05, baseline_ref="cfg_base"
    )
    assert request["selection_mode"] == "keep_accuracy_reduce_cost"
    assert request["baseline_ref"] == "cfg_base"
    assert request["schema_version"] == "traigent.guaranteed_selection_request.v1"


def test_request_builder_mode2_omits_baseline() -> None:
    request = build_guaranteed_selection_request("accuracy_then_cost", delta=0.02, epsilon=0.1)
    assert request["selection_mode"] == "accuracy_then_cost"
    assert "baseline_ref" not in request


def test_request_builder_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="selection_mode"):
        build_guaranteed_selection_request("maximize_accuracy", delta=0.05)
