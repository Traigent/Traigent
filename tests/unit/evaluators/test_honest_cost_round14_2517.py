"""Honest cost round 14 (#2517, owner decision A): reported spend is a floor
that survives every writer, and an unmeasured cost is never a measured zero.

Each test reproduces one executed probe from the pre-merge review.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from traigent.api.types import OptimizationResult, TrialResult, TrialStatus
from traigent.cloud.session_budgets import (
    ensure_cost_metric_for_budgeted_completed_submission as ensure,
)
from traigent.core.evaluator_wrapper import CustomEvaluatorWrapper
from traigent.integrations.observability import workflow_traces as wt
from traigent.observability import add_agent_span
from traigent.utils.trial_costs import COST_FLOOR_WIRE_KEY, known_spend_lower_bound

_SPAN_REQUIRED = {
    "span_id",
    "trace_id",
    "configuration_run_id",
    "span_name",
    "span_type",
    "start_time",
    "status",
    "input_tokens",
    "output_tokens",
    "cost_usd",
}
_SPAN_ALLOWED = _SPAN_REQUIRED | {
    "parent_span_id",
    "node_id",
    "end_time",
    "error_message",
    "decision_reason",
    "input_data",
    "output_data",
    "metadata",
}
_FLAG_KEY = "traigent.cost_unmeasured"


# --- B1: custom evaluation keeps the reported floor ---------------------------
def test_b1_custom_evaluation_keeps_reported_floor_with_flag() -> None:
    rows: list[dict[str, Any]] = [
        {"total_cost": 0.40, "input_cost": 0.25, "output_cost": 0.15},
        {"cost_unmeasured": 1.0},
    ]
    aggregated = CustomEvaluatorWrapper._aggregate_llm_metrics(
        SimpleNamespace(), rows, []
    )

    assert aggregated["cost_unmeasured"] == 1.0
    assert "total_cost" not in aggregated
    assert aggregated["cost_lower_bound"] == pytest.approx(0.40)

    # Budget submission: the floor travels with the flag, never a bare zero.
    client = SimpleNamespace(_cost_budget_armed_sessions={"s1"})
    metrics = dict(aggregated)
    ensure(client=client, session_id="s1", metrics=metrics, status="COMPLETED")
    assert metrics["cost"] == metrics[COST_FLOOR_WIRE_KEY]  # TEMPORARY bridge (#2688)
    assert metrics[COST_FLOOR_WIRE_KEY] == pytest.approx(0.40)
    assert metrics["cost_unmeasured"] == 1.0


def test_b1_no_reported_spend_has_no_floor() -> None:
    aggregated = CustomEvaluatorWrapper._aggregate_llm_metrics(
        SimpleNamespace(), [{"cost_unmeasured": 1.0}], []
    )
    assert aggregated["cost_unmeasured"] == 1.0
    assert "cost_lower_bound" not in aggregated


# --- B2: a flagged zero never overrides the reported floor -------------------
def test_b2_flagged_zero_cost_is_lifted_to_the_reported_floor() -> None:
    client = SimpleNamespace(_cost_budget_armed_sessions={"s1"})
    metrics = {"cost": 0, "cost_lower_bound": 0.4, "cost_unmeasured": 1}
    ensure(client=client, session_id="s1", metrics=metrics, status="COMPLETED")
    assert metrics["cost"] == metrics[COST_FLOOR_WIRE_KEY]  # TEMPORARY bridge (#2688)
    assert metrics[COST_FLOOR_WIRE_KEY] == pytest.approx(0.4)
    assert metrics["cost_unmeasured"] == 1


def test_b2_session_manager_lifts_flagged_zero_too() -> None:
    from traigent.core.backend_session_manager import BackendSessionManager

    manager = BackendSessionManager.__new__(BackendSessionManager)
    manager._session_cost_budget_armed = {"s1"}
    payload = {"cost": 0.0, "cost_lower_bound": 0.4, "cost_unmeasured": 1.0}
    trial = SimpleNamespace(status=_completed(), metrics=dict(payload), metadata={})
    manager._ensure_budget_cost_metric(
        session_id="s1", trial_result=trial, metrics_payload=payload, metadata={}
    )
    assert payload["cost"] == payload[COST_FLOOR_WIRE_KEY]  # TEMPORARY bridge (#2688)
    assert payload[COST_FLOOR_WIRE_KEY] == pytest.approx(0.4)
    assert payload["cost_unmeasured"] == 1.0


def _completed() -> Any:
    return TrialStatus.COMPLETED


def test_b2_measured_zero_is_untouched() -> None:
    client = SimpleNamespace(_cost_budget_armed_sessions={"s1"})
    metrics = {"cost": 0.0}
    assert not ensure(
        client=client, session_id="s1", metrics=metrics, status="COMPLETED"
    )
    assert metrics == {"cost": 0.0}


# --- B3/B4: span payload satisfies the schema contract -----------------------
def _installed_trace_schema() -> Path | None:
    """The one intended schema: the copy shipped in the installed schema package.

    Never glob the repo tree: stray or vendored copies elsewhere are not the
    contract and their relative refs need not resolve.
    """
    try:
        import traigent_schema
    except ImportError:  # pragma: no cover - schema package not installed
        return None
    candidate = (
        Path(traigent_schema.__file__).resolve().parent
        / "schemas"
        / "execution"
        / "workflow_trace_schema.json"
    )
    return candidate if candidate.is_file() else None


def _validate_against_schema(payload: dict[str, Any], schema_file: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    referencing = pytest.importorskip("referencing")
    from referencing.jsonschema import DRAFT7

    root = schema_file.parent.resolve()

    def retrieve(uri: str) -> Any:
        # Relative refs such as ../status_schema.json resolve against the
        # schema's own directory, and may not escape the schema package.
        target = (root / uri).resolve()
        target.relative_to(root.parent)
        return referencing.Resource.from_contents(
            json.loads(target.read_text()), default_specification=DRAFT7
        )

    registry = referencing.Registry(retrieve=retrieve)
    document = json.loads(schema_file.read_text())
    span_schema = {
        **document["definitions"]["SpanPayload"],
        "definitions": document["definitions"],
    }
    jsonschema.Draft7Validator(span_schema, registry=registry).validate(payload)


def _assert_schema_shape(payload: dict[str, Any]) -> None:
    schema_file = _installed_trace_schema()
    if schema_file is not None:
        _validate_against_schema(payload, schema_file)
    assert _SPAN_REQUIRED <= set(payload), _SPAN_REQUIRED - set(payload)
    assert set(payload) <= _SPAN_ALLOWED, set(payload) - _SPAN_ALLOWED
    assert isinstance(payload["cost_usd"], (int, float))
    assert payload["cost_usd"] >= 0


def _tracker(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(wt, "requests", MagicMock())
    monkeypatch.setattr(wt, "REQUESTS_AVAILABLE", True)
    return wt.WorkflowTracesTracker(
        backend_url="http://localhost:5000", auth_token="t", auto_send=False
    )


def test_b3_unmeasured_span_emits_numeric_cost_and_namespaced_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = _tracker(monkeypatch)
    with tracker.trace_trial("config_1"):
        tracker.add_span(
            span_id="s1", span_name="n", span_type="node", start_time=datetime.now(UTC)
        )
        tracker.add_span(
            span_id="s2",
            span_name="n",
            span_type="node",
            start_time=datetime.now(UTC),
            cost_usd=0.0,
        )
        tracker.add_span(
            span_id="s3",
            span_name="n",
            span_type="node",
            start_time=datetime.now(UTC),
            cost_usd=0.25,
        )
        payloads = [span.to_dict() for span in tracker._spans]

    unreported, reported_zero, reported = payloads
    for payload in payloads:
        _assert_schema_shape(payload)
    assert unreported["cost_usd"] == 0.0
    assert unreported["metadata"][_FLAG_KEY] is True
    assert "cost_unmeasured" not in unreported  # never top level
    assert reported_zero["cost_usd"] == 0.0
    assert _FLAG_KEY not in reported_zero.get("metadata", {})
    assert reported["cost_usd"] == 0.25
    assert _FLAG_KEY not in reported.get("metadata", {})


def test_b3_unmeasured_span_carries_reported_floor() -> None:
    span = wt.SpanPayload(
        span_id="s",
        trace_id="t",
        configuration_run_id="c",
        span_name="n",
        span_type="node",
        start_time="2026-01-01T00:00:00+00:00",
        cost_usd=None,
        cost_lower_bound=0.4,
    )
    payload = span.to_dict()
    _assert_schema_shape(payload)
    assert payload["cost_usd"] == pytest.approx(0.4)
    assert payload["metadata"][_FLAG_KEY] is True


def test_b4_agent_span_without_cost_is_marked_unmeasured() -> None:
    from tests.unit.observability.test_agent_spans import _active_trial, _manager

    manager = _manager()
    with _active_trial(manager):
        add_agent_span("llm-node", input_tokens=10)
        add_agent_span("measured-node", input_tokens=10, cost_usd=0.0)

    spans = manager._group_spans_by_config_run()["config-run-1"]
    omitted, measured_zero = (s.to_dict() for s in spans)
    _assert_schema_shape(omitted)
    _assert_schema_shape(measured_zero)
    assert omitted["cost_usd"] == 0.0
    assert omitted["metadata"][_FLAG_KEY] is True
    assert measured_zero["cost_usd"] == 0.0
    assert _FLAG_KEY not in measured_zero.get("metadata", {})


# --- F1: run summaries surface the preserved floor ---------------------------
def _flagged_trial(trial_id: str, **metrics: Any) -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        config={"id": trial_id},
        metrics={"cost_unmeasured": 1.0, **metrics},
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime.now(UTC),
    )


def _result(trials: list[TrialResult]) -> OptimizationResult:
    return OptimizationResult(
        trials=trials,
        best_config={"id": "a"},
        best_score=0.5,
        optimization_id="o",
        duration=1.0,
        convergence_info={},
        status="completed",
        objectives=["accuracy"],
        algorithm="grid",
        timestamp=datetime.now(),
    )


def test_f1_summaries_surface_floor_preserved_on_a_flagged_trial() -> None:
    result = _result([_flagged_trial("a", cost_lower_bound=0.4)])
    assert known_spend_lower_bound(result.trials[0].metrics) == pytest.approx(0.4)

    summary = result.get_summary()
    stats = result.experiment_stats
    assert stats.total_cost is None
    assert stats.cost_lower_bound == pytest.approx(0.4)
    assert summary["total_cost"] is None
    assert summary["cost_lower_bound"] == pytest.approx(0.4)
