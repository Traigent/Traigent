"""Budget-armed submissions: the floor is its own field, never ``metrics.cost``.

For a flagged (unmeasured) trial the SDK sends ``cost_unmeasured = 1.0`` plus
the spend actually reported (an exact lower bound, 0.0 when nothing was
reported) under ``COST_FLOOR_WIRE_KEY``. ``metrics["cost"]`` stays absent: an
unknown cost is never a measured number (#2688).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from traigent.cloud.session_budgets import (
    ensure_cost_metric_for_budgeted_completed_submission as ensure,
)
from traigent.core.backend_session_manager import BackendSessionManager
from traigent.utils.trial_costs import COST_FLOOR_WIRE_KEY, known_spend_lower_bound


def _client() -> SimpleNamespace:
    return SimpleNamespace(_cost_budget_armed_sessions={"s1"})


def test_flagged_trial_without_number_carries_zero_floor_and_flag() -> None:
    metrics = {"accuracy": 1.0, "cost_unmeasured": 1.0}
    ensure(client=_client(), session_id="s1", metrics=metrics, status="COMPLETED")
    assert metrics["cost"] == metrics[COST_FLOOR_WIRE_KEY]  # TEMPORARY bridge (#2688)
    assert metrics[COST_FLOOR_WIRE_KEY] == 0.0
    assert metrics["cost_unmeasured"] == 1.0


def test_flagged_trial_sends_reported_spend_as_floor_field_with_flag() -> None:
    metrics = {"accuracy": 1.0, "cost_unmeasured": 1.0}
    ensure(
        client=_client(),
        session_id="s1",
        metrics=metrics,
        status="COMPLETED",
        telemetry_sources=({"total_cost": 0.4},),
    )
    assert metrics["cost"] == metrics[COST_FLOOR_WIRE_KEY]  # TEMPORARY bridge (#2688)
    assert metrics[COST_FLOOR_WIRE_KEY] == pytest.approx(0.4)
    assert metrics["cost_unmeasured"] == 1.0


def test_flagged_trial_moves_existing_cost_into_floor_field() -> None:
    metrics = {"cost": 0.3, "cost_unmeasured": 1.0}
    ensure(client=_client(), session_id="s1", metrics=metrics, status="COMPLETED")
    # Armed session: floor in its own field plus the TEMPORARY cost bridge.
    assert metrics == {COST_FLOOR_WIRE_KEY: 0.3, "cost": 0.3, "cost_unmeasured": 1.0}


def test_floor_never_lowers_reported_spend_budget_counts_floor() -> None:
    metrics = {"cost": 0.0, "cost_lower_bound": 0.7, "cost_unmeasured": 1.0}
    ensure(
        client=_client(),
        session_id="s1",
        metrics=metrics,
        status="COMPLETED",
        telemetry_sources=({"total_cost": 0.4},),
    )
    assert metrics["cost"] == metrics[COST_FLOOR_WIRE_KEY]  # TEMPORARY bridge (#2688)
    assert metrics[COST_FLOOR_WIRE_KEY] == pytest.approx(0.7)


def test_measured_trial_unchanged_and_unflagged() -> None:
    metrics = {"accuracy": 1.0, "cost": 0.25}
    mutated = ensure(
        client=_client(), session_id="s1", metrics=metrics, status="COMPLETED"
    )
    assert mutated is False
    assert metrics == {"accuracy": 1.0, "cost": 0.25}


def test_measured_zero_stays_unflagged() -> None:
    metrics = {"accuracy": 1.0, "cost": 0.0}
    mutated = ensure(
        client=_client(), session_id="s1", metrics=metrics, status="COMPLETED"
    )
    assert mutated is False
    assert metrics == {"accuracy": 1.0, "cost": 0.0}
    assert "cost_unmeasured" not in metrics


def test_flagged_trial_on_unarmed_session_omits_cost() -> None:
    metrics = {"accuracy": 1.0, "cost_unmeasured": 1.0}
    ensure(
        client=SimpleNamespace(_cost_budget_armed_sessions=set()),
        session_id="s1",
        metrics=metrics,
        status="COMPLETED",
    )
    assert "cost" not in metrics
    assert metrics[COST_FLOOR_WIRE_KEY] == 0.0
    assert metrics["cost_unmeasured"] == 1.0


def test_known_spend_lower_bound_defaults_to_zero() -> None:
    assert known_spend_lower_bound({}, None) == 0.0


def _manager(armed: set[str]) -> BackendSessionManager:
    manager = BackendSessionManager.__new__(BackendSessionManager)
    manager._session_cost_budget_armed = armed
    manager._cost_budget_zero_backfill_logged = False
    return manager


def _trial(metrics: dict) -> SimpleNamespace:
    from traigent.api.types import TrialStatus

    return SimpleNamespace(status=TrialStatus.COMPLETED, metrics=metrics, metadata={})


def test_session_manager_flagged_trial_armed_sends_floor_field_and_flag() -> None:
    payload = {"accuracy": 1.0, "cost_unmeasured": 1.0}
    _manager({"s1"})._ensure_budget_cost_metric(
        session_id="s1",
        trial_result=_trial(dict(payload)),
        metrics_payload=payload,
        metadata={},
    )
    assert payload["cost"] == payload[COST_FLOOR_WIRE_KEY]  # TEMPORARY bridge (#2688)
    assert payload[COST_FLOOR_WIRE_KEY] == 0.0
    assert payload["cost_unmeasured"] == 1.0


def test_session_manager_flagged_trial_unarmed_omits_cost() -> None:
    payload = {"accuracy": 1.0, "cost_unmeasured": 1.0}
    _manager(set())._ensure_budget_cost_metric(
        session_id="s1",
        trial_result=_trial(dict(payload)),
        metrics_payload=payload,
        metadata={},
    )
    assert "cost" not in payload
    assert payload[COST_FLOOR_WIRE_KEY] == 0.0
