"""Budget-armed submissions carry a numeric cost AND the unmeasured flag.

The backend refuses a completed trial on a budget-armed session without a
numeric ``metrics.cost``. For a flagged trial the SDK therefore sends the spend
actually reported (a lower bound, 0.0 when nothing was reported) together with
``cost_unmeasured = 1.0``; the flag, never the number, is the signal.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from traigent.cloud.session_budgets import (
    ensure_cost_metric_for_budgeted_completed_submission as ensure,
)
from traigent.core.backend_session_manager import BackendSessionManager
from traigent.utils.trial_costs import known_spend_lower_bound


def _client() -> SimpleNamespace:
    return SimpleNamespace(_cost_budget_armed_sessions={"s1"})


def test_flagged_trial_without_number_carries_zero_lower_bound_and_flag() -> None:
    metrics = {"accuracy": 1.0, "cost_unmeasured": 1.0}
    ensure(client=_client(), session_id="s1", metrics=metrics, status="COMPLETED")
    assert metrics["cost"] == 0.0
    assert metrics["cost_unmeasured"] == 1.0


def test_flagged_trial_sends_reported_spend_as_lower_bound_with_flag() -> None:
    metrics = {"accuracy": 1.0, "cost_unmeasured": 1.0}
    ensure(
        client=_client(),
        session_id="s1",
        metrics=metrics,
        status="COMPLETED",
        telemetry_sources=({"total_cost": 0.4},),
    )
    assert metrics["cost"] == pytest.approx(0.4)
    assert metrics["cost_unmeasured"] == 1.0


def test_flagged_trial_keeps_existing_cost_and_flag() -> None:
    metrics = {"cost": 0.3, "cost_unmeasured": 1.0}
    ensure(client=_client(), session_id="s1", metrics=metrics, status="COMPLETED")
    assert metrics == {"cost": 0.3, "cost_unmeasured": 1.0}


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


def test_session_manager_flagged_trial_armed_sends_number_and_flag() -> None:
    payload = {"accuracy": 1.0, "cost_unmeasured": 1.0}
    _manager({"s1"})._ensure_budget_cost_metric(
        session_id="s1",
        trial_result=_trial(dict(payload)),
        metrics_payload=payload,
        metadata={},
    )
    assert payload["cost"] == 0.0
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
