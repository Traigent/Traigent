"""Round 19 (#2517): the cost limit bounds known spend in unknown-cost mode."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from traigent.api.types import TrialResult, TrialStatus
from traigent.core.cost_enforcement import (
    CostEnforcer,
    CostEnforcerConfig,
    unmeasured_cost_ran_message,
    unmeasured_cost_stop_message,
)
from traigent.core.execution_budget import ExecutionBudget
from traigent.core.orchestrator import OptimizationOrchestrator

_COST_ENV = (
    "TRAIGENT_STRICT_COST_ACCOUNTING",
    "TRAIGENT_REQUIRE_COST_TRACKING",
    "TRAIGENT_RUN_COST_LIMIT",
)


def _flagged_trial(floor: float = 0.5, trial_id: str = "t") -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        config={},
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime.now(UTC),
        metrics={
            "accuracy": 0.9,
            "cost_unmeasured": 1.0,
            "cost_lower_bound": floor,
        },
    )


def _unknown_mode_enforcer(
    monkeypatch: pytest.MonkeyPatch, limit: float
) -> CostEnforcer:
    for key in _COST_ENV:
        monkeypatch.delenv(key, raising=False)
    enforcer = CostEnforcer(CostEnforcerConfig(limit=limit))
    permit = enforcer.acquire_permit()
    enforcer.track_cost(cost=None, permit=permit)  # enters unknown-cost mode
    return enforcer


def test_unwaived_unknown_cost_run_stops_when_known_spend_reaches_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """probe_limit_e2e shape: $1 limit, $0.50 floors, no max_trials."""
    for key in _COST_ENV:
        monkeypatch.delenv(key, raising=False)
    enforcer = CostEnforcer(CostEnforcerConfig(limit=1.0))
    ran = 0
    for _ in range(10):
        permit = enforcer.acquire_permit()
        if not permit.active:
            break
        enforcer.track_cost(cost=0.5, permit=permit, cost_unmeasured=True)
        ran += 1
        if enforcer.is_limit_reached:
            break
    assert enforcer.get_status().unknown_cost_mode
    assert ran == 2
    assert enforcer.accumulated_cost == pytest.approx(1.0)
    assert enforcer.is_limit_reached
    assert not enforcer.acquire_permit().active


def test_trial_count_fallback_still_bounds_unseen_spend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enforcer = _unknown_mode_enforcer(monkeypatch, limit=1000.0)
    limit = enforcer.config.fallback_trial_limit
    for _ in range(limit):
        permit = enforcer.acquire_permit()
        if not permit.active:
            break
        enforcer.track_cost(cost=None, permit=permit)
    assert enforcer.is_limit_reached
    assert not enforcer.acquire_permit().active


def test_stop_detail_says_known_spend_reached_the_limit() -> None:
    status = SimpleNamespace(
        unknown_cost_mode=True,
        unmeasured_trial_count=2,
        trial_count=2,
        accumulated_cost_usd=20.0,
        limit_usd=1.0,
    )
    enforcer = SimpleNamespace(
        unknown_cost_cap_waived=False,
        get_status=lambda: status,
        config=SimpleNamespace(fallback_trial_limit=10),
    )
    orch = SimpleNamespace(
        _stop_reason="cost_limit", _trials=[1, 2], cost_enforcer=enforcer
    )
    detail = OptimizationOrchestrator._describe_stop(orch)
    assert detail is not None
    assert "cannot bound spend" not in detail
    assert "at least $20.00" in detail
    assert "$1.00" in detail


def test_stop_message_says_limit_reached_when_known_spend_exceeds_it() -> None:
    msg = unmeasured_cost_stop_message(2, 10, 1.0, 1, known_spend=1.5)
    assert "cannot bound spend" not in msg
    assert "at least $1.50" in msg
    assert "cost limit ($1.00) was reached" in msg
    # Without known spend at the limit the original wording stands.
    assert "cannot bound spend" in unmeasured_cost_stop_message(2, 10, 1.0, 1)


def test_sized_run_warning_does_not_claim_limit_failed_when_it_stopped_run() -> None:
    msg = unmeasured_cost_ran_message(3, 1, known_spend=2.0, cost_limit=1.0)
    assert "did not bound" not in msg
    assert "at least $2.00" in msg
    assert "did not bound" in unmeasured_cost_ran_message(3, 1)


def test_untracked_marker_reaches_the_execution_budget() -> None:
    """A flagged trial is charged its floor but stays 'untracked' in the ledger."""
    budget = ExecutionBudget(max_cost_usd=100.0)
    enforcer = SimpleNamespace(track_cost_async=_noop_async)
    orch = SimpleNamespace(
        cost_enforcer=enforcer,
        execution_budget=budget,
        _extract_trial_cost=lambda t: 0.5,
        _examples_of=lambda t: 1,
    )
    asyncio.run(OptimizationOrchestrator._account_trial_spend(orch, _flagged_trial()))
    snap = budget.snapshot()
    assert snap.consumed_cost == pytest.approx(0.5)
    assert snap.untracked_trials == 1


async def _noop_async(**_kwargs) -> None:
    return None


def test_completion_log_says_at_least_when_cost_is_a_floor() -> None:
    src = _completion_log_text(unmeasured=2, cost=1.5)
    assert "at least $1.5000" in src
    assert "some cost unmeasured" in src
    exact = _completion_log_text(unmeasured=0, cost=1.5)
    assert "at least" not in exact
    assert "total cost: $1.5000" in exact


def _completion_log_text(*, unmeasured: int, cost: float) -> str:
    """Drive the real completion-log code path via the orchestrator method."""
    status = SimpleNamespace(
        accumulated_cost_usd=cost, unmeasured_trial_count=unmeasured
    )
    records: list[str] = []

    class _H(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    logger = logging.getLogger("traigent.core.orchestrator")
    handler = _H()
    old = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        orch = SimpleNamespace(
            cost_enforcer=SimpleNamespace(get_status=lambda: status),
            _optimization_id="opt",
            _trials=[1],
        )
        OptimizationOrchestrator._log_completion_cost(
            orch, SimpleNamespace(best_score=0.9)
        )
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old)
    return " ".join(records)
