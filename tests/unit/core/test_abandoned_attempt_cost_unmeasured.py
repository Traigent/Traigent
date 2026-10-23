"""An abandoned vendor attempt is unmeasured spend for the run (SDK-41, P1-A).

An exhausted call that becomes ``VendorPauseError`` produces no ``TrialResult``,
so its provider spend is unknown. It must never be scored as free: strict cost
tracking fails closed, and a later successful trial cannot leave the run total
looking fully measured.
"""

from __future__ import annotations

import types
from unittest.mock import AsyncMock

import pytest

from tests.unit.core.test_trial_lifecycle import MockOrchestrator, create_mock_dataset
from traigent.api.types import OptimizationResult
from traigent.core.cost_enforcement import (
    CostEnforcer,
    CostEnforcerConfig,
    CostTrackingRequiredError,
)
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.core.trial_lifecycle import TrialLifecycle
from traigent.utils.exceptions import RateLimitError, VendorPauseError
from traigent.utils.langchain_interceptor import capture_scope


def _result(total_cost: float | None = 0.5) -> OptimizationResult:
    return OptimizationResult(
        trials=[],
        best_config={},
        best_score=1.0,
        optimization_id="o",
        duration=1.0,
        convergence_info={},
        status="completed",
        objectives=["accuracy"],
        algorithm="grid",
        timestamp=__import__("datetime").datetime.now(),
        total_cost=total_cost,
        metrics={"total_cost": total_cost} if total_cost is not None else {},
    )


def test_abandoned_unmeasured_attempt_makes_run_total_unknown() -> None:
    orch = types.SimpleNamespace(
        _abandoned_vendor_attempts=[
            {"trial_id": "t1", "vendor_call_unmeasured_attempts": 1},
            {"trial_id": "t2", "vendor_call_unmeasured_attempts": 0},
        ]
    )
    result = _result(0.5)

    OptimizationOrchestrator._apply_abandoned_attempts_cost(orch, result)

    assert result.total_cost is None
    assert "total_cost" not in result.metrics
    assert result.metrics["cost_floor_usd"] == pytest.approx(0.5)
    assert result.metrics["cost_unmeasured_trials"] == 1.0


def test_fully_measured_abandoned_attempt_leaves_total_alone() -> None:
    orch = types.SimpleNamespace(
        _abandoned_vendor_attempts=[
            {"trial_id": "t1", "vendor_call_unmeasured_attempts": 0}
        ]
    )
    result = _result(0.5)
    OptimizationOrchestrator._apply_abandoned_attempts_cost(orch, result)
    assert result.total_cost == 0.5


async def _pause_trial(monkeypatch, *, strict: bool):
    from traigent.utils.langchain_interceptor import record_vendor_attempt

    monkeypatch.setenv("TRAIGENT_REQUIRE_COST_TRACKING", "true" if strict else "false")
    monkeypatch.delenv("TRAIGENT_STRICT_COST_ACCOUNTING", raising=False)
    orch = MockOrchestrator()
    orch._abandoned_vendor_attempts = []
    orch.cost_enforcer = CostEnforcer(
        config=CostEnforcerConfig(limit=10.0, approved=True)
    )
    orch._account_untracked_trial_attempt = types.MethodType(
        OptimizationOrchestrator._account_untracked_trial_attempt, orch
    )
    orch.execution_budget = None
    lifecycle = TrialLifecycle(orch)

    async def evaluate(*a, **k):
        with capture_scope():
            record_vendor_attempt(
                {"retried": True, "usage_known": False, "outcome": "error"}
            )
        raise RateLimitError("429")

    orch.evaluator.evaluate = AsyncMock(side_effect=evaluate)
    lifecycle._collect_workflow_span = lambda *a, **k: None
    return lifecycle, orch


@pytest.mark.asyncio
async def test_sequential_pause_with_abandoned_attempt_fails_closed_in_strict_mode(
    monkeypatch,
) -> None:
    lifecycle, orch = await _pause_trial(monkeypatch, strict=True)

    async def func(input_data):
        return "x"

    with pytest.raises(CostTrackingRequiredError):
        await lifecycle.run_sequential_trial(
            func=func,
            dataset=create_mock_dataset(),
            session_id=None,
            function_name="f",
            trial_count=0,
        )
    assert orch.cost_enforcer.get_status().unmeasured_trial_count == 1


@pytest.mark.asyncio
async def test_sequential_pause_counts_unmeasured_and_keeps_pause_when_not_strict(
    monkeypatch,
) -> None:
    lifecycle, orch = await _pause_trial(monkeypatch, strict=False)

    async def func(input_data):
        return "x"

    with pytest.raises(VendorPauseError):
        await lifecycle.run_sequential_trial(
            func=func,
            dataset=create_mock_dataset(),
            session_id=None,
            function_name="f",
            trial_count=0,
        )
    status = orch.cost_enforcer.get_status()
    assert status.unmeasured_trial_count == 1
    assert status.in_flight_count == 0  # permit released exactly once


def _measured_trial_result(ledger: dict[str, float]) -> OptimizationResult:
    from datetime import UTC, datetime

    from traigent.api.types import TrialResult, TrialStatus

    trial = TrialResult(
        trial_id="t-ok",
        config={"v": "a"},
        metrics={"accuracy": 1.0, "total_cost": 0.5},
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime.now(UTC),
        metadata={},
    )
    result = _result(None)
    result.trials = [trial]
    result.metrics = dict(ledger)
    return result


def test_public_summaries_honor_run_level_unmeasured_ledger() -> None:
    result = _measured_trial_result(
        {"cost_unmeasured_trials": 1.0, "cost_floor_usd": 0.7}
    )

    stats = result.experiment_stats
    summary = result.get_summary()

    assert stats.total_cost is None
    assert stats.cost_unmeasured_trials == 1
    assert stats.cost_lower_bound == pytest.approx(0.7)
    assert summary["total_cost"] is None
    assert summary["cost_unmeasured_trials"] == 1
    assert summary["cost_lower_bound"] == pytest.approx(0.7)


def test_measured_run_without_ledger_keeps_numeric_total() -> None:
    result = _measured_trial_result({})
    assert result.get_summary()["total_cost"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_abandoned_winner_stability_rerun_counts_once() -> None:
    """The lifecycle files the attempt in the abandoned ledger AND the
    stability counter records it: one attempt must count once."""
    import asyncio

    entry = {"trial_id": "lifecycle-id", "vendor_call_unmeasured_attempts": 1}
    orch = types.SimpleNamespace(
        _abandoned_vendor_attempts=[entry],
        _winner_stability_ledger_start=0,
        _winner_stability_recorded_ids={},
        _winner_stability_unmeasured_reruns=0,
        _state_lock=asyncio.Lock(),
        _account_untracked_trial_attempt=AsyncMock(),
    )
    orch._winner_stability_mark_recorded = types.MethodType(
        OptimizationOrchestrator._winner_stability_mark_recorded, orch
    )

    await OptimizationOrchestrator._account_winner_stability_rerun_attempt(
        orch,
        rerun_trial=None,
        rerun_dispatched=True,
        permit=None,
        trial_id="winner_stability_o_1",
    )
    # Cancellation path: rebuild the result (counts the abandoned ledger), then
    # add the stability attempt.
    result = _result(0.5)
    OptimizationOrchestrator._apply_abandoned_attempts_cost(orch, result)
    OptimizationOrchestrator._add_winner_stability_cost_to_result(
        result, 0.0, orch._winner_stability_unmeasured_reruns
    )

    assert result.metrics["cost_unmeasured_trials"] == 1.0
