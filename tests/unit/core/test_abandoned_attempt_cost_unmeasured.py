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
