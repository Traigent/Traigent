"""Cost caps must not overshoot because of float rounding of equal costs (#3118)."""

from __future__ import annotations

import pytest

from traigent.api.types import TrialResult, TrialStatus
from traigent.core.cost_enforcement import CostEnforcer, CostEnforcerConfig
from traigent.core.execution_budget import ExecutionBudget
from traigent.core.stop_conditions import MetricLimitStopCondition


@pytest.mark.parametrize(
    ("limit", "cost", "expected"), [(1.0, 0.1, 10), (0.9, 0.3, 3), (0.8, 0.1, 8)]
)
def test_cost_enforcer_reports_limit_reached_at_exact_cap(
    limit: float, cost: float, expected: int
) -> None:
    enforcer = CostEnforcer(CostEnforcerConfig(limit=limit, approved=True))
    for _ in range(expected):
        assert not enforcer.is_limit_reached
        permit = enforcer.acquire_permit()
        assert permit.is_granted
        enforcer.track_cost(cost, permit=permit)
    assert enforcer.is_limit_reached


@pytest.mark.parametrize(
    ("limit", "cost", "expected"), [(1.0, 0.1, 10), (0.9, 0.3, 3), (0.8, 0.1, 8)]
)
def test_execution_budget_exhausts_at_exact_cap(
    limit: float, cost: float, expected: int
) -> None:
    budget = ExecutionBudget(max_cost_usd=limit)
    trials = 0
    while budget.exhausted_dimension is None and trials < 50:
        budget.debit_trial(cost=cost)
        trials += 1
    assert trials == expected


def test_metric_limit_stop_condition_stops_at_exact_cap() -> None:
    cond = MetricLimitStopCondition(limit=1.0, metric_name="total_cost")
    trials: list[TrialResult] = []
    for i in range(1, 20):
        trials.append(
            TrialResult(
                trial_id=str(i),
                config={"p": i},
                metrics={"total_cost": 0.1},
                status=TrialStatus.COMPLETED,
                duration=0.1,
                timestamp=None,
            )
        )
        if cond.should_stop(trials):
            assert i == 10
            return
    pytest.fail("never stopped")
