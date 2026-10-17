"""Round-8 findings on merged PR 2532 (owner ruling A, tracked spend).

1. ``EvaluationResult`` round-trips ``tracked_spend`` through to_dict/from_dict.
2. A winner-stability rerun with unknown cost makes the run total unavailable.
3. A post-evaluation (constraint) failure keeps the tracker snapshot so the
   failed trial's floor and uncertainty reach run aggregation.
"""

from __future__ import annotations

from typing import Any

import pytest

from traigent.config.types import TraigentConfig
from traigent.core.execution_budget import ExecutionBudget
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.evaluators.base import BaseEvaluator, EvaluationResult

from tests.unit.core.test_winner_stability_rerun import (
    StabilityEvaluator,
    StabilityOptimizer,
    mock_function,  # noqa: F401  (fixture)
    sample_dataset,  # noqa: F401  (fixture)
    isolated_optimization_logs,  # noqa: F401  (autouse fixture)
)

_FLOOR = 0.0000135


# 1 -- EvaluationResult round trip ---------------------------------------------
def _result(tracked: dict[str, Any] | None) -> EvaluationResult:
    metrics = {"accuracy": 1.0, "cost": 200.0, "total_cost": 200.0}
    return EvaluationResult(
        config={},
        aggregated_metrics=dict(metrics),
        total_examples=1,
        successful_examples=1,
        duration=0.1,
        metrics=dict(metrics),
        tracked_spend=tracked,
    )


def test_tracked_spend_survives_round_trip_and_debit_follows_tracker() -> None:
    tracked = {"unmeasured": False, "cost": _FLOOR, "lower_bound": 0.0}
    restored = EvaluationResult.from_dict(_result(tracked).to_dict())
    assert restored.tracked_spend == tracked
    assert BaseEvaluator._execution_budget_cost(restored) == pytest.approx(_FLOOR)


def test_unknown_tracked_spend_survives_round_trip() -> None:
    tracked = {"unmeasured": True, "cost": None, "lower_bound": 0.2}
    restored = EvaluationResult.from_dict(_result(tracked).to_dict())
    assert restored.tracked_spend == tracked
    assert BaseEvaluator._execution_budget_cost(restored) == pytest.approx(0.2)


def test_dict_without_tracked_spend_still_loads() -> None:
    payload = _result(None).to_dict()
    payload.pop("tracked_spend", None)
    assert EvaluationResult.from_dict(payload).tracked_spend is None


# 2 -- winner-stability rerun uncertainty --------------------------------------
class _TrackedEvaluator(StabilityEvaluator):
    """Trials measured at $0.10 each; every rerun has unknown cost ($0.20 floor)."""

    async def evaluate(self, *args: Any, **kwargs: Any) -> EvaluationResult:
        result = await super().evaluate(*args, **kwargs)
        if self.evaluation_count <= 3:
            result.tracked_spend = {
                "unmeasured": False,
                "cost": 0.1,
                "lower_bound": 0.0,
            }
        else:
            result.tracked_spend = {
                "unmeasured": True,
                "cost": None,
                "lower_bound": 0.2,
            }
            result.metrics["cost_unmeasured"] = 1.0
            result.aggregated_metrics["cost_unmeasured"] = 1.0
        return result


def _orchestrator(evaluator: BaseEvaluator, **kwargs: Any) -> OptimizationOrchestrator:
    optimizer = StabilityOptimizer({"param1": [0, 1, 2]}, ["accuracy"])
    return OptimizationOrchestrator(
        optimizer=optimizer,
        evaluator=evaluator,
        max_trials=3,
        config=TraigentConfig(no_egress=True, enable_usage_analytics=False),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_unknown_rerun_makes_total_unavailable(
    sample_dataset,  # noqa: F811
    mock_function,  # noqa: F811
) -> None:
    orchestrator = _orchestrator(_TrackedEvaluator(), winner_stability_reps=1)
    orchestrator.execution_budget = ExecutionBudget(max_cost_usd=10, max_examples=50)

    result = await orchestrator.optimize(mock_function, sample_dataset)

    assert result.total_cost is None
    assert "total_cost" not in result.metrics
    assert result.metrics["cost_unmeasured"] == 1.0
    assert result.metrics["cost_lower_bound"] == pytest.approx(0.5)


# 3 -- post-evaluation failure keeps the tracker snapshot ----------------------
class _PostEvalEvaluator(StabilityEvaluator):
    """Trials 1-2 measured $0.50; trial 3 (param1=2) has unknown cost, floor $0.20."""

    async def evaluate(self, *args: Any, **kwargs: Any) -> EvaluationResult:
        result = await super().evaluate(*args, **kwargs)
        if self.evaluation_count <= 2:
            result.tracked_spend = {
                "unmeasured": False,
                "cost": 0.5,
                "lower_bound": 0.0,
            }
        else:
            result.tracked_spend = {
                "unmeasured": True,
                "cost": None,
                "lower_bound": 0.2,
            }
            result.metrics["cost_unmeasured"] = 1.0
            result.aggregated_metrics["cost_unmeasured"] = 1.0
        return result


@pytest.mark.asyncio
async def test_failed_post_eval_trial_keeps_tracker_in_run_totals(
    sample_dataset,  # noqa: F811
    mock_function,  # noqa: F811
) -> None:
    orchestrator = _orchestrator(_PostEvalEvaluator())
    orchestrator._constraints_post_eval = [
        lambda config, metrics: config.get("param1") != 2
    ]

    result = await orchestrator.optimize(mock_function, sample_dataset)

    failed = [t for t in orchestrator._trials if t.status.value == "failed"]
    assert len(failed) == 1
    assert failed[0].tracked_spend is not None
    assert failed[0].tracked_spend["unmeasured"] is True
    assert result.total_cost is None
    assert "total_cost" not in result.metrics
    assert result.metrics["cost_unmeasured"] == 1.0
    assert result.metrics["cost_lower_bound"] == pytest.approx(1.2)
