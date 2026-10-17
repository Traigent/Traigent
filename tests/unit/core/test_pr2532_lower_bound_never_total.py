"""Owner ruling A: a cost lower bound is never presented as the run total."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from traigent.api.types import TrialResult, TrialStatus
from traigent.core.metric_registry import MetricRegistry
from traigent.core.metrics_aggregator import aggregate_metrics
from traigent.core.orchestrator import OptimizationOrchestrator


def _trial(trial_id: str, tracked: dict) -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        config={},
        metrics={"accuracy": 0.5},
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime.now(UTC),
        metadata={},
        tracked_spend=tracked,
    )


_UNKNOWN = {"unmeasured": True, "cost": None, "lower_bound": 0.2}


def test_unknown_trial_total_cost_is_unavailable_and_floor_exposed() -> None:
    agg = aggregate_metrics([_trial("t1", _UNKNOWN)], MetricRegistry.default())
    assert agg.total_cost is None
    assert "total_cost" not in agg.processed_metrics
    assert agg.processed_metrics["cost_unmeasured"] == 1.0
    assert agg.processed_metrics["cost_lower_bound"] == pytest.approx(0.2)


def test_mixed_run_floor_is_measured_sum_plus_unknown_floors() -> None:
    trials = [
        _trial("t1", {"unmeasured": False, "cost": 0.5}),
        _trial("t2", _UNKNOWN),
    ]
    agg = aggregate_metrics(trials, MetricRegistry.default())
    assert agg.total_cost is None
    assert "total_cost" not in agg.processed_metrics
    assert agg.processed_metrics["cost_unmeasured"] == 1.0
    assert agg.processed_metrics["cost_lower_bound"] == pytest.approx(0.7)


def test_fully_measured_run_keeps_finite_total() -> None:
    agg = aggregate_metrics(
        [_trial("t1", {"unmeasured": False, "cost": 0.5})], MetricRegistry.default()
    )
    assert agg.total_cost == pytest.approx(0.5)
    assert agg.processed_metrics["total_cost"] == pytest.approx(0.5)
    assert "cost_unmeasured" not in agg.processed_metrics
    assert "cost_lower_bound" not in agg.processed_metrics


def test_stability_cost_on_unknown_result_raises_floor_not_total() -> None:
    class _R:
        total_cost = None
        metrics = {"cost_unmeasured": 1.0, "cost_lower_bound": 0.2}

    result = _R()
    OptimizationOrchestrator._add_winner_stability_cost_to_result(result, 0.1)
    assert result.total_cost is None
    assert "total_cost" not in result.metrics
    assert result.metrics["cost_lower_bound"] == pytest.approx(0.3)
