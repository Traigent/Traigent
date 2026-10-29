"""#2819: get_optimization_insights honours objective direction.

A cost or latency objective used to name the costliest trial "best" and report
a 9900% improvement; a trial missing the primary metric was scored 0.0.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from traigent.api.types import (
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)
from traigent.utils.insights import get_optimization_insights


def _trial(trial_id: str, model: str, metrics: dict[str, float]) -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        config={"model": model},
        metrics=metrics,
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime.now(),
        metadata={},
    )


def _result(
    trials: list[TrialResult],
    objectives: list[str],
    metadata: dict | None = None,
) -> OptimizationResult:
    return OptimizationResult(
        trials=trials,
        best_config=trials[0].config,
        best_score=0.0,
        optimization_id="insights_2819",
        duration=1.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=objectives,
        algorithm="random",
        timestamp=datetime.now(),
        metadata=metadata or {},
    )


def _cost_trials() -> list[TrialResult]:
    return [
        _trial("t1", "m1", {"cost": 0.001, "accuracy": 0.80}),
        _trial("t2", "m2", {"cost": 0.010, "accuracy": 0.82}),
        _trial("t3", "m3", {"cost": 0.100, "accuracy": 0.83}),
    ]


def test_cost_primary_names_the_cheapest_trial_best() -> None:
    insights = get_optimization_insights(_result(_cost_trials(), ["cost", "accuracy"]))

    top = insights["top_configurations"]
    assert top[0]["config"] == {"model": "m1"}
    assert [row["config"]["model"] for row in top] == ["m1", "m2", "m3"]
    summary = insights["performance_summary"]
    assert summary["best_score"] == 0.001
    assert summary["worst_score"] == 0.1
    assert summary["multi_objective_analysis"]["cost"]["best"] == 0.001
    assert summary["multi_objective_analysis"]["accuracy"]["best"] == 0.83
    # best is 99% cheaper than the worst, not a 9900% "improvement"
    assert summary["improvement"] == pytest.approx(0.99)
    assert top[0]["relative_performance"] == 1.0
    assert top[2]["relative_performance"] < 1.0


def test_recommendations_never_name_the_worst_trial() -> None:
    insights = get_optimization_insights(_result(_cost_trials(), ["cost", "accuracy"]))

    text = " ".join(insights["recommendations"])
    assert "model=m1" in text
    assert "model=m3" not in text


def test_maximize_primary_is_unchanged() -> None:
    insights = get_optimization_insights(_result(_cost_trials(), ["accuracy", "cost"]))

    assert insights["top_configurations"][0]["config"] == {"model": "m3"}
    assert insights["performance_summary"]["best_score"] == 0.83


def test_trial_missing_the_primary_metric_is_excluded_not_zero() -> None:
    trials = _cost_trials() + [_trial("t4", "m4", {"accuracy": 0.5})]
    insights = get_optimization_insights(_result(trials, ["cost", "accuracy"]))

    ids = [row["trial_id"] for row in insights["top_configurations"]]
    assert "t4" not in ids
    summary = insights["performance_summary"]
    assert summary["best_score"] == 0.001  # not the missing trial's 0.0
    assert summary["average_score"] == pytest.approx((0.001 + 0.01 + 0.1) / 3)


def test_run_where_nothing_reports_the_primary_metric_is_an_error_result() -> None:
    trials = [_trial("t1", "m1", {"accuracy": 0.5})]
    insights = get_optimization_insights(_result(trials, ["cost", "accuracy"]))

    assert "cost" in insights["error"]
    assert insights["top_configurations"] == []


def test_declared_orientation_for_a_custom_metric_is_used() -> None:
    trials = [
        _trial("t1", "m1", {"loss": 0.9}),
        _trial("t2", "m2", {"loss": 0.1}),
    ]
    result = _result(
        trials, ["loss"], metadata={"objective_orientations": {"loss": "minimize"}}
    )

    insights = get_optimization_insights(result)

    assert insights["top_configurations"][0]["config"] == {"model": "m2"}
    assert insights["parameter_insights"]["model"]["best_value"] == "m2"


def test_undeclared_custom_primary_objective_fails_closed() -> None:
    trials = [_trial("t1", "m1", {"loss": 0.9}), _trial("t2", "m2", {"loss": 0.1})]

    with pytest.raises(ValueError, match="no declared orientation"):
        get_optimization_insights(_result(trials, ["loss"]))


def test_parameter_best_value_follows_direction() -> None:
    insights = get_optimization_insights(_result(_cost_trials(), ["cost"]))

    assert insights["parameter_insights"]["model"]["best_value"] == "m1"
