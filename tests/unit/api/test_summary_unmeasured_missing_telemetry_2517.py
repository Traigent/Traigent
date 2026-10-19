"""Missing attempt/cost telemetry is unmeasured, not $0 (Traigent#2517)."""

from __future__ import annotations

import pytest

from traigent.api.types import OptimizationResult, TrialResult, TrialStatus


def _trial(status, metrics=None, metadata=None, tid="t"):
    return TrialResult(
        trial_id=tid,
        config={"model": "m"},
        metrics=metrics or {},
        status=status,
        duration=1.0,
        timestamp=__import__("datetime").datetime.now(),
        metadata=metadata or {},
    )


def _result(trials):
    return OptimizationResult(
        trials=trials,
        best_config={"model": "m"},
        best_score=0.5,
        optimization_id="o",
        duration=1.0,
        convergence_info={},
        status="completed",
        objectives=["accuracy"],
        algorithm="grid",
        timestamp=__import__("datetime").datetime.now(),
    )


@pytest.mark.parametrize(
    "status", [TrialStatus.FAILED, TrialStatus.PRUNED, TrialStatus.CANCELLED]
)
def test_terminal_trial_without_progress_telemetry_is_unmeasured(status):
    result = _result([_trial(status)])
    stats = result.experiment_stats
    summary = result.get_summary()
    assert stats.total_cost is None
    assert stats.cost_per_configuration is None
    assert stats.cost_unmeasured_trials == 1
    assert summary["total_cost"] is None
    assert summary["cost_unmeasured_trials"] == 1


def test_completed_trial_without_cost_metric_is_unmeasured():
    result = _result([_trial(TrialStatus.COMPLETED, {"accuracy": 0.9})])
    stats = result.experiment_stats
    assert stats.total_cost is None
    assert stats.cost_per_configuration is None
    assert stats.cost_unmeasured_trials == 1


def test_explicit_zero_attempts_stays_measured():
    result = _result([_trial(TrialStatus.FAILED, metadata={"examples_attempted": 0})])
    stats = result.experiment_stats
    assert stats.cost_unmeasured_trials == 0
    assert stats.total_cost == 0.0


def test_reported_zero_dollar_cost_stays_measured():
    result = _result([_trial(TrialStatus.COMPLETED, {"total_cost": 0.0})])
    stats = result.experiment_stats
    assert stats.cost_unmeasured_trials == 0
    assert stats.total_cost == 0.0
    assert stats.cost_per_configuration == 0.0
