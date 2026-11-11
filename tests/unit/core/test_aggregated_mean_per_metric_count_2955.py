"""Aggregated best-config means divide by the replicates that reported a metric (#2955).

A replicate missing a secondary metric used to pull that metric's mean toward
0 (its sum was divided by the replicate count), which flipped tie-breaks and
reported wrong ``best_metrics`` in aggregated mode.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from traigent.api.types import TrialResult, TrialStatus
from traigent.core.result_selection import select_best_configuration

_COUNTER = 0
ORIENTATIONS = {"accuracy": "maximize", "f1": "maximize"}


def _trial(model: str, metrics: dict[str, float]) -> TrialResult:
    global _COUNTER
    _COUNTER += 1
    return TrialResult(
        trial_id=f"t{_COUNTER}",
        config={"model": model},
        metrics=metrics,
        status=TrialStatus.COMPLETED,
        duration=0.1,
        timestamp=datetime.now(),
        metadata={},
    )


def _trials() -> list[TrialResult]:
    return [
        _trial("A", {"accuracy": 0.8, "f1": 0.9}),
        _trial("A", {"accuracy": 0.8}),
        _trial("B", {"accuracy": 0.8, "f1": 0.5}),
    ]


@pytest.mark.parametrize("aggregate", [True, False])
def test_tie_break_uses_mean_over_reporting_replicates(aggregate: bool) -> None:
    result = select_best_configuration(
        trials=_trials(),
        primary_objective="accuracy",
        config_space_keys={"model"},
        aggregate_configs=aggregate,
        objective_order=["accuracy", "f1"],
        comparability_mode="legacy",
        objective_orientations=ORIENTATIONS,
    )
    assert result.best_config == {"model": "A"}


def test_aggregated_best_metrics_mean_ignores_missing_replicate() -> None:
    result = select_best_configuration(
        trials=_trials(),
        primary_objective="accuracy",
        config_space_keys={"model"},
        aggregate_configs=True,
        objective_order=["accuracy", "f1"],
        comparability_mode="legacy",
        objective_orientations=ORIENTATIONS,
    )
    assert result.session_summary is not None
    metrics = result.session_summary["metrics"]
    assert metrics["accuracy"] == pytest.approx(0.8)
    assert metrics["run_f1"] == pytest.approx(0.9)
    assert result.session_summary["samples_per_config"] is not None
