"""Traigent#3107: ASCII plots skip NaN/inf metrics instead of crashing."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from traigent.api.types import (
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)
from traigent.visualization.plots import create_quick_plot


def _result(bad: float) -> OptimizationResult:
    trials = [
        TrialResult(
            trial_id=f"t{i}",
            config={"i": i},
            metrics={"accuracy": acc, "cost": cost},
            status=TrialStatus.COMPLETED,
            duration=1.0,
            timestamp=datetime.now(UTC),
            metadata={},
        )
        for i, (acc, cost) in enumerate([(0.5, 1.0), (bad, 2.0), (0.7, 0.5)])
    ]
    return OptimizationResult(
        trials=trials,
        best_config=trials[0].config,
        best_score=0.5,
        optimization_id="opt-3107",
        duration=1.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy", "cost"],
        algorithm="test_algo",
        timestamp=datetime.now(UTC),
        metadata={
            "objective_orientations": {"accuracy": "maximize", "cost": "minimize"}
        },
    )


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
@pytest.mark.parametrize("kind", ["progress", "pareto", "report"])
def test_quick_plot_survives_nonfinite_metric(bad: float, kind: str) -> None:
    assert isinstance(create_quick_plot(_result(bad), kind), str)
