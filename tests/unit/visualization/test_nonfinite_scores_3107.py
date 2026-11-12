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


def _ascii(scores: list[float], label: str) -> str:
    from traigent.visualization.plots import PlotGenerator

    gen = PlotGenerator(use_matplotlib=False)
    # Mimic _extract_progress_series: running best that NaN/inf can poison.
    best: list[float] = []
    cur = None
    for s in scores:
        cur = s if cur is None else (max(cur, s) if "max" in label else min(cur, s))
        best.append(cur)
    return gen._plot_progress_ascii(
        list(range(1, len(scores) + 1)), scores, best, label
    )


def test_leading_nan_keeps_later_finite_points() -> None:
    out = _ascii([float("nan"), 0.2, 0.9], "Accuracy (max)")
    assert "Score range: 0.20 -> 0.90" in out
    assert "Final best score: 0.900" in out


def test_plus_inf_does_not_hide_later_points_when_maximizing() -> None:
    out = _ascii([0.1, float("inf"), 0.9], "Accuracy (max)")
    assert "Score range: 0.10 -> 0.90" in out
    assert "Final best score: 0.900" in out


def test_nan_trial_cannot_displace_finite_trial_from_pareto_front() -> None:
    from unittest.mock import patch

    from traigent.visualization.plots import PlotGenerator

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
        for i, (acc, cost) in enumerate([(float("nan"), 0.0), (0.9, 1.0)])
    ]
    result = OptimizationResult(
        trials=trials,
        best_config=trials[1].config,
        best_score=0.9,
        optimization_id="opt-3107b",
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
    gen = PlotGenerator(use_matplotlib=False)
    with patch.object(gen, "_plot_pareto_ascii", return_value="ok") as ascii_:
        gen.plot_pareto_front(result, "accuracy", "cost")
    pareto_x, pareto_y = ascii_.call_args.args[2], ascii_.call_args.args[3]
    assert (pareto_x, pareto_y) == ([0.9], [1.0])
