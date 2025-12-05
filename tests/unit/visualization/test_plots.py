"""Tests for visualization plots."""

from datetime import datetime, timezone
from unittest.mock import patch

from traigent.api.types import (
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)
from traigent.visualization.plots import PlotGenerator


def _make_trial(trial_id: str, accuracy: float, cost: float) -> TrialResult:
    """Helper to create trial results."""
    return TrialResult(
        trial_id=trial_id,
        config={"trial": trial_id},
        metrics={"accuracy": accuracy, "cost": cost},
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime.now(timezone.utc),
        metadata={},
    )


def _make_result(
    trials: list[TrialResult],
    orientations: dict[str, str],
    objectives: list[str] | None = None,
) -> OptimizationResult:
    """Helper to create optimization results."""
    return OptimizationResult(
        trials=trials,
        best_config=trials[0].config if trials else {},
        best_score=0.0,
        optimization_id="opt-123",
        duration=1.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=objectives or ["accuracy", "cost"],
        algorithm="test_algo",
        timestamp=datetime.now(timezone.utc),
        metadata={"objective_orientations": orientations},
    )


@patch("traigent.visualization.plots.ParetoFrontCalculator")
def test_plot_pareto_front_respects_orientation_metadata(mock_calculator):
    """Ensure Pareto front calculator receives orientation overrides."""
    instance = mock_calculator.return_value
    instance.calculate_pareto_front.return_value = []

    result = _make_result(
        trials=[
            _make_trial("t1", 0.9, 0.4),
            _make_trial("t2", 0.85, 0.2),
        ],
        orientations={"accuracy": "maximize", "cost": "minimize"},
    )

    plotter = PlotGenerator(use_matplotlib=False)
    plotter.plot_pareto_front(result, "accuracy", "cost")

    mock_calculator.assert_called_once()
    kwargs = mock_calculator.call_args.kwargs
    assert kwargs["maximize"] == {"accuracy": True, "cost": False}


def test_plot_pareto_front_ascii_outputs_orientation():
    """ASCII plot should reflect objective orientations and correct Pareto size."""
    trials = [
        _make_trial("t1", 0.9, 0.4),
        _make_trial("t2", 0.85, 0.2),
        _make_trial("t3", 0.8, 0.7),
    ]
    result = _make_result(
        trials=trials,
        orientations={"accuracy": "maximize", "cost": "minimize"},
    )

    plotter = PlotGenerator(use_matplotlib=False)
    output = plotter.plot_pareto_front(result, "accuracy", "cost")

    assert "Accuracy (max)" in output
    assert "Cost (min)" in output
    assert "Pareto points found: 2" in output


def test_progress_plot_respects_minimize_orientation(monkeypatch):
    """Progress plot should treat minimize objectives as improvements when scores drop."""
    trials = [
        TrialResult(
            trial_id="t1",
            config={"trial": "t1"},
            metrics={"loss": 0.5},
            status=TrialStatus.COMPLETED,
            duration=1.0,
            timestamp=datetime.now(timezone.utc),
        ),
        TrialResult(
            trial_id="t2",
            config={"trial": "t2"},
            metrics={"loss": 0.3},
            status=TrialStatus.COMPLETED,
            duration=1.0,
            timestamp=datetime.now(timezone.utc),
        ),
        TrialResult(
            trial_id="t3",
            config={"trial": "t3"},
            metrics={"loss": 0.4},
            status=TrialStatus.COMPLETED,
            duration=1.0,
            timestamp=datetime.now(timezone.utc),
        ),
    ]

    result = _make_result(
        trials=trials,
        orientations={"loss": "minimize"},
        objectives=["loss"],
    )

    plotter = PlotGenerator(use_matplotlib=False)
    with patch.object(plotter, "_plot_progress_ascii") as mock_ascii:
        mock_ascii.return_value = "ok"
        output = plotter.plot_optimization_progress(result)

    assert output == "ok"
    mock_ascii.assert_called_once()
    trial_numbers, scores, best_scores, label = mock_ascii.call_args.args
    assert trial_numbers == [1, 2, 3]
    assert scores == [0.5, 0.3, 0.4]
    assert best_scores == [0.5, 0.3, 0.3]
    assert label == "Loss (min)"
