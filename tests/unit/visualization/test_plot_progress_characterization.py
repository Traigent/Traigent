"""Characterization tests for PlotGenerator.plot_optimization_progress."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from traigent.api.types import (
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)
from traigent.visualization.plots import PlotGenerator


def _trial(metrics, status=TrialStatus.COMPLETED):
    return TrialResult(
        trial_id="t",
        config={},
        metrics=metrics,
        status=status,
        duration=1.0,
        timestamp=datetime.now(UTC),
        metadata={},
    )


def _result(trials, objectives=None, orientations=None):
    return OptimizationResult(
        trials=trials,
        best_config={},
        best_score=0.0,
        optimization_id="o",
        duration=1.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=objectives if objectives is not None else ["accuracy"],
        algorithm="a",
        timestamp=datetime.now(UTC),
        metadata={"objective_orientations": orientations or {}},
    )


def _capture(result, objective=None, matplotlib=False):
    gen = PlotGenerator(use_matplotlib=False)
    gen._matplotlib_available = matplotlib
    with (
        patch.object(gen, "_plot_progress_ascii", return_value="ascii") as ascii_,
        patch.object(gen, "_plot_progress_matplotlib", return_value="mpl") as mpl,
    ):
        out = gen.plot_optimization_progress(result, objective)
    return out, ascii_, mpl


def test_no_trials_message():
    gen = PlotGenerator(use_matplotlib=False)
    assert gen.plot_optimization_progress(_result([])) == "No trials to plot"


def test_no_scored_trials_message_uses_objective_name():
    trials = [
        _trial({"accuracy": 0.5}, TrialStatus.FAILED),
        _trial({}),
        _trial({"other": 1}),
        _trial({"accuracy": "bad"}),
        _trial({"accuracy": None}),
        _trial({"accuracy": [1]}),
    ]
    out, ascii_, mpl = _capture(_result(trials))
    assert out == "No completed trials with objective 'accuracy'"
    ascii_.assert_not_called()
    mpl.assert_not_called()


def test_maximize_running_best_and_indices_skip_unusable_trials():
    trials = [
        _trial({"accuracy": 0.2}),
        _trial({"accuracy": 0.9}, TrialStatus.FAILED),
        _trial({"accuracy": "0.5"}),  # numeric strings are coerced
        _trial({"accuracy": None}),
        _trial({"accuracy": 0.1}),
        _trial({"accuracy": 0.7}),
    ]
    out, ascii_, _ = _capture(_result(trials, orientations={"accuracy": "maximize"}))
    assert out == "ascii"
    nums, scores, best, label = ascii_.call_args.args
    assert nums == [1, 3, 5, 6]
    assert scores == [0.2, 0.5, 0.1, 0.7]
    assert best == [0.2, 0.5, 0.5, 0.7]
    assert label == "Accuracy (max)"


def test_minimize_running_best():
    trials = [_trial({"cost": 5.0}), _trial({"cost": 7.0}), _trial({"cost": 3.0})]
    out, ascii_, _ = _capture(
        _result(trials, objectives=["cost"], orientations={"cost": "minimize"})
    )
    nums, scores, best, _label = ascii_.call_args.args
    assert nums == [1, 2, 3]
    assert scores == [5.0, 7.0, 3.0]
    assert best == [5.0, 5.0, 3.0]


def test_default_objective_is_first_objective_or_score():
    trials = [_trial({"accuracy": 1.0, "score": 2.0})]
    _, ascii_, _ = _capture(_result(trials, objectives=["accuracy"]))
    assert ascii_.call_args.args[1] == [1.0]
    _, ascii_, _ = _capture(
        _result(trials, objectives=[], orientations={"score": "maximize"})
    )
    assert ascii_.call_args.args[1] == [2.0]


def test_explicit_objective_overrides_default():
    trials = [_trial({"accuracy": 1.0, "cost": 9.0})]
    _, ascii_, _ = _capture(
        _result(trials, orientations={"cost": "minimize", "accuracy": "maximize"}),
        objective="cost",
    )
    assert ascii_.call_args.args[1] == [9.0]


def test_matplotlib_branch_receives_objective_and_label():
    trials = [_trial({"accuracy": 1.0})]
    out, ascii_, mpl = _capture(_result(trials), matplotlib=True)
    assert out == "mpl"
    ascii_.assert_not_called()
    nums, scores, best, objective, label = mpl.call_args.args
    assert (nums, scores, best, objective) == ([1], [1.0], [1.0], "accuracy")
    assert label == "Accuracy (max)"


def test_nan_score_is_kept_as_float():
    trials = [_trial({"accuracy": float("nan")}), _trial({"accuracy": 0.5})]
    _, ascii_, _ = _capture(_result(trials))
    scores = ascii_.call_args.args[1]
    assert scores[0] != scores[0] and scores[1] == 0.5


@pytest.mark.parametrize("raw", [True])
def test_bool_metric_is_coerced_to_float(raw):
    _, ascii_, _ = _capture(_result([_trial({"accuracy": raw})]))
    assert ascii_.call_args.args[1] == [1.0]
