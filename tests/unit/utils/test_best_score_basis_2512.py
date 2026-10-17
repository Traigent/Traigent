"""#2512: the final summary names the mean when the winner was repeated.

On connected runs ``best_score`` is the mean over the winning setup's
repeated evaluations (#1854), while the progress line shows the best single
trial. The summary now says so instead of printing an unexplained number.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from traigent.api.types import (
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)
from traigent.utils.callbacks import ManagedProgressCallback, ProgressBarCallback


def _trial(trial_id: str, accuracy: float) -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        config={"model": "m"},
        metrics={"accuracy": accuracy},
        status=TrialStatus.COMPLETED,
        duration=0.0,
        timestamp=datetime.now(UTC),
    )


def _result(summary: dict[str, Any] | None, best_score: float) -> OptimizationResult:
    return OptimizationResult(
        trials=[_trial("t1", 0.780), _trial("t2", 0.683), _trial("t3", 0.5)],
        best_config={"model": "m"},
        best_score=best_score,
        optimization_id="opt",
        duration=1.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="grid",
        timestamp=datetime.now(UTC),
        metadata={} if summary is None else {"session_summary": summary},
    )


_REPEATED = {
    "selection_mode": "aggregated_mean",
    "primary_objective": "accuracy",
    "winning_trial_ids": ["t1", "t2"],
}


@pytest.mark.parametrize(
    "callback", [ManagedProgressCallback, ProgressBarCallback], ids=lambda c: c.__name__
)
def test_repeated_winner_summary_names_the_mean(callback, capsys) -> None:
    callback().on_optimization_complete(_result(_REPEATED, 0.7315))
    assert "(mean of 2 runs: 0.780, 0.683)" in capsys.readouterr().out


def test_weighted_selection_names_the_mean_without_primary_values(capsys) -> None:
    summary = {**_REPEATED, "weighted_selection": {"enabled": True}}
    ManagedProgressCallback().on_optimization_complete(_result(summary, 0.6))
    out = capsys.readouterr().out
    assert "(mean of 2 runs)" in out
    assert "0.780" not in out


@pytest.mark.parametrize(
    "summary",
    [
        None,
        {**_REPEATED, "winning_trial_ids": ["t1"]},
        {**_REPEATED, "selection_mode": "best_trial"},
    ],
    ids=["no-summary", "single-run", "not-averaged"],
)
def test_single_or_unaveraged_winner_is_unlabelled(summary, capsys) -> None:
    ManagedProgressCallback().on_optimization_complete(_result(summary, 0.78))
    assert "mean of" not in capsys.readouterr().out
