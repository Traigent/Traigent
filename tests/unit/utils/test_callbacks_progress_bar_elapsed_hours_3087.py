"""ProgressBarCallback keeps hours in elapsed time (#3087)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from traigent.api.types import TrialResult, TrialStatus
from traigent.utils.callbacks import ProgressBarCallback, ProgressInfo


def _trial() -> TrialResult:
    return TrialResult(
        trial_id="t1",
        config={"a": 1},
        metrics={"accuracy": 0.5},
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime.now(UTC),
    )


def _progress(best: float | None = 0.5, elapsed: float = 1.0) -> ProgressInfo:
    return ProgressInfo(
        current_trial=1,
        total_trials=1,
        completed_trials=1,
        successful_trials=1,
        failed_trials=0,
        best_score=best,
        best_config=None,
        elapsed_time=elapsed,
        estimated_remaining=0.0,
        current_algorithm="grid",
    )


# --- #3087 -----------------------------------------------------------------
def test_progress_bar_elapsed_keeps_hours(capsys: pytest.CaptureFixture[str]) -> None:
    cb = ProgressBarCallback(width=10, update_interval=0)
    cb.on_optimization_start({"a": [1]}, ["accuracy"], "grid")
    cb.on_trial_complete(_trial(), _progress(elapsed=3725.0))
    out = capsys.readouterr().out
    assert "1:02:05" in out
    assert "02:05" not in out.replace("1:02:05", "")


def test_progress_bar_final_line_keeps_hours(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cb = ProgressBarCallback(width=10, update_interval=0)
    cb._print_final_progress_bar(_progress(elapsed=3725.0))
    assert "1:02:05" in capsys.readouterr().out
