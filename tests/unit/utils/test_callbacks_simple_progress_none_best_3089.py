"""SimpleProgressCallback tolerates a None best score (#3089)."""

from __future__ import annotations

from datetime import UTC, datetime

from traigent.api.types import TrialResult, TrialStatus
from traigent.utils.callbacks import ProgressInfo, SimpleProgressCallback


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


# --- #3089 -----------------------------------------------------------------
def test_simple_progress_none_best_score_does_not_raise() -> None:
    lines: list[str] = []
    cb = SimpleProgressCallback(output=lines.append)
    cb.on_optimization_start({"a": [1]}, ["accuracy"], "grid")
    cb.on_trial_start(0, {"a": 1})
    cb.on_trial_complete(_trial(), _progress(best=None))
    completed = [line for line in lines if "Completed. Score" in line]
    assert completed and "50.0%" in completed[0]
    assert "n/a" in completed[0]
