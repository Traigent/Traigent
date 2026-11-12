"""Completion lines name the trial that finished, not the last one started (#2984)."""

from __future__ import annotations

from datetime import UTC, datetime

from traigent.api.types import TrialResult, TrialStatus
from traigent.utils.callbacks import (
    DetailedProgressCallback,
    ProgressInfo,
    SimpleProgressCallback,
)

CONFIGS = [{"m": "a"}, {"m": "b"}, {"m": "c"}]


def _trial(index: int) -> TrialResult:
    return TrialResult(
        trial_id=f"t{index}",
        config=CONFIGS[index],
        metrics={"accuracy": 0.5},
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime.now(UTC),
    )


def _progress(done: int) -> ProgressInfo:
    return ProgressInfo(
        current_trial=done,
        total_trials=3,
        completed_trials=done,
        successful_trials=done,
        failed_trials=0,
        best_score=0.5,
        best_config=CONFIGS[0],
        elapsed_time=1.0,
        estimated_remaining=1.0,
        current_algorithm="grid",
    )


def test_simple_callback_reports_each_completed_trial_number() -> None:
    lines: list[str] = []
    callback = SimpleProgressCallback(output=lines.append, show_details=True)
    callback.on_optimization_start({"m": ["a", "b", "c"]}, ["accuracy"], "grid")
    for number, config in enumerate(CONFIGS):
        callback.on_trial_start(number, config)
    lines.clear()
    for index in (2, 0, 1):  # completion order differs from start order
        callback.on_trial_complete(_trial(index), _progress(1))
    assert [line.split("]")[0] for line in lines] == ["[3/3", "[1/3", "[2/3"]


def test_detailed_callback_reports_each_completed_trial_number(capsys) -> None:
    callback = DetailedProgressCallback()
    for number, config in enumerate(CONFIGS):
        callback.on_trial_start(number, config)
    capsys.readouterr()
    for index in (1, 0, 2):
        callback.on_trial_complete(_trial(index), _progress(1))
    completed = [
        line for line in capsys.readouterr().out.splitlines() if "completed" in line
    ]
    assert [line.split()[2].split("/")[0] for line in completed] == ["2", "1", "3"]


def test_identical_configs_complete_in_start_order() -> None:
    lines: list[str] = []
    callback = SimpleProgressCallback(output=lines.append, show_details=True)
    same = {"m": "a"}
    for number in range(3):
        callback.on_trial_start(number, dict(same))
    lines.clear()
    for index in range(3):
        trial = _trial(0)
        callback.on_trial_complete(trial, _progress(index + 1))
    assert [line.split("]")[0] for line in lines] == ["[1/3", "[2/3", "[3/3"]
