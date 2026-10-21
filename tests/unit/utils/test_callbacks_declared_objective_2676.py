"""Trial-reporting surfaces must read the declared objective, not accuracy (#2676)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from traigent.api.types import TrialResult, TrialStatus
from traigent.core.objectives import create_default_objectives
from traigent.utils.callbacks import (
    ProgressInfo,
    SimpleProgressCallback,
    StatisticsCallback,
)
from traigent.utils.optimization_logger import OptimizationLogger


def _trial(tid: str, style: str, overlap: float) -> TrialResult:
    return TrialResult(
        trial_id=tid,
        config={"style": style},
        metrics={"overlap": overlap, "accuracy": 1.0},
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime.now(UTC),
    )


def _progress() -> ProgressInfo:
    return ProgressInfo(
        current_trial=1,
        total_trials=2,
        completed_trials=1,
        successful_trials=1,
        failed_trials=0,
        best_score=1.0,
        best_config={"style": "upper"},
        elapsed_time=1.0,
        estimated_remaining=1.0,
        current_algorithm="grid",
    )


def test_statistics_callback_scores_declared_objective() -> None:
    cb = StatisticsCallback()
    cb.on_optimization_start({"style": ["upper", "lower"]}, ["overlap"], "grid")
    for tid, style, value in (("t1", "upper", 1.0), ("t2", "lower", 0.5)):
        cb.on_trial_start(0, {"style": style})
        cb.on_trial_complete(_trial(tid, style, value), _progress())
    assert cb.stats["scores_by_trial"] == [1.0, 0.5]
    assert cb.get_parameter_importance()["style"] > 0.0


def test_statistics_callback_missing_objective_is_unknown_not_zero() -> None:
    cb = StatisticsCallback()
    cb.on_optimization_start({"style": ["a"]}, ["missing_metric"], "grid")
    cb.on_trial_complete(_trial("t1", "a", 1.0), _progress())
    assert cb.stats["scores_by_trial"] == [None]


def test_simple_progress_prints_declared_objective_and_keeps_zero() -> None:
    lines: list[str] = []
    cb = SimpleProgressCallback(output=lines.append)
    cb.on_optimization_start({"style": ["a", "b"]}, ["overlap"], "grid")
    cb.on_trial_start(1, {"style": "a"})
    cb.on_trial_complete(_trial("t1", "a", 0.5), _progress())
    cb.on_trial_complete(_trial("t2", "b", 0.0), _progress())
    completed = [line for line in lines if "Completed. Score" in line]
    assert "50.0%" in completed[0]
    assert "Score: 0.0%" in completed[1]


def test_trial_summary_header_uses_declared_objective(tmp_path: Path) -> None:
    log = OptimizationLogger(
        experiment_name="exp",
        session_id="sess12345678",
        execution_mode="local",
        base_path=tmp_path,
    )
    log.log_session_start(
        {},
        create_default_objectives(["overlap"], orientations={"overlap": "maximize"}),
        "grid",
    )
    (log.run_path / "trials").mkdir(parents=True, exist_ok=True)
    log._append_trial_summary(_trial("t1", "upper", 0.5), None)
    text = (log.run_path / "trials" / "trial_summary.txt").read_text()
    assert "overlap=0.5" in text
    assert "accuracy=" not in text
