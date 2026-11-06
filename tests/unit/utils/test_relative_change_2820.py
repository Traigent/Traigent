"""#2820: session improvement is direction-aware and never hides a real 0.0."""

from __future__ import annotations

from pathlib import Path

import pytest

from traigent.storage.local_storage import LocalStorageManager
from traigent.utils.relative_change import relative_change, relative_change_with_reason


@pytest.mark.parametrize(
    ("baseline", "best", "orientation", "expected"),
    [
        (0.0, 1.0, "maximize", None),
        (0.5, 1.0, "maximize", 1.0),
        (0.5, 0.0, "maximize", -1.0),
        (-1.0, -0.5, "maximize", 0.5),
        (0.5, 0.5, "maximize", 0.0),
        (0.5, 0.3, "minimize", pytest.approx(0.4)),
        (0.5, 0.7, "minimize", pytest.approx(-0.4)),
        (None, 1.0, "maximize", None),
        (0.5, None, "maximize", None),
        (float("nan"), 1.0, "maximize", None),
        (0.5, float("inf"), "maximize", None),
        (0.5, 1.0, "band", None),
        (0.5, 1.0, None, None),
    ],
)
def test_relative_change(baseline, best, orientation, expected) -> None:
    assert relative_change(baseline, best, orientation) == expected


def test_reason_is_given_exactly_when_undefined() -> None:
    assert relative_change_with_reason(0.0, 1.0, "maximize") == (None, "baseline 0")
    value, reason = relative_change_with_reason(0.5, 1.0, "maximize")
    assert value == 1.0 and reason is None


def _summary(tmp_path: Path, baseline: float, best: float, config: dict | None) -> dict:
    storage = LocalStorageManager(str(tmp_path))
    session_id = storage.create_session("fn", optimization_config=config)
    storage.add_trial_result(session_id, {"x": 1}, 0.1)
    session = storage.load_session(session_id)
    session.baseline_score = baseline
    session.best_score = best
    storage._save_session(session)
    summary = storage.get_session_summary(session_id)
    assert summary is not None
    return summary


@pytest.mark.parametrize(
    ("baseline", "best", "expected"),
    [
        (0.0, 1.0, None),
        (0.5, 1.0, 1.0),
        (0.5, 0.0, -1.0),
        (-1.0, -0.5, 0.5),
        (0.5, 0.5, 0.0),
    ],
)
def test_summary_maximize_rows(tmp_path, baseline, best, expected) -> None:
    assert _summary(tmp_path, baseline, best, None)["improvement"] == expected


@pytest.mark.parametrize(
    ("baseline", "best", "expected"),
    [(0.5, 0.3, pytest.approx(0.4)), (0.5, 0.7, pytest.approx(-0.4))],
)
def test_summary_minimize_cost_rows(tmp_path, baseline, best, expected) -> None:
    summary = _summary(tmp_path, baseline, best, {"objectives": ["cost"]})
    assert summary["improvement"] == expected


def test_average_improvement_keeps_a_zero_average(tmp_path) -> None:
    from datetime import UTC, datetime
    from unittest.mock import patch

    from traigent.config.types import ExecutionMode, TraigentConfig
    from traigent.storage.local_storage import OptimizationSession
    from traigent.utils.local_analytics import LocalAnalytics

    analytics = LocalAnalytics(
        TraigentConfig(
            execution_mode=ExecutionMode.LOCAL.value,
            enable_usage_analytics=True,
            local_storage_path=str(tmp_path),
            anonymous_user_id="u",
        )
    )
    now = datetime.now(UTC).isoformat()
    sessions = [
        OptimizationSession(
            session_id="s1",
            function_name="f",
            created_at=now,
            updated_at=now,
            status="completed",
            total_trials=1,
            completed_trials=1,
        )
    ]
    with (
        patch.object(analytics.storage, "list_sessions", return_value=sessions),
        patch.object(
            analytics.storage, "get_session_summary", return_value={"improvement": 0.0}
        ),
    ):
        stats = analytics.collect_usage_stats()
    assert stats["avg_improvement_percent"] == 0.0
