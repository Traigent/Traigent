"""Characterization for DetailedProgressCallback.on_trial_complete output."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import traigent.utils.callbacks as cb
from traigent.utils.callbacks import DetailedProgressCallback


@pytest.fixture
def printed(monkeypatch):
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    monkeypatch.setattr(cb, "_safe_print", lambda *a, **k: calls.append((a, k)))
    return calls


def _progress(total=10, completed=5, best=None):
    return SimpleNamespace(
        total_trials=total, completed_trials=completed, best_score=best
    )


def _trial(status="completed", metrics=None):
    return SimpleNamespace(status=status, metrics=metrics)


def _callback(trial_count=3, total=0, show_metrics=True):
    c = DetailedProgressCallback(show_metrics=show_metrics)
    c.trial_count = trial_count
    c.total_trials = total
    return c


def _lines(printed):
    return [a[0] if a else None for a, _ in printed]


def test_full_report_lines_and_flush(printed):
    c = _callback(trial_count=3, total=20)
    c.on_trial_complete(
        _trial(metrics={"acc": 0.5, "n": 4, "name": "x"}),
        _progress(total=10, completed=5, best=0.123456),
    )
    assert _lines(printed) == [
        "✅ Trial 3/20 completed",
        "   Metrics:",
        "     • acc: 0.500",
        "     • n: 4",
        "     • name: x",
        "   🏆 Best score so far: 0.123",
        f"   Progress: [{'█' * 20}{'░' * 20}] 50%",
        None,
    ]
    assert printed[-1] == ((), {"flush": True})


def test_failed_icon_and_total_fallbacks(printed):
    c = _callback(trial_count=2, total=0)
    c.on_trial_complete(_trial(status="failed"), _progress(total=7, completed=1))
    assert _lines(printed)[0] == "❌ Trial 2/7 completed"


@pytest.mark.parametrize("progress_total", [0, -3, None])
def test_unknown_total_shows_question_mark(printed, progress_total):
    c = _callback(trial_count=1, total=0)
    c.on_trial_complete(_trial(), _progress(total=progress_total, completed=0))
    assert _lines(printed)[0] == "✅ Trial 1/? completed"


def test_metrics_hidden_when_disabled_or_empty(printed):
    c = _callback(show_metrics=False)
    c.on_trial_complete(_trial(metrics={"a": 1}), _progress())
    assert "   Metrics:" not in _lines(printed)
    printed.clear()
    c = _callback()
    c.on_trial_complete(_trial(metrics={}), _progress())
    c.on_trial_complete(_trial(metrics=None), _progress())
    assert "   Metrics:" not in _lines(printed)


def test_best_score_zero_is_still_printed(printed):
    _callback().on_trial_complete(_trial(), _progress(best=0.0))
    assert "   🏆 Best score so far: 0.000" in _lines(printed)


@pytest.mark.parametrize(
    "progress_total,own_total,completed,trial_count,expected_bar,percent",
    [
        (10, 99, 5, 3, 20, "50%"),  # progress total wins
        (0, 8, 2, 4, 20, "50%"),  # own total used when progress total unusable
        (0, 0, 0, 3, 40, "100%"),  # derived denominator = max(completed, count, 1)
        (4, 0, 9, 1, 40, "100%"),  # ratio capped at 1.0
        (10, 0, 0, 0, 0, "0%"),
        (-1, 0, 0, 0, 0, "0%"),  # denominator falls back to 1; nothing completed
        (3, 0, 1, 0, 13, "33%"),
    ],
)
def test_progress_bar_math(
    printed, progress_total, own_total, completed, trial_count, expected_bar, percent
):
    c = _callback(trial_count=trial_count, total=own_total)
    c.on_trial_complete(_trial(), _progress(total=progress_total, completed=completed))
    bar_line = _lines(printed)[-2]
    assert bar_line == (
        f"   Progress: [{'█' * expected_bar}{'░' * (40 - expected_bar)}] {percent}"
    )


def test_string_total_raises_before_printing(printed):
    c = _callback(trial_count=2, total=0)
    with pytest.raises(TypeError):
        c.on_trial_complete(_trial(), _progress(total="10", completed=1))
    assert printed == []
