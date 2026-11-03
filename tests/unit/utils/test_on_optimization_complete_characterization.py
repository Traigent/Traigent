"""Characterization tests for ProgressBarCallback.on_optimization_complete."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from traigent.utils.callbacks import ProgressBarCallback


def _result(**overrides):
    base = {
        "stop_reason": None,
        "status": "completed",
        "metadata": {},
        "best_score": 0.8,
        "duration": 12.34,
        "success_rate": 0.5,
        "run_label": None,
        "cloud_url": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _lines(cb, result):
    printed = []
    with (
        patch(
            "traigent.utils.callbacks._safe_print",
            side_effect=lambda *a, **k: printed.append((a, k)),
        ),
        patch.object(cb, "_print_results_table") as table,
    ):
        cb.on_optimization_complete(result)
    table.assert_called_once()
    return [a[0] if a else "" for a, _ in printed], printed


def test_basic_summary_without_progress():
    cb = ProgressBarCallback()
    lines, _ = _lines(cb, _result())
    assert lines == [
        "\n",
        "✅ Optimization complete!",
        "🏆 Best score: 0.800",
        "⏱️  Total time: 12.3s",
        "📈 Success rate: 50.0%",
    ]


def test_best_score_hidden_when_show_best_false_and_na_when_none():
    lines, _ = _lines(ProgressBarCallback(show_best=False), _result())
    assert not any("Best score" in line for line in lines)
    lines, _ = _lines(ProgressBarCallback(), _result(best_score=None))
    assert "🏆 Best score: N/A" in lines


def test_user_cancel_message_and_stop_reason_line():
    lines, _ = _lines(ProgressBarCallback(), _result(stop_reason="user_cancelled"))
    assert "🛑 Optimization CANCELLED BY USER; results are partial." in lines
    assert "🛑 Stop reason: user_cancelled" in lines
    assert "✅ Optimization complete!" not in lines


@pytest.mark.parametrize(
    ("kwargs", "hint"),
    [
        ({"stop_reason": "timeout", "metadata": {"timeout": 30}}, " (30s)"),
        ({"stop_reason": "timeout", "metadata": {"timeout": 0}}, ""),
        ({"stop_reason": "timeout", "metadata": {}}, ""),
        ({"stop_reason": "timeout", "metadata": None}, ""),
        ({"status": "cancelled", "metadata": {"timeout": 5}}, " (5s)"),
    ],
)
def test_timeout_or_cancelled_status(kwargs, hint):
    lines, _ = _lines(ProgressBarCallback(), _result(**kwargs))
    assert f"⚠️ Optimization stopped: timeout reached{hint}." in lines
    # "timeout" never produces a separate stop-reason line
    assert not any(line.startswith("🛑 Stop reason: timeout") for line in lines)


def test_other_stop_reason_is_printed():
    lines, _ = _lines(ProgressBarCallback(), _result(stop_reason="max_trials"))
    assert lines[-1] == "🛑 Stop reason: max_trials"
    assert "✅ Optimization complete!" in lines


def test_run_label_and_cloud_url():
    lines, _ = _lines(
        ProgressBarCallback(), _result(run_label="r1", cloud_url="http://x")
    )
    assert lines[-2:] == ["🔖 Run: r1", "🔗 View: http://x"]


def test_offline_hint_only_with_label_and_offline_mode_and_no_url():
    cb = ProgressBarCallback()
    hint = "   Run locally — sync to cloud with `traigent sync`"
    lines, _ = _lines(cb, _result(run_label="r", metadata={"offline_mode": True}))
    assert lines[-1] == hint
    lines, _ = _lines(cb, _result(metadata={"offline_mode": True}))
    assert hint not in lines
    lines, _ = _lines(cb, _result(run_label="r", metadata={}))
    assert hint not in lines
    lines, _ = _lines(cb, _result(run_label="r", metadata=None))
    assert hint not in lines
    lines, _ = _lines(
        cb, _result(run_label="r", cloud_url="u", metadata={"offline_mode": True})
    )
    assert hint not in lines


def test_final_progress_bar_is_drawn_first_when_progress_recorded():
    cb = ProgressBarCallback(width=4)
    cb._last_progress = SimpleNamespace(
        elapsed_time=65,
        best_score=0.5,
        total_trials=10,
        successful_trials=7,
        failed_trials=3,
    )
    cb._timed_out_examples = 2
    lines, printed = _lines(cb, _result())
    first_args, first_kwargs = printed[0]
    assert first_args[0] == (
        "\r[████] 100.0% (10/10) ✅ 7 ❌ 3 ⌛ 2 timed out ⏱️  01:05 🏆 0.500"
    )
    assert first_kwargs == {"flush": True}
    assert lines[1] == "\n"


def test_final_progress_bar_without_best_and_with_na():
    cb = ProgressBarCallback(width=2, show_best=False)
    cb._last_progress = SimpleNamespace(
        elapsed_time=0,
        best_score=None,
        total_trials=1,
        successful_trials=1,
        failed_trials=0,
    )
    lines, _ = _lines(cb, _result())
    assert lines[0] == "\r[██] 100.0% (1/1) ✅ 1 ❌ 0 ⏱️  00:00"
    cb2 = ProgressBarCallback(width=2)
    cb2._last_progress = cb._last_progress
    lines, _ = _lines(cb2, _result())
    assert lines[0].endswith(" 🏆 N/A")
