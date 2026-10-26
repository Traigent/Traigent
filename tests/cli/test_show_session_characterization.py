"""Characterization tests pinning the ``show_session`` CLI command output."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from traigent.cli.local_commands import show_session

MOD = "traigent.cli.local_commands"


def _trial(i, score, error=None):
    return SimpleNamespace(
        trial_id=f"t{i}", config={"k": i}, score=score, timestamp="ts", error=error
    )


def _session(**over):
    base = dict(
        session_id="s1",
        function_name="fn",
        created_at="c",
        updated_at="u",
        status="completed",
        total_trials=2,
        completed_trials=2,
        best_config={"model": "a"},
        best_score=0.91234,
        baseline_score=0.5,
        optimization_config={"o": 1},
        trials=[_trial(1, 0.5), _trial(2, None, error="boom")],
    )
    base.update(over)
    return SimpleNamespace(**base)


def _run(session, args=(), summary=None, local=True, hint=None, exc=None):
    storage = MagicMock()
    storage.load_session.return_value = session
    storage.get_session_summary.return_value = summary
    if exc is not None:
        storage.load_session.side_effect = exc
    config = MagicMock()
    config.is_local_mode.return_value = local
    hint = hint or MagicMock()
    with (
        patch(f"{MOD}.TraigentConfig") as cfg_cls,
        patch(f"{MOD}.LocalStorageManager", return_value=storage),
        patch(f"{MOD}.show_upgrade_hint", hint),
    ):
        cfg_cls.from_environment.return_value = config
        result = CliRunner().invoke(show_session, ["s1", *args])
    return result, storage, hint


def test_json_format_full_payload():
    result, _, _ = _run(_session(), ["--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["best_score"] == 0.91234
    assert data["trials"] == [
        {
            "trial_id": "t1",
            "config": {"k": 1},
            "score": 0.5,
            "timestamp": "ts",
            "error": None,
        },
        {
            "trial_id": "t2",
            "config": {"k": 2},
            "score": None,
            "timestamp": "ts",
            "error": "boom",
        },
    ]
    assert list(data)[:3] == ["session_id", "function_name", "created_at"]


def test_json_format_trials_none_is_empty_list():
    result, storage, _ = _run(_session(trials=None), ["--format", "json"])
    assert json.loads(result.output)["trials"] == []
    storage.get_session_summary.assert_not_called()


def test_detailed_format_with_improvement_and_history_marks():
    result, storage, hint = _run(
        _session(), ["--format", "detailed"], summary={"improvement": 0.25}
    )
    assert result.exit_code == 0
    assert result.output == (
        "\n📊 Session Details: s1\n" + "=" * 60 + "\n"
        "Function: fn\nStatus: completed\nCreated: c\nUpdated: u\nTrials: 2\n"
        "Best Score: 0.9123\nImprovement: 25.0%\n"
        "\n🎯 Best Configuration:\n  model: a\n"
        "\n📈 Trial History:\n  ✅ Trial t1: 0.5000\n  ❌ Trial t2: N/A\n"
    )
    storage.get_session_summary.assert_called_once_with("s1")
    hint.assert_not_called()


def test_detailed_history_shows_only_last_five():
    trials = [_trial(i, i / 10) for i in range(8)]
    result, _, _ = _run(_session(trials=trials), ["--format", "detailed"])
    assert "Trial t2:" not in result.output
    assert "Trial t3:" in result.output and "Trial t7:" in result.output


def test_detailed_without_score_config_or_trials():
    result, _, _ = _run(
        _session(best_score=None, best_config=None, trials=[]),
        ["--format", "detailed"],
        summary=None,
    )
    assert result.output.endswith("Trials: 2\n")


def test_detailed_zero_improvement_and_none_summary_are_silent():
    r1, _, _ = _run(_session(), ["--format", "detailed"], summary={"improvement": 0})
    r2, _, _ = _run(_session(), ["--format", "detailed"], summary=None)
    assert "Improvement" not in r1.output and "Improvement" not in r2.output
    assert "Best Score: 0.9123" in r2.output


def test_summary_format_local_completed_shows_hint():
    result, _, hint = _run(_session(), summary={"improvement": 0.1})
    assert result.output == (
        "\n📊 fn Optimization\nStatus: completed | Trials: 2\n"
        "Best Score: 0.9123\nImprovement: 10.0%\nBest Config: {'model': 'a'}\n"
    )
    hint.assert_called_once_with("cli_usage", trial_count=2, best_score=0.91234)


def test_summary_hint_failure_is_swallowed():
    result, _, hint = _run(_session(), hint=MagicMock(side_effect=RuntimeError("x")))
    assert result.exit_code == 0
    assert "Best Config" in result.output
    hint.assert_called_once()


@pytest.mark.parametrize(
    "kwargs",
    [dict(local=False), dict(session=_session(status="running"))],
)
def test_summary_no_hint_when_not_local_or_not_completed(kwargs):
    session = kwargs.pop("session", _session())
    _, _, hint = _run(session, **kwargs)
    hint.assert_not_called()


def test_summary_without_score_or_config():
    result, _, _ = _run(_session(best_score=None, best_config={}))
    assert result.output == "\n📊 fn Optimization\nStatus: completed | Trials: 2\n"


def test_not_found_exits_one_with_message():
    result, storage, _ = _run(None)
    assert result.exit_code == 1
    assert "Session 's1' not found." in result.output
    storage.get_session_summary.assert_not_called()


def test_exception_reported_and_exits_one():
    result, _, _ = _run(_session(), exc=RuntimeError("disk"))
    assert result.exit_code == 1
    assert "Error showing session: disk" in result.output


def test_odd_input_none_status_summary_hint_args():
    result, _, hint = _run(
        _session(status=None, best_score=None, completed_trials=None)
    )
    assert "Status: None | Trials: None" in result.output
    hint.assert_not_called()


def test_odd_best_score_not_float_in_detailed_is_reported_as_error():
    result, _, _ = _run(_session(best_score="high"), ["--format", "detailed"])
    assert result.exit_code == 1
    assert "Error showing session:" in result.output
