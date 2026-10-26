"""Characterization of TraigentWandBTracker.log_optimization_result."""

from __future__ import annotations

import json
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

import pytest

import traigent.integrations.observability.wandb as mod
from traigent.integrations.observability.wandb import TraigentWandBTracker


@pytest.fixture
def wb(monkeypatch, tmp_path):
    m = MagicMock(name="wandb")
    saved = {}

    def _save(path):
        try:
            with open(path) as f:
                saved[path] = json.load(f)
        except FileNotFoundError:
            saved[path] = None

    m.save.side_effect = _save
    m._saved = saved
    monkeypatch.setattr(mod, "wandb", m)
    monkeypatch.chdir(tmp_path)
    return m


def _tracker(active=True):
    t = object.__new__(TraigentWandBTracker)
    t.current_run = NS(id="rid") if active else None
    return t


def _trial(status, metrics=None, tid="t"):
    ns = NS(
        status=NS(value=status),
        trial_id=tid,
        config={"a": 1},
        duration=1.5,
    )
    ns.metrics = metrics if metrics is not None else {}
    return ns


def _result(**kw):
    base = dict(
        best_config={},
        best_metrics={},
        trials=[],
        duration=2.5,
        status=NS(value="completed"),
        metadata={},
    )
    base.update(kw)
    return NS(**base)


def test_no_active_run_does_nothing(wb):
    with patch.object(mod, "logger") as lg:
        _tracker(active=False).log_optimization_result(_result())
    lg.warning.assert_called_once_with(
        "No active W&B run. Call start_optimization_run first."
    )
    assert wb.mock_calls == []


def test_minimal_result_summary_and_artifact(wb, tmp_path):
    _tracker().log_optimization_result(_result())
    wb.log.assert_called_once_with(
        {
            "optimization/total_trials": 0,
            "optimization/duration": 2.5,
            "optimization/status": "completed",
        }
    )
    assert wb._saved["optimization_result.json"] == {
        "best_config": {},
        "best_metrics": {},
        "total_trials": 0,
        "duration": 2.5,
        "status": "completed",
        "metadata": {},
        "trial_summary": [],
    }
    assert not (tmp_path / "optimization_result.json").exists()


def test_full_summary_golden(wb):
    trials = [
        _trial("completed", {"acc": 0.5, "other": 1}, "t1"),
        _trial("completed", {"acc": 1.0}, "t2"),
        _trial("failed", {"acc": 0.0}, "t3"),
        _trial("completed", {}, "t4"),
    ]
    res = _result(
        best_config={"model": "m", "t": 0.2},
        best_metrics={"acc": 1.0},
        trials=trials,
        metadata={"k": "v"},
    )
    _tracker().log_optimization_result(res)
    summary = wb.log.call_args.args[0]
    assert summary == {
        "optimization/total_trials": 4,
        "optimization/duration": 2.5,
        "optimization/status": "completed",
        "best_config/model": "m",
        "best_config/t": 0.2,
        "best_metrics/acc": 1.0,
        "optimization/successful_trials": 3,
        "optimization/success_rate": 0.75,
        "statistics/acc_mean": 0.75,
        "statistics/acc_std": 0.25,
        "statistics/acc_min": 0.5,
        "statistics/acc_max": 1.0,
        "metadata/k": "v",
    }
    assert list(summary)[:3] == [
        "optimization/total_trials",
        "optimization/duration",
        "optimization/status",
    ]
    assert [
        t["trial_id"] for t in wb._saved["optimization_result.json"]["trial_summary"]
    ] == [
        "t1",
        "t2",
        "t3",
        "t4",
    ]


def test_no_statistics_without_successful_trials_or_best_metrics(wb):
    res = _result(best_metrics={"acc": 1.0}, trials=[_trial("failed", {"acc": 1.0})])
    _tracker().log_optimization_result(res)
    summary = wb.log.call_args.args[0]
    assert summary["optimization/successful_trials"] == 0
    assert summary["optimization/success_rate"] == 0.0
    assert not [k for k in summary if k.startswith("statistics/")]

    wb.reset_mock()
    res = _result(trials=[_trial("completed", {"acc": 1.0})])
    _tracker().log_optimization_result(res)
    assert not [k for k in wb.log.call_args.args[0] if k.startswith("statistics/")]


def test_metric_absent_from_all_trials_has_no_statistics(wb):
    res = _result(best_metrics={"acc": 1.0}, trials=[_trial("completed", {"x": 1})])
    _tracker().log_optimization_result(res)
    assert not [k for k in wb.log.call_args.args[0] if k.startswith("statistics/")]


def test_trial_without_status_attr_excluded_from_successful(wb):
    odd = NS(trial_id="o", config={}, metrics={}, duration=0, status=NS(value="x"))
    del odd.metrics
    res = _result(trials=[odd])
    # summary_data filtering uses hasattr(status); artifact serialisation then
    # needs metrics, so it raises AttributeError after the summary was logged.
    with pytest.raises(AttributeError):
        _tracker().log_optimization_result(res)
    assert wb.log.call_args.args[0]["optimization/successful_trials"] == 0


def test_dataset_saved_only_when_it_exists(wb, tmp_path):
    ds = tmp_path / "ds.jsonl"
    ds.write_text("{}")
    _tracker().log_optimization_result(_result(), dataset_path=str(ds))
    assert [c.args[0] for c in wb.save.call_args_list] == [
        "optimization_result.json",
        str(ds),
    ]
    wb.reset_mock()
    _tracker().log_optimization_result(_result(), dataset_path=str(tmp_path / "nope"))
    assert [c.args[0] for c in wb.save.call_args_list] == ["optimization_result.json"]
    wb.reset_mock()
    _tracker().log_optimization_result(_result(), dataset_path="")
    assert [c.args[0] for c in wb.save.call_args_list] == ["optimization_result.json"]


def test_logs_info_with_run_id(wb):
    with patch.object(mod, "logger") as lg:
        _tracker().log_optimization_result(_result())
    lg.info.assert_called_once_with("Logged optimization result to W&B run: rid")
