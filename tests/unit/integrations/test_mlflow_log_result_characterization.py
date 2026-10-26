"""Characterization of TraigentMLflowTracker.log_optimization_result."""

from __future__ import annotations

from types import SimpleNamespace as NS
from unittest.mock import MagicMock, call

import pytest

import traigent.integrations.observability.mlflow as mod
from traigent.integrations.observability.mlflow import TraigentMLflowTracker


@pytest.fixture
def ml(monkeypatch):
    m = MagicMock(name="mlflow")
    monkeypatch.setattr(mod, "mlflow", m)
    return m


def _tracker(active=True):
    t = object.__new__(TraigentMLflowTracker)
    t.current_run = NS(info=NS(run_id="rid")) if active else None
    return t


def _trial(status, metrics=None):
    ns = NS(status=NS(value=status))
    if metrics is not None:
        ns.metrics = metrics
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


def test_no_active_run_does_nothing(ml):
    _tracker(active=False).log_optimization_result(_result())
    assert ml.mock_calls == []


def test_minimal_result_calls_exact(ml):
    _tracker().log_optimization_result(_result())
    assert ml.log_metric.call_args_list == [
        call("total_trials", 0),
        call("optimization_duration", 2.5),
    ]
    assert ml.log_param.call_args_list == [call("optimization_status", "completed")]
    ml.log_dict.assert_called_once_with(
        {
            "best_config": {},
            "best_metrics": {},
            "total_trials": 0,
            "duration": 2.5,
            "status": "completed",
            "metadata": {},
        },
        "optimization_result.json",
    )
    ml.log_artifact.assert_not_called()


def test_best_config_metrics_and_trial_statistics_order(ml):
    trials = [
        _trial("completed", {"acc": 1.0}),
        _trial("failed", {"acc": 0.0}),
        _trial("completed", {"acc": 3.0, "other": 1}),
        _trial("completed", {}),
        NS(),  # no status attribute -> not successful
    ]
    r = _result(best_config={"m": "x"}, best_metrics={"acc": 3.0}, trials=trials)
    _tracker().log_optimization_result(r)
    assert ml.log_param.call_args_list == [
        call("best_m", "x"),
        call("optimization_status", "completed"),
    ]
    names = [c.args[0] for c in ml.log_metric.call_args_list]
    assert names == [
        "best_acc",
        "total_trials",
        "optimization_duration",
        "successful_trials",
        "success_rate",
        "acc_mean",
        "acc_std",
    ]
    values = {c.args[0]: c.args[1] for c in ml.log_metric.call_args_list}
    assert values["successful_trials"] == 3
    assert values["success_rate"] == pytest.approx(3 / 5)
    assert values["acc_mean"] == 2.0
    assert values["acc_std"] == pytest.approx(1.0)


def test_no_stats_when_no_successful_trials_or_no_best_metrics(ml):
    _tracker().log_optimization_result(
        _result(trials=[_trial("failed", {"a": 1})], best_metrics={"a": 1})
    )
    names = [c.args[0] for c in ml.log_metric.call_args_list]
    assert names == [
        "best_a",
        "total_trials",
        "optimization_duration",
        "successful_trials",
        "success_rate",
    ]
    ml.reset_mock()
    _tracker().log_optimization_result(_result(trials=[_trial("completed", {"a": 1})]))
    names = [c.args[0] for c in ml.log_metric.call_args_list]
    assert "a_mean" not in names and "successful_trials" in names


def test_metric_with_no_values_is_skipped(ml):
    r = _result(trials=[_trial("completed", {"x": 1})], best_metrics={"y": 1})
    _tracker().log_optimization_result(r)
    names = [c.args[0] for c in ml.log_metric.call_args_list]
    assert "y_mean" not in names and "y_std" not in names


def test_metadata_numeric_as_metric_else_param_string(ml):
    r = _result(metadata={"n": 3, "f": 0.5, "s": "t", "b": True, "l": [1]})
    _tracker().log_optimization_result(r)
    metrics = [c for c in ml.log_metric.call_args_list if c.args[0].startswith("metadata_")]
    params = [c for c in ml.log_param.call_args_list if c.args[0].startswith("metadata_")]
    assert metrics == [
        call("metadata_n", 3),
        call("metadata_f", 0.5),
        call("metadata_b", True),
    ]
    assert params == [call("metadata_s", "t"), call("metadata_l", "[1]")]


def test_dataset_artifact_only_when_path_exists(ml, tmp_path):
    f = tmp_path / "d.jsonl"
    f.write_text("{}")
    _tracker().log_optimization_result(_result(), dataset_path=str(f))
    ml.log_artifact.assert_called_once_with(str(f), "datasets")
    ml.reset_mock()
    _tracker().log_optimization_result(_result(), dataset_path=str(tmp_path / "no"))
    ml.log_artifact.assert_not_called()
    _tracker().log_optimization_result(_result(), dataset_path="")
    ml.log_artifact.assert_not_called()
