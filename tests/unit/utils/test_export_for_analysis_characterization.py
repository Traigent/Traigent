"""Characterization tests for OptimizationAnalyzer.export_for_analysis."""

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from traigent.utils.optimization_analyzer import OptimizationAnalyzer


@pytest.fixture
def analyzer(tmp_path: Path) -> OptimizationAnalyzer:
    return OptimizationAnalyzer(base_path=tmp_path)


def _runs(*pairs):
    return pd.DataFrame([{"experiment": e, "run_id": r} for e, r in pairs])


def _export(analyzer, runs, loads, tmp_path, fmt="json", exp="exp"):
    calls = []

    def fake_load(experiment, run_id):
        calls.append((experiment, run_id))
        return loads[run_id]

    with (
        patch.object(OptimizationAnalyzer, "_get_all_runs", return_value=runs),
        patch.object(analyzer, "load_run", side_effect=fake_load),
    ):
        out = analyzer.export_for_analysis(exp, format=fmt, output_dir=tmp_path)
    return out, calls


def test_empty_runs_frame_returns_none(analyzer, tmp_path):
    out, calls = _export(analyzer, pd.DataFrame(), {}, tmp_path)
    assert out is None and calls == []


def test_unknown_experiment_returns_none_without_loading(analyzer, tmp_path):
    out, calls = _export(analyzer, _runs(("other", "r1")), {}, tmp_path)
    assert out is None and calls == []


def test_runs_without_trials_key_or_trials_return_none(analyzer, tmp_path):
    loads = {"r1": {}, "r2": {"trials": []}, "r3": {"session": {}}}
    out, calls = _export(
        analyzer, _runs(("exp", "r1"), ("exp", "r2"), ("exp", "r3")), loads, tmp_path
    )
    assert out is None
    assert calls == [("exp", "r1"), ("exp", "r2"), ("exp", "r3")]
    assert list(tmp_path.iterdir()) == []


def test_row_columns_and_order(analyzer, tmp_path):
    loads = {
        "r1": {
            "trials": [
                {
                    "trial_id": "t1",
                    "status": "ok",
                    "duration": 1.5,
                    "config": {"temp": 0.1, "model": "m"},
                    "metrics": {"acc": 0.9},
                },
                {"trial_id": "t2"},
            ]
        },
        "r2": {"trials": [{"status": "failed", "metrics": {"cost": 2}}]},
    }
    out, calls = _export(
        analyzer, _runs(("exp", "r1"), ("x", "rx"), ("exp", "r2")), loads, tmp_path
    )
    assert calls == [("exp", "r1"), ("exp", "r2")]
    assert Path(out).parent == tmp_path
    assert Path(out).suffix == ".json"
    assert Path(out).name.startswith("exp_")
    rows = json.loads(Path(out).read_text())
    assert [r["run_id"] for r in rows] == ["r1", "r1", "r2"]
    assert rows[0] == {
        "run_id": "r1",
        "trial_id": "t1",
        "status": "ok",
        "duration": 1.5,
        "config_temp": 0.1,
        "config_model": "m",
        "metric_acc": 0.9,
        "metric_cost": None,
    }
    assert rows[1]["trial_id"] == "t2" and rows[1]["status"] is None
    assert rows[2]["metric_cost"] == 2 and rows[2]["trial_id"] is None


def test_csv_format_writes_without_index(analyzer, tmp_path):
    loads = {"r1": {"trials": [{"trial_id": "t1", "config": {"a": 1}}]}}
    out, _ = _export(analyzer, _runs(("exp", "r1")), loads, tmp_path, fmt="csv")
    assert Path(out).suffix == ".csv"
    df = pd.read_csv(out)
    assert list(df.columns) == ["run_id", "trial_id", "status", "duration", "config_a"]


def test_unsupported_format_returns_none_after_loading(analyzer, tmp_path):
    loads = {"r1": {"trials": [{"trial_id": "t1"}]}}
    out, calls = _export(analyzer, _runs(("exp", "r1")), loads, tmp_path, fmt="xml")
    assert out is None
    assert calls == [("exp", "r1")]
    assert list(tmp_path.iterdir()) == []


def test_default_output_dir_is_cwd(analyzer, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    loads = {"r1": {"trials": [{"trial_id": "t1"}]}}
    with (
        patch.object(
            OptimizationAnalyzer, "_get_all_runs", return_value=_runs(("exp", "r1"))
        ),
        patch.object(analyzer, "load_run", return_value=loads["r1"]),
    ):
        out = analyzer.export_for_analysis("exp", format="csv")
    assert Path(out).parent == tmp_path
