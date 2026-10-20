"""Regression tests for examples/ci-pipeline/export_results.py (#2635, sol r5)."""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


export_results = _load("examples/ci-pipeline/export_results.py", "ci_export_results")


def _write_run(base: Path, trials: list[dict], total: int | None = None) -> Path:
    run = base / "f_random_60"
    run.mkdir(parents=True)
    meta = {"function_name": "f", "total_trials": total or len(trials)}
    (run / "metadata.json").write_text(json.dumps(meta))
    with gzip.open(run / "trials.json.gz", "wt") as fh:
        json.dump(trials, fh)
    # The truncated, success-only sample must never be what is exported.
    (run / "trials_summary.json").write_text("[]")
    return run


def _trial(i: int, status: str = "completed", **metrics) -> dict:
    return {"config": {"i": i}, "metrics": metrics, "status": status, "score": 1.0}


def test_exports_every_trial_including_failed_beyond_fifty(tmp_path):
    trials = [_trial(i, total_cost=0.01) for i in range(55)]
    trials.append(_trial(55, status="failed"))
    _write_run(tmp_path, trials)
    export_results.export(tmp_path)
    out = json.loads((tmp_path / "latest.json").read_text())["trials"]
    assert len(out) == 56
    assert out[-1]["status"] == "failed"
    assert out[-1]["cost"] is None and out[-1]["cost_status"] == "unmeasured"


def test_unknown_cost_is_never_zero(tmp_path):
    _write_run(
        tmp_path,
        [
            _trial(0, total_cost=0.25),
            _trial(1, cost_unmeasured=1.0, cost_lower_bound=0.1),
            _trial(2, total_cost=0.0, cost_unpriced=1.0),
            _trial(3),
        ],
    )
    export_results.export(tmp_path)
    t = json.loads((tmp_path / "latest.json").read_text())["trials"]
    assert (t[0]["cost"], t[0]["cost_status"]) == (0.25, "measured")
    assert t[1]["cost"] is None and t[1]["cost_status"] == "unmeasured"
    assert t[1]["cost_lower_bound"] == 0.1
    assert t[2]["cost"] is None and t[2]["cost_status"] == "unpriced"
    assert t[3]["cost"] is None and t[3]["cost_status"] == "unmeasured"


@pytest.mark.parametrize("case", ["missing", "corrupt", "count_mismatch"])
def test_incomplete_persistence_fails_closed(tmp_path, case):
    run = _write_run(tmp_path, [_trial(0), _trial(1)], total=2)
    if case == "missing":
        (run / "trials.json.gz").unlink()
    elif case == "corrupt":
        (run / "trials.json.gz").write_bytes(b"not gzip")
    else:
        (run / "metadata.json").write_text(json.dumps({"total_trials": 9}))
    with pytest.raises(SystemExit) as exc:
        export_results.export(tmp_path)
    assert isinstance(exc.value.code, str) and "incomplete" in exc.value.code
    assert not (tmp_path / "latest.json").exists()


def test_downstream_evaluate_treats_unknown_cost_as_unmeasured():
    ev = _load("scripts/auto_tune/evaluate_performance.py", "ci_eval_perf")
    results = {
        "trials": [
            {"score": 1.0, "cost": 0.5, "cost_status": "measured"},
            {"score": 1.0, "cost": None, "cost_status": "unmeasured"},
            {"score": 0.0, "cost": 0.0, "cost_status": "unpriced"},
        ]
    }
    m = ev.calculate_metrics(results)
    assert m["total_cost"] == 0.5
    assert (m["cost_trials_measured"], m["cost_trials_unmeasured"]) == (1, 2)
    none = ev.calculate_metrics(
        {"trials": [{"score": 1.0, "cost_status": "unmeasured"}]}
    )
    assert none["total_cost"] is None
    cmp = ev.compare_with_baseline(none, {"accuracy": 1.0, "total_cost": 0.0}, 0.05)
    assert "total_cost" not in cmp["metrics"]
    assert not cmp["has_regression"]
