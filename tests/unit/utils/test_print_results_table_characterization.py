"""Golden-output characterization of print_results_table (colours made visible)."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from traigent.api.types import (
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)
from traigent.utils import results_table
from traigent.utils.results_table import print_results_table

GOLDEN = Path(__file__).parent / "golden" / "print_results_table.json"


def _trial(i, config, metrics, ok=True, attempted=4, successful=None, metadata=None):
    meta = {
        "examples_attempted": attempted,
        "successful_examples": (attempted if ok else 0)
        if successful is None
        else successful,
    }
    meta.update(metadata or {})
    return TrialResult(
        trial_id=f"t{i}",
        config=config,
        metrics=metrics,
        status=TrialStatus.COMPLETED,
        duration=0.1,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        metadata=meta,
    )


def _result(trials, metadata=None, best_config=None, best_score=None):
    return OptimizationResult(
        trials=trials,
        best_config=best_config if best_config is not None else {},
        best_score=best_score,
        optimization_id="o",
        duration=1.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="random",
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        metadata=metadata if metadata is not None else {"mock_llm_run": False},
    )


def _scenarios() -> dict[str, tuple[Any, ...]]:
    t1 = _trial(1, {"model": "a", "t": 0.5, "flag": True}, {"accuracy": 0.5, "cost": 0.02})
    t2 = _trial(2, {"model": "bb", "t": 1.0, "flag": False}, {"accuracy": 0.9, "cost": 0.05})
    space = {"model": ["a", "bb"], "t": [0.5, 1.0], "flag": [True, False]}
    ok = _result(
        [t1, t2],
        best_config=t2.config,
        best_score=0.9,
        metadata={"mock_llm_run": False, "stop_detail": "stopped by cost limit"},
    )
    failed = _result(
        [
            _trial(1, {"model": "a"}, {"accuracy": 0.0}, ok=False),
            _trial(2, {"model": "b"}, {"accuracy": 0.0}, ok=False),
        ]
    )
    unmeasured = _result(
        [
            _trial(1, {"model": "a"}, {"accuracy": 0.5, "cost_unmeasured": 1.0}),
            _trial(2, {"model": "b"}, {"accuracy": 0.6, "cost_unmeasured": 1.0}),
        ]
    )
    noex = _result(
        [
            _trial(1, {"model": "a"}, {"accuracy": 0.5}, attempted=None),
            _trial(2, {"model": "b"}, {"accuracy": 0.7}, attempted=None),
        ]
    )
    for t in noex.trials:
        t.metadata = {}
    mixed_counts = _result(
        [
            _trial(1, {"model": "a"}, {"accuracy": 0.5}),
            _trial(2, {"model": "b"}, {"accuracy": 0.7}, attempted=12, successful=10),
        ]
    )
    mock = _result(
        [t1, t2], metadata={"mock_llm_run": True, "stop_detail": ""}, best_config=t2.config
    )
    nometa = _result([t1, t2], best_config=t2.config)
    nometa.metadata = None
    return {
        "ok_best": (ok, space, ["accuracy", "cost"], dict(mode_label=" MOCK ")),
        "ok_no_show_best": (ok, space, ["accuracy"], dict(show_best=False)),
        "failed": (failed, {"model": []}, ["accuracy"], {}),
        "unmeasured_cost": (unmeasured, {"model": []}, ["accuracy", "cost"], {}),
        "no_examples_cols": (noex, {"model": []}, ["accuracy"], {}),
        "mixed_counts": (mixed_counts, {"model": []}, ["accuracy"], {}),
        "overrides": (
            ok,
            space,
            ["accuracy"],
            dict(metric_overrides={"accuracy": [0.25, 0.75]}),
        ),
        "mock_notice": (mock, space, ["accuracy"], {}),
        "metadata_none": (nometa, space, ["accuracy"], {}),
        "empty": (_result([]), {}, ["accuracy"], {}),
    }


def _render(name, capsys, monkeypatch):
    monkeypatch.setattr(
        results_table, "_get_colors", lambda: ("<B>", "<G>", "<Y>", "<C>", "<D>", "<R>")
    )
    monkeypatch.setattr(results_table, "_mock_run_active", lambda: False)
    res, space, objectives, kwargs = _scenarios()[name]
    print_results_table(res, space, objectives, **kwargs)
    return capsys.readouterr().out


@pytest.mark.parametrize("name", list(_scenarios()))
def test_output_matches_golden(name, capsys, monkeypatch):
    out = _render(name, capsys, monkeypatch)
    data = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
    if os.environ.get("REGEN_GOLDEN") == "1":
        data[name] = out
        GOLDEN.write_text(json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False))
        return
    assert out == data[name]
