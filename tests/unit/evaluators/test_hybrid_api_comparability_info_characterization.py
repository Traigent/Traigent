"""Golden characterization of HybridAPIEvaluator._build_comparability_info."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from traigent.evaluators.hybrid_api import HybridAPIEvaluator, HybridExampleResult

GOLDEN = Path(__file__).parent / "golden" / "hybrid_comparability_info.json"


def _r(eid="e", metrics=None, err=None, meta=None, cost=0.0, lat=0.0, un=False):
    return HybridExampleResult(
        example_id=eid,
        metrics=metrics if metrics is not None else {},
        error=err,
        metadata=meta or {},
        cost_usd=cost,
        latency_ms=lat,
        cost_unmeasured=un,
    )


EV = {"evaluation_mode": "evaluated"}
# name -> (results, metric_counts, primary_objective, mode)
SCENARIOS = {
    "empty": ([], {}, "accuracy", "warn"),
    "empty_cost_objective": ([], {}, "cost", "warn"),
    "full_evaluated": (
        [_r("a", {"accuracy": 1.0}, meta=EV), _r("b", {"accuracy": 0.5}, meta=EV)],
        {"accuracy": 2},
        "accuracy",
        "warn",
    ),
    "legacy_forces_eligible": (
        [_r("a", {}, meta=EV)],
        {},
        "accuracy",
        "legacy",
    ),
    "partial_coverage": (
        [_r("a", {"accuracy": 1.0}, meta=EV), _r("b", {}, meta=EV)],
        {"accuracy": 1},
        "accuracy",
        "warn",
    ),
    "failed_rows": (
        [_r("a", {"accuracy": 1.0}, meta=EV), _r("b", {"accuracy": 1.0}, err="x")],
        {"accuracy": 1},
        "accuracy",
        "strict",
    ),
    "mixed_modes": (
        [
            _r("a", {"accuracy": 1.0}, meta={"evaluation_mode": "evaluated"}),
            _r("b", {"accuracy": 1.0}, meta={"evaluation_mode": "heuristic"}),
        ],
        {"accuracy": 2},
        "accuracy",
        "warn",
    ),
    "mixed_derivation_paths": (
        [
            _r("a", {"accuracy": 1.0}, meta={**EV, "accuracy_derivation_path": "p1"}),
            _r("b", {"accuracy": 1.0}, meta={**EV, "accuracy_derivation_path": "p2"}),
        ],
        {"accuracy": 2},
        "accuracy",
        "warn",
    ),
    "split_mean_and_split_keys": (
        [
            _r("a", {"x_accuracy": 0.5, "y_accuracy": 0.7}, meta=EV),
            _r("b", {"x_accuracy": 0.5}, meta=EV),
        ],
        {"x_accuracy": 2, "y_accuracy": 1},
        "x_accuracy",
        "warn",
    ),
    "metadata_derivation_path": (
        [
            _r(
                "a",
                {"accuracy": 1.0},
                meta={**EV, "accuracy_derivation_path": " custom "},
            )
        ],
        {"accuracy": 1},
        "accuracy",
        "warn",
    ),
    "metadata_path_none_falls_back": (
        [
            _r(
                "a",
                {"overall_accuracy": 1.0},
                meta={**EV, "accuracy_derivation_path": "none"},
            )
        ],
        {"overall_accuracy": 1},
        "overall_accuracy",
        "warn",
    ),
    "cost_objective_unmeasured": (
        [_r("a", {}, cost=0.1, un=True), _r("b", {}, cost=0.2)],
        {"cost": 1},
        "cost",
        "warn",
    ),
    "latency_objective": (
        [_r("a", {}, lat=5.0), _r("b", {}, lat=0.0)],
        {"latency": 2},
        "latency",
        "warn",
    ),
    "unknown_objective": (
        [_r("a", {"f1": 0.5})],
        {"f1": 1},
        "f1",
        "warn",
    ),
    "metric_value_none_not_counted": (
        [_r("a", {"accuracy": None}, meta=EV)],
        {"accuracy": 0},
        "accuracy",
        "warn",
    ),
    "duplicate_ids": (
        [_r("dup", {}, err="x"), _r("dup", {}, meta=EV)],
        {},
        "score",
        "warn",
    ),
}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_matches_golden(name, monkeypatch):
    results, counts, objective, mode = SCENARIOS[name]
    monkeypatch.setenv("TRAIGENT_COMPARABILITY_MODE", mode)
    ev = HybridAPIEvaluator(api_endpoint="http://unused.invalid")
    got = json.loads(
        json.dumps(
            ev._build_comparability_info(results, counts, primary_objective=objective)
        )
    )
    data = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
    if os.environ.get("REGEN_GOLDEN") == "1":
        data[name] = got
        GOLDEN.write_text(json.dumps(data, indent=1, sort_keys=True))
        return
    assert got == data[name]


def test_odd_unparseable_primary_metric_value_raises():
    ev = HybridAPIEvaluator(api_endpoint="http://unused.invalid")
    with pytest.raises(ValueError):
        ev._build_comparability_info(
            [_r("a", {"accuracy": "high"}, meta=EV)], {}, primary_objective="accuracy"
        )
