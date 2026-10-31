"""Golden characterization of _compute_aggregated_metrics_with_comparability."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from traigent.evaluators.hybrid_api import HybridAPIEvaluator, HybridExampleResult

GOLDEN = Path(__file__).parent / "golden" / "hybrid_aggregated_comparability.json"


def _r(metrics=None, cost=0.0, lat=0.0, err=None, un=False):
    return HybridExampleResult(
        example_id="e",
        metrics=metrics or {},
        cost_usd=cost,
        latency_ms=lat,
        error=err,
        cost_unmeasured=un,
    )


SCENARIOS = {
    "empty": ([], 0.0),
    "basic": (
        [
            _r({"accuracy": 0.5, "f1": 0.2}, 0.1, 100.0),
            _r({"accuracy": 1.0, "f1": 0.4}, 0.3, 300.0),
        ],
        0.4,
    ),
    "failed_row": (
        [
            _r({"accuracy": 0.8}, 0.02, 200.0),
            _r({"accuracy": 0.0, "z": 5.0}, 0.0, 0.0, "bad"),
        ],
        0.02,
    ),
    "unknown_cost": (
        [_r({"accuracy": 1.0}, 0.2, 10.0), _r({"accuracy": 0.0}, 0.0, 20.0, un=True)],
        0.2,
    ),
    "all_unmeasured": ([_r({"accuracy": 1.0}, 0.0, 0.0, un=True)], 0.0),
    "split_acc": ([_r({"a_accuracy": 0.2, "b_accuracy": 0.6})], 0.0),
    "reserved_payload": (
        [
            _r(
                {
                    "cost": 9.0,
                    "success_rate": 0.0,
                    "score": 0.7,
                    "cost_unmeasured": 1.0,
                },
                0.1,
                5.0,
            )
        ],
        0.1,
    ),
    "no_latency": ([_r({"f1": 0.5}, 0.1)], 0.1),
    "all_failed": ([_r({"accuracy": 1.0}, 0.1, 5.0, "x")], 0.1),
    "two_unmeasured": (
        [
            _r({"accuracy": 1.0}, 0.0, 0.0, un=True),
            _r({"accuracy": 1.0}, 0.0, 0.0, un=True),
        ],
        0.0,
    ),
    "no_metrics_at_all": ([_r({}, 0.0, 0.0)], 0.0),
}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_matches_golden(name, monkeypatch):
    monkeypatch.setenv("TRAIGENT_COMPARABILITY_MODE", "warn")
    ev = HybridAPIEvaluator(api_endpoint="http://unused.invalid")
    results, total_cost = SCENARIOS[name]
    aggregated, comparability = ev._compute_aggregated_metrics_with_comparability(
        results, total_cost
    )
    got = json.loads(
        json.dumps({"aggregated": aggregated, "comparability": comparability})
    )
    data = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
    if os.environ.get("REGEN_GOLDEN") == "1":
        data[name] = got
        GOLDEN.write_text(json.dumps(data, indent=1, sort_keys=True))
        return
    assert got == data[name]


def test_odd_non_numeric_metric_value_raises_type_error():
    ev = HybridAPIEvaluator(api_endpoint="http://unused.invalid")
    with pytest.raises(TypeError):
        ev._compute_aggregated_metrics_with_comparability(
            [_r({"accuracy": "high"})], 0.0
        )


def test_odd_failed_row_with_non_numeric_metric_is_ignored():
    ev = HybridAPIEvaluator(api_endpoint="http://unused.invalid")
    aggregated, _ = ev._compute_aggregated_metrics_with_comparability(
        [_r({"accuracy": "high"}, err="x")], 0.0
    )
    assert aggregated["success_rate"] == 0.0
