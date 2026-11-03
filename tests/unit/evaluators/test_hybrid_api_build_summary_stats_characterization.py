"""Characterization tests pinning HybridAPIEvaluator._build_summary_stats."""

from __future__ import annotations

import pytest

from traigent.evaluators.hybrid_api import HybridAPIEvaluator, HybridExampleResult


@pytest.fixture
def ev() -> HybridAPIEvaluator:
    return HybridAPIEvaluator(api_endpoint="http://unused.invalid")


def _r(metrics=None, cost=0.0, latency=0.0, error=None, unmeasured=False):
    return HybridExampleResult(
        example_id="e",
        metrics=metrics if metrics is not None else {},
        cost_usd=cost,
        latency_ms=latency,
        error=error,
        cost_unmeasured=unmeasured,
    )


def _means(stats):
    return {k: (v["count"], round(v["mean"], 6)) for k, v in stats["metrics"].items()}


def test_empty_results_returns_none(ev):
    assert ev._build_summary_stats([], 1.0) is None


def test_basic_shape_and_derived_metrics(ev):
    results = [
        _r({"accuracy": 0.5}, cost=0.1, latency=100.0),
        _r({"accuracy": 1.0}, cost=0.3, latency=300.0),
    ]
    stats = ev._build_summary_stats(results, 2.5)
    assert _means(stats) == {
        "accuracy": (2.0, 0.75),
        "score": (2.0, 0.75),
        "cost": (2.0, 0.2),
        "total_cost": (2.0, 0.2),
        "latency": (2.0, 200.0),
        "response_time_ms": (2.0, 200.0),
        "success_rate": (2.0, 1.0),
    }
    assert stats["execution_time"] == 2.5
    assert stats["total_examples"] == 2
    meta = stats["metadata"]
    assert meta["aggregation_method"] == "pandas.describe"
    assert meta["source"] == "hybrid_api"
    assert "comparability" not in meta


def test_comparability_included_only_when_dict(ev):
    results = [_r({"accuracy": 1.0})]
    assert ev._build_summary_stats(results, 1.0, {"mode": "warn"})["metadata"][
        "comparability"
    ] == {"mode": "warn"}
    assert (
        "comparability" not in ev._build_summary_stats(results, 1.0, ["x"])["metadata"]
    )


def test_failed_example_contributes_only_success_cost_latency(ev):
    results = [
        _r({"accuracy": 0.8}, cost=0.02, latency=200.0),
        _r({"accuracy": 0.0, "foo": 9.0}, cost=0.5, latency=50.0, error="bad"),
    ]
    m = _means(ev._build_summary_stats(results, 1.0))
    assert m["accuracy"] == (1.0, 0.8)
    assert "foo" not in m
    assert m["success_rate"] == (2.0, 0.5)
    # cost/latency of the errored row are still collected
    assert m["latency"] == (2.0, 125.0)
    assert m["cost"] == (2.0, 0.26)


def test_unmeasured_cost_row_makes_cost_unknown(ev):
    results = [_r({"accuracy": 1.0}, cost=0.2), _r({"accuracy": 1.0}, unmeasured=True)]
    m = _means(ev._build_summary_stats(results, 1.0))
    assert "cost" not in m and "total_cost" not in m


def test_unmeasured_flag_excludes_cost_value(ev):
    m = _means(
        ev._build_summary_stats([_r({"accuracy": 1.0}, cost=0.4, unmeasured=True)], 1.0)
    )
    assert "cost" not in m


def test_zero_cost_and_zero_latency_are_not_measurements(ev):
    m = _means(ev._build_summary_stats([_r({"accuracy": 1.0})], 1.0))
    assert set(m) == {"accuracy", "score", "success_rate"}


def test_payload_metrics_filtering_and_priority(ev):
    results = [
        _r(
            {
                "accuracy": 0.2,
                "f1": 0.5,
                "flag": True,
                "name": "x",
                "none": None,
                "success_rate": 0.0,
                "cost": 99.0,
            },
            cost=0.1,
            latency=10.0,
        )
    ]
    m = _means(ev._build_summary_stats(results, 1.0))
    assert m["f1"] == (1.0, 0.5)
    assert "flag" not in m and "name" not in m and "none" not in m
    # Traigent-computed keys are reserved: payload values are ignored
    assert m["success_rate"] == (1.0, 1.0)
    assert m["cost"] == (1.0, 0.1)


def test_explicit_accuracy_in_payload_kept_over_derived(ev):
    # accuracy comes via payload metrics loop first, so "accuracy" in metric_values
    results = [_r({"accuracy": 0.3, "score": 0.9})]
    m = _means(ev._build_summary_stats(results, 1.0))
    assert m["accuracy"] == (1.0, 0.3)
    assert m["score"] == (1.0, 0.9)


def test_accuracy_derived_from_split_metrics(ev):
    results = [_r({"a_accuracy": 0.2, "b_accuracy": 0.6})]
    m = _means(ev._build_summary_stats(results, 1.0))
    assert m["accuracy"] == (1.0, 0.4)
    assert m["score"] == (1.0, 0.4)
    assert m["a_accuracy"] == (1.0, 0.2)


def test_no_accuracy_no_score(ev):
    m = _means(ev._build_summary_stats([_r({"f1": 0.5})], 1.0))
    assert "accuracy" not in m and "score" not in m


def test_all_failed_still_reports_success_rate_zero(ev):
    stats = ev._build_summary_stats([_r({"accuracy": 1.0}, error="x")], 1.0)
    assert _means(stats) == {"success_rate": (1.0, 0.0)}


def test_odd_none_comparability_and_negative_duration(ev):
    stats = ev._build_summary_stats([_r({"accuracy": 1.0})], -3, None)
    assert stats["execution_time"] == -3.0
    assert "comparability" not in stats["metadata"]
