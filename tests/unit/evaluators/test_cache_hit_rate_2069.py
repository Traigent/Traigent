"""Cached input tokens and cache-hit rate per trial (Traigent#2069)."""

from __future__ import annotations

import datetime
from types import SimpleNamespace

from traigent.core.cache_usage import normalize_cache_usage
from traigent.api.types import OptimizationResult, TrialResult, TrialStatus
from traigent.evaluators.local import LocalEvaluator
from traigent.evaluators.metrics_tracker import ExampleMetrics, MetricsTracker
from traigent.evaluators.metrics_tracker import ResponseHandler


def _openai_response(prompt, cached):
    usage = SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=5,
        total_tokens=prompt + 5,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
    )
    return SimpleNamespace(usage=usage)


def _tracker(*rows):
    tracker = MetricsTracker()
    for read, total in rows:
        m = ExampleMetrics()
        m.cache_read_tokens = read
        m.cache_total_input_tokens = total
        tracker.add_example_metrics(m)
    return tracker


def test_mixed_cached_and_uncached_calls_sum_and_rate():
    out = _tracker((800, 1000), (0, 1000)).aggregate_cache_usage()
    assert out["cached_input_tokens"] == 800
    assert out["cache_hit_rate"] == 0.4


def test_unknown_provider_cache_data_is_none_not_zero():
    out = _tracker((800, 1000), (None, None)).aggregate_cache_usage()
    assert out == {"cached_input_tokens": None, "cache_hit_rate": None}
    assert MetricsTracker().aggregate_cache_usage()["cached_input_tokens"] is None


def test_response_handler_extracts_openai_subset_semantics():
    read, total = ResponseHandler._extract_cache_counts(_openai_response(2006, 1920))
    assert (read, total) == (1920, 2006)


def test_response_without_cache_fields_is_unknown():
    resp = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=1))
    assert ResponseHandler._extract_cache_counts(resp) == (None, None)


def test_with_usage_meta_populates_example_cache_fields():
    # Same envelope with_usage(provider_usage=...) builds in optimization mode.
    cache = normalize_cache_usage({"inputTokens": 6, "cacheReadInputTokens": 4609})
    meta = {
        "total_cost": 0.001,
        "usage": {"input_tokens": cache.input_tokens},
        "cache_usage": cache.as_metadata(),
    }
    metrics = ExampleMetrics()
    LocalEvaluator._inject_cache_usage_from_meta(meta, metrics)
    assert metrics.cache_read_tokens == 4609
    assert metrics.cache_total_input_tokens == 4615


def test_meta_without_cache_usage_stays_unknown():
    metrics = ExampleMetrics()
    LocalEvaluator._inject_cache_usage_from_meta(
        {"total_cost": 0.1, "usage": {"input_tokens": 5}}, metrics
    )
    assert metrics.cache_read_tokens is None


def _trial(tid, rate):
    meta = {} if rate is None else {"cache_hit_rate": rate}
    return TrialResult(
        trial_id=tid,
        config={"model": "m"},
        metrics={"accuracy": 0.5, "cost": 0.1},
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime.datetime.now(),
        metadata=meta,
    )


def _result(trials):
    return OptimizationResult(
        trials=trials,
        best_config={"model": "m"},
        best_score=0.5,
        optimization_id="o",
        duration=1.0,
        convergence_info={},
        status="completed",
        objectives=["accuracy"],
        algorithm="grid",
        timestamp=datetime.datetime.now(),
    )


def test_summary_note_when_rates_differ_materially():
    summary = _result([_trial("a", 0.0), _trial("b", 0.6)]).get_summary()
    assert "run order" in summary["cache_note"]
    assert "0%" in summary["cache_note"] and "60%" in summary["cache_note"]


def test_summary_no_note_for_close_or_unknown_rates():
    assert (
        "cache_note" not in _result([_trial("a", 0.5), _trial("b", 0.6)]).get_summary()
    )
    assert (
        "cache_note" not in _result([_trial("a", None), _trial("b", 0.9)]).get_summary()
    )
