"""Cached prompt tokens are priced at the cached-input rate (Traigent#3253)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from traigent.evaluators.metrics_tracker import extract_llm_metrics
from traigent.utils import cost_calculator
from traigent.utils.cost_calculator import cost_from_tokens, get_cached_input_rate

MODEL = "gpt-4o-mini"


def _response(prompt, cached, completion=10):
    usage = SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
    )
    return SimpleNamespace(model=MODEL, usage=usage)


def test_cached_rate_comes_from_price_table():
    rate = get_cached_input_rate(MODEL)
    in_cost, _ = cost_from_tokens(1, 0, MODEL)
    assert rate is not None
    assert 0 < rate < in_cost


def test_cached_prompt_tokens_priced_at_cached_rate():
    metrics = extract_llm_metrics(_response(1000, 800), model_name=MODEL)
    in_rate, _ = cost_from_tokens(1, 0, MODEL)
    out_cost = cost_from_tokens(0, 10, MODEL)[1]
    expected_in = 200 * in_rate + 800 * get_cached_input_rate(MODEL)
    assert metrics.cost.input_cost == pytest.approx(expected_in)
    assert metrics.cost.total_cost == pytest.approx(expected_in + out_cost)
    assert metrics.cost.total_cost < cost_from_tokens(1000, 10, MODEL)[0] + out_cost
    assert metrics.cost.cache_rate_unknown is False
    assert metrics.cache_read_tokens == 800


def test_no_cached_tokens_cost_unchanged():
    metrics = extract_llm_metrics(_response(1000, 0), model_name=MODEL)
    in_cost, out_cost = cost_from_tokens(1000, 10, MODEL)
    assert metrics.cost.total_cost == pytest.approx(in_cost + out_cost)
    assert metrics.cost.cache_rate_unknown is False


def test_unknown_cached_rate_keeps_full_rate_and_is_marked(monkeypatch):
    monkeypatch.setattr(cost_calculator, "get_cached_input_rate", lambda m: None)
    metrics = extract_llm_metrics(_response(1000, 800), model_name=MODEL)
    in_cost, out_cost = cost_from_tokens(1000, 10, MODEL)
    # Conservative: never below the full-rate figure.
    assert metrics.cost.total_cost == pytest.approx(in_cost + out_cost)
    assert metrics.cost.cache_rate_unknown is True


def test_cached_rate_above_input_rate_is_not_trusted(monkeypatch):
    monkeypatch.setattr(cost_calculator, "get_cached_input_rate", lambda m: 1.0)
    metrics = extract_llm_metrics(_response(1000, 800), model_name=MODEL)
    in_cost, out_cost = cost_from_tokens(1000, 10, MODEL)
    assert metrics.cost.total_cost == pytest.approx(in_cost + out_cost)
    assert metrics.cost.cache_rate_unknown is True


def test_get_cached_input_rate_unknown_model_is_none():
    assert get_cached_input_rate("definitely-not-a-model-3253") is None


def test_context_tiered_entry_has_no_trusted_cached_rate(monkeypatch):
    entry = {
        "input_cost_per_token": 3e-6,
        "cache_read_input_token_cost": 3e-7,
        "cache_read_input_token_cost_above_200k_tokens": 6e-7,
    }
    monkeypatch.setitem(cost_calculator.litellm.model_cost, "tiered-3253", entry)
    assert get_cached_input_rate("tiered-3253") is None


def test_reported_cost_reconciliation_uses_cached_rate():
    from traigent.evaluators.metrics_tracker import _token_derived_cost

    metrics = extract_llm_metrics(_response(1000, 800), model_name=MODEL)
    derived = _token_derived_cost(metrics, MODEL)
    assert derived is not None
    assert derived[2] == pytest.approx(metrics.cost.total_cost)


def test_cache_rate_unknown_survives_call_merge():
    from traigent.evaluators.base import SimpleScoringEvaluator

    a = extract_llm_metrics(_response(100, 0), model_name=MODEL)
    b = extract_llm_metrics(_response(100, 0), model_name=MODEL)
    b.cost.cache_rate_unknown = True
    SimpleScoringEvaluator._merge_call_metrics(a, b)
    assert a.cost.cache_rate_unknown is True


def _unknown_rate_metrics(monkeypatch, reported_fraction_of_upper):
    from traigent.evaluators.metrics_tracker import (
        ExampleMetrics,
        _reconcile_reported_cost_with_tokens,
    )

    monkeypatch.setattr(cost_calculator, "get_cached_input_rate", lambda m: None)
    metrics = extract_llm_metrics(_response(1000, 800), model_name=MODEL)
    upper = metrics.cost.total_cost
    m = ExampleMetrics()
    m.tokens = metrics.tokens
    m.cache_read_tokens = metrics.cache_read_tokens
    m.cache_total_input_tokens = metrics.cache_total_input_tokens
    m.cost.total_cost = upper * reported_fraction_of_upper
    _reconcile_reported_cost_with_tokens(m, MODEL)
    fresh, out = cost_from_tokens(200, 10, MODEL)
    return m, upper, fresh + out


def test_unknown_rate_reported_between_bounds_is_accepted(monkeypatch):
    m, upper, lower = _unknown_rate_metrics(monkeypatch, 0.5)
    assert lower < upper * 0.5 < upper
    assert m.cost.total_cost == pytest.approx(upper * 0.5)
    assert m.cost.cache_rate_unknown is False


def test_unknown_rate_reported_below_lower_bound_is_clamped(monkeypatch):
    m, upper, lower = _unknown_rate_metrics(monkeypatch, 0.01)
    assert upper * 0.01 < lower
    assert m.cost.total_cost == pytest.approx(lower)
    assert m.cost.cache_rate_unknown is True  # lower bound only: bill may be higher


def test_known_rate_under_report_still_clamped_to_derived(monkeypatch):
    from traigent.evaluators.metrics_tracker import (
        ExampleMetrics,
        _reconcile_reported_cost_with_tokens,
    )

    metrics = extract_llm_metrics(_response(1000, 800), model_name=MODEL)
    derived = metrics.cost.total_cost
    m = ExampleMetrics()
    m.tokens = metrics.tokens
    m.cache_read_tokens = metrics.cache_read_tokens
    m.cache_total_input_tokens = metrics.cache_total_input_tokens
    m.cost.total_cost = derived * 0.01
    _reconcile_reported_cost_with_tokens(m, MODEL)
    assert m.cost.total_cost == pytest.approx(derived)
    assert m.cost.cache_rate_unknown is False
