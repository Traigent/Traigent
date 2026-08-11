"""Regression tests for Traigent#1964: per-trial cost excluded errored examples
that still incurred real, billable cost.

A provider call that ERRORED can still have burned real tokens before failing
-- the LLM call itself succeeded and incurred cost; something downstream
(output parsing, a scoring function, the eval harness) raised afterward.
Three cost aggregation sites shared the same "successful examples only" bug,
one of which is the value actually surfaced as the live ``cost`` key for the
default (local) evaluator:

* ``BaseEvaluator._compute_cost`` -- the metric-registry default MEAN.
* ``MetricsTracker.aggregate_metrics``'s ``total_cost`` stat -- feeds
  ``cost_per_example_mean``.
* ``MetricsTracker.format_for_backend``'s ``cost_total`` -- the per-trial
  TOTAL that becomes the live ``cost`` key for ``LocalEvaluator`` (it
  overwrites whatever ``_compute_cost`` computed via
  ``_merge_comprehensive_metrics``).

``_compute_latency`` is deliberately left alone: it already includes any
example with a recorded positive ``execution_time`` regardless of error
status -- only its docstring was stale, claiming "successful examples" when
the actual filter was time-based. This file exists to prove ``cost`` now
agrees with that behaviour instead of disagreeing with it.

A fourth site sharing the identical bug was found during the pre-merge
review of this fix (a knowingly-omitted "tertiary insights" path flagged as
DO_NOT_MERGE): ``MetricsTracker.format_as_summary_stats`` -- the
pandas.describe()-compatible payload generated for privacy-preserving /
hybrid submission mode -- filtered ``input_cost``/``output_cost``/
``total_cost`` (and token/response-time stats) to ``m.success`` examples
only, silently dropping the same real, billable spend from an errored-but-
costly example that the other three sites were fixed to include.
``LocalEvaluator._extract_llm_metrics_for_output`` populates cost/tokens on
every ``ExampleMetrics`` from the actual output BEFORE ``example_metric.
success`` is set from the error, so an errored-downstream example genuinely
does carry real cost/token data that a success-only filter throws away.
"""

from __future__ import annotations

import pytest

from traigent.evaluators.base import BaseEvaluator
from traigent.evaluators.metrics_tracker import (
    CostMetrics,
    ExampleMetrics,
    MetricsTracker,
)


class _DummyEvaluator(BaseEvaluator):
    async def evaluate(self, func, config, dataset, **kwargs):  # noqa: D401, ANN001
        raise NotImplementedError


def _mixed_example_metrics() -> list[ExampleMetrics]:
    """2 successes at 0.01 each, one error that still burned 0.03, one error
    that burned nothing. True total: 0.05. True mean over all 4: 0.0125.

    The old, buggy "successful only" aggregation saw only the first two:
    sum 0.02, mean 0.01 -- silently dropping the 0.03 a failed call actually
    spent.
    """
    return [
        ExampleMetrics(cost=CostMetrics(input_cost=0.01), success=True),
        ExampleMetrics(cost=CostMetrics(input_cost=0.01), success=True),
        ExampleMetrics(cost=CostMetrics(input_cost=0.03), success=False, error="boom"),
        ExampleMetrics(cost=CostMetrics(), success=False, error="boom2"),
    ]


def _mixed_tracker() -> MetricsTracker:
    tracker = MetricsTracker()
    tracker.start_tracking()
    for metrics in _mixed_example_metrics():
        tracker.add_example_metrics(metrics)
    tracker.end_tracking()
    return tracker


def test_compute_cost_includes_errored_examples_that_burned_real_tokens():
    evaluator = _DummyEvaluator()
    example_metrics = _mixed_example_metrics()
    errors = [None, None, "boom", "boom2"]
    outputs = [None] * 4
    expected = [None] * 4

    cost = evaluator._compute_cost(
        outputs, expected, errors, example_metrics=example_metrics
    )

    # True mean over all 4: (0.01+0.01+0.03+0.0)/4 = 0.0125.
    # Old buggy code averaged only the 2 successful examples: 0.01.
    assert cost == pytest.approx(0.0125)
    assert cost != pytest.approx(0.01)


def test_format_for_backend_cost_sum_includes_errored_examples():
    formatted = _mixed_tracker().format_for_backend()

    # True total: 0.01 + 0.01 + 0.03 + 0.0 = 0.05.
    # Old buggy code summed only the 2 successful examples: 0.02.
    assert formatted["cost"] == pytest.approx(0.05)
    assert formatted["cost"] != pytest.approx(0.02)


def test_format_for_backend_cost_per_example_mean_includes_errored_examples():
    formatted = _mixed_tracker().format_for_backend()

    assert formatted["cost_per_example_mean"] == pytest.approx(0.0125)
    assert formatted["cost_per_example_mean"] != pytest.approx(0.01)


def test_aggregate_metrics_total_cost_stat_includes_errored_examples():
    aggregated = _mixed_tracker().aggregate_metrics()

    assert aggregated["total_cost"]["mean"] == pytest.approx(0.0125)
    assert aggregated["total_cost"]["mean"] != pytest.approx(0.01)


def test_cost_and_latency_now_agree_on_which_examples_exist():
    """The disagreement the issue names directly: latency already counted a
    positive-execution_time errored example; cost now does too, using the
    same "we have a real measurement" gate rather than an error-based one.
    """
    from traigent.evaluators.metrics_tracker import ResponseMetrics

    evaluator = _DummyEvaluator()
    example_results = [
        type("R", (), {"execution_time": 1.0})(),  # success, 1s
        type("R", (), {"execution_time": 2.0})(),  # errored but took 2s
    ]
    example_metrics = [
        ExampleMetrics(
            cost=CostMetrics(input_cost=0.01),
            response=ResponseMetrics(response_time_ms=1000.0),
            success=True,
        ),
        ExampleMetrics(
            cost=CostMetrics(input_cost=0.02),
            response=ResponseMetrics(response_time_ms=2000.0),
            success=False,
            error="boom",
        ),
    ]
    errors = [None, "boom"]
    outputs = [None, None]
    expected = [None, None]

    latency = evaluator._compute_latency(
        outputs, expected, errors, example_results=example_results
    )
    cost = evaluator._compute_cost(
        outputs, expected, errors, example_metrics=example_metrics
    )

    # Latency already included both (1000+2000)/2 = 1500ms.
    assert latency == pytest.approx(1500.0)
    # Cost now includes both too: (0.01+0.02)/2 = 0.015, not 0.01 (successful
    # example only, the old behaviour that disagreed with latency).
    assert cost == pytest.approx(0.015)
    assert cost != pytest.approx(0.01)


def test_format_as_summary_stats_includes_cost_from_errored_examples():
    """The 4th aggregation site found during pre-merge review: the
    privacy-mode/hybrid-submission summary_stats payload must not silently
    drop cost/tokens from an example that errored AFTER the provider call
    already burned real, billable tokens.
    """
    tracker = _mixed_tracker()

    summary_stats = tracker.format_as_summary_stats()

    total_cost_stats = summary_stats["metrics"]["total_cost"]
    input_tokens_stats = summary_stats["metrics"]["input_tokens"]

    # 4 examples total (2 success + 2 errored, one of which burned 0.03).
    # Old (buggy): filtered to the 2 successful examples -> count=2,
    # sum=0.02, mean=0.01 -- the $0.03 burned by the errored call at index 2
    # is invisible.
    # Fixed: count=4, sum=0.05, mean=0.0125 -- the true spend.
    assert total_cost_stats["count"] == 4
    assert total_cost_stats["mean"] == pytest.approx(0.0125)
    assert total_cost_stats["mean"] != pytest.approx(0.01)

    # Token counts must agree: all 4 examples counted, not just the 2
    # successful ones (every ExampleMetrics here has 0 tokens recorded, so
    # this asserts on count/shape rather than a nonzero mean).
    assert input_tokens_stats["count"] == 4
