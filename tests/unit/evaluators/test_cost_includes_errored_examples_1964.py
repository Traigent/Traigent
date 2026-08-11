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
Its own regression test originally pinned only ``total_cost`` and an
``input_tokens`` COUNT (using all-zero token fixtures for the other six
fields), so a one-line regression restoring the success-only filter on any
field but ``total_cost`` would have gone unobserved. It has been
strengthened below to use non-zero, distinct values for every field and
assert all seven (``input_cost``, ``output_cost``, ``input_tokens``,
``output_tokens``, ``total_tokens``, ``response_time_ms``,
``tokens_per_second``).

A FIFTH site sharing the identical bug was found while re-reviewing PR
#2160's response to the DO_NOT_MERGE gate above (specifically while
answering "is there a path the sweep still has not enumerated?"):
``MetricsTracker.aggregate_metrics`` itself. Its ``total_cost``/
``input_cost``/``output_cost`` lists were already fixed to read every
example (see the docstring above), but its ``input_tokens``/
``output_tokens``/``total_tokens``/``response_time_ms``/
``tokens_per_second`` lists were still built from ``successful_metrics``
only -- the identical bug, unfixed, on the non-cost fields. Worse: an
`if not successful_metrics: return self._empty_aggregated_metrics()` early
return meant a trial where EVERY example errored (but still burned real
cost/tokens) lost its cost stats too, DESPITE the #1964 fix, because the
already-fixed per-example cost lists were never reached. This function
feeds the LIVE ``input_tokens``/``output_tokens``/``total_tokens``/
``response_time_ms``/``tokens_per_second``/``cost_per_example_mean`` keys
via ``format_for_backend()``, which unconditionally overwrites
``LocalEvaluator.evaluate()``'s aggregated metrics
(``_merge_comprehensive_metrics``) -- so both gaps were live and
user-facing, not merely theoretical. See
``test_aggregate_metrics_tokens_and_response_time_include_errored_examples``
and
``test_aggregate_metrics_all_failed_trial_still_reports_real_cost_and_tokens``
below.
"""

from __future__ import annotations

import pytest

from traigent.evaluators.base import BaseEvaluator
from traigent.evaluators.metrics_tracker import (
    CostMetrics,
    ExampleMetrics,
    MetricsTracker,
    ResponseMetrics,
    TokenMetrics,
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


def _full_field_example_metrics() -> list[ExampleMetrics]:
    """2 successes + 2 downstream-errored examples, EVERY field non-zero and
    distinct per example, so filtering to ``m.success`` changes the count
    AND the mean of every single field -- not just ``total_cost``.

    input_tokens:      [100, 120,  90,  60]  sum=370  mean=92.5
    output_tokens:      [40,  50,  30,  20]  sum=140  mean=35.0
    total_tokens:      [140, 170, 120,  80]  sum=510  mean=127.5
    response_time_ms: [150, 170, 130,  90]  sum=540  mean=135.0
    tokens_per_second:  [20,  22,  18,  15]  sum=75   mean=18.75
    input_cost:      [.010,.012,.009,.006]  sum=.037 mean=.00925
    output_cost:     [.004,.005,.003,.002]  sum=.014 mean=.0035
    total_cost:      [.014,.017,.012,.008]  sum=.051 mean=.01275

    Old (buggy, success-only) would see just the first two rows of each
    column above -- a different, HIGHER count/mean for every field.
    """
    return [
        ExampleMetrics(
            tokens=TokenMetrics(input_tokens=100, output_tokens=40),
            response=ResponseMetrics(response_time_ms=150.0, tokens_per_second=20.0),
            cost=CostMetrics(input_cost=0.010, output_cost=0.004),
            success=True,
        ),
        ExampleMetrics(
            tokens=TokenMetrics(input_tokens=120, output_tokens=50),
            response=ResponseMetrics(response_time_ms=170.0, tokens_per_second=22.0),
            cost=CostMetrics(input_cost=0.012, output_cost=0.005),
            success=True,
        ),
        ExampleMetrics(
            tokens=TokenMetrics(input_tokens=90, output_tokens=30),
            response=ResponseMetrics(response_time_ms=130.0, tokens_per_second=18.0),
            cost=CostMetrics(input_cost=0.009, output_cost=0.003),
            success=False,
            error="boom",
        ),
        ExampleMetrics(
            tokens=TokenMetrics(input_tokens=60, output_tokens=20),
            response=ResponseMetrics(response_time_ms=90.0, tokens_per_second=15.0),
            cost=CostMetrics(input_cost=0.006, output_cost=0.002),
            success=False,
            error="boom2",
        ),
    ]


def _full_field_tracker() -> MetricsTracker:
    tracker = MetricsTracker()
    tracker.start_tracking()
    for metrics in _full_field_example_metrics():
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


def test_aggregate_metrics_tokens_and_response_time_include_errored_examples():
    """The FIFTH aggregation site (see module docstring): unlike its
    ``input_cost``/``output_cost``/``total_cost`` siblings a few lines below
    it in the same function, ``aggregate_metrics``'s ``input_tokens``/
    ``output_tokens``/``total_tokens``/``response_time_ms``/
    ``tokens_per_second`` lists were still built from ``successful_metrics``
    only. This is the live, user-facing value: ``format_for_backend()``
    reads these straight through and ``LocalEvaluator.evaluate()``
    unconditionally overwrites its own aggregated metrics with them
    (``_merge_comprehensive_metrics``).
    """
    tracker = MetricsTracker()
    tracker.start_tracking()
    tracker.add_example_metrics(
        ExampleMetrics(
            tokens=TokenMetrics(input_tokens=100, output_tokens=40),
            response=ResponseMetrics(response_time_ms=150.0, tokens_per_second=20.0),
            cost=CostMetrics(input_cost=0.010, output_cost=0.004),
            success=True,
        )
    )
    tracker.add_example_metrics(
        ExampleMetrics(
            # Errored AFTER burning real tokens/time downstream.
            tokens=TokenMetrics(input_tokens=90, output_tokens=30),
            response=ResponseMetrics(response_time_ms=130.0, tokens_per_second=18.0),
            cost=CostMetrics(input_cost=0.009, output_cost=0.003),
            success=False,
            error="boom",
        )
    )
    tracker.end_tracking()

    aggregated = tracker.aggregate_metrics()

    # True mean over BOTH examples. Old (buggy): the successful example only.
    assert aggregated["input_tokens"]["mean"] == pytest.approx(95.0)
    assert aggregated["input_tokens"]["mean"] != pytest.approx(100.0)
    assert aggregated["output_tokens"]["mean"] == pytest.approx(35.0)
    assert aggregated["output_tokens"]["mean"] != pytest.approx(40.0)
    assert aggregated["total_tokens"]["mean"] == pytest.approx(130.0)
    assert aggregated["total_tokens"]["mean"] != pytest.approx(140.0)
    assert aggregated["response_time_ms"]["mean"] == pytest.approx(140.0)
    assert aggregated["response_time_ms"]["mean"] != pytest.approx(150.0)
    assert aggregated["tokens_per_second"]["mean"] == pytest.approx(19.0)
    assert aggregated["tokens_per_second"]["mean"] != pytest.approx(20.0)

    # format_for_backend surfaces these as LIVE, user-facing values -- the
    # path an optimizer objective or the portal actually reads.
    formatted = tracker.format_for_backend()
    assert formatted["input_tokens"] == pytest.approx(95.0)
    assert formatted["response_time_ms"] == pytest.approx(140.0)
    assert formatted["tokens_per_second"] == pytest.approx(19.0)


def test_aggregate_metrics_all_failed_trial_still_reports_real_cost_and_tokens():
    """Second half of the fifth-site bug: ``aggregate_metrics`` early-
    returned ``_empty_aggregated_metrics()`` whenever ``successful_metrics``
    was EMPTY -- so a trial where EVERY example errored, but still burned
    real cost/tokens, silently lost its ``cost_per_example_mean`` (and
    every other ``aggregate_metrics`` stat) to 0.0, even though the
    already-#1964-fixed per-example cost lists a few lines further down were
    never reached to compute it. ``format_for_backend()['cost']`` (a direct
    sum that bypasses ``aggregate_metrics`` entirely) already reported the
    real total correctly, so the SAME trial's ``cost`` and
    ``cost_per_example_mean`` keys disagreed with each other.
    """
    tracker = MetricsTracker()
    tracker.start_tracking()
    tracker.add_example_metrics(
        ExampleMetrics(
            tokens=TokenMetrics(input_tokens=100, output_tokens=40),
            response=ResponseMetrics(response_time_ms=150.0),
            cost=CostMetrics(input_cost=0.010, output_cost=0.004),
            success=False,
            error="boom1",
        )
    )
    tracker.add_example_metrics(
        ExampleMetrics(
            tokens=TokenMetrics(input_tokens=90, output_tokens=30),
            response=ResponseMetrics(response_time_ms=130.0),
            cost=CostMetrics(input_cost=0.009, output_cost=0.003),
            success=False,
            error="boom2",
        )
    )
    tracker.end_tracking()

    formatted = tracker.format_for_backend()

    # True total: 0.014 + 0.012 = 0.026; true per-example mean: 0.013.
    # Old (buggy): the all-failed early return zeroed cost_per_example_mean
    # (and input_tokens) to 0.0 even though `cost` (the direct sum) already
    # correctly reported 0.026 for the identical trial.
    assert formatted["cost"] == pytest.approx(0.026)
    assert formatted["cost_per_example_mean"] == pytest.approx(0.013)
    assert formatted["cost_per_example_mean"] != pytest.approx(0.0)
    assert formatted["input_tokens"] == pytest.approx(95.0)
    assert formatted["input_tokens"] != pytest.approx(0.0)
    assert formatted["response_time_ms"] == pytest.approx(140.0)
    assert formatted["response_time_ms"] != pytest.approx(0.0)


def test_format_as_summary_stats_includes_cost_from_errored_examples():
    """The 4th aggregation site found during pre-merge review: the
    privacy-mode/hybrid-submission summary_stats payload must not silently
    drop cost/tokens/response-time from an example that errored AFTER the
    provider call already burned real, billable measurements.

    Uses ``_full_field_example_metrics`` (non-zero, distinct values for
    every field across all 4 rows) rather than the zero-token
    ``_mixed_tracker`` fixture used elsewhere in this file: with all-zero
    tokens, a regression that restored the success-only filter on
    ``input_tokens``/``output_tokens``/``response_time_ms``/
    ``tokens_per_second`` would still report the SAME (zero) mean either
    way -- only the count would move, and the original version of this test
    didn't even assert that for most fields. Every one of the seven changed
    fields is asserted on both count and mean here so each is independently
    observable.
    """
    tracker = _full_field_tracker()

    summary_stats = tracker.format_as_summary_stats()
    metrics = summary_stats["metrics"]

    # See _full_field_example_metrics for the per-row values this derives
    # from. "Old (buggy)" below is the success-only mean over rows 0-1 only.
    expectations = {
        "input_cost": (4, 0.00925, 0.011),
        "output_cost": (4, 0.0035, 0.0045),
        "total_cost": (4, 0.01275, 0.0155),
        "input_tokens": (4, 92.5, 110.0),
        "output_tokens": (4, 35.0, 45.0),
        "total_tokens": (4, 127.5, 155.0),
        "response_time_ms": (4, 135.0, 160.0),
        "tokens_per_second": (4, 18.75, 21.0),
    }
    for field, (count, fixed_mean, buggy_mean) in expectations.items():
        stats = metrics[field]
        assert stats["count"] == count, field
        assert stats["mean"] == pytest.approx(fixed_mean), field
        assert stats["mean"] != pytest.approx(buggy_mean), field
