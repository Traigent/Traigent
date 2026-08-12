"""Regression tests for the third gpt-5.6-sol DO_NOT_MERGE round on PR #2160
(BLOCKER 1, high): the #1964/#1965/#2111 fix round replaced one wrong number
with a different wrong number.

The original Traigent#1964 bug excluded ERRORED-BUT-MEASURED examples (a
provider call that burned real tokens/cost before something downstream
failed) from cost/token/response-time aggregation -- silently UNDER-counting
real spend. The fix (and its sibling sweeps in #1965/#2111 and the prior two
rounds of this PR) corrected that by aggregating over EVERY tracked
``ExampleMetrics``, with no measurement predicate at all. That over-corrects
in the opposite direction: an example whose provider call raised BEFORE
producing any output at all (``LocalEvaluator._extract_llm_metrics_for_
output``'s ``output is None`` guard) never had anything extracted -- its
zeros are the ABSENCE of a measurement, not a measured zero. Blanket-including
those rows drags the reported MEAN down for every trial with any
never-measured example, because zero is not neutral in a mean. This is the
identical defect class as the original #1964 bug, in the opposite direction.

The fix: ``ExampleMetrics`` gains a ``measured: bool = True`` field.
``LocalEvaluator._extract_llm_metrics_for_output`` sets it ``False`` on its
``output is None`` early return (the sole genuinely-no-measurement site in
the local lane); every other ``ExampleMetrics`` construction site (tests,
response handlers, the errored-but-measured #1964 fixtures) keeps the
default ``True`` and is therefore UNAFFECTED by this change --
``measured=True`` is exactly the #1964 scenario ("errored but still has a
real cost/token/response-time value") and must keep aggregating.
``MetricsTracker.aggregate_metrics``/``format_for_backend``/
``format_as_summary_stats`` and ``BaseEvaluator._compute_cost`` now exclude
``measured=False`` rows from MEAN/median/std cost, token, and response-time
statistics. Sums (the per-trial TOTAL ``cost`` key) are numerically
unaffected either way -- a zero contributes nothing to a sum -- but are
filtered too, for consistency.

``BaseEvaluator._compute_latency`` is intentionally NOT switched to the same
flag: it reads a DIFFERENT data structure (``ExampleResult.execution_time``,
wall-clock function-call time, set regardless of success) where a magnitude
check (``execution_time > 0``) is already the correct "was this measured"
signal -- see the reconciliation note on ``_compute_latency`` in base.py.
"""

from __future__ import annotations

from typing import Any

import pytest

from traigent.evaluators.base import BaseEvaluator, Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator
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


def _measured_and_unmeasured_example_metrics() -> list[ExampleMetrics]:
    """1 measured row (real cost/tokens/response-time) + 1 unmeasured row
    whose zeros are the ABSENCE of a measurement (the ``output is None``
    shape). True mean over the MEASURED row alone: cost=0.02, input_tokens=
    100, response_time_ms=200.0. The buggy "average every row" behaviour
    would halve every one of these (mean over 2 rows, one of them zero).
    """
    return [
        ExampleMetrics(
            tokens=TokenMetrics(input_tokens=100, output_tokens=50),
            response=ResponseMetrics(response_time_ms=200.0),
            cost=CostMetrics(input_cost=0.02),
            success=True,
        ),
        ExampleMetrics(
            # The `output is None` shape: the call raised before producing
            # ANY output, so nothing was ever extracted. Zeros here are the
            # ABSENCE of a measurement.
            success=False,
            error="boom - no output at all",
            measured=False,
        ),
    ]


def _measured_and_unmeasured_tracker() -> MetricsTracker:
    tracker = MetricsTracker()
    tracker.start_tracking()
    for metrics in _measured_and_unmeasured_example_metrics():
        tracker.add_example_metrics(metrics)
    tracker.end_tracking()
    return tracker


# ---------------------------------------------------------------------------
# The exact gap sol named: mix a measured row with an unmeasured zero-default
# row and assert the mean reflects only the measured one.
# ---------------------------------------------------------------------------


def test_aggregate_metrics_excludes_unmeasured_zero_default_from_mean():
    aggregated = _measured_and_unmeasured_tracker().aggregate_metrics()

    # True mean over the ONE measured row. Buggy (blanket-average-in-zeros):
    # halved, because the unmeasured row's zero is averaged in as a real
    # zero over a denominator of 2.
    assert aggregated["total_cost"]["mean"] == pytest.approx(0.02)
    assert aggregated["total_cost"]["mean"] != pytest.approx(0.01)
    assert aggregated["input_tokens"]["mean"] == pytest.approx(100.0)
    assert aggregated["input_tokens"]["mean"] != pytest.approx(50.0)
    assert aggregated["output_tokens"]["mean"] == pytest.approx(50.0)
    assert aggregated["output_tokens"]["mean"] != pytest.approx(25.0)
    assert aggregated["response_time_ms"]["mean"] == pytest.approx(200.0)
    assert aggregated["response_time_ms"]["mean"] != pytest.approx(100.0)

    # total_examples still counts BOTH rows (attempt count is unaffected);
    # only the measurement-derived means exclude the unmeasured row.
    assert aggregated["total_examples"] == 2
    assert aggregated["successful_examples"] == 1


def test_aggregate_metrics_excludes_unmeasured_response_time_even_if_nonzero():
    """An unmeasured row can still pick up a NONZERO ``response_time_ms``
    from the wall-clock overwrite ``LocalEvaluator._update_example_metric_
    from_result`` applies in detailed mode -- that method sets
    ``response_time_ms`` from ``ExampleResult.execution_time`` (real
    wall-clock time to fail) REGARDLESS of whether the call ever produced
    output. ``measured`` is a FLAG, not a magnitude check, so this
    nonzero-but-still-unmeasured value must still be excluded from the
    mean -- otherwise a fast-failing example's harness-level "time to raise"
    would silently substitute for the (nonexistent) provider-response
    latency the field is meant to represent.
    """
    tracker = MetricsTracker()
    tracker.start_tracking()
    tracker.add_example_metrics(
        ExampleMetrics(response=ResponseMetrics(response_time_ms=200.0), success=True)
    )
    tracker.add_example_metrics(
        ExampleMetrics(
            response=ResponseMetrics(response_time_ms=5.0),  # wall-clock time-to-fail
            success=False,
            error="boom",
            measured=False,
        )
    )
    tracker.end_tracking()

    aggregated = tracker.aggregate_metrics()

    assert aggregated["response_time_ms"]["mean"] == pytest.approx(200.0)
    assert aggregated["response_time_ms"]["mean"] != pytest.approx(102.5)


def test_format_for_backend_cost_per_example_mean_excludes_unmeasured():
    formatted = _measured_and_unmeasured_tracker().format_for_backend()

    assert formatted["cost_per_example_mean"] == pytest.approx(0.02)
    assert formatted["cost_per_example_mean"] != pytest.approx(0.01)
    assert formatted["input_tokens"] == pytest.approx(100.0)
    assert formatted["input_tokens"] != pytest.approx(50.0)
    assert formatted["response_time_ms"] == pytest.approx(200.0)
    assert formatted["response_time_ms"] != pytest.approx(100.0)


def test_format_for_backend_cost_sum_unaffected_by_unmeasured():
    """The per-trial TOTAL ``cost`` (a SUM, not a mean) is numerically
    identical whether or not the unmeasured zero-default row is included --
    a zero contributes nothing to a sum. This is the expected, harmless
    case; the bug this file fixes is specifically about MEANS.
    """
    formatted = _measured_and_unmeasured_tracker().format_for_backend()

    assert formatted["cost"] == pytest.approx(0.02)


def test_format_as_summary_stats_excludes_unmeasured_from_mean_and_count():
    summary_stats = _measured_and_unmeasured_tracker().format_as_summary_stats()
    metrics = summary_stats["metrics"]

    # count == 1: only the measured row contributes to token/cost/
    # response-time stats. Buggy: count == 2, mean halved.
    assert metrics["total_cost"]["count"] == 1
    assert metrics["total_cost"]["mean"] == pytest.approx(0.02)
    assert metrics["total_cost"]["mean"] != pytest.approx(0.01)
    assert metrics["input_tokens"]["count"] == 1
    assert metrics["input_tokens"]["mean"] == pytest.approx(100.0)
    assert metrics["response_time_ms"]["count"] == 1
    assert metrics["response_time_ms"]["mean"] == pytest.approx(200.0)

    # Accuracy is UNAFFECTED: #1963's denominator fix is orthogonal to
    # whether the row left a cost/token measurement behind. Both rows still
    # count (1 success, 1 failure -> accuracy mean 0.5, count 2).
    assert metrics["accuracy"]["count"] == 2
    assert metrics["accuracy"]["mean"] == pytest.approx(0.5)


def test_format_as_summary_stats_all_unmeasured_still_emits_builtin_keys():
    """Edge case: every tracked example is unmeasured (e.g. every call in
    the trial raised before producing output). The built-in cost/token/
    response-time keys must still be present in ``summary_stats["metrics"]``
    (with a zero-count describe structure) rather than silently vanishing --
    downstream consumers rely on their presence.
    """
    tracker = MetricsTracker()
    tracker.start_tracking()
    tracker.add_example_metrics(
        ExampleMetrics(success=False, error="boom1", measured=False)
    )
    tracker.add_example_metrics(
        ExampleMetrics(success=False, error="boom2", measured=False)
    )
    tracker.end_tracking()

    summary_stats = tracker.format_as_summary_stats()
    metrics = summary_stats["metrics"]

    for key in (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "response_time_ms",
        "input_cost",
        "output_cost",
        "total_cost",
    ):
        assert key in metrics, key
        assert metrics[key]["count"] == 0, key
        assert metrics[key]["mean"] == 0.0, key

    # Accuracy still reflects both (unsuccessful) attempts.
    assert metrics["accuracy"]["count"] == 2
    assert metrics["accuracy"]["mean"] == pytest.approx(0.0)


def test_compute_cost_excludes_unmeasured_examples():
    evaluator = _DummyEvaluator()
    example_metrics = _measured_and_unmeasured_example_metrics()
    errors = [None, "boom - no output at all"]
    outputs = [None, None]
    expected = [None, None]

    cost = evaluator._compute_cost(
        outputs, expected, errors, example_metrics=example_metrics
    )

    assert cost == pytest.approx(0.02)
    assert cost != pytest.approx(0.01)


def test_compute_cost_still_includes_measured_but_errored_examples():
    """Regression guard for Traigent#1964 itself: an errored example that
    WAS measured (the provider call succeeded and burned real cost before a
    downstream step failed) must still count -- this file's fix must not
    reintroduce the original #1964 under-count while fixing the opposite
    over-count.
    """
    evaluator = _DummyEvaluator()
    example_metrics = [
        ExampleMetrics(cost=CostMetrics(input_cost=0.01), success=True),
        ExampleMetrics(
            cost=CostMetrics(input_cost=0.03),
            success=False,
            error="downstream failure after a real response",
            # measured left at its True default: a real response was
            # extracted before the downstream failure.
        ),
    ]
    errors = [None, "downstream failure after a real response"]
    outputs = [None, None]
    expected = [None, None]

    cost = evaluator._compute_cost(
        outputs, expected, errors, example_metrics=example_metrics
    )

    # Both rows count: (0.01 + 0.03) / 2 = 0.02, not 0.01 (which would be
    # the pre-#1964 "successful only" bug reappearing).
    assert cost == pytest.approx(0.02)
    assert cost != pytest.approx(0.01)


@pytest.mark.asyncio
async def test_local_evaluator_evaluate_excludes_unmeasured_example_from_cost_and_token_means():
    """End-to-end through the real ``evaluate()`` path and the LIVE call
    site (``LocalEvaluator._extract_llm_metrics_for_output``'s ``output is
    None`` guard -> ``MetricsTracker.aggregate_metrics``/
    ``format_for_backend``, which unconditionally overwrites
    ``LocalEvaluator.evaluate()``'s own aggregated metrics via
    ``_merge_comprehensive_metrics``).

    Row 0's agent call raises BEFORE returning anything -- the SDK's own
    ``_execute_function`` always nulls ``actual_output`` when it records an
    error, so ``_extract_llm_metrics_for_output`` takes its ``output is
    None`` branch and marks the row ``measured=False``. Row 1 returns a
    dict carrying ``__traigent_meta__`` (the real, public way for a user's
    function to report cost/tokens directly), so it is genuinely measured.

    NEGATIVE CONTROL: reverting the `measured`-filter in
    `MetricsTracker.aggregate_metrics`/`format_for_backend` back to
    unconditional iteration over `self.example_metrics` (the pre-fix
    behaviour) turns this RED (cost_per_example_mean == 0.01, input_tokens
    == 50.0). See the worker report for the executed negative control.
    """

    async def agent(input_data: dict) -> Any:
        if input_data["idx"] == 0:
            raise ValueError("boom - no output at all")
        return {
            "text": "match",
            "__traigent_meta__": {
                "usage": {"input_tokens": 100, "output_tokens": 50},
                "total_cost": 0.02,
            },
        }

    evaluator = LocalEvaluator(metrics=["accuracy", "cost"], detailed=True)
    dataset = Dataset(
        [EvaluationExample({"idx": i}, "match") for i in range(2)],
        name="measured_vs_unmeasured_e2e_2160",
    )

    result = await evaluator.evaluate(agent, {}, dataset)

    # Buggy (blanket-average-in-zeros): mean = (0.0 + 0.02) / 2 = 0.01 and
    # input_tokens mean = (0 + 100) / 2 = 50.0. Fixed: only the measured row
    # (row 1) counts.
    assert result.metrics["cost_per_example_mean"] == pytest.approx(0.02)
    assert result.metrics["cost_per_example_mean"] != pytest.approx(0.01)
    assert result.metrics["input_tokens"] == pytest.approx(100.0)
    assert result.metrics["input_tokens"] != pytest.approx(50.0)

    # The per-trial TOTAL `cost` (a sum) is unaffected either way -- both
    # report the same real spend, 0.02.
    assert result.metrics["cost"] == pytest.approx(0.02)
