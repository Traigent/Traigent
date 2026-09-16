"""Regression tests for Traigent#2263.

``LocalEvaluator._estimate_string_tokens`` fills ``input_tokens``/
``output_tokens``/``total_tokens`` from character counts (1 token per 4
characters, minimum 1) whenever usage extraction captured nothing AND the
optimized function returned a plain string. That fallback is legitimate --
privacy mode needs SOME length-derived number when it cannot retain raw
prompts -- but the counts it produces are a guess, not a measurement, and
they used to land on ``ExampleMetrics.tokens`` indistinguishable from a real
captured count. A cost objective or the trial summary table then could not
tell "no LLM call was captured" from "a call that used N real tokens".

The fix: ``TokenMetrics`` gains an ``estimated: bool = False`` field, set by
``_estimate_string_tokens``, threaded through ``MetricsTracker.
format_for_backend`` as a trial-level ``tokens_estimated`` flag (1.0/0.0,
same wire-safe convention as ``CostMetrics.unpriced``/``cost_unpriced``,
#1597/#1741), and defended at the cost-pricing entry point
(``_calculate_cost_for_metrics``) so a future reordering cannot start pricing
fabricated tokens as if they were real usage.

Each test below is red on the pre-fix code: reverting the ``estimated``
field and its wiring removes ``tokens_estimated`` from ``formatted``/the flag
never gets set/the pricing guard never fires.
"""

from __future__ import annotations

from typing import Any

import pytest

from traigent.cloud.dtos import MeasuresDict
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator
from traigent.evaluators.metrics_tracker import (
    RESERVED_METRIC_KEYS,
    CostMetrics,
    ExampleMetrics,
    MetricsTracker,
    ResponseMetrics,
    TokenMetrics,
    _calculate_cost_for_metrics,
)


class TestEstimateStringTokensSetsFlag:
    """``LocalEvaluator._estimate_string_tokens`` itself."""

    def test_estimated_flag_is_set_and_totals_still_correct(self):
        evaluator = LocalEvaluator(metrics=["accuracy"])
        metric = ExampleMetrics()

        evaluator._estimate_string_tokens(metric, "A", {"idx": 0}, config={})

        assert metric.tokens.estimated is True
        assert metric.tokens.output_tokens == 1  # max(1, len("A") // 4)
        assert metric.tokens.total_tokens == (
            metric.tokens.input_tokens + metric.tokens.output_tokens
        )

    def test_a_real_captured_response_never_sets_the_flag(self):
        """Sanity check on the negative case: a normal ``ExampleMetrics``
        built from a real handler-extracted response defaults to
        ``estimated=False`` and is unaffected by this change."""
        metric = ExampleMetrics(tokens=TokenMetrics(input_tokens=50, output_tokens=10))

        assert metric.tokens.estimated is False


class TestCostPricingExcludesEstimatedTokens:
    """Defensive guard in ``_calculate_cost_for_metrics`` (Traigent#2263)."""

    def test_estimated_tokens_are_never_priced(self):
        metrics = ExampleMetrics(
            tokens=TokenMetrics(input_tokens=1000, output_tokens=500, estimated=True)
        )

        _calculate_cost_for_metrics(metrics, "gpt-4o-mini", None, None)

        # Without the guard, 1000/500 real-shaped tokens against a known
        # model would price to a non-zero cost -- this pins that estimated
        # tokens stay $0 regardless of model.
        assert metrics.cost.total_cost == 0.0

    def test_non_estimated_tokens_still_price_normally(self):
        """Negative control: the guard must not swallow real pricing."""
        metrics = ExampleMetrics(
            tokens=TokenMetrics(input_tokens=1000, output_tokens=500, estimated=False)
        )

        _calculate_cost_for_metrics(metrics, "gpt-4o-mini", None, None)

        assert metrics.cost.total_cost > 0.0


class TestFormatForBackendTokensEstimated:
    """LocalEvaluator lane: ``MetricsTracker.format_for_backend``."""

    def _measured_example(self) -> ExampleMetrics:
        return ExampleMetrics(
            tokens=TokenMetrics(input_tokens=100, output_tokens=50),
            response=ResponseMetrics(response_time_ms=1000),
            cost=CostMetrics(input_cost=0.001, output_cost=0.002),
            success=True,
        )

    def _estimated_example(self) -> ExampleMetrics:
        # Same shape ``_estimate_string_tokens`` produces: non-zero tokens,
        # zero cost, flagged estimated.
        return ExampleMetrics(
            tokens=TokenMetrics(input_tokens=2, output_tokens=1, estimated=True),
            response=ResponseMetrics(response_time_ms=5),
            cost=CostMetrics(input_cost=0.0, output_cost=0.0),
            success=True,
        )

    def test_all_measured_examples_flag_false(self):
        tracker = MetricsTracker()
        tracker.start_tracking()
        for _ in range(3):
            tracker.add_example_metrics(self._measured_example())
        tracker.end_tracking()

        formatted = tracker.format_for_backend()

        assert "tokens_estimated" in formatted
        assert formatted["tokens_estimated"] == 0.0
        assert not isinstance(formatted["tokens_estimated"], bool)

    def test_one_estimated_example_flags_trial_true(self):
        tracker = MetricsTracker()
        tracker.start_tracking()
        tracker.add_example_metrics(self._measured_example())
        tracker.add_example_metrics(self._estimated_example())
        tracker.end_tracking()

        formatted = tracker.format_for_backend()

        assert formatted["tokens_estimated"] == 1.0
        assert not isinstance(formatted["tokens_estimated"], bool)
        # The flag is additive: the mean token/cost fields still include the
        # estimated row's numbers verbatim, they are just now labelled.
        assert formatted["cost"] == pytest.approx(0.003)

    def test_tokens_estimated_is_reserved_and_never_dropped(self):
        assert "tokens_estimated" in RESERVED_METRIC_KEYS

    def test_flag_is_wire_valid_in_a_measures_dict(self):
        """Must survive the backend's numeric-only wire contract -- a bare
        Python ``bool`` for ``tokens_estimated`` would raise ``TypeError``
        the moment a trial's metrics are submitted (see the identical
        ``cost_unpriced`` guarantee this mirrors, #1741)."""
        tracker = MetricsTracker()
        tracker.start_tracking()
        tracker.add_example_metrics(self._estimated_example())
        tracker.end_tracking()

        formatted = tracker.format_for_backend()

        measures = MeasuresDict({"tokens_estimated": formatted["tokens_estimated"]})
        assert measures["tokens_estimated"] == 1.0


@pytest.mark.asyncio
async def test_local_evaluator_evaluate_flags_string_length_fallback_end_to_end():
    """End-to-end through the real ``evaluate()`` path: an agent that returns
    a plain string with no captured LLM usage, and a ``model`` present in the
    config (the exact scenario in the issue). Before the fix, ``result.
    metrics`` had non-zero ``input_tokens``/``output_tokens``/``total_tokens``
    with no way to tell they were fabricated. After the fix, the same means
    are reported alongside ``tokens_estimated == 1.0``, and cost stays $0
    (never fabricated-priced) rather than pricing the guessed tokens.
    """

    async def agent(_input: dict[str, Any]) -> str:
        return "A"

    evaluator = LocalEvaluator(metrics=["accuracy", "cost"], detailed=True)
    dataset = Dataset(
        [EvaluationExample({"idx": 0}, "A")], name="estimated_tokens_e2e_2263"
    )

    result = await evaluator.evaluate(agent, {"model": "gpt-4o-mini"}, dataset)

    assert result.metrics["tokens_estimated"] == 1.0
    assert result.metrics["total_tokens"] > 0.0
    # The estimate is never priced: cost stays exactly $0, not a guessed spend.
    assert result.metrics["cost"] == 0.0
