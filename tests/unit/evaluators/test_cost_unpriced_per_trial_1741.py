"""Regression tests for #1741 (follow-up to #1597/#1407).

``CostMetrics.unpriced`` (set when non-strict cost accounting could not
price a model and recorded $0 despite real token usage, #1597) previously
stopped at the per-example ``ExampleMetrics.cost`` object and the
run-level warning/metadata. Per-trial consumers -- the trial summary
table, ``result.trials[i]``, Pareto/cost-objective logic -- still saw a
bare $0 for an unpriceable model, indistinguishable from a genuinely free
one.

These tests pin that a trial-level ``cost_unpriced`` flag is now threaded
through BOTH evaluator lanes that build trial metrics:

* ``MetricsTracker.format_for_backend`` (the ``LocalEvaluator`` lane).
* ``BaseEvaluator._build_llm_metrics_dict`` /
  ``_add_llm_metrics_to_example`` / ``_aggregate_llm_metrics`` (the
  ``SimpleScoringEvaluator`` lane).

Each test is red on the pre-fix code: reverting the ``cost_unpriced``
additions removes the key from ``formatted``/``example_metrics``/
``aggregated`` entirely, which fails the ``in``/``== 1.0`` assertions
below.

The flag is asserted as ``1.0``/``0.0`` (float), never Python ``bool``:
the wire-format ``MeasuresDict`` (``traigent.cloud.dtos.MeasuresDict``)
explicitly rejects ``bool`` measures for JSON Schema parity, so a naive
``True``/``False`` implementation would pass every in-process assertion
here yet raise ``TypeError`` the moment a trial's metrics are submitted
(pinned by ``test_flag_is_wire_valid_in_a_measures_dict`` below, which
caught exactly that defect during development of this fix).
"""

from __future__ import annotations

from traigent.cloud.dtos import MeasuresDict
from traigent.evaluators.base import SimpleScoringEvaluator
from traigent.evaluators.metrics_tracker import (
    RESERVED_METRIC_KEYS,
    CostMetrics,
    ExampleMetrics,
    MetricsTracker,
    ResponseMetrics,
    TokenMetrics,
)


class TestFormatForBackendCostUnpriced:
    """LocalEvaluator lane: ``MetricsTracker.format_for_backend``."""

    def _priced_example(self) -> ExampleMetrics:
        return ExampleMetrics(
            tokens=TokenMetrics(input_tokens=100, output_tokens=50),
            response=ResponseMetrics(response_time_ms=1000),
            cost=CostMetrics(input_cost=0.001, output_cost=0.002),
            success=True,
        )

    def _unpriced_example(self) -> ExampleMetrics:
        # Real token usage, $0 cost, flagged unpriced -- same shape
        # ``_calculate_cost_for_metrics`` produces for an unknown model
        # under non-strict cost accounting (#1597).
        return ExampleMetrics(
            tokens=TokenMetrics(input_tokens=120, output_tokens=40),
            response=ResponseMetrics(response_time_ms=900),
            cost=CostMetrics(input_cost=0.0, output_cost=0.0, unpriced=True),
            success=True,
        )

    def test_all_priced_examples_flag_false(self):
        tracker = MetricsTracker()
        tracker.start_tracking()
        for _ in range(3):
            tracker.add_example_metrics(self._priced_example())
        tracker.end_tracking()

        formatted = tracker.format_for_backend()

        assert "cost_unpriced" in formatted
        assert formatted["cost_unpriced"] == 0.0
        assert not isinstance(formatted["cost_unpriced"], bool)

    def test_one_unpriced_example_flags_trial_true(self):
        tracker = MetricsTracker()
        tracker.start_tracking()
        tracker.add_example_metrics(self._priced_example())
        tracker.add_example_metrics(self._unpriced_example())
        tracker.add_example_metrics(self._priced_example())
        tracker.end_tracking()

        formatted = tracker.format_for_backend()

        assert formatted["cost_unpriced"] == 1.0
        assert not isinstance(formatted["cost_unpriced"], bool)
        # The trial's cost total is still a real (zero-inclusive) sum, not
        # replaced or hidden by the flag -- the flag is additive metadata.
        assert formatted["cost"] == 0.001 + 0.002 + 0.0 + 0.001 + 0.002

    def test_cost_unpriced_is_reserved_and_never_dropped(self):
        assert "cost_unpriced" in RESERVED_METRIC_KEYS

    def test_flag_is_wire_valid_in_a_measures_dict(self):
        """The flag must survive the backend's numeric-only wire contract.

        ``MeasuresDict`` rejects Python ``bool`` values even though
        ``bool`` is an ``int`` subclass ("bool is not treated as numeric
        for contract parity with JSON Schema"). A trial whose metrics
        include a bare ``True``/``False`` for ``cost_unpriced`` would
        raise ``TypeError`` here instead of submitting.
        """
        tracker = MetricsTracker()
        tracker.start_tracking()
        tracker.add_example_metrics(self._unpriced_example())
        tracker.end_tracking()

        formatted = tracker.format_for_backend()

        # Must not raise.
        measures = MeasuresDict({"cost_unpriced": formatted["cost_unpriced"]})
        assert measures["cost_unpriced"] == 1.0


class TestSimpleScoringEvaluatorCostUnpriced:
    """SimpleScoringEvaluator lane: ``evaluators/base.py`` helpers."""

    def _evaluator(self) -> SimpleScoringEvaluator:
        return SimpleScoringEvaluator(metrics=["score"])

    def test_build_llm_metrics_dict_carries_unpriced_flag(self):
        evaluator = self._evaluator()
        metrics_obj = ExampleMetrics(
            tokens=TokenMetrics(input_tokens=50, output_tokens=10),
            cost=CostMetrics(input_cost=0.0, output_cost=0.0, unpriced=True),
        )

        llm_metrics = evaluator._build_llm_metrics_dict(
            metrics_obj, "some/unknown-model"
        )

        assert llm_metrics["cost_unpriced"] == 1.0
        assert not isinstance(llm_metrics["cost_unpriced"], bool)

    def test_build_llm_metrics_dict_defaults_false_when_priced(self):
        evaluator = self._evaluator()
        metrics_obj = ExampleMetrics(
            tokens=TokenMetrics(input_tokens=50, output_tokens=10),
            cost=CostMetrics(input_cost=0.001, output_cost=0.0),
        )

        llm_metrics = evaluator._build_llm_metrics_dict(metrics_obj, "gpt-4o-mini")

        assert llm_metrics["cost_unpriced"] == 0.0
        assert not isinstance(llm_metrics["cost_unpriced"], bool)

    def test_add_llm_metrics_to_example_threads_flag(self):
        evaluator = self._evaluator()
        example_metrics: dict[str, object] = {}
        evaluator._add_llm_metrics_to_example(
            example_metrics, {"cost_unpriced": 1.0, "total_cost": 0.0}
        )

        assert example_metrics["cost_unpriced"] == 1.0
        assert not isinstance(example_metrics["cost_unpriced"], bool)

    def test_aggregate_llm_metrics_true_if_any_example_unpriced(self):
        evaluator = self._evaluator()
        all_metrics = [
            {"total_cost": 0.001, "cost_unpriced": 0.0},
            {"total_cost": 0.0, "cost_unpriced": 1.0},
            {"total_cost": 0.002, "cost_unpriced": 0.0},
        ]

        aggregated = evaluator._aggregate_llm_metrics(all_metrics, example_results=[])

        assert aggregated["cost_unpriced"] == 1.0
        assert not isinstance(aggregated["cost_unpriced"], bool)

    def test_aggregate_llm_metrics_false_when_all_priced(self):
        evaluator = self._evaluator()
        all_metrics = [
            {"total_cost": 0.001, "cost_unpriced": 0.0},
            {"total_cost": 0.002, "cost_unpriced": 0.0},
        ]

        aggregated = evaluator._aggregate_llm_metrics(all_metrics, example_results=[])

        assert aggregated["cost_unpriced"] == 0.0

    def test_aggregate_llm_metrics_false_when_no_examples_carry_the_key(self):
        """Back-compat: entries without ``cost_unpriced`` (e.g. a failed
        example's metrics dict) must not make the trial look unpriced."""
        evaluator = self._evaluator()
        all_metrics: list[dict[str, object] | None] = [{}, None]

        aggregated = evaluator._aggregate_llm_metrics(all_metrics, example_results=[])

        assert aggregated["cost_unpriced"] == 0.0
