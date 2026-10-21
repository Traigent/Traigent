"""Built-in accuracy must not be fabricated from execution success (#2495)."""

from __future__ import annotations

import pytest

from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator
from traigent.evaluators.metrics_tracker import ExampleMetrics, MetricsTracker


def test_tracker_only_success_fallback_is_unchanged() -> None:
    """Legacy tracker-only callers (no dataset context) keep the fallback."""
    tracker = MetricsTracker()
    tracker.add_example_metrics(ExampleMetrics(success=True))
    assert tracker.format_for_backend()["accuracy"] == 1.0


def test_unavailable_accuracy_is_not_reported_as_perfect() -> None:
    tracker = MetricsTracker()
    tracker.add_example_metrics(ExampleMetrics(success=True, accuracy_unavailable=True))
    formatted = tracker.format_for_backend()
    assert formatted.get("accuracy") is None
    assert formatted.get("score") is None


def test_mixed_rows_average_only_measured_accuracy() -> None:
    tracker = MetricsTracker()
    measured = ExampleMetrics(success=True)
    measured.custom_metrics["accuracy"] = 0.0
    tracker.add_example_metrics(measured)
    tracker.add_example_metrics(ExampleMetrics(success=True, accuracy_unavailable=True))
    formatted = tracker.format_for_backend()
    assert formatted["accuracy"] == 0.0


@pytest.mark.asyncio
async def test_local_evaluator_none_output_with_gold_is_not_perfect() -> None:
    dataset = Dataset([EvaluationExample({"q": "a"}, "gold")], name="d")
    evaluator = LocalEvaluator(metrics=["accuracy"])
    result = await evaluator.evaluate(lambda _q: None, {}, dataset)
    assert result.metrics.get("accuracy") != 1.0
    assert result.metrics.get("score") != 1.0
