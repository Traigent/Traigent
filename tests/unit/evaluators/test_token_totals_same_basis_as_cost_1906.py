"""Regression for #1906: token fields must be per-trial TOTALS, like ``cost``."""

from __future__ import annotations

import pytest

from traigent.evaluators.metrics_tracker import (
    CostMetrics,
    ExampleMetrics,
    MetricsTracker,
    ResponseMetrics,
    TokenMetrics,
)


def _tracker(rows: list[tuple[int, int, bool]]) -> MetricsTracker:
    tracker = MetricsTracker()
    tracker.start_tracking()
    for inp, out, measured in rows:
        tracker.add_example_metrics(
            ExampleMetrics(
                tokens=TokenMetrics(input_tokens=inp, output_tokens=out),
                response=ResponseMetrics(response_time_ms=10),
                cost=CostMetrics(input_cost=0.01, output_cost=0.0),
                success=True,
                measured=measured,
            )
        )
    tracker.end_tracking()
    return tracker


def test_tokens_are_totals_not_means() -> None:
    formatted = _tracker([(100, 50, True)] * 4).format_for_backend()

    assert formatted["input_tokens"] == pytest.approx(400)
    assert formatted["output_tokens"] == pytest.approx(200)
    assert formatted["total_tokens"] == pytest.approx(600)
    # Same basis as cost: total over the same examples.
    assert formatted["cost"] == pytest.approx(0.04)


def test_token_totals_exclude_unmeasured_rows() -> None:
    formatted = _tracker(
        [(100, 50, True), (100, 50, True), (0, 0, False)]
    ).format_for_backend()

    assert formatted["total_tokens"] == pytest.approx(300)
