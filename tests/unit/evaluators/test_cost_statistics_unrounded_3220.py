"""Cost statistics are not rounded to 6 decimals (#3220).

``cost_per_example_mean`` is a registered minimize objective; rounding the
per-example mean to 6 decimals turned cheap-model costs into 0.0 or values
several percent off while the exact total sat next to it.
"""

from __future__ import annotations

import pytest

from traigent.evaluators.metrics_tracker import (
    CostMetrics,
    ExampleMetrics,
    MetricsTracker,
    ResponseMetrics,
    TokenMetrics,
)


def _tracker(per_example_cost: float, n: int) -> MetricsTracker:
    tracker = MetricsTracker()
    tracker.start_tracking()
    for _ in range(n):
        tracker.add_example_metrics(
            ExampleMetrics(
                tokens=TokenMetrics(30, 10),
                response=ResponseMetrics(response_time_ms=100.0),
                cost=CostMetrics(
                    input_cost=per_example_cost / 2,
                    output_cost=per_example_cost / 2,
                    total_cost=per_example_cost,
                ),
            )
        )
    tracker.end_tracking()
    return tracker


@pytest.mark.parametrize("per_example_cost", [4e-7, 1.05e-5, 3.3e-9])
def test_cost_mean_matches_total_over_examples(per_example_cost: float) -> None:
    tracker = _tracker(per_example_cost, 10)
    stats = tracker.aggregate_metrics()["total_cost"]
    assert stats["mean"] == pytest.approx(per_example_cost, rel=1e-9)
    assert stats["median"] == pytest.approx(per_example_cost, rel=1e-9)
    payload = tracker.format_for_backend()
    assert payload["cost_per_example_mean"] == pytest.approx(per_example_cost, rel=1e-9)
    assert payload["cost_per_example_mean"] * 10 == pytest.approx(
        payload["cost"], rel=1e-9
    )


def test_small_std_is_not_zeroed() -> None:
    tracker = MetricsTracker()
    stats = tracker.calculate_statistics([1e-7, 3e-7])
    assert stats["mean"] == pytest.approx(2e-7, rel=1e-12)
    assert stats["std"] == pytest.approx(1.4142135623730951e-07, rel=1e-9)
