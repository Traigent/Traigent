"""Custom metric aggregation is chosen by whole name tokens, not substrings (#3053)."""

from __future__ import annotations

import pytest

from traigent.evaluators.base import (
    SimpleScoringEvaluator,
    quantile_for_metric_name,
)
from traigent.evaluators.local import LocalEvaluator

VALUES = [1.0] * 9 + [100.0]
MEAN = sum(VALUES) / len(VALUES)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("p95_latency", 0.95),
        ("latency_p95", 0.95),
        ("P95", 0.95),
        ("latency-p99-ms", 0.99),
        ("p90_latency", 0.90),
        ("p50_latency", 0.50),
        ("max_latency", 1.0),
        ("latencyP95", 0.95),
        ("maxLatency", 1.0),
        ("latency_max", 1.0),
        ("step95_score", None),
        ("p950_score", None),
        ("maxim_score", None),
        ("accuracy", None),
        ("p9_latency", None),
    ],
)
def test_quantile_for_metric_name_uses_whole_tokens(
    name: str, expected: float | None
) -> None:
    assert quantile_for_metric_name(name) == expected


def _local(name: str) -> tuple[LocalEvaluator, list[object]]:
    from types import SimpleNamespace

    evaluator = LocalEvaluator(
        metrics=["accuracy"], metric_functions={name: lambda *a, **k: 0.0}
    )
    rows = [SimpleNamespace(metrics={name: v}) for v in VALUES]
    return evaluator, rows


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("p95_latency", 55.45),
        ("p99_latency", 1.0 + 0.91 * 99.0),
        ("max_latency", 100.0),
        ("top95_recall", MEAN),
        ("p950_score", MEAN),
    ],
)
def test_local_custom_metric_aggregation(name: str, expected: float) -> None:
    evaluator, rows = _local(name)
    result = evaluator._compute_aggregated_custom_metrics(rows)
    assert result[name] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("max_latency", 100.0),
        ("top95_recall", MEAN),
        ("p950_score", MEAN),
        ("latency_p95", 55.45),
    ],
)
def test_simple_scoring_aggregation(name: str, expected: float) -> None:
    evaluator = SimpleScoringEvaluator(
        scoring_function=lambda output, expected: 1.0, metrics=[name]
    )
    result = evaluator._aggregate_custom_metrics([{name: v} for v in VALUES])
    assert result[name] == pytest.approx(expected)
