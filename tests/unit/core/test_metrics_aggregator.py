from __future__ import annotations

from datetime import UTC, datetime

import pytest

from traigent.api.types import TrialResult, TrialStatus
from traigent.core.metric_registry import MetricRegistry, MetricSpec
from traigent.core.metrics_aggregator import (
    AggregatedMetrics,
    aggregate_metrics,
    build_safeguards_telemetry,
)


class DummyEvalResult:
    def __init__(
        self, aggregated_metrics: dict[str, float], total_examples: int
    ) -> None:
        self.aggregated_metrics = aggregated_metrics
        self.total_examples = total_examples
        self.metrics = {"accuracy": 0.8}
        self.outputs = []
        self.success_rate = 0.8
        self.has_errors = False


def make_trial(
    *,
    trial_id: str,
    metrics: dict[str, float],
    status: TrialStatus = TrialStatus.COMPLETED,
    metadata: dict | None = None,
    duration: float = 1.0,
) -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        config={},
        metrics=metrics,
        status=status,
        duration=duration,
        timestamp=datetime.now(UTC),
        metadata=metadata or {},
    )


def test_aggregate_metrics_includes_mandatory_values() -> None:
    registry = MetricRegistry.default()

    eval_result = DummyEvalResult(
        aggregated_metrics={
            "total_cost": {"mean": 0.01},
            "total_tokens": {"mean": 50},
        },
        total_examples=10,
    )
    trial = make_trial(
        trial_id="t1",
        metrics={"examples_attempted": 10},
        metadata={"evaluation_result": eval_result},
        duration=2.5,
    )

    aggregated = aggregate_metrics([trial], registry)

    assert isinstance(aggregated, AggregatedMetrics)
    assert pytest.approx(aggregated.total_cost) == pytest.approx(0.1)
    assert aggregated.total_tokens == 500
    assert aggregated.processed_metrics["total_cost"] == pytest.approx(0.1)
    assert aggregated.processed_metrics["total_tokens"] == 500
    assert aggregated.processed_metrics["total_duration"] == pytest.approx(2.5)
    assert aggregated.processed_metrics["examples_attempted_total"] == 10


def test_aggregate_metrics_respects_registry_strategies() -> None:
    registry = MetricRegistry.default()
    registry.register(MetricSpec(name="precision", aggregator="mean"))
    registry.register(MetricSpec(name="latency", aggregator="last"))

    trial_a = make_trial(
        trial_id="a",
        metrics={
            "precision": 0.6,
            "latency": 100,
        },
    )
    trial_b = make_trial(
        trial_id="b",
        metrics={
            "precision": 0.9,
            "latency": 120,
        },
    )

    aggregated = aggregate_metrics([trial_a, trial_b], registry)

    assert aggregated.processed_metrics["precision"] == pytest.approx(0.75)
    assert aggregated.processed_metrics["latency"] == 120


def test_mandatory_metrics_prefer_trial_values() -> None:
    registry = MetricRegistry.default()

    eval_result = DummyEvalResult(
        aggregated_metrics={
            "total_cost": {"mean": 0.01},
            "total_tokens": {"mean": 50},
        },
        total_examples=10,
    )
    trial = make_trial(
        trial_id="prefer-trial-metrics",
        metrics={"total_cost": 2.0, "total_tokens": 20},
        metadata={"evaluation_result": eval_result},
    )

    aggregated = aggregate_metrics([trial], registry)

    assert aggregated.total_cost == pytest.approx(2.0)
    assert aggregated.total_tokens == 20
    assert aggregated.processed_metrics["total_cost"] == pytest.approx(2.0)
    assert aggregated.processed_metrics["total_tokens"] == 20


def test_build_safeguards_telemetry_defaults_to_allow_repeats() -> None:
    telemetry = build_safeguards_telemetry(
        trials_prevented=2,
        configs_deduplicated=3,
        examples_capped=1,
        cached_results_reused=0,
        ci_blocks=0,
        cache_policy_used=None,
    )

    assert telemetry == {
        "trials_prevented": 2,
        "configs_deduplicated": 3,
        "examples_capped": 1,
        "cached_results_reused": 0,
        "ci_blocks": 0,
        "cache_policy": "allow_repeats",
    }
