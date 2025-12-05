"""Helpers for aggregating trial metrics and telemetry."""

# Traceability: CONC-Layer-Core CONC-Quality-Maintainability FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from traigent.api.types import TrialResult, TrialStatus
from traigent.core.mandatory_metrics import MandatoryMetricsCollector
from traigent.core.metric_registry import MetricRegistry
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class AggregatedMetrics:
    """Aggregated metrics extracted from a collection of trials."""

    processed_metrics: dict[str, Any]
    total_cost: float
    total_tokens: int


def aggregate_metrics(
    trials: Iterable[TrialResult],
    registry: MetricRegistry,
) -> AggregatedMetrics:
    """Aggregate metrics across completed trials.

    Args:
        trials: Iterable of trial results to aggregate.
        registry: Metric registry specifying aggregation strategies.

    Returns:
        AggregatedMetrics containing processed metrics, total cost, and total tokens.
    """

    collector = MandatoryMetricsCollector()
    aggregated_values: dict[str, list[float]] = {}

    for trial in trials:
        collector.accumulate(trial)

        if trial.status != TrialStatus.COMPLETED:
            continue

        for key, value in (trial.metrics or {}).items():
            if value is None or registry.is_mandatory(key):
                continue
            try:
                float_value = float(value)
            except (TypeError, ValueError):
                logger.debug(
                    "Skipping non-numeric metric %s for trial %s", key, trial.trial_id
                )
                continue
            aggregated_values.setdefault(key, []).append(float_value)

    processed_metrics: dict[str, Any] = {}
    for key, values in aggregated_values.items():
        if not values:
            continue
        aggregator = registry.aggregator_for(key)
        if aggregator == "mean":
            processed_metrics[key] = sum(values) / len(values)
        elif aggregator == "sum":
            processed_metrics[key] = sum(values)
        elif aggregator == "last":
            processed_metrics[key] = values[-1]
        else:
            logger.debug(
                "Unknown aggregator '%s' for metric %s; falling back to mean",
                aggregator,
                key,
            )
            processed_metrics[key] = sum(values) / len(values)

    totals = collector.totals()
    processed_metrics.update(totals.as_metrics_dict())

    return AggregatedMetrics(
        processed_metrics=processed_metrics,
        total_cost=totals.total_cost,
        total_tokens=totals.total_tokens,
    )


def build_safeguards_telemetry(
    *,
    trials_prevented: int,
    configs_deduplicated: int,
    examples_capped: int,
    cached_results_reused: int,
    ci_blocks: int,
    cache_policy_used: str | None,
) -> dict[str, Any]:
    """Build the safeguards telemetry payload for optimization results."""

    return {
        "trials_prevented": trials_prevented,
        "configs_deduplicated": configs_deduplicated,
        "examples_capped": examples_capped,
        "cached_results_reused": cached_results_reused,
        "ci_blocks": ci_blocks,
        "cache_policy": cache_policy_used or "allow_repeats",
    }


__all__ = [
    "AggregatedMetrics",
    "aggregate_metrics",
    "build_safeguards_telemetry",
]
