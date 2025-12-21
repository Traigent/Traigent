"""Latency tracking for Haystack pipeline evaluations.

This module provides latency statistics computation for Haystack pipelines,
computing percentiles (p50, p95, p99) and other latency metrics.

Example usage:
    from traigent.integrations.haystack.latency_tracking import (
        LatencyStats,
        compute_latency_stats,
        extract_latencies_from_results,
    )

    # Compute stats from execution times
    latencies = [0.1, 0.15, 0.2, 0.25, 0.3]  # seconds
    stats = compute_latency_stats(latencies)
    print(f"P50: {stats.p50_ms:.1f}ms, P95: {stats.p95_ms:.1f}ms")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from traigent.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LatencyStats:
    """Latency statistics for a pipeline run.

    All latency values are in milliseconds for consistency with
    standard monitoring and alerting conventions.

    Attributes:
        p50_ms: 50th percentile (median) latency in ms.
        p95_ms: 95th percentile latency in ms.
        p99_ms: 99th percentile latency in ms.
        mean_ms: Mean latency in ms.
        min_ms: Minimum latency in ms.
        max_ms: Maximum latency in ms.
        total_ms: Sum of all latencies in ms.
        count: Number of samples used for calculation.
    """

    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    mean_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    total_ms: float = 0.0
    count: int = 0


def _percentile(sorted_values: list[float], p: float) -> float:
    """Compute percentile using linear interpolation.

    This implements the same algorithm as numpy.percentile with
    interpolation='linear', but without the numpy dependency.

    Args:
        sorted_values: Pre-sorted list of values.
        p: Percentile to compute (0-100).

    Returns:
        The computed percentile value.
    """
    if not sorted_values:
        return 0.0

    n = len(sorted_values)

    if n == 1:
        return sorted_values[0]

    # Calculate the index
    k = (p / 100) * (n - 1)
    f = int(k)
    c = f + 1

    if c >= n:
        return sorted_values[-1]

    # Linear interpolation
    d = k - f
    return sorted_values[f] * (1 - d) + sorted_values[c] * d


def compute_latency_stats(
    latencies_seconds: list[float],
    include_zeros: bool = False,
) -> LatencyStats:
    """Compute latency statistics from a list of latencies.

    Args:
        latencies_seconds: List of latency values in seconds.
        include_zeros: Whether to include zero values in stats.
            Default False - zeros typically indicate errors.

    Returns:
        LatencyStats with percentiles and aggregate metrics.
    """
    # Filter values
    if include_zeros:
        values = [v for v in latencies_seconds if v >= 0]
    else:
        values = [v for v in latencies_seconds if v > 0]

    if not values:
        logger.debug("No valid latency values for statistics computation")
        return LatencyStats()

    # Convert to milliseconds
    values_ms = [v * 1000 for v in values]

    # Sort for percentile computation
    sorted_values = sorted(values_ms)

    # Compute stats
    total = sum(sorted_values)
    mean = total / len(sorted_values)

    return LatencyStats(
        p50_ms=_percentile(sorted_values, 50),
        p95_ms=_percentile(sorted_values, 95),
        p99_ms=_percentile(sorted_values, 99),
        mean_ms=mean,
        min_ms=sorted_values[0],
        max_ms=sorted_values[-1],
        total_ms=total,
        count=len(sorted_values),
    )


def extract_latencies_from_results(
    example_results: list[Any],
    include_failed: bool = True,
) -> list[float]:
    """Extract latency values from a list of example results.

    Args:
        example_results: List of ExampleResult objects from execute_with_config.
        include_failed: Whether to include latencies from failed examples.
            Default True - failed examples still have meaningful latency.

    Returns:
        List of latency values in seconds.
    """
    latencies = []

    for result in example_results:
        # Get execution_time attribute
        execution_time = getattr(result, "execution_time", 0.0)

        # Check if we should include this result
        if include_failed:
            latencies.append(execution_time)
        else:
            success = getattr(result, "success", True)
            if success:
                latencies.append(execution_time)

    return latencies


def get_latency_metrics(stats: LatencyStats) -> dict[str, float]:
    """Convert LatencyStats to metrics dict for EvaluationResult.

    The returned dict uses standard metric names that can be used
    in constraints and optimization objectives.

    Args:
        stats: Latency statistics to convert.

    Returns:
        Dict with latency metrics ready for aggregated_metrics.
    """
    return {
        "latency_p50_ms": stats.p50_ms,
        "latency_p95_ms": stats.p95_ms,
        "latency_p99_ms": stats.p99_ms,
        "latency_mean_ms": stats.mean_ms,
        "latency_min_ms": stats.min_ms,
        "latency_max_ms": stats.max_ms,
        "total_latency_ms": stats.total_ms,
        "latency_count": stats.count,
    }
