#!/usr/bin/env python3
"""Example: Latency Tracking for Haystack Pipelines.

This example demonstrates how to track latency percentiles (p50, p95, p99)
for Haystack pipeline evaluations.

Coverage: Epic 4, Story 4.2 (Track Latency Metrics Per Run)
"""

from __future__ import annotations

import asyncio
import random
import time
from unittest.mock import MagicMock


def create_mock_pipeline_with_variable_latency():
    """Create a mock Haystack pipeline with variable response times."""
    pipeline = MagicMock()

    generator = MagicMock()
    generator.__class__.__name__ = "OpenAIGenerator"
    generator.model = "gpt-4o"
    generator.temperature = 0.7

    def get_component(name):
        if name == "generator":
            return generator
        return None

    pipeline.get_component.side_effect = get_component

    # Mock run that simulates variable latency
    call_count = [0]

    def run_with_latency(**kwargs):
        call_count[0] += 1
        # Simulate variable latency (50-300ms)
        latency = 0.05 + random.random() * 0.25  # noqa: S311
        time.sleep(latency)
        return {
            "llm": {
                "replies": [f"Response {call_count[0]}"],
                "meta": [
                    {
                        "model": "gpt-4o",
                        "usage": {
                            "prompt_tokens": 50,
                            "completion_tokens": 30,
                        },
                    }
                ],
            }
        }

    pipeline.run.side_effect = run_with_latency

    return pipeline


def example_latency_stats_computation():
    """Demonstrate computing latency statistics from timing data."""
    print("=" * 60)
    print("Example 1: Latency Statistics Computation")
    print("=" * 60)

    from traigent.integrations.haystack import LatencyStats, compute_latency_stats

    # Simulate latency measurements (in seconds)
    latencies = [
        0.050,  # 50ms
        0.075,  # 75ms
        0.100,  # 100ms
        0.120,  # 120ms
        0.150,  # 150ms
        0.180,  # 180ms
        0.200,  # 200ms
        0.250,  # 250ms
        0.300,  # 300ms
        0.500,  # 500ms (outlier)
    ]

    stats = compute_latency_stats(latencies)

    print(f"\nLatency statistics from {stats.count} samples:")
    print(f"  P50 (median): {stats.p50_ms:.1f}ms")
    print(f"  P95: {stats.p95_ms:.1f}ms")
    print(f"  P99: {stats.p99_ms:.1f}ms")
    print(f"  Mean: {stats.mean_ms:.1f}ms")
    print(f"  Min: {stats.min_ms:.1f}ms")
    print(f"  Max: {stats.max_ms:.1f}ms")
    print(f"  Total: {stats.total_ms:.1f}ms")

    print("\n")


def example_extract_latencies():
    """Demonstrate extracting latencies from execution results."""
    print("=" * 60)
    print("Example 2: Extract Latencies from Results")
    print("=" * 60)

    from traigent.integrations.haystack import (
        compute_latency_stats,
        extract_latencies_from_results,
    )
    from traigent.integrations.haystack.execution import ExampleResult

    # Simulate execution results with timing
    results = [
        ExampleResult(
            example_index=0,
            input={"query": "q1"},
            output={"reply": "r1"},
            success=True,
            execution_time=0.10,  # 100ms
        ),
        ExampleResult(
            example_index=1,
            input={"query": "q2"},
            output={"reply": "r2"},
            success=True,
            execution_time=0.15,  # 150ms
        ),
        ExampleResult(
            example_index=2,
            input={"query": "q3"},
            output=None,
            success=False,  # Failed example
            error="Timeout",
            execution_time=0.50,  # 500ms (timed out)
        ),
        ExampleResult(
            example_index=3,
            input={"query": "q4"},
            output={"reply": "r4"},
            success=True,
            execution_time=0.12,  # 120ms
        ),
    ]

    # Include failed examples in latency stats (default)
    latencies_all = extract_latencies_from_results(results, include_failed=True)
    stats_all = compute_latency_stats(latencies_all)

    print(f"\nIncluding failed examples ({stats_all.count} samples):")
    print(f"  P50: {stats_all.p50_ms:.1f}ms")
    print(f"  P95: {stats_all.p95_ms:.1f}ms")
    print(f"  Max: {stats_all.max_ms:.1f}ms (includes timeout)")

    # Exclude failed examples
    latencies_success = extract_latencies_from_results(results, include_failed=False)
    stats_success = compute_latency_stats(latencies_success)

    print(f"\nExcluding failed examples ({stats_success.count} samples):")
    print(f"  P50: {stats_success.p50_ms:.1f}ms")
    print(f"  P95: {stats_success.p95_ms:.1f}ms")
    print(f"  Max: {stats_success.max_ms:.1f}ms (successful only)")

    print("\n")


def example_latency_metrics():
    """Demonstrate converting stats to metrics dict."""
    print("=" * 60)
    print("Example 3: Latency Metrics for EvaluationResult")
    print("=" * 60)

    from traigent.integrations.haystack import (
        compute_latency_stats,
        get_latency_metrics,
    )

    latencies = [0.1, 0.12, 0.15, 0.18, 0.2]  # 100-200ms range
    stats = compute_latency_stats(latencies)

    # Convert to metrics dict (same format as aggregated_metrics)
    metrics = get_latency_metrics(stats)

    print(f"\nMetrics dict for aggregated_metrics:")
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    print("\n  These keys can be used in constraints:")
    print("  - Constraint('latency_p95_ms', '<=', 200)")
    print("  - Constraint('latency_p99_ms', '<=', 500)")

    print("\n")


async def example_evaluator_with_latency():
    """Demonstrate HaystackEvaluator with latency tracking."""
    print("=" * 60)
    print("Example 4: HaystackEvaluator with Latency Tracking")
    print("=" * 60)

    from traigent.integrations.haystack import EvaluationDataset, HaystackEvaluator

    pipeline = create_mock_pipeline_with_variable_latency()

    dataset = EvaluationDataset.from_dicts(
        [
            {"input": {"query": "What is AI?"}, "expected": "AI response"},
            {"input": {"query": "What is ML?"}, "expected": "ML response"},
            {"input": {"query": "What is DL?"}, "expected": "DL response"},
            {"input": {"query": "What is NLP?"}, "expected": "NLP response"},
            {"input": {"query": "What is CV?"}, "expected": "CV response"},
        ]
    )

    # Create evaluator with latency tracking enabled (default)
    evaluator = HaystackEvaluator(
        pipeline=pipeline,
        haystack_dataset=dataset,
        output_key="llm.replies",
        track_latency=True,  # This is the default
        track_costs=True,  # Both cost and latency tracked
    )

    # Run evaluation
    result = await evaluator.evaluate(
        func=pipeline.run,
        config={"generator.temperature": 0.7},
        dataset=dataset.to_core_dataset(),
    )

    print(f"\nEvaluation completed:")
    print(f"  Examples: {result.total_examples}")
    print(f"  Duration: {result.duration:.3f}s")

    print(f"\nLatency metrics in result:")
    latency_keys = [
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
        "latency_mean_ms",
        "latency_min_ms",
        "latency_max_ms",
        "latency_count",
    ]
    for key in latency_keys:
        value = result.aggregated_metrics.get(key, "N/A")
        if isinstance(value, float):
            print(f"  {key}: {value:.1f}")
        else:
            print(f"  {key}: {value}")

    print("\n")


async def example_both_cost_and_latency():
    """Demonstrate tracking both cost and latency."""
    print("=" * 60)
    print("Example 5: Combined Cost and Latency Tracking")
    print("=" * 60)

    from traigent.integrations.haystack import EvaluationDataset, HaystackEvaluator

    pipeline = create_mock_pipeline_with_variable_latency()

    dataset = EvaluationDataset.from_dicts(
        [
            {"input": {"query": "Test 1"}, "expected": "Result 1"},
            {"input": {"query": "Test 2"}, "expected": "Result 2"},
            {"input": {"query": "Test 3"}, "expected": "Result 3"},
        ]
    )

    evaluator = HaystackEvaluator(
        pipeline=pipeline,
        haystack_dataset=dataset,
        output_key="llm.replies",
        track_costs=True,
        track_latency=True,
    )

    result = await evaluator.evaluate(
        func=pipeline.run,
        config={},
        dataset=dataset.to_core_dataset(),
    )

    print(f"\nCombined metrics summary:")
    metrics = result.aggregated_metrics

    # Cost metrics
    print(f"\n  Cost:")
    print(f"    Total tokens: {metrics.get('total_tokens', 0)}")
    print(f"    Total cost: ${metrics.get('total_cost', 0):.6f}")

    # Latency metrics
    print(f"\n  Latency:")
    print(f"    P50: {metrics.get('latency_p50_ms', 0):.1f}ms")
    print(f"    P95: {metrics.get('latency_p95_ms', 0):.1f}ms")
    print(f"    P99: {metrics.get('latency_p99_ms', 0):.1f}ms")

    print("\n  These can be used for multi-objective optimization:")
    print("  - Maximize quality")
    print("  - Minimize cost (total_cost)")
    print("  - Constrain latency (latency_p95_ms <= 300)")

    print("\n")


def main():
    """Run all examples."""
    # Set seed for reproducible latency simulation
    random.seed(42)

    example_latency_stats_computation()
    example_extract_latencies()
    example_latency_metrics()
    asyncio.run(example_evaluator_with_latency())
    asyncio.run(example_both_cost_and_latency())

    print("All latency tracking examples completed successfully!")


if __name__ == "__main__":
    main()
