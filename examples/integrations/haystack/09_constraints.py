#!/usr/bin/env python3
"""Example: Metric Constraints for Haystack Pipelines.

This example demonstrates how to define and use cost, latency, and
quality constraints for Haystack pipeline evaluations.

Coverage: Epic 4, Story 4.3 (Define Cost and Latency Constraints)
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock


def create_mock_pipeline():
    """Create a mock Haystack pipeline with token usage."""
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

    call_count = [0]

    def run_with_usage(**kwargs):
        call_count[0] += 1
        import time

        time.sleep(0.01)  # Small delay for latency
        return {
            "llm": {
                "replies": [f"Response {call_count[0]}"],
                "meta": [
                    {
                        "model": "gpt-4o",
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 50,
                        },
                    }
                ],
            }
        }

    pipeline.run.side_effect = run_with_usage

    return pipeline


def example_basic_constraints():
    """Demonstrate basic constraint creation."""
    print("=" * 60)
    print("Example 1: Basic Constraint Creation")
    print("=" * 60)

    from traigent.integrations.haystack import MetricConstraint

    # Create a cost constraint
    cost_limit = MetricConstraint(
        metric_name="total_cost",
        op="<=",
        threshold=0.05,
    )
    print(f"\nCost constraint: {cost_limit.name}")
    print(f"  Metric: {cost_limit.metric_name}")
    print(f"  Operator: {cost_limit.op}")
    print(f"  Threshold: {cost_limit.threshold}")

    # Create a latency constraint
    latency_limit = MetricConstraint(
        metric_name="latency_p95_ms",
        op="<=",
        threshold=500,
        name="max_p95_latency",
    )
    print(f"\nLatency constraint: {latency_limit.name}")

    # Create an accuracy constraint
    accuracy_min = MetricConstraint(
        metric_name="accuracy",
        op=">=",
        threshold=0.8,
    )
    print(f"\nAccuracy constraint: {accuracy_min.name}")

    # Test constraint checking
    metrics = {"total_cost": 0.03, "latency_p95_ms": 300, "accuracy": 0.85}

    print(f"\nChecking against metrics: {metrics}")
    print(f"  Cost satisfied: {cost_limit.check(metrics)}")
    print(f"  Latency satisfied: {latency_limit.check(metrics)}")
    print(f"  Accuracy satisfied: {accuracy_min.check(metrics)}")

    print("\n")


def example_helper_functions():
    """Demonstrate constraint helper functions."""
    print("=" * 60)
    print("Example 2: Constraint Helper Functions")
    print("=" * 60)

    from traigent.integrations.haystack import (
        cost_constraint,
        latency_constraint,
        quality_constraint,
    )

    # Cost constraint helper
    cost = cost_constraint(max_cost=0.10)
    print(f"\ncost_constraint(max_cost=0.10):")
    print(f"  Name: {cost.name}")
    print(f"  Metric: {cost.metric_name} {cost.op} {cost.threshold}")

    # Latency constraint helper (returns list)
    latencies = latency_constraint(p50_ms=100, p95_ms=500, p99_ms=1000)
    print(f"\nlatency_constraint(p50_ms=100, p95_ms=500, p99_ms=1000):")
    for c in latencies:
        print(f"  - {c.metric_name} {c.op} {c.threshold}")

    # Quality constraint helper (returns list)
    quality = quality_constraint("accuracy", min_value=0.8)
    print(f"\nquality_constraint('accuracy', min_value=0.8):")
    for c in quality:
        print(f"  - {c.metric_name} {c.op} {c.threshold}")

    # Range constraint
    score_range = quality_constraint("score", min_value=0.0, max_value=1.0)
    print(f"\nquality_constraint('score', min_value=0.0, max_value=1.0):")
    for c in score_range:
        print(f"  - {c.metric_name} {c.op} {c.threshold}")

    print("\n")


def example_constraint_checking():
    """Demonstrate checking multiple constraints."""
    print("=" * 60)
    print("Example 3: Batch Constraint Checking")
    print("=" * 60)

    from traigent.integrations.haystack import (
        MetricConstraint,
        check_constraints,
        cost_constraint,
        latency_constraint,
    )

    # Define multiple constraints
    constraints = [
        cost_constraint(max_cost=0.05),
        *latency_constraint(p95_ms=500),
        MetricConstraint("accuracy", ">=", 0.8),
    ]

    print(f"\nDefined {len(constraints)} constraints:")
    for c in constraints:
        print(f"  - {c.name}")

    # Check against satisfied metrics
    good_metrics = {
        "total_cost": 0.03,
        "latency_p95_ms": 300,
        "accuracy": 0.9,
    }

    result = check_constraints(constraints, good_metrics)
    print(f"\nGood metrics: {good_metrics}")
    print(f"  All satisfied: {result.all_satisfied}")
    print(f"  Satisfied: {result.satisfied_count}/{result.total_count}")

    # Check against violated metrics
    bad_metrics = {
        "total_cost": 0.10,  # Exceeded
        "latency_p95_ms": 600,  # Exceeded
        "accuracy": 0.7,  # Below minimum
    }

    result = check_constraints(constraints, bad_metrics)
    print(f"\nBad metrics: {bad_metrics}")
    print(f"  All satisfied: {result.all_satisfied}")
    print(f"  Satisfied: {result.satisfied_count}/{result.total_count}")
    print(f"  Violations:")
    for msg in result.violation_messages:
        print(f"    - {msg}")

    print("\n")


def example_violation_details():
    """Demonstrate accessing violation details."""
    print("=" * 60)
    print("Example 4: Violation Details")
    print("=" * 60)

    from traigent.integrations.haystack import (
        check_constraints,
        cost_constraint,
        latency_constraint,
    )

    constraints = [
        cost_constraint(max_cost=0.05),
        *latency_constraint(p95_ms=500),
    ]

    metrics = {
        "total_cost": 0.08,  # Violated
        "latency_p95_ms": 300,  # OK
    }

    result = check_constraints(constraints, metrics)

    print(f"\nMetrics: {metrics}")
    print(f"\nViolation details:")
    for violation in result.violations:
        print(f"\n  Constraint: {violation.constraint.name}")
        print(f"  Metric: {violation.constraint.metric_name}")
        print(
            f"  Threshold: {violation.constraint.op} {violation.constraint.threshold}"
        )
        print(f"  Actual value: {violation.actual_value}")
        print(f"  Message: {violation.message}")

    print("\n")


async def example_evaluator_with_constraints():
    """Demonstrate HaystackEvaluator with constraints."""
    print("=" * 60)
    print("Example 5: HaystackEvaluator with Constraints")
    print("=" * 60)

    from traigent.integrations.haystack import (
        EvaluationDataset,
        HaystackEvaluator,
        cost_constraint,
        latency_constraint,
    )

    pipeline = create_mock_pipeline()

    dataset = EvaluationDataset.from_dicts(
        [
            {"input": {"query": "What is AI?"}, "expected": "Artificial Intelligence"},
            {"input": {"query": "What is ML?"}, "expected": "Machine Learning"},
        ]
    )

    # Define constraints that will be checked after evaluation
    constraints = [
        cost_constraint(max_cost=1.0),  # Generous limit
        *latency_constraint(p95_ms=1000),  # Generous limit
    ]

    # Create evaluator with constraints
    evaluator = HaystackEvaluator(
        pipeline=pipeline,
        haystack_dataset=dataset,
        output_key="llm.replies",
        track_costs=True,
        track_latency=True,
        constraints=constraints,  # Pass constraints here
    )

    result = await evaluator.evaluate(
        func=pipeline.run,
        config={"generator.temperature": 0.7},
        dataset=dataset.to_core_dataset(),
    )

    print(f"\nEvaluation completed:")
    print(f"  Examples: {result.total_examples}")

    print(f"\nCost & latency metrics:")
    print(f"  Total cost: ${result.aggregated_metrics.get('total_cost', 0):.6f}")
    print(f"  P95 latency: {result.aggregated_metrics.get('latency_p95_ms', 0):.1f}ms")

    print(f"\nConstraint results:")
    print(f"  Satisfied: {result.aggregated_metrics.get('constraints_satisfied')}")
    print(f"  Checked: {result.aggregated_metrics.get('constraints_checked')}")
    print(f"  Passed: {result.aggregated_metrics.get('constraints_passed')}")

    print("\n")


async def example_constraint_violation():
    """Demonstrate constraint violation in evaluation."""
    print("=" * 60)
    print("Example 6: Constraint Violation in Evaluation")
    print("=" * 60)

    from traigent.integrations.haystack import (
        EvaluationDataset,
        HaystackEvaluator,
        cost_constraint,
    )

    pipeline = create_mock_pipeline()

    dataset = EvaluationDataset.from_dicts(
        [{"input": {"query": "Test"}, "expected": "Result"}]
    )

    # Define a very strict cost constraint that will be violated
    constraints = [
        cost_constraint(max_cost=0.0000001),  # Impossibly low
    ]

    evaluator = HaystackEvaluator(
        pipeline=pipeline,
        haystack_dataset=dataset,
        output_key="llm.replies",
        track_costs=True,
        constraints=constraints,
    )

    result = await evaluator.evaluate(
        func=pipeline.run,
        config={},
        dataset=dataset.to_core_dataset(),
    )

    print(f"\nEvaluation with strict cost constraint:")
    print(f"  Actual cost: ${result.aggregated_metrics.get('total_cost', 0):.6f}")
    print(f"  Constraint threshold: $0.0000001")
    print(f"  Satisfied: {result.aggregated_metrics.get('constraints_satisfied')}")

    # In real usage, you might filter or flag results based on constraints
    if not result.aggregated_metrics.get("constraints_satisfied"):
        print("\n  This configuration would be marked as constraint-violating!")

    print("\n")


def example_combining_constraints():
    """Demonstrate combining constraints for production use."""
    print("=" * 60)
    print("Example 7: Production-Ready Constraint Setup")
    print("=" * 60)

    from traigent.integrations.haystack import (
        MetricConstraint,
        cost_constraint,
        latency_constraint,
        quality_constraint,
    )

    # Production constraints for an LLM-based Q&A system
    production_constraints = [
        # Cost: max $0.05 per evaluation
        cost_constraint(max_cost=0.05),
        # Latency: p50 < 200ms, p95 < 500ms, p99 < 1s
        *latency_constraint(p50_ms=200, p95_ms=500, p99_ms=1000),
        # Quality: accuracy >= 80%
        *quality_constraint("accuracy", min_value=0.8),
        # Custom: answer relevance score >= 0.7
        MetricConstraint("relevance_score", ">=", 0.7),
    ]

    print(f"\nProduction constraints ({len(production_constraints)} total):")
    print("\n  Cost constraints:")
    for c in production_constraints:
        if "cost" in c.metric_name:
            print(f"    - {c.name}")

    print("\n  Latency constraints:")
    for c in production_constraints:
        if "latency" in c.metric_name:
            print(f"    - {c.name}")

    print("\n  Quality constraints:")
    for c in production_constraints:
        if c.metric_name in ("accuracy", "relevance_score"):
            print(f"    - {c.name}")

    print("\n  These would be passed to HaystackEvaluator:")
    print("    evaluator = HaystackEvaluator(")
    print("        pipeline=pipeline,")
    print("        haystack_dataset=dataset,")
    print("        constraints=production_constraints,")
    print("    )")

    print("\n")


def create_slow_mock_pipeline():
    """Create a mock pipeline with slow responses."""
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

    def slow_run(**kwargs):
        import time

        time.sleep(0.05)  # 50ms - will exceed 20ms constraint
        return {
            "llm": {
                "replies": ["Response"],
                "meta": [
                    {
                        "model": "gpt-4o",
                        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                    }
                ],
            }
        }

    pipeline.run.side_effect = slow_run

    return pipeline


async def example_early_stopping():
    """Demonstrate early stopping on constraint violation."""
    print("=" * 60)
    print("Example 8: Early Stopping on Constraint Violation")
    print("=" * 60)

    from traigent.integrations.haystack import (
        EvaluationDataset,
        HaystackEvaluator,
        latency_constraint,
    )

    pipeline = create_slow_mock_pipeline()

    # Create dataset with many examples
    dataset = EvaluationDataset.from_dicts(
        [
            {"input": {"query": f"Question {i}"}, "expected": f"Answer {i}"}
            for i in range(10)  # 10 examples
        ]
    )

    # Define a strict latency constraint that will be violated
    constraints = latency_constraint(p95_ms=20)  # 20ms limit

    # Create evaluator with early stopping enabled
    evaluator = HaystackEvaluator(
        pipeline=pipeline,
        haystack_dataset=dataset,
        output_key="llm.replies",
        track_latency=True,
        constraints=constraints,
        early_stop_on_violation=True,  # Enable early stopping
        violation_threshold=0.5,  # Stop when >50% violate
        min_examples_before_stop=3,  # Wait for at least 3 examples
    )

    print("\nConfiguration:")
    print("  - 10 examples to evaluate")
    print("  - Latency constraint: p95 <= 20ms")
    print("  - Early stopping: enabled")
    print("  - Violation threshold: 50%")
    print("  - Min examples before stop: 3")
    print("\n  (Pipeline simulates 50ms latency per call)")

    result = await evaluator.evaluate(
        func=pipeline.run,
        config={},
        dataset=dataset.to_core_dataset(),
    )

    print("\nResults:")
    print(f"  Examples evaluated: {result.total_examples} of 10")
    print(f"  Stopped early: {result.aggregated_metrics.get('stopped_early', False)}")
    print(f"  Latency p95: {result.aggregated_metrics.get('latency_p95_ms', 0):.1f}ms")
    print(
        f"  Constraints satisfied: {result.aggregated_metrics.get('constraints_satisfied')}"
    )

    if result.aggregated_metrics.get("stopped_early"):
        print("\n  Early stopping saved compute by not running all examples!")

    print("\n")


def example_filter_by_constraints():
    """Demonstrate filtering results by constraint satisfaction."""
    print("=" * 60)
    print("Example 9: Filtering Results by Constraint Satisfaction")
    print("=" * 60)

    from traigent.evaluators.base import EvaluationResult
    from traigent.integrations.haystack import (
        filter_by_constraints,
        get_best_satisfying,
    )

    # Simulate multiple evaluation results from different configs
    results = [
        EvaluationResult(
            config={"temperature": 0.5},
            aggregated_metrics={
                "accuracy": 0.75,
                "total_cost": 0.02,
                "constraints_satisfied": False,  # Accuracy too low
            },
        ),
        EvaluationResult(
            config={"temperature": 0.7},
            aggregated_metrics={
                "accuracy": 0.85,
                "total_cost": 0.03,
                "constraints_satisfied": True,  # All constraints met
            },
        ),
        EvaluationResult(
            config={"temperature": 0.9},
            aggregated_metrics={
                "accuracy": 0.90,
                "total_cost": 0.08,
                "constraints_satisfied": False,  # Cost too high
            },
        ),
        EvaluationResult(
            config={"temperature": 0.8},
            aggregated_metrics={
                "accuracy": 0.88,
                "total_cost": 0.04,
                "constraints_satisfied": True,  # All constraints met
            },
        ),
    ]

    print(f"\nAll results ({len(results)} configurations):")
    for r in results:
        print(
            f"  temp={r.config['temperature']}: "
            f"accuracy={r.aggregated_metrics['accuracy']:.2f}, "
            f"cost=${r.aggregated_metrics['total_cost']:.2f}, "
            f"satisfied={r.aggregated_metrics['constraints_satisfied']}"
        )

    # Filter to only constraint-satisfying results
    satisfying = filter_by_constraints(results)
    print(f"\nFiltered results ({len(satisfying)} satisfying constraints):")
    for r in satisfying:
        print(
            f"  temp={r.config['temperature']}: "
            f"accuracy={r.aggregated_metrics['accuracy']:.2f}"
        )

    # Get the best satisfying result by accuracy
    best = get_best_satisfying(results, metric="accuracy", maximize=True)
    if best:
        print("\nBest constraint-satisfying config (by accuracy):")
        print(f"  Config: {best.config}")
        print(f"  Accuracy: {best.aggregated_metrics['accuracy']:.2f}")
        print(f"  Cost: ${best.aggregated_metrics['total_cost']:.2f}")

    # Get the cheapest satisfying result
    cheapest = get_best_satisfying(results, metric="total_cost", maximize=False)
    if cheapest:
        print("\nCheapest constraint-satisfying config:")
        print(f"  Config: {cheapest.config}")
        print(f"  Accuracy: {cheapest.aggregated_metrics['accuracy']:.2f}")
        print(f"  Cost: ${cheapest.aggregated_metrics['total_cost']:.2f}")

    print("\n")


async def example_early_stopping_disabled():
    """Demonstrate early stopping disabled for comparison."""
    print("=" * 60)
    print("Example 10: Early Stopping Disabled (Comparison)")
    print("=" * 60)

    from traigent.integrations.haystack import (
        EvaluationDataset,
        HaystackEvaluator,
        latency_constraint,
    )

    pipeline = create_slow_mock_pipeline()

    dataset = EvaluationDataset.from_dicts(
        [
            {"input": {"query": f"Question {i}"}, "expected": f"Answer {i}"}
            for i in range(5)  # 5 examples
        ]
    )

    constraints = latency_constraint(p95_ms=20)

    # Create evaluator WITHOUT early stopping
    evaluator = HaystackEvaluator(
        pipeline=pipeline,
        haystack_dataset=dataset,
        output_key="llm.replies",
        track_latency=True,
        constraints=constraints,
        early_stop_on_violation=False,  # Disabled (default)
    )

    print("\nConfiguration:")
    print("  - 5 examples to evaluate")
    print("  - Latency constraint: p95 <= 20ms")
    print("  - Early stopping: DISABLED")

    result = await evaluator.evaluate(
        func=pipeline.run,
        config={},
        dataset=dataset.to_core_dataset(),
    )

    print("\nResults:")
    print(f"  Examples evaluated: {result.total_examples} of 5 (all executed)")
    print(f"  Stopped early: {result.aggregated_metrics.get('stopped_early', False)}")
    print(
        f"  Constraints satisfied: {result.aggregated_metrics.get('constraints_satisfied')}"
    )

    print("\n  Without early stopping, all examples run even when constraints fail.")

    print("\n")


def main():
    """Run all examples."""
    example_basic_constraints()
    example_helper_functions()
    example_constraint_checking()
    example_violation_details()
    asyncio.run(example_evaluator_with_constraints())
    asyncio.run(example_constraint_violation())
    example_combining_constraints()
    asyncio.run(example_early_stopping())
    example_filter_by_constraints()
    asyncio.run(example_early_stopping_disabled())

    print("All constraint examples completed successfully!")


if __name__ == "__main__":
    main()
