#!/usr/bin/env python3
"""
TVL 0.9 Tutorial: Multi-Objective Optimization Analysis

This script demonstrates how to analyze multi-objective optimization
results and understand Pareto trade-offs.

Key concepts:
- Pareto dominance: A solution dominates another if it's at least as good
  on all objectives and strictly better on at least one.
- Pareto front: The set of non-dominated solutions.
- Epsilon-Pareto: Requires improvement of at least epsilon to dominate.

Run with (from repo root): .venv/bin/python examples/tvl/tutorials/03_multi_objective/analyze_tradeoffs.py
"""

import random
from pathlib import Path

from traigent.tvl import load_tvl_spec
from traigent.tvl.statistics import hypervolume_improvement

SPEC_PATH = Path(__file__).parent / "multi_objective_rag.tvl.yml"


def simulate_evaluation(config: dict) -> dict:
    """Simulate metric evaluation for a configuration."""
    # Base metrics depend on model choice
    model_quality = {
        "gpt-4o": 0.92,
        "gpt-4o-mini": 0.85,
        "gpt-3.5-turbo": 0.75,
    }
    model_cost = {
        "gpt-4o": 0.005,
        "gpt-4o-mini": 0.00015,
        "gpt-3.5-turbo": 0.0005,
    }
    model_latency = {
        "gpt-4o": 2000,
        "gpt-4o-mini": 500,
        "gpt-3.5-turbo": 300,
    }

    model = config["model"]

    # Accuracy: affected by model, temperature, and retrieval
    accuracy = model_quality[model]
    accuracy -= config["temperature"] * 0.05  # Higher temp = less precise
    accuracy += config["retrieval_k"] * 0.01  # More context = better
    if config.get("use_cot", False):
        accuracy += 0.03  # CoT helps
    accuracy += random.gauss(0, 0.02)
    accuracy = max(0.5, min(1.0, accuracy))

    # Consistency: inversely related to temperature
    consistency = 0.95 - config["temperature"] * 0.15
    consistency += random.gauss(0, 0.02)
    consistency = max(0.5, min(1.0, consistency))

    # Latency: affected by model, tokens, and retrieval
    latency = model_latency[model]
    latency += config["max_tokens"] * 0.5
    latency += config["retrieval_k"] * 20
    latency += random.gauss(0, 50)
    latency = max(100, latency)

    # Cost: model cost * tokens
    cost = model_cost[model] * (config["max_tokens"] / 1000)
    if config.get("use_cot", False):
        cost *= 1.5  # CoT uses more tokens

    return {
        "answer_accuracy": accuracy,
        "consistency_score": consistency,
        "latency_ms": latency,
        "cost_per_query": cost,
    }


def is_pareto_dominated(point_a: list, point_b: list, directions: list) -> bool:
    """Check if point_a is dominated by point_b."""
    at_least_as_good = True
    strictly_better = False

    for i, direction in enumerate(directions):
        if direction == "maximize":
            if point_b[i] < point_a[i]:
                at_least_as_good = False
            if point_b[i] > point_a[i]:
                strictly_better = True
        else:  # minimize
            if point_b[i] > point_a[i]:
                at_least_as_good = False
            if point_b[i] < point_a[i]:
                strictly_better = True

    return at_least_as_good and strictly_better


def find_pareto_front(
    points: list[tuple[dict, list]], directions: list
) -> list[tuple[dict, list]]:
    """Find the Pareto front from a set of points."""
    front = []
    for config, metrics in points:
        dominated = False
        for _, other_metrics in points:
            if is_pareto_dominated(metrics, other_metrics, directions):
                dominated = True
                break
        if not dominated:
            front.append((config, metrics))
    return front


def main():
    """Demonstrate multi-objective analysis."""
    print("=" * 60)
    print("TVL 0.9 Tutorial: Multi-Objective Optimization")
    print("=" * 60)

    # Load spec
    print("\n1. Loading TVL Spec...")
    spec = load_tvl_spec(spec_path=SPEC_PATH)

    # Extract objective info
    objectives = spec.objective_schema.objectives
    print(f"\n   Objectives ({len(objectives)}):")
    directions = []
    for obj in objectives:
        print(f"   - {obj.name}: {obj.orientation} (weight={obj.weight})")
        directions.append(obj.orientation)

    # Extract promotion policy
    policy = spec.promotion_policy
    if policy:
        print("\n   Promotion Policy:")
        print(f"   - Dominance: {policy.dominance}")
        print(f"   - Alpha: {policy.alpha}")
        print(f"   - Adjustment: {policy.adjust}")
        print(f"   - Min effects: {policy.min_effect}")

    # Generate sample configurations
    print("\n2. Simulating Optimization Run...")
    random.seed(42)

    configs = [
        {
            "model": "gpt-4o",
            "temperature": 0.3,
            "max_tokens": 1024,
            "retrieval_k": 7,
            "use_cot": True,
        },
        {
            "model": "gpt-4o",
            "temperature": 0.7,
            "max_tokens": 512,
            "retrieval_k": 5,
            "use_cot": True,
        },
        {
            "model": "gpt-4o-mini",
            "temperature": 0.5,
            "max_tokens": 512,
            "retrieval_k": 5,
            "use_cot": False,
        },
        {
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 768,
            "retrieval_k": 6,
            "use_cot": False,
        },
        {
            "model": "gpt-4o-mini",
            "temperature": 0.3,
            "max_tokens": 1024,
            "retrieval_k": 8,
            "use_cot": True,
        },
        {
            "model": "gpt-3.5-turbo",
            "temperature": 0.5,
            "max_tokens": 512,
            "retrieval_k": 5,
            "use_cot": False,
        },
        {
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 256,
            "retrieval_k": 3,
            "use_cot": False,
        },
        {
            "model": "gpt-3.5-turbo",
            "temperature": 0.3,
            "max_tokens": 768,
            "retrieval_k": 7,
            "use_cot": False,
        },
    ]

    # Evaluate all configs
    results = []
    for config in configs:
        metrics = simulate_evaluation(config)
        metrics_list = [
            metrics["answer_accuracy"],
            metrics["consistency_score"],
            metrics["latency_ms"],
            metrics["cost_per_query"],
        ]
        results.append((config, metrics_list))

    print(f"   Evaluated {len(results)} configurations")

    # Find Pareto front
    print("\n3. Finding Pareto Front...")
    pareto_front = find_pareto_front(results, directions)
    print(f"   Found {len(pareto_front)} Pareto-optimal solutions")

    print("\n   Pareto-Optimal Configurations:")
    for i, (config, metrics) in enumerate(pareto_front):
        print(f"\n   Solution {i+1}: {config['model']}")
        print(
            f"     Config: temp={config['temperature']}, tokens={config['max_tokens']}, k={config['retrieval_k']}"
        )
        print(f"     Accuracy:    {metrics[0]:.3f}")
        print(f"     Consistency: {metrics[1]:.3f}")
        print(f"     Latency:     {metrics[2]:.0f}ms")
        print(f"     Cost:        ${metrics[3]:.5f}")

    # Demonstrate trade-offs
    print("\n4. Understanding Trade-offs...")
    print("\n   The Pareto front shows the fundamental trade-offs:")
    print("   - Higher quality models (GPT-4o) cost more and are slower")
    print("   - Lower temperature improves consistency but may reduce creativity")
    print("   - More retrieval (higher k) improves accuracy but increases latency")
    print("\n   There's no single 'best' solution - it depends on your priorities!")

    # Demonstrate hypervolume
    print("\n5. Hypervolume Analysis...")
    # Reference point (worst acceptable values)
    reference = [0.5, 0.5, 5000, 0.01]  # [min_acc, min_cons, max_lat, max_cost]

    # Add points one by one and show improvement
    print("   Adding solutions to Pareto front:")
    accumulated_front = []
    for i, (config, metrics) in enumerate(pareto_front):
        improvement = hypervolume_improvement(
            metrics, accumulated_front, reference, directions
        )
        accumulated_front.append(metrics)
        print(
            f"   + Solution {i+1} ({config['model']}): HV improvement = {improvement:.4f}"
        )

    print("\n" + "=" * 60)
    print("Tutorial complete! Next: 04_promotion_policy")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130)
