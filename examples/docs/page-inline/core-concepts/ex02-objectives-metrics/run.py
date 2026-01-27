#!/usr/bin/env python3
"""Runner for objectives and metrics example."""

from __future__ import annotations

import asyncio
import json
import os
import sys

# Add parent directory to path to import the module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from objectives_metrics import (
    analyze_tradeoffs,
    demonstrate_objective_types,
)


async def main() -> None:
    """Run the objectives and metrics example."""
    results_file = "results.json"

    print("Running Traigent Objectives & Metrics Example")
    print("=" * 50)

    # Demonstrate objective types
    demonstrate_objective_types()

    # Analyze tradeoffs
    analyze_tradeoffs()

    # Create results for display
    results_data = {
        "concept": "Objectives & Metrics",
        "description": "Defining what success looks like for optimization",
        "objective_types": {
            "single_objective": {
                "example": "objectives=['cost']",
                "description": "Optimize for one goal only",
                "use_case": "When you have a clear priority",
            },
            "multi_objective": {
                "example": "objectives=['cost', 'quality', 'speed']",
                "description": "Balance multiple competing goals",
                "use_case": "Real-world scenarios with tradeoffs",
            },
            "weighted_objectives": {
                "example": "weights={'cost': 0.3, 'quality': 0.5, 'speed': 0.2}",
                "description": "Specify relative importance",
                "use_case": "Fine-tune the balance between objectives",
            },
            "custom_metrics": {
                "example": "custom_metrics={'satisfaction': my_custom_function}",
                "description": "Define domain-specific metrics",
                "use_case": "Specialized evaluation criteria",
            },
        },
        "built_in_objectives": [
            "cost - Minimize API costs",
            "latency - Minimize response time",
            "accuracy - Maximize correctness",
            "quality - Maximize output quality",
            "response_time - Minimize total processing time",
        ],
        "constraints_example": {
            "max_cost_per_call": 0.05,
            "min_quality_score": 0.7,
            "max_response_time": 2.0,
        },
        "tradeoff_analysis": {
            "cost_vs_quality": {
                "cheap": {"model": "gpt-3.5-turbo", "cost": "$0.002", "quality": "70%"},
                "balanced": {"model": "gpt-4o-mini", "cost": "$0.01", "quality": "85%"},
                "premium": {"model": "gpt-4o", "cost": "$0.03", "quality": "95%"},
            },
            "speed_vs_accuracy": {
                "fast": {"tokens": 100, "time": "1s", "accuracy": "75%"},
                "balanced": {"tokens": 200, "time": "2s", "accuracy": "85%"},
                "thorough": {"tokens": 500, "time": "4s", "accuracy": "92%"},
            },
        },
        "pareto_optimization": {
            "description": "Traigent finds Pareto-optimal solutions",
            "explanation": "Solutions where improving one objective would worsen another",
            "benefit": "Get the best possible tradeoffs for your use case",
        },
        "key_insights": [
            "Different use cases require different optimization goals",
            "Multi-objective optimization handles real-world tradeoffs",
            "Custom metrics allow domain-specific optimization",
            "Constraints ensure solutions meet minimum requirements",
            "Pareto frontier shows optimal tradeoff points",
        ],
    }

    # Save results
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\nWrote {os.path.abspath(results_file)}")
    print("\nExample optimization scenarios created:")
    print("1. Cost-optimized bot (single objective)")
    print("2. Balanced support bot (multi-objective)")
    print("3. Quality-constrained bot (custom metrics)")


if __name__ == "__main__":
    asyncio.run(main())
