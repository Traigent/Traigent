#!/usr/bin/env python3
"""Example 4: Multi-Objective - Balance accuracy, cost, and latency."""

import asyncio
import time

import traigent
from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema

traigent.initialize(execution_mode="edge_analytics")

OBJECTIVES = ObjectiveSchema.from_objectives([
    ObjectiveDefinition("accuracy", orientation="maximize", weight=0.5),
    ObjectiveDefinition("cost", orientation="minimize", weight=0.3),
    ObjectiveDefinition("latency", orientation="minimize", weight=0.2),
])


@traigent.optimize(
    eval_dataset="../datasets/classification.jsonl",
    objectives=OBJECTIVES,
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "temperature": [0.0, 0.3],
        "use_cot": [True, False],
    },
    execution_mode="edge_analytics",
)
def classify_text(text: str) -> str:
    """Text classification with multiple objectives."""
    config = traigent.get_config()
    model = config.get("model", "gpt-3.5-turbo")
    use_cot = config.get("use_cot", False)

    # Simulate latency differences
    if "gpt-4o" in model:
        time.sleep(0.05)
    else:
        time.sleep(0.02)

    return "positive" if use_cot else "neutral"


async def main() -> None:
    print("Traigent Example 4: Multi-Objective Optimization")
    print("=" * 50)
    print("Balancing accuracy (50%), cost (30%), latency (20%).\n")

    results = await classify_text.optimize(algorithm="random", max_trials=8)

    print("\nOptimal Configuration:")
    print(f"  Model: {results.best_config.get('model')}")
    print(f"  Temperature: {results.best_config.get('temperature')}")
    print(f"  Chain-of-Thought: {results.best_config.get('use_cot')}")

    print("\nPerformance:")
    print(f"  Accuracy: {results.best_metrics.get('accuracy', 0):.2%}")
    print(f"  Cost: ${results.best_metrics.get('cost', 0):.6f}")
    print(f"  Latency: {results.best_metrics.get('latency', 0):.3f}s")


if __name__ == "__main__":
    asyncio.run(main())
