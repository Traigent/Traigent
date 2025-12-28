#!/usr/bin/env python3
"""Example 4: Multi-Objective - Balance accuracy, cost, and latency.

Usage:
    export OPENAI_API_KEY="your-key"
    python 04_multi_objective.py
"""

import asyncio

from langchain_openai import ChatOpenAI

import traigent
from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema

OBJECTIVES = ObjectiveSchema.from_objectives([
    ObjectiveDefinition("accuracy", orientation="maximize", weight=0.5),
    ObjectiveDefinition("cost", orientation="minimize", weight=0.3),
    ObjectiveDefinition("latency", orientation="minimize", weight=0.2),
])


@traigent.optimize(
    eval_dataset="./classification.jsonl",
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

    llm = ChatOpenAI(
        model=config.get("model"),
        temperature=config.get("temperature"),
    )

    if config.get("use_cot"):
        prompt = f"""Think step by step to classify sentiment.
Text: {text}
Classification (positive/negative/neutral):"""
    else:
        prompt = f"Classify sentiment: {text}\nAnswer:"

    response = llm.invoke(prompt)
    return str(response.content).strip().lower()


async def main() -> None:
    print("Traigent Example 4: Multi-Objective Optimization")
    print("=" * 50)
    print("Balancing accuracy (50%), cost (30%), latency (20%).\n")

    results = await classify_text.optimize(algorithm="random", max_trials=10, random_seed=42)

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
