#!/usr/bin/env python3
"""Example 1: Simple Optimization - Basic model and temperature tuning."""

import asyncio
from pathlib import Path

import traigent

traigent.initialize(execution_mode="edge_analytics")

# Get the directory containing this script
SCRIPT_DIR = Path(__file__).parent


@traigent.optimize(
    eval_dataset=str(SCRIPT_DIR / "simple_questions.jsonl"),
    objectives=["accuracy", "cost"],
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "gpt-4.1-nano"],
        "temperature": [0.1, 0.7],
    },
    execution_mode="edge_analytics",
)
def answer_question(question: str) -> str:
    """Simple Q&A function with mock responses."""
    q = question.lower()
    if "2+2" in q:
        return "4"
    elif "capital" in q and "france" in q:
        return "Paris"
    elif "machine learning" in q:
        return "A method where computers learn from data"
    elif "color" in q:
        return "red"
    return "I don't know"


async def main() -> None:
    print("Traigent Example 1: Simple Optimization")
    print("=" * 50)

    results = await answer_question.optimize(algorithm="grid", max_trials=8, random_seed=42)

    print("\nBest Configuration:")
    print(f"  Model: {results.best_config.get('model')}")
    print(f"  Temperature: {results.best_config.get('temperature')}")
    print("\nPerformance:")
    print(f"  Accuracy: {results.best_metrics.get('accuracy', 0):.2%}")
    print(f"  Cost: ${results.best_metrics.get('cost', 0):.6f}")


if __name__ == "__main__":
    asyncio.run(main())
