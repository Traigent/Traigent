#!/usr/bin/env python3
"""Example 6: Custom Evaluator - Define your own success metrics."""

import asyncio

import traigent

traigent.initialize(execution_mode="edge_analytics")


def code_evaluator(output: str, expected: str) -> float:
    """Custom evaluator for code generation quality."""
    score = 0.0
    if "def " in output:
        score += 0.4
    if output.strip() and "error" not in output.lower():
        score += 0.3
    if '"""' in output or "#" in output:
        score += 0.3
    return min(score, 1.0)


@traigent.optimize(
    eval_dataset="./code_gen.jsonl",
    objectives=["accuracy", "cost"],
    custom_evaluator=code_evaluator,
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
        "temperature": [0.0, 0.2, 0.5],
        "style": ["verbose", "concise", "documented"],
    },
)
def generate_code(task: str) -> str:
    """Generate code with configurable style."""
    config = traigent.get_config()
    style = config.get("style", "concise")

    print(f"  Generating {style} code...")

    if style == "verbose":
        return '''def calculate_sum(numbers):
    """Calculate the sum of a list."""
    total = 0
    for n in numbers:
        total += n
    return total'''
    elif style == "documented":
        return '''def calculate_sum(numbers):
    # Sum all numbers
    return sum(numbers)'''
    return "def calculate_sum(nums): return sum(nums)"


async def main() -> None:
    print("Traigent Example 6: Custom Evaluator")
    print("=" * 50)
    print("Scoring: function def (40%), no errors (30%), docs (30%).\n")

    results = await generate_code.optimize(algorithm="grid", max_trials=8)

    print("\nBest Code Generation Config:")
    print(f"  Model: {results.best_config.get('model')}")
    print(f"  Temperature: {results.best_config.get('temperature')}")
    print(f"  Style: {results.best_config.get('style')}")

    print(f"\nCustom Score: {results.best_metrics.get('accuracy', 0):.2%}")


if __name__ == "__main__":
    asyncio.run(main())
