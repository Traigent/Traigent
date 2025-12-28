#!/usr/bin/env python3
"""Example 6: Custom Evaluator - Define your own success metrics.

Usage:
    export OPENAI_API_KEY="your-key"
    python 06_custom_evaluator.py
"""

import asyncio

from langchain_openai import ChatOpenAI

import traigent


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
    eval_dataset="../datasets/code_gen.jsonl",
    objectives=["accuracy", "cost"],
    custom_evaluator=code_evaluator,
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
        "temperature": [0.0, 0.2, 0.5],
        "style": ["verbose", "concise", "documented"],
    },
    execution_mode="edge_analytics",
)
def generate_code(task: str) -> str:
    """Generate code with configurable style."""
    config = traigent.get_config()
    style = config.get("style", "concise")

    style_instructions = {
        "verbose": "Include detailed comments and documentation",
        "concise": "Write compact, efficient code",
        "documented": "Add docstrings and inline comments",
    }

    llm = ChatOpenAI(
        model=config.get("model", "gpt-3.5-turbo"),
        temperature=config.get("temperature", 0.2),
    )

    prompt = f"Generate Python code for: {task}\nStyle: {style_instructions[style]}"
    response = llm.invoke(prompt)
    return str(response.content)


async def main() -> None:
    print("Traigent Example 6: Custom Evaluator")
    print("=" * 50)
    print("Scoring: function def (40%), no errors (30%), docs (30%).\n")

    results = await generate_code.optimize(algorithm="grid", max_trials=24)

    print("\nBest Code Generation Config:")
    print(f"  Model: {results.best_config.get('model')}")
    print(f"  Temperature: {results.best_config.get('temperature')}")
    print(f"  Style: {results.best_config.get('style')}")

    print(f"\nCustom Score: {results.best_metrics.get('accuracy', 0):.2%}")


if __name__ == "__main__":
    asyncio.run(main())
