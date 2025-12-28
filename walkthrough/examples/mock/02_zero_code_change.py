#!/usr/bin/env python3
"""Example 2: Zero Code Change - Seamless mode intercepts hardcoded values."""

import asyncio

import traigent

traigent.initialize(execution_mode="edge_analytics")


@traigent.optimize(
    eval_dataset="./simple_questions.jsonl",
    objectives=["accuracy", "cost"],
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "temperature": [0.1, 0.5, 0.9],
    },
    injection_mode="seamless",
    execution_mode="edge_analytics",
)
def answer_question(question: str) -> str:
    """Your existing code - Traigent overrides the hardcoded values below."""
    config = traigent.get_config()
    model = config.get("model", "gpt-3.5-turbo")
    print(f"  Testing: {model} with temp={config.get('temperature', 0.7)}")
    return f"Answer from {model}"


async def main() -> None:
    print("Traigent Example 2: Zero Code Change")
    print("=" * 50)
    print("Seamless mode overrides hardcoded LLM parameters.\n")

    results = await answer_question.optimize(algorithm="random", max_trials=6)

    print("\nBest Configuration:")
    print(f"  Model: {results.best_config.get('model')}")
    print(f"  Temperature: {results.best_config.get('temperature')}")
    print("\nYour original code stayed exactly the same!")


if __name__ == "__main__":
    asyncio.run(main())
