#!/usr/bin/env python3
"""Example 1: Simple Optimization - Basic model and temperature tuning.

Usage:
    export OPENAI_API_KEY="your-key"
    python 01_simple.py
"""

import asyncio

from langchain_openai import ChatOpenAI

import traigent


@traigent.optimize(
    eval_dataset="./simple_questions.jsonl",
    objectives=["accuracy", "cost"],
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
        "temperature": [0.1, 0.5, 0.9],
    },
    execution_mode="edge_analytics",
)
def answer_question(question: str) -> str:
    """Simple Q&A function using OpenAI."""
    config = traigent.get_config()
    llm = ChatOpenAI(
        model=config["model"],
        temperature=config["temperature"],
    )
    response = llm.invoke(f"Answer concisely: {question}")
    return str(response.content)


async def main() -> None:
    print("Traigent Example 1: Simple Optimization")
    print("=" * 50)

    results = await answer_question.optimize(algorithm="grid", max_trials=10, random_seed=42)

    print("\nBest Configuration:")
    print(f"  Model: {results.best_config.get('model')}")
    print(f"  Temperature: {results.best_config.get('temperature')}")
    print("\nPerformance:")
    print(f"  Accuracy: {results.best_metrics.get('accuracy', 0):.2%}")
    print(f"  Cost: ${results.best_metrics.get('cost', 0):.6f}")


if __name__ == "__main__":
    asyncio.run(main())
