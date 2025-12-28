#!/usr/bin/env python3
"""Example 2: Zero Code Change - Seamless mode intercepts hardcoded values.

Usage:
    export OPENAI_API_KEY="your-key"
    python 02_zero_code_change.py
"""

import asyncio

from langchain_openai import ChatOpenAI

import traigent


@traigent.optimize(
    eval_dataset="../datasets/simple_questions.jsonl",
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
    # These hardcoded values will be overridden by Traigent!
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    response = llm.invoke(f"Answer: {question}")
    return str(response.content)


async def main() -> None:
    print("Traigent Example 2: Zero Code Change")
    print("=" * 50)
    print("Seamless mode overrides hardcoded LLM parameters.\n")

    results = await answer_question.optimize(algorithm="random", max_trials=9)

    print("\nBest Configuration:")
    print(f"  Model: {results.best_config.get('model')}")
    print(f"  Temperature: {results.best_config.get('temperature')}")
    print("\nYour original code stayed exactly the same!")


if __name__ == "__main__":
    asyncio.run(main())
