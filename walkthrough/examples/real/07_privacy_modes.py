#!/usr/bin/env python3
"""Example 7: Privacy Modes - Local, Cloud, and Hybrid execution.

Usage:
    export OPENAI_API_KEY="your-key"
    python 07_privacy_modes.py
"""

import asyncio

from langchain_openai import ChatOpenAI

import traigent


@traigent.optimize(
    eval_dataset="../datasets/simple_questions.jsonl",
    objectives=["accuracy"],
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
        "temperature": [0.1, 0.5],
    },
    execution_mode="edge_analytics",
    local_storage_path="./local_results",
)
def local_mode(text: str) -> str:
    """Local mode - all data stays on your machine."""
    config = traigent.get_config()
    llm = ChatOpenAI(
        model=config.get("model"),
        temperature=config.get("temperature"),
    )
    return str(llm.invoke(text).content)


@traigent.optimize(
    eval_dataset="../datasets/simple_questions.jsonl",
    objectives=["accuracy", "cost"],
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "temperature": [0.0, 0.3, 0.6],
    },
    execution_mode="cloud",
)
def cloud_mode(text: str) -> str:
    """Cloud mode - Bayesian optimization via Traigent cloud."""
    config = traigent.get_config()
    llm = ChatOpenAI(
        model=config.get("model"),
        temperature=config.get("temperature"),
    )
    return str(llm.invoke(text).content)


@traigent.optimize(
    eval_dataset="../datasets/simple_questions.jsonl",
    objectives=["accuracy", "cost"],
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
        "temperature": [0.1, 0.5],
    },
    execution_mode="hybrid",
    privacy_enabled=True,
)
def hybrid_mode(text: str) -> str:
    """Hybrid mode - local execution with cloud optimization."""
    config = traigent.get_config()
    llm = ChatOpenAI(
        model=config.get("model"),
        temperature=config.get("temperature"),
    )
    return str(llm.invoke(text).content)


async def main() -> None:
    print("Traigent Example 7: Privacy Modes")
    print("=" * 50)

    print("\nLOCAL - All data stays on your machine")
    local_results = await local_mode.optimize(algorithm="grid", max_trials=4)
    print(f"  Best: {local_results.best_config}")

    print("\nCLOUD - Advanced Bayesian optimization")
    cloud_results = await cloud_mode.optimize(algorithm="bayesian", max_trials=12)
    print(f"  Best: {cloud_results.best_config}")

    print("\nHYBRID - Local execution, cloud intelligence")
    hybrid_results = await hybrid_mode.optimize(algorithm="random", max_trials=6)
    print(f"  Best: {hybrid_results.best_config}")

    print("\nChoose the mode that fits your needs!")


if __name__ == "__main__":
    asyncio.run(main())
