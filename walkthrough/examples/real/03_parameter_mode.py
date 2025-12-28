#!/usr/bin/env python3
"""Example 3: Parameter Mode - Explicit configuration control.

Usage:
    export OPENAI_API_KEY="your-key"
    python 03_parameter_mode.py
"""

import asyncio

from langchain_openai import ChatOpenAI

import traigent
from traigent import TraigentConfig


@traigent.optimize(
    eval_dataset="../datasets/simple_questions.jsonl",
    objectives=["accuracy", "cost"],
    injection_mode="parameter",
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
        "temperature": [0.0, 0.5, 1.0],
        "max_tokens": [50, 150, 300],
        "use_system_prompt": [True, False],
    },
    execution_mode="edge_analytics",
)
def answer_with_control(question: str, config: TraigentConfig) -> str:
    """Function with explicit configuration parameter."""
    model = config.get("model", "gpt-3.5-turbo")
    temperature = config.get("temperature", 0.5)
    max_tokens = config.get("max_tokens", 150)
    use_system = config.get("use_system_prompt", True)

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if use_system:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ]
    else:
        messages = [{"role": "user", "content": question}]

    response = llm.invoke(messages)
    return str(response.content)


async def main() -> None:
    print("Traigent Example 3: Parameter Mode")
    print("=" * 50)
    print("Full control with explicit configuration parameter.\n")

    results = await answer_with_control.optimize(algorithm="random", max_trials=12)

    print("\nOptimal Configuration:")
    for key, value in results.best_config.items():
        print(f"  {key}: {value}")

    print(f"\nAccuracy: {results.best_metrics.get('accuracy', 0):.2%}")


if __name__ == "__main__":
    asyncio.run(main())
