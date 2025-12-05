#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportGeneralTypeIssues=false, reportUndefinedVariable=false, reportArgumentType=false
# mypy: ignore-errors
# flake8: noqa
# ruff: noqa
# pylint: disable=all

"""Example 3: Parameter Mode - Explicit configuration control."""

import asyncio

from _shared import add_repo_root_to_sys_path, dataset_path, ensure_dataset, init_mock_mode

add_repo_root_to_sys_path(__file__)

PARAMETER_DATASET = dataset_path(__file__, "simple_questions.jsonl")
ensure_dataset(
    PARAMETER_DATASET,
    [
        {
            "input": {"question": "What is Python?"},
            "expected_output": "A programming language",
        },
        {
            "input": {"question": "Explain quantum computing"},
            "expected_output": "Computing using quantum mechanics",
        },
        {
            "input": {"question": "When to use system prompts?"},
            "expected_output": "To steer assistant behavior",
        },
    ],
)

import traigent
from traigent import TraigentConfig

MOCK = init_mock_mode()


@traigent.optimize(
    injection_mode="parameter",  # Explicit parameter mode
    eval_dataset=str(PARAMETER_DATASET),
    objectives=["accuracy", "cost"],
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

    # You have full control over configuration
    model = config.get("model", "gpt-3.5-turbo")
    temperature = config.get("temperature", 0.5)
    max_tokens = config.get("max_tokens", 150)
    use_system = config.get("use_system_prompt", True)

    print(
        f"  Using: model={model}, temp={temperature}, tokens={max_tokens}, system={use_system}"
    )

    if MOCK:
        # Show how different configs work
        if temperature < 0.3:
            style = "factual"
        elif temperature > 0.7:
            style = "creative"
        else:
            style = "balanced"

        return f"{style} answer from {model} (max {max_tokens} tokens)"

    # Real implementation
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=model, temperature=temperature, model_kwargs={"max_tokens": max_tokens})

    # Use system prompt based on configuration
    if use_system:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ]
    else:
        messages = [{"role": "user", "content": question}]

    response = llm.invoke(messages)
    return response.content


async def main():
    print("🎯 TraiGent Example 3: Parameter Mode")
    print("=" * 50)
    print("🎮 Full control with explicit configuration parameter\n")

    print("Configuration space to explore:")
    print("  • Models: 2 options")
    print("  • Temperature: 3 levels (factual/balanced/creative)")
    print("  • Max tokens: 3 lengths")
    print("  • System prompt: on/off")
    print(f"  Total combinations: {2*3*3*2} = 36\n")

    # Run optimization
    print("🔍 Starting optimization...\n")
    results = await answer_with_control.optimize(
        algorithm="random", max_trials=12 if not MOCK else 6
    )

    # Display results
    print("\n" + "=" * 50)
    print("🏆 OPTIMAL CONFIGURATION:")
    print("=" * 50)
    for key, value in results.best_config.items():
        print(f"{key:20s}: {value}")

    print(f"\nAccuracy: {results.best_metrics.get('accuracy', 0):.2%}")
    print(f"Cost: ${results.best_metrics.get('cost', 0):.6f}")

    print("\n💡 Parameter mode gives you:")
    print("  • Full control over configuration")
    print("  • Type safety with TraigentConfig")
    print("  • Easy debugging and testing")
    print("  • Complex conditional logic based on config")


if __name__ == "__main__":
    asyncio.run(main())
