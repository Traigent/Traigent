#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportGeneralTypeIssues=false, reportUndefinedVariable=false, reportArgumentType=false
# mypy: ignore-errors
# flake8: noqa
# ruff: noqa
# pylint: disable=all
"""Example 2: Zero Code Changes - TraiGent's seamless mode magic."""

import asyncio

from _shared import add_repo_root_to_sys_path, dataset_path, ensure_dataset, init_mock_mode

add_repo_root_to_sys_path(__file__)

ZERO_CODE_DATASET = dataset_path(__file__, "simple_questions.jsonl")
ensure_dataset(
    ZERO_CODE_DATASET,
    [
        {
            "input": {"question": "What is AI?"},
            "expected_output": "Artificial Intelligence",
        },
        {"input": {"question": "What is 2+2?"}, "expected_output": "4"},
        {
            "input": {"question": "Name a TraiGent feature."},
            "expected_output": "optimization",
        },
    ],
)

import traigent

MOCK = init_mock_mode()


# Your EXISTING code - completely unchanged!
def answer_question_original(question: str) -> str:
    """This is your existing function - NO CHANGES NEEDED!"""
    if MOCK:
        # Mock implementation
        return "Mock answer"

    from langchain_openai import ChatOpenAI

    # These hardcoded values will be overridden by TraiGent!
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    response = llm.invoke(f"Answer: {question}")
    return response.content


# Now add TraiGent - your code above stays EXACTLY the same!
@traigent.optimize(
    eval_dataset=str(ZERO_CODE_DATASET),
    objectives=["accuracy", "cost"],
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "temperature": [0.1, 0.5, 0.9],
    },
    injection_mode="seamless",  # This is the magic!
    execution_mode="edge_analytics",
)
def answer_question_optimized(question: str) -> str:
    """EXACT SAME CODE as above - just copy-pasted!"""
    if MOCK:
        # Mock will show different configs being tested
        config = traigent.get_current_config()
        model = config.get("model", "gpt-3.5-turbo")
        temp = config.get("temperature", 0.7)
        print(f"  Testing: {model} with temp={temp}")
        return f"Answer from {model}"

    from langchain_openai import ChatOpenAI

    # TraiGent automatically overrides these values!
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    response = llm.invoke(f"Answer: {question}")
    return response.content


async def main():
    print("🎯 TraiGent Example 2: Zero Code Changes Demo")
    print("=" * 50)
    print("🪄 Watch TraiGent optimize your EXISTING code without changes!\n")

    print("🔍 TraiGent will test these configurations:")
    print("   Models: gpt-3.5-turbo, gpt-4o-mini, gpt-4o")
    print("   Temperatures: 0.1, 0.5, 0.9")
    print("\nYour hardcoded values (gpt-3.5-turbo, temp=0.7) will be overridden!\n")

    # Run optimization
    results = await answer_question_optimized.optimize(
        algorithm="random", max_trials=9 if not MOCK else 6
    )

    # Show results
    print("\n" + "=" * 50)
    print("🏆 BEST CONFIGURATION FOUND:")
    print("=" * 50)
    print(f"Model: {results.best_config.get('model')}")
    print(f"Temperature: {results.best_config.get('temperature')}")
    print("\n📊 Performance:")
    print(f"Accuracy: {results.best_metrics.get('accuracy', 0):.2%}")
    print(f"Cost: ${results.best_metrics.get('cost', 0):.6f}")

    print("\n💡 Key Insight: Your original code stayed exactly the same!")
    print("   TraiGent seamlessly intercepted and optimized the LLM calls.")


if __name__ == "__main__":
    asyncio.run(main())
