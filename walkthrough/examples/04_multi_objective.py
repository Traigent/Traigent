#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportGeneralTypeIssues=false, reportUndefinedVariable=false, reportArgumentType=false
# mypy: ignore-errors
# flake8: noqa
# ruff: noqa
# pylint: disable=all

"""Example 4: Multi-Objective Optimization - Balance multiple goals."""

import asyncio
import time

from _shared import add_repo_root_to_sys_path, dataset_path, ensure_dataset, init_mock_mode

add_repo_root_to_sys_path(__file__)

CLASSIFICATION_DATASET = dataset_path(__file__, "classification.jsonl")
ensure_dataset(
    CLASSIFICATION_DATASET,
    [
        {
            "input": {"text": "This product is amazing!"},
            "expected_output": "positive",
        },
        {
            "input": {"text": "Terrible experience, very disappointed"},
            "expected_output": "negative",
        },
        {
            "input": {"text": "It works as expected"},
            "expected_output": "neutral",
        },
        {
            "input": {"text": "Best purchase ever!"},
            "expected_output": "positive",
        },
        {
            "input": {"text": "Quality could be better"},
            "expected_output": "negative",
        },
    ],
)

import traigent
from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema

MOCK = init_mock_mode()


TRIPLE_OBJECTIVE_SCHEMA = ObjectiveSchema.from_objectives(
    [
        ObjectiveDefinition("accuracy", orientation="maximize", weight=0.5),
        ObjectiveDefinition("cost", orientation="minimize", weight=0.3),
        ObjectiveDefinition("latency", orientation="minimize", weight=0.2),
    ]
)


@traigent.optimize(
    eval_dataset=str(CLASSIFICATION_DATASET),
    objectives=TRIPLE_OBJECTIVE_SCHEMA,
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "temperature": [0.0, 0.3],
        "max_tokens": [50, 150],
        "use_cot": [True, False],  # Chain of thought
    },
    execution_mode="edge_analytics",
)
def classify_text(text: str) -> str:
    """Text classification with multiple objectives."""

    config = traigent.get_current_config()

    if MOCK:
        # Simulate different performance characteristics
        model = config.get("model", "gpt-3.5-turbo")
        use_cot = config.get("use_cot", False)

        # Simulate latency
        if "gpt-4o" in model:
            time.sleep(0.1)  # Slower
        else:
            time.sleep(0.05)  # Faster

        # Simulate accuracy differences
        if use_cot:
            return "positive"  # More accurate with CoT
        return "neutral"

    # Real implementation
    from langchain_openai import ChatOpenAI

    start_time = time.time()

    llm = ChatOpenAI(
        model=config.get("model"),
        temperature=config.get("temperature"),
        model_kwargs={"max_tokens": config.get("max_tokens")},
    )

    # Use chain of thought if enabled
    if config.get("use_cot"):
        prompt = f"""Think step by step to classify this text's sentiment.
        Text: {text}

        Step 1: Identify key emotional words
        Step 2: Consider context
        Step 3: Determine overall sentiment

        Classification (positive/negative/neutral):"""
    else:
        prompt = f"Classify sentiment: {text}\nAnswer (positive/negative/neutral):"

    response = llm.invoke(prompt)

    # Track latency
    latency = time.time() - start_time

    return response.content.strip().lower()


async def main():
    print("🎯 TraiGent Example 4: Multi-Objective Optimization")
    print("=" * 50)
    print("⚖️  Balancing accuracy, cost, and latency\n")

    print("🎯 Objectives and weights:")
    print("  • Accuracy (50%) - Most important")
    print("  • Cost (30%) - Keep expenses low")
    print("  • Latency (20%) - Fast responses\n")

    print("Trade-offs to explore:")
    print("  • GPT-4 is accurate but expensive")
    print("  • GPT-3.5 is fast and cheap but less accurate")
    print("  • Chain-of-thought improves accuracy but increases latency\n")

    # Run optimization
    print("🔍 Finding optimal balance...\n")
    results = await classify_text.optimize(
        algorithm="random", max_trials=12 if not MOCK else 8
    )

    # Display results
    print("\n" + "=" * 50)
    print("🏆 OPTIMAL CONFIGURATION (Best Balance):")
    print("=" * 50)

    best_config = results.best_config
    print(f"Model: {best_config.get('model')}")
    print(f"Temperature: {best_config.get('temperature')}")
    print(f"Max Tokens: {best_config.get('max_tokens')}")
    print(f"Use Chain-of-Thought: {best_config.get('use_cot')}")

    print("\n📊 Performance Metrics:")
    metrics = results.best_metrics
    print(f"Accuracy: {metrics.get('accuracy', 0):.2%}")
    print(f"Cost per call: ${metrics.get('cost', 0):.6f}")
    print(f"Latency: {metrics.get('latency', 0):.3f}s")

    print("\n📊 Trade-off Analysis:")
    if MOCK:
        # Show mock trade-off insights
        if "gpt-4" in str(best_config.get("model")):
            print("✅ High accuracy achieved")
            print("⚠️  Higher cost accepted for quality")
        elif "gpt-3.5" in str(best_config.get("model")):
            print("✅ Cost-effective solution")
            print("ℹ️  Balanced accuracy for the price")

        if best_config.get("use_cot"):
            print("✅ Chain-of-thought improves reasoning")
            print("⚠️  Slightly slower responses")

    print("\n💡 Multi-objective optimization finds the best compromise!")


if __name__ == "__main__":
    asyncio.run(main())
