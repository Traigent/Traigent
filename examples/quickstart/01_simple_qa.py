#!/usr/bin/env python
"""
Traigent Quickstart Example 1: Simple Q&A Agent

Run with (from repo root):
    TRAIGENT_MOCK_LLM=true python examples/quickstart/01_simple_qa.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Enable mock mode only when no API key is available
if not os.environ.get("OPENAI_API_KEY"):
    os.environ.setdefault("TRAIGENT_MOCK_LLM", "true")

# Set results folder to local directory
os.environ.setdefault(
    "TRAIGENT_RESULTS_FOLDER", str(Path(__file__).parent / ".traigent_results")
)

ROOT_DIR = Path(__file__).resolve().parents[2]
os.environ.setdefault("TRAIGENT_DATASET_ROOT", str(ROOT_DIR))

# Allow running from repo root without installation
try:
    import traigent
except ImportError:
    sys.path.insert(0, str(ROOT_DIR))
    import traigent

from traigent.api.decorators import EvaluationOptions, ExecutionOptions  # noqa: E402

os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")


# Path to dataset (relative to this file)
DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "datasets" / "quickstart" / "qa_samples.jsonl"
)


@traigent.optimize(
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],  # Tuned Variable #1
        "temperature": [0.1, 0.5, 0.9],  # Tuned Variable #2
    },
    objectives=["accuracy", "cost"],  # What to optimize for
    evaluation=EvaluationOptions(eval_dataset=str(DATASET_PATH)),
    execution=ExecutionOptions(execution_mode="edge_analytics"),
    max_trials=5,  # Limit trials for quick demo
)
def simple_qa_agent(question: str) -> str:
    """Simple Q&A agent with Tuned Variables.

    In mock mode, this returns a simulated response.
    With real API keys, it calls the actual LLM.
    """
    if os.environ.get("TRAIGENT_MOCK_LLM", "").lower() in ("1", "true", "yes"):
        mock_answers = {
            "What is the capital of France?": "Paris",
            "What is 2 + 2?": "4",
            "Who wrote Romeo and Juliet?": "William Shakespeare",
        }
        return mock_answers.get(question, "I don't know")

    from langchain_openai import ChatOpenAI

    config = traigent.get_config()
    llm = ChatOpenAI(
        model=config.get("model", "gpt-3.5-turbo"),
        temperature=config.get("temperature", 0.7),
    )
    response = llm.invoke(f"Question: {question}\nAnswer:")
    return response.content


async def main():
    print("=" * 60)
    print("Traigent Quickstart: Simple Q&A Agent Optimization")
    print("=" * 60)
    print()

    print(f"Dataset: {DATASET_PATH}")
    print(f"Mock mode: {os.environ.get('TRAIGENT_MOCK_LLM', 'false')}")
    print()

    # Run optimization
    print("Starting optimization...")
    results = await simple_qa_agent.optimize()

    print()
    print("=" * 60)
    print("Optimization Complete!")
    print("=" * 60)
    print()
    print(f"Best Score: {results.best_score}")
    print(f"Best Configuration: {results.best_config}")
    print()

    # Show all trial results if available
    if hasattr(results, "trials") and results.trials:
        print("All Trials:")
        print("-" * 40)
        for i, trial in enumerate(results.trials, 1):
            # Handle different trial result formats
            score = getattr(trial, "score", None) or getattr(trial, "metrics", {}).get(
                "score", "N/A"
            )
            config = getattr(trial, "config", getattr(trial, "configuration", {}))
            print(f"  Trial {i}: {config} -> score={score}")

    print()
    print("Next steps:")
    print("  1. Try with real API keys (set OPENAI_API_KEY)")
    print("  2. Disable mock mode (TRAIGENT_MOCK_LLM=false)")
    print("  3. Explore more examples in examples/core/")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130) from None
