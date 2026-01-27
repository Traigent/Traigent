#!/usr/bin/env python
"""
Traigent Quickstart Example 3: Custom Objectives

This example shows how to define custom objective weights and orientations.
Based on the README.md custom objectives example.

Run with (from repo root):
    TRAIGENT_MOCK_LLM=true .venv/bin/python examples/quickstart/03_custom_objectives.py
"""

import asyncio
import os
from pathlib import Path

# Ensure mock mode for testing without API keys
os.environ.setdefault("TRAIGENT_MOCK_LLM", "true")

# Set results folder to local directory
os.environ.setdefault(
    "TRAIGENT_RESULTS_FOLDER", str(Path(__file__).parent / ".traigent_results")
)

ROOT_DIR = Path(__file__).resolve().parents[2]
os.environ.setdefault("TRAIGENT_DATASET_ROOT", str(ROOT_DIR))

import traigent  # noqa: E402
from traigent.api.decorators import EvaluationOptions, ExecutionOptions  # noqa: E402
from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema  # noqa: E402

os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")


# Path to dataset (shared quickstart dataset)
DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "datasets" / "quickstart" / "qa_samples.jsonl"
)

# Define custom objectives with explicit weights and orientations
custom_objectives = ObjectiveSchema.from_objectives(
    [
        ObjectiveDefinition("accuracy", orientation="maximize", weight=0.7),
        ObjectiveDefinition("cost", orientation="minimize", weight=0.3),
    ]
)


@traigent.optimize(
    objectives=custom_objectives,
    configuration_space={
        # Use tuple for continuous ranges, list for categorical
        "temperature": (0.0, 1.0),  # Continuous range
        "top_p": (0.1, 1.0),  # Continuous range
        "model": ["gpt-3.5-turbo", "gpt-4o-mini"],  # Categorical
    },
    evaluation=EvaluationOptions(eval_dataset=str(DATASET_PATH)),
    execution=ExecutionOptions(execution_mode="edge_analytics"),
    max_trials=5,
)
def weighted_agent(question: str) -> str:
    """Agent optimized with custom-weighted objectives.

    This demonstrates:
    - Custom objective weights (70% accuracy, 30% cost)
    - Continuous parameter spaces (temperature, top_p)
    - Mixed configuration space (continuous + categorical)
    """
    # Mock response for demo
    mock_answers = {
        "What is the capital of France?": "Paris",
        "What is 2 + 2?": "4",
        "Who wrote Romeo and Juliet?": "William Shakespeare",
    }
    return mock_answers.get(question, "I don't know")


async def main():
    print("=" * 60)
    print("Traigent Quickstart: Custom Objectives Example")
    print("=" * 60)
    print()

    print("Custom Objectives Configuration:")
    print("-" * 40)
    print("  accuracy: maximize, weight=0.7 (70%)")
    print("  cost:     minimize, weight=0.3 (30%)")
    print()

    print("Configuration Space:")
    print("-" * 40)
    print("  temperature: continuous (0.0, 1.0)")
    print("  top_p:       continuous (0.1, 1.0)")
    print("  model:       categorical [gpt-3.5-turbo, gpt-4o-mini]")
    print()

    print(f"Dataset: {DATASET_PATH}")
    print(f"Mock mode: {os.environ.get('TRAIGENT_MOCK_LLM', 'false')}")
    print()

    # Run optimization
    print("Starting multi-objective optimization...")
    print("(Balancing accuracy vs cost with custom weights)")
    print()

    results = await weighted_agent.optimize()

    print()
    print("=" * 60)
    print("Optimization Complete!")
    print("=" * 60)
    print()
    print(f"Best Score: {results.best_score}")
    print(f"Best Configuration: {results.best_config}")
    print()

    # Explain the trade-off
    print("Understanding the Results:")
    print("-" * 40)
    print("  The optimizer found a configuration that balances:")
    print("  - High accuracy (weighted 70%)")
    print("  - Low cost (weighted 30%)")
    print()
    print("  This means the solution prioritizes accuracy")
    print("  but still considers cost optimization.")
    print()

    print("Try adjusting weights to see different trade-offs:")
    print("  - accuracy=0.5, cost=0.5 -> Equal balance")
    print("  - accuracy=0.3, cost=0.7 -> Cost-focused")
    print("  - accuracy=0.9, cost=0.1 -> Accuracy-focused")


if __name__ == "__main__":
    asyncio.run(main())
