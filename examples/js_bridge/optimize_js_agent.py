#!/usr/bin/env python3
"""
Python-Orchestrated JS Agent Optimization Example

This example demonstrates how to use Traigent's Python SDK to orchestrate
optimization of a JavaScript agent. The Python SDK handles:

- Configuration sampling (grid, random, bayesian, optuna)
- Dataset management and subset selection
- Parallel trial execution (via process pool)
- Budget guardrails and early stopping
- Result aggregation and best config selection

The JS agent handles:
- Actual trial execution (LLM calls, processing)
- Computing metrics per trial

Prerequisites:
1. Provide a built JS trial module via TRAIGENT_JS_MODULE, or keep a sibling
   `traigent-js` checkout with the demo built under `demos/agent-app/dist/`
2. Set environment: export TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true
3. Run: python optimize_js_agent.py

See docs/guides/js-bridge.md for full documentation.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import traigent
from traigent.api.decorators import ExecutionOptions

# Initialize Traigent in edge_analytics mode (local execution with analytics)
traigent.initialize(execution_mode="edge_analytics")

# Path to the compiled JS module (override with TRAIGENT_JS_MODULE)
DEFAULT_JS_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "traigent-js"
    / "demos"
    / "agent-app"
    / "dist"
    / "run-trial.js"
)
JS_MODULE_PATH = os.getenv("TRAIGENT_JS_MODULE", str(DEFAULT_JS_MODULE_PATH))
DEFAULT_JS_RUNNER_PATH = (
    Path(__file__).resolve().parents[3] / "traigent-js" / "dist" / "cli" / "runner.js"
)
JS_RUNNER_PATH = os.getenv("TRAIGENT_JS_RUNNER", str(DEFAULT_JS_RUNNER_PATH))
USE_NPX = not os.path.exists(JS_RUNNER_PATH)

# Create a simple dataset file for the demo
DATASET_PATH = Path(__file__).parent / "sentiment_dataset.jsonl"


def create_demo_dataset():
    """Create a simple JSONL dataset for the demo."""
    import json

    examples = [
        {"input": {"text": "This product is amazing!"}, "output": "positive"},
        {
            "input": {"text": "Terrible experience, very disappointed."},
            "output": "negative",
        },
        {"input": {"text": "The item arrived on time."}, "output": "neutral"},
        {"input": {"text": "Best purchase I've ever made!"}, "output": "positive"},
        {"input": {"text": "Complete waste of money."}, "output": "negative"},
        {"input": {"text": "It does what it's supposed to do."}, "output": "neutral"},
        {"input": {"text": "Absolutely love it!"}, "output": "positive"},
        {"input": {"text": "Would not recommend."}, "output": "negative"},
        {"input": {"text": "Average product."}, "output": "neutral"},
        {"input": {"text": "Five stars! Perfect!"}, "output": "positive"},
    ]

    with open(DATASET_PATH, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


# Create dataset at import time (before decorator validation)
create_demo_dataset()


# Define the optimizable function with JS runtime
@traigent.optimize(
    # Configuration space: what parameters to tune
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "temperature": [0.0, 0.3, 0.5, 0.7, 1.0],
        "system_prompt": ["concise", "detailed", "cot"],
    },
    # Objectives: what to optimize for
    objectives=["accuracy", "cost"],
    # Evaluation settings
    evaluation={
        "eval_dataset": str(DATASET_PATH),
    },
    # Execution settings for JS runtime
    execution=ExecutionOptions(
        runtime="node",  # Use Node.js runtime
        js_module=JS_MODULE_PATH,  # Path to compiled JS module
        js_function="runTrial",  # Exported function name
        js_timeout=60.0,  # 60 second timeout per trial
        js_parallel_workers=2,  # Run 2 Node.js processes in parallel
        js_use_npx=USE_NPX,
        js_runner_path=None if USE_NPX else JS_RUNNER_PATH,
    ),
    # Injection mode must be PARAMETER for JS runtime
    injection={"injection_mode": "parameter"},
)
def sentiment_classifier(text: str, config: dict | None = None) -> str:
    """Classify sentiment of text.

    Note: This function body is NOT executed when runtime="node".
    The JS module handles all trial execution. This is just a placeholder
    to define the function signature.

    Args:
        text: Input text to classify
        config: Configuration injected by Traigent (not used in JS mode)
    """
    pass


async def main():
    """Run the optimization."""
    print("=" * 70)
    print("TRAIGENT JS BRIDGE DEMO - Python-Orchestrated Optimization")
    print("=" * 70)

    # Check if JS module exists
    if not os.path.exists(JS_MODULE_PATH):
        print(f"\nJS module not found at: {JS_MODULE_PATH}")
        print("\nSet TRAIGENT_JS_MODULE to a compiled run-trial.js file, or build")
        print("the companion JS demo in a sibling ../traigent-js checkout:")
        print("  cd ../traigent-js/demos/agent-app")
        print("  npm install && npm run build")
        return

    print(f"\nJS Module: {JS_MODULE_PATH}")
    print(f"JS Runner: {'npx traigent-js' if USE_NPX else JS_RUNNER_PATH}")
    print(f"Dataset: {DATASET_PATH}")
    print("\nConfiguration Space:")
    print("  Models:      gpt-3.5-turbo, gpt-4o-mini, gpt-4o")
    print("  Temperature: 0.0, 0.3, 0.5, 0.7, 1.0")
    print("  Prompts:     concise, detailed, cot")
    print("\nObjectives: accuracy (maximize), cost (minimize)")
    print("\n" + "-" * 70)

    # Run optimization
    result = await sentiment_classifier.optimize(
        algorithm="random",  # Use random search
        max_trials=6,  # Run 6 trials
    )

    # Print results
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(
        f"\nTotal trials: {len(result.trials) if hasattr(result, 'trials') else 'N/A'}"
    )
    print(f"Stop reason: {getattr(result, 'stop_reason', 'N/A')}")

    if hasattr(result, "best_config") and result.best_config:
        print("\nBest Configuration:")
        for key, value in result.best_config.items():
            print(f"  {key}: {value}")

    if hasattr(result, "best_metrics") and result.best_metrics:
        print("\nBest Metrics:")
        for key, value in result.best_metrics.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
