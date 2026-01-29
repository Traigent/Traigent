#!/usr/bin/env python3
"""
Traigent Python Orchestrator - JS Agent Parallel Execution Demo

This script demonstrates using the Python @traigent.optimize decorator to
orchestrate optimization of a JavaScript agent with parallel trial execution.

Key Features Shown:
- Python orchestrator managing JS trial execution
- Parallel trial execution via JSProcessPool (js_parallel_workers=4)
- Cost budget enforcement with early stopping
- Integration with Traigent cloud backend

Prerequisites:
1. Build the JS demo: cd demos/agent-app && npm run build
2. Install Traigent Python SDK: pip install traigent

Usage:
    python run_with_python.py

Environment Variables:
    TRAIGENT_API_KEY: Your Traigent API key (optional for local mode)
    TRAIGENT_EXECUTION_MODE: "edge_analytics" (default) or "mock"
"""

import asyncio
import os
import sys
from pathlib import Path

# Ensure we can import traigent from the SDK
sdk_path = Path(__file__).parent.parent.parent.parent / "Traigent"
if sdk_path.exists():
    sys.path.insert(0, str(sdk_path))

import traigent


async def main():
    """Run optimization with parallel JS trial execution."""

    # Initialize Traigent SDK
    traigent.initialize(
        execution_mode=os.getenv("TRAIGENT_EXECUTION_MODE", "edge_analytics"),
    )

    # Path to the compiled JS trial module
    # Relative to the demos/agent-app directory
    js_module_path = "./dist/trial.js"

    # Define the optimization with parallel JS execution
    @traigent.optimize(
        # Execution configuration - use Node.js runtime with parallel workers
        execution={
            "runtime": "node",
            "js_module": js_module_path,
            "js_function": "runTrial",
            "js_timeout": 30.0,  # 30 second timeout per trial
            "js_parallel_workers": 4,  # Run 4 Node.js processes in parallel
        },
        # Dataset for sampling (indices sent to JS runtime)
        eval_dataset="./dataset.jsonl",
        # Configuration space - must match AgentConfig in agent.ts
        configuration_space={
            "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
            "temperature": [0.0, 0.3, 0.5, 0.7, 1.0],
            "system_prompt": ["concise", "detailed", "cot"],
        },
        # Objectives to optimize
        objectives=["accuracy", "cost"],
        # Runtime overrides
        max_trials=12,  # Run 12 trials total
        plateau_window=3,  # Stop if no improvement for 3 trials
    )
    def sentiment_agent(text: str) -> str:
        """
        Placeholder for the sentiment classification agent.

        This function is not called directly - the JS trial function handles
        actual execution. This decorator wrapper enables Python orchestration.

        Args:
            text: Input text for sentiment classification

        Returns:
            Sentiment label: "positive", "negative", or "neutral"
        """
        # The actual implementation is in agent.ts
        # This is just a type signature for the Python decorator
        pass

    # Print banner
    print("\n" + "=" * 70)
    print("  TRAIGENT PYTHON ORCHESTRATOR - PARALLEL JS EXECUTION DEMO")
    print("=" * 70)
    print(f"\nJS Module: {js_module_path}")
    print("Parallel Workers: 4")
    print("Max Trials: 12")
    print("Cost Budget: $0.10")
    print("\nSearch Space:")
    print("  - Models: gpt-3.5-turbo, gpt-4o-mini, gpt-4o")
    print("  - Temperature: 0.0, 0.3, 0.5, 0.7, 1.0")
    print("  - Prompt Type: concise, detailed, cot")
    print("\n" + "-" * 70)

    # Run optimization
    print("\nStarting parallel optimization...\n")

    try:
        result = await sentiment_agent.optimize()

        # Print results
        print("\n" + "=" * 70)
        print("  OPTIMIZATION COMPLETE")
        print("=" * 70)

        if result.best_config:
            print("\nBest Configuration Found:")
            print(f"  Model:       {result.best_config.get('model')}")
            print(f"  Temperature: {result.best_config.get('temperature')}")
            print(f"  Prompt:      {result.best_config.get('system_prompt')}")

        if result.best_metrics:
            print("\nBest Metrics:")
            for name, value in result.best_metrics.items():
                if isinstance(value, float):
                    if name == "cost":
                        print(f"  {name}: ${value:.6f}")
                    elif name == "accuracy":
                        print(f"  {name}: {value * 100:.1f}%")
                    else:
                        print(f"  {name}: {value:.2f}")
                else:
                    print(f"  {name}: {value}")

        print(f"\nTrials Completed: {result.trials_completed}")
        print(f"Stop Reason: {result.stop_reason}")

        if hasattr(result, "total_cost"):
            print(f"Total Cost: ${result.total_cost:.6f}")

    except Exception as e:
        print(f"\nError during optimization: {e}")
        raise

    print("\nDemo complete!")


if __name__ == "__main__":
    asyncio.run(main())
