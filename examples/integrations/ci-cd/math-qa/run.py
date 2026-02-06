#!/usr/bin/env python3
"""CI/CD runner for math Q&A example with proper SDK usage."""

import asyncio
import json
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "math_qa.jsonl"

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from math_qa import solve_arithmetic

from traigent.evaluators.base import Dataset  # noqa: E402
from traigent.evaluators.local import LocalEvaluator  # noqa: E402

os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")


def load_saved_config():
    """Load the saved production configuration."""
    config_path = Path(__file__).parent / "saved_config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    else:
        # Default config if none exists
        return {
            "model": "gpt-3.5-turbo",
            "temperature": 0.0,
            "prompt_style": "direct",
            "max_tokens": 50,
        }


async def evaluate_config(config, output_path):
    """Evaluate a specific configuration using SDK's evaluator directly.

    This is the proper way to evaluate without optimization.
    """
    # Load dataset
    dataset = Dataset.from_jsonl(str(DATASET_PATH))

    # Create evaluator
    evaluator = LocalEvaluator()

    # Apply config to function
    solve_arithmetic.set_config(config)

    # Evaluate directly - no optimization needed
    result = await evaluator.evaluate(
        func=solve_arithmetic.func,  # Get unwrapped function
        config=config,
        dataset=dataset,
    )

    # Format results in standard structure
    results = {
        "mode": "evaluation",
        "config": config,
        "metrics": {
            "accuracy": result.metrics.get("accuracy", 0.0),
            "cost": result.metrics.get("cost", 0.0),
            "response_time": result.metrics.get("response_time_ms", 0.0),
            "total_examples": len(dataset),
            "successful_examples": result.metrics.get("successful_examples", 0),
        },
        "example_results": (
            [
                {
                    "input": ex.input,
                    "output": ex.output,
                    "expected": ex.expected,
                    "metrics": ex.metrics,
                }
                for ex in result.example_results[:3]  # Include first 3 for debugging
            ]
            if hasattr(result, "example_results")
            else []
        ),
    }

    # Save results
    output_dir = Path("results-ci/math")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Evaluation complete. Results saved to {output_dir / output_path}")
    return results


async def run_optimization(max_trials=10):
    """Run full optimization to find better configurations."""
    # Use the decorated function's optimize method
    results = await solve_arithmetic.optimize(max_trials=max_trials)

    # Format results
    output = {
        "mode": "optimization",
        "best_config": results.best_config,
        "best_score": results.best_score,
        "objectives": results.objectives,
        "total_trials": len(results.trials),
        "metrics": (
            {
                "accuracy": results.best_metrics.get("accuracy", 0.0),
                "cost": results.best_metrics.get("cost", 0.0),
                "response_time": results.best_metrics.get("response_time_ms", 0.0),
            }
            if hasattr(results, "best_metrics")
            else {}
        ),
        "improvement_potential": {},
    }

    # Calculate improvement potential vs saved config
    saved_config = load_saved_config()
    if saved_config != results.best_config:
        output["improvement_potential"]["config_differs"] = True
        output["improvement_potential"]["changes"] = {
            k: {"from": saved_config.get(k), "to": v}
            for k, v in results.best_config.items()
            if saved_config.get(k) != v
        }

    # Save results
    output_dir = Path("results-ci/math")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "tuned.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"Optimization complete. Results saved to {output_dir}/tuned.json")
    return output


def main():
    """Main entry point with mode selection."""
    mode = os.environ.get("MODE", "eval").lower()

    # Set mock mode for CI
    if os.environ.get("CI") == "true":
        os.environ["TRAIGENT_MOCK_LLM"] = "true"

    print(f"Running in {mode} mode...")
    print(f"Mock mode: {os.environ.get('TRAIGENT_MOCK_LLM', 'false')}")

    if mode == "eval":
        # Evaluate saved configuration
        config = load_saved_config()
        print(f"Evaluating config: {config}")

        # Run async evaluation
        result = asyncio.run(evaluate_config(config, "current.json"))

        print(f"Accuracy: {result['metrics']['accuracy']:.2%}")
        print(f"Cost: ${result['metrics']['cost']:.4f}")
        print(f"Response time: {result['metrics']['response_time']:.1f}ms")

    elif mode == "tune":
        # Run optimization
        max_trials = int(os.environ.get("MAX_TRIALS", "10"))
        print(f"Running optimization with {max_trials} trials...")

        result = asyncio.run(run_optimization(max_trials))

        print(f"Best config found: {result['best_config']}")
        print(f"Best score: {result['best_score']:.3f}")

        if result["improvement_potential"]:
            print("\n⚠️  Better configuration found!")
            print("Changes needed in saved_config.json:")
            for param, change in result["improvement_potential"]["changes"].items():
                print(f"  {param}: {change['from']} → {change['to']}")

    else:
        print(f"Unknown mode: {mode}")
        print("Use MODE=eval or MODE=tune")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130) from None
