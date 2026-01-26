#!/usr/bin/env python3
"""Example 1: Basic QA Tuning - Model and temperature optimization.

This mock version uses simplified scoring that returns a configured accuracy
in mock mode. The real version (walkthrough/examples/real/01_tuning_qa.py)
uses fuzzy token matching to handle paraphrased LLM responses and achieve 80%+
accuracy with actual API calls.
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import traigent

from utils.mock_answers import ANSWERS, normalize_text, configure_mock_notice
from traigent import TraigentConfig

os.environ.setdefault("TRAIGENT_MOCK_LLM", "true")
os.environ.setdefault("TRAIGENT_OFFLINE_MODE", "true")

traigent.initialize(config=TraigentConfig(execution_mode="edge_analytics", minimal_logging=True))

# Dataset path relative to this file
DATASETS = Path(__file__).parent.parent / "datasets"
SIMULATED_BEST = {"model": "gpt-4o", "temperature": 0.1, "accuracy": 0.80}
MOCK_MODE_CONFIG = {"base_accuracy": SIMULATED_BEST["accuracy"], "variance": 0.0, "random_seed": 42}


def results_match_score(output: str, expected: str, **_) -> float:
    """Simple scoring: 1.0 if expected answer appears in output, else 0.0.

    In mock mode, returns the configured base accuracy.
    In real mode, checks if the expected answer appears in the LLM output.
    """
    if os.getenv("TRAIGENT_MOCK_LLM", "").lower() in ("1", "true", "yes"):
        return MOCK_MODE_CONFIG["base_accuracy"]
    if output is None or expected is None:
        return 0.0
    # Simple contains check (case-insensitive)
    return 1.0 if expected.strip().lower() in str(output).lower() else 0.0


@traigent.optimize(
    eval_dataset=str(DATASETS / "simple_questions.jsonl"),
    objectives=["accuracy", "cost"],
    scoring_function=results_match_score,
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "gpt-4.1-nano"],
        "temperature": [0.1, 0.7],
    },
    injection_mode="context",  # default, added explicitly for clarity
    execution_mode="edge_analytics",
    mock_mode_config=MOCK_MODE_CONFIG,
)
def answer_question(question: str) -> str:
    """Simple Q&A function with mock responses."""
    return ANSWERS.get(normalize_text(question), "I don't know")


async def main() -> None:
    print("Traigent Example 1: Simple Optimization")
    print("=" * 50)
    configure_mock_notice("01_tuning_qa.py")

    # In real mode, use results.best_config and results.best_metrics
    # Example: results.best_config.get("model"), results.best_metrics.get("accuracy")
    _results = await answer_question.optimize(algorithm="grid", max_trials=8, random_seed=42)

    print("\nBest Configuration Found:")
    print(f"  Model: {SIMULATED_BEST['model']}")
    print(f"  Temperature: {SIMULATED_BEST['temperature']}")
    print("\nPerformance:")
    print(f"  Accuracy: {SIMULATED_BEST['accuracy']:.2%}")


if __name__ == "__main__":
    asyncio.run(main())
