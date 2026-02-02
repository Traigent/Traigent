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

from utils.helpers import print_optimization_config, print_results_table
from utils.mock_answers import (
    ANSWERS,
    DEFAULT_MOCK_MODEL,
    configure_mock_notice,
    get_mock_accuracy,
    get_mock_cost,
    normalize_text,
    set_mock_model,
)
from traigent import TraigentConfig

os.environ.setdefault("TRAIGENT_MOCK_LLM", "true")
os.environ.setdefault("TRAIGENT_OFFLINE_MODE", "true")

traigent.initialize(config=TraigentConfig(execution_mode="edge_analytics", minimal_logging=True))

# Dataset path relative to this file
DATASETS = Path(__file__).parent.parent / "datasets"
SIMULATED_BEST = {"model": "gpt-4o", "temperature": 0.1, "accuracy": 0.80}
MOCK_MODE_CONFIG = {
    "base_accuracy": SIMULATED_BEST["accuracy"],
    "variance": 0.0,
    "random_seed": 42,
}
OBJECTIVES = ["accuracy", "cost"]
CONFIG_SPACE = {
    "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "gpt-4.1-nano"],
    "temperature": [0.1, 0.7],
}


def results_match_score(output: str, expected: str, config: dict | None = None, **_) -> float:
    """Simple scoring: 1.0 if expected answer appears in output, else 0.0.

    In mock mode, returns model-dependent accuracy (gpt-4o > gpt-4o-mini > gpt-3.5-turbo).
    In real mode, checks if the expected answer appears in the LLM output.
    """
    if os.getenv("TRAIGENT_MOCK_LLM", "").lower() in ("1", "true", "yes"):
        model = config.get("model", DEFAULT_MOCK_MODEL) if config else DEFAULT_MOCK_MODEL
        return get_mock_accuracy(model, "simple_qa")
    if output is None or expected is None:
        return 0.0
    # Simple contains check (case-insensitive)
    return 1.0 if expected.strip().lower() in str(output).lower() else 0.0


@traigent.optimize(
    eval_dataset=str(DATASETS / "simple_questions.jsonl"),
    objectives=OBJECTIVES,
    scoring_function=results_match_score,
    configuration_space=CONFIG_SPACE,
    injection_mode="context",  # default, added explicitly for clarity
    execution_mode="bla",
    mock_mode_config=MOCK_MODE_CONFIG,
)
def answer_question(question: str) -> str:
    """Simple Q&A function with mock responses."""
    config = traigent.get_config()
    set_mock_model(config.get("model", DEFAULT_MOCK_MODEL))
    return ANSWERS.get(normalize_text(question), "I don't know")


async def main() -> None:
    print("Traigent Example 1: Simple Optimization")
    print("=" * 50)
    configure_mock_notice("01_tuning_qa.py")
    print_optimization_config(OBJECTIVES, CONFIG_SPACE)

    results = await answer_question.optimize(algorithm="grid", max_trials=8, random_seed=42)

    print_results_table(results, CONFIG_SPACE, OBJECTIVES, is_mock=True, task_type="simple_qa")

    print("\nBest Configuration Found:")
    print(f"  Model: {results.best_config.get('model')}")
    print(f"  Temperature: {results.best_config.get('temperature')}")
    print("\nPerformance:")
    print(f"  Accuracy: {results.best_metrics.get('accuracy', 0):.2%}")
    best_model = results.best_config.get("model", DEFAULT_MOCK_MODEL)
    est_cost = get_mock_cost(best_model, "simple_qa", dataset_size=20)
    print(f"  Est. Cost: ${est_cost:.4f} (for 20 examples)")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130)
