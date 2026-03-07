#!/usr/bin/env python3
"""Example 6: Custom Evaluator - Define your own success metrics."""

import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import print_optimization_config, print_results_table
from utils.mock_answers import (
    DEFAULT_MOCK_MODEL,
    configure_mock_notice,
    get_mock_accuracy,
    get_mock_cost,
    set_mock_model,
)

import traigent
from traigent import TraigentConfig

os.environ.setdefault("TRAIGENT_MOCK_LLM", "true")

traigent.initialize(config=TraigentConfig(execution_mode="edge_analytics", minimal_logging=True))

# Dataset path relative to this file
DATASETS = Path(__file__).parent.parent / "datasets"
SIMULATED_BEST = {
    "model": "gpt-3.5-turbo",
    "temperature": 0.0,
    "style": "documented",
    "accuracy": 0.8890,
}
MOCK_MODE_CONFIG = {
    "base_accuracy": SIMULATED_BEST["accuracy"],
    "variance": 0.0,
    "random_seed": 42,
}
OBJECTIVES = ["accuracy", "cost"]
CONFIG_SPACE = {
    "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
    "temperature": [0.0, 0.2, 0.5],
    "style": ["verbose", "concise", "documented"],
}
_GENERATION_COUNTS = {"verbose": 0, "concise": 0, "documented": 0}
_LOG_EVERY = int(os.getenv("TRAIGENT_GEN_LOG_EVERY", "0"))
_DATASET_WARNING_FILTER_ADDED = False


def code_evaluator(output: str, expected: str, config: dict | None = None, **_) -> float:
    """Custom evaluator for code generation quality.

    In mock mode, returns model-dependent accuracy.
    """
    if os.getenv("TRAIGENT_MOCK_LLM", "").lower() in ("1", "true", "yes"):
        model = config.get("model", DEFAULT_MOCK_MODEL) if config else DEFAULT_MOCK_MODEL
        return get_mock_accuracy(model, "code_generation")
    score = 0.0
    if "def " in output:
        score += 0.4
    if output.strip() and "error" not in output.lower():
        score += 0.3
    if '"""' in output or "#" in output:
        score += 0.3
    return min(score, 1.0)


def _suppress_code_gen_warning() -> None:
    global _DATASET_WARNING_FILTER_ADDED
    if _DATASET_WARNING_FILTER_ADDED:
        return
    base_logger = logging.getLogger("traigent.evaluators.base")

    class _Filter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            message = record.getMessage()
            if "code_gen.jsonl" in message and "has no expected outputs" in message:
                return False
            return True

    base_logger.addFilter(_Filter())
    _DATASET_WARNING_FILTER_ADDED = True


@traigent.optimize(
    eval_dataset=str(DATASETS / "code_gen.jsonl"),
    objectives=OBJECTIVES,
    scoring_function=code_evaluator,
    configuration_space=CONFIG_SPACE,
    injection_mode="context",  # default, added explicitly for clarity
    execution_mode="edge_analytics",
    mock_mode_config=MOCK_MODE_CONFIG,
)
def generate_code(task: str) -> str:
    """Generate code with configurable style."""
    config = traigent.get_config()
    model = config.get("model", DEFAULT_MOCK_MODEL)
    style = config.get("style", "concise")

    set_mock_model(model)

    _GENERATION_COUNTS[style] += 1
    if _LOG_EVERY > 0:
        if _GENERATION_COUNTS[style] % _LOG_EVERY == 0:
            print(f"  Generating {style} code... ({_GENERATION_COUNTS[style]})")
    elif _GENERATION_COUNTS[style] == 1:
        print(f"  Generating {style} code...")

    if style == "verbose":
        return '''def calculate_sum(numbers):
    """Calculate the sum of a list."""
    total = 0
    for n in numbers:
        total += n
    return total'''
    elif style == "documented":
        return '''def calculate_sum(numbers):
    # Sum all numbers
    return sum(numbers)'''
    return "def calculate_sum(nums): return sum(nums)"


async def main() -> None:
    print("Traigent Example 6: Custom Evaluator")
    print("=" * 50)
    configure_mock_notice("06_custom_evaluator.py")
    print("Scoring: function def (40%), no errors (30%), documentation (30%).")
    print_optimization_config(OBJECTIVES, CONFIG_SPACE)

    _suppress_code_gen_warning()

    results = await generate_code.optimize(
        algorithm="grid", max_trials=8, random_seed=42
    )

    print_results_table(results, CONFIG_SPACE, OBJECTIVES, is_mock=True, task_type="code_generation", dataset_size=10)

    print("\nBest Configuration Found:")
    print(f"  Model: {results.best_config.get('model')}")
    print(f"  Temperature: {results.best_config.get('temperature')}")
    print(f"  Style: {results.best_config.get('style')}")
    print("\nPerformance:")
    print(f"  Custom Score: {results.best_metrics.get('accuracy', 0):.2%}")
    best_model = results.best_config.get("model", DEFAULT_MOCK_MODEL)
    est_cost = get_mock_cost(best_model, "code_generation", dataset_size=10)
    print(f"  Est. Cost: ${est_cost:.4f} (for 10 examples)")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130)
