#!/usr/bin/env python3
"""Example 8: Privacy Modes - Local-only privacy-first execution (current)."""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

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

import traigent
from traigent import TraigentConfig

os.environ.setdefault("TRAIGENT_MOCK_LLM", "true")

traigent.initialize(config=TraigentConfig(execution_mode="edge_analytics", minimal_logging=True))

# Dataset path relative to this file
DATASETS = Path(__file__).parent.parent / "datasets"
RESULTS_DIR = os.getenv("TRAIGENT_RESULTS_FOLDER", "./local_results")
SIMULATED_BEST = {"model": "gpt-3.5-turbo", "temperature": 0.1}
MOCK_MODE_CONFIG = {"base_accuracy": 0.9, "variance": 0.0, "random_seed": 42}
OBJECTIVES = ["accuracy"]
CONFIG_SPACE = {
    "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
    "temperature": [0.1, 0.5],
}


def results_match_score(output: str, expected: str, config: dict | None = None, **_) -> float:
    """Simple scoring - mock returns model-dependent accuracy."""
    if os.getenv("TRAIGENT_MOCK_LLM", "").lower() in ("1", "true", "yes"):
        model = config.get("model", DEFAULT_MOCK_MODEL) if config else DEFAULT_MOCK_MODEL
        return get_mock_accuracy(model, "privacy_test")
    # Real mode: simple contains check
    if output is None or expected is None:
        return 0.0
    return 1.0 if str(expected).strip().lower() in str(output).lower() else 0.0


@traigent.optimize(
    eval_dataset=str(DATASETS / "simple_questions.jsonl"),
    objectives=OBJECTIVES,
    scoring_function=results_match_score,
    configuration_space=CONFIG_SPACE,
    injection_mode="context",  # default, added explicitly for clarity
    execution_mode="edge_analytics",
    local_storage_path=RESULTS_DIR,
    mock_mode_config=MOCK_MODE_CONFIG,
)
def local_mode(question: str) -> str:
    """Local mode - all data stays on your machine."""
    config = traigent.get_config()
    set_mock_model(config.get("model", DEFAULT_MOCK_MODEL))
    return ANSWERS.get(normalize_text(question), "I don't know")


async def main() -> None:
    print("Traigent Example 8: Privacy Modes (local-only for now)")
    print("=" * 50)
    configure_mock_notice("08_privacy_modes.py")
    print_optimization_config(OBJECTIVES, CONFIG_SPACE)

    print("\nLOCAL - All data stays on your machine")

    results = await local_mode.optimize(
        algorithm="grid", max_trials=2, random_seed=42
    )

    print_results_table(results, CONFIG_SPACE, OBJECTIVES, is_mock=True, task_type="simple_qa")

    print("\nBest Configuration Found:")
    print(f"  Model: {results.best_config.get('model')}")
    print(f"  Temperature: {results.best_config.get('temperature')}")
    print("\nPerformance:")
    print(f"  Accuracy: {results.best_metrics.get('accuracy', 0):.2%}")
    best_model = results.best_config.get("model", DEFAULT_MOCK_MODEL)
    est_cost = get_mock_cost(best_model, "simple_qa", dataset_size=20)
    print(f"  Est. Cost: ${est_cost:.4f} (for 20 examples)")
    print("\nLocal Storage:")
    print(f"  Results stored in: {RESULTS_DIR}")
    print("  Look inside: sessions/ and experiments/ for saved runs")
    print("\nThis walkthrough focuses on privacy-first local execution.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130)
