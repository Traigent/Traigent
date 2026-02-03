#!/usr/bin/env python3
"""Example 4: Multi-Objective - Balance accuracy, cost, and latency."""

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import traigent
from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema

from utils.helpers import print_optimization_config, print_results_table
from utils.mock_answers import (
    CLASSIFICATION_LABELS,
    DEFAULT_MOCK_MODEL,
    configure_mock_notice,
    get_mock_accuracy,
    get_mock_cost,
    get_mock_latency,
    normalize_text,
    set_mock_model,
)
from traigent import TraigentConfig

os.environ.setdefault("TRAIGENT_MOCK_LLM", "true")
os.environ.setdefault("TRAIGENT_OFFLINE_MODE", "true")

traigent.initialize(config=TraigentConfig(execution_mode="edge_analytics", minimal_logging=True))

# Dataset path relative to this file
DATASETS = Path(__file__).parent.parent / "datasets"
SIMULATED_BEST = {
    "model": "gpt-4o",
    "temperature": 0.0,
    "use_cot": True,
    "accuracy": 0.95,
    "cost": 0.000150,  # Simulated cost per evaluation (realistic for gpt-4o)
    "latency": 0.065,  # Simulated latency in seconds
}
MOCK_MODE_CONFIG = {
    "base_accuracy": SIMULATED_BEST["accuracy"],
    "variance": 0.0,
    "random_seed": 42,
}
CONFIG_SPACE = {
    "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
    "temperature": [0.0, 0.3],
    "use_cot": [True, False],
}

OBJECTIVES = ObjectiveSchema.from_objectives([
    ObjectiveDefinition("accuracy", orientation="maximize", weight=0.5),
    ObjectiveDefinition("cost", orientation="minimize", weight=0.3),
    ObjectiveDefinition("latency", orientation="minimize", weight=0.2),
])


def mock_accuracy_score(output: str, expected: str, config: dict | None = None, **_) -> float:
    """Scoring function with config-dependent mock accuracy."""
    if os.getenv("TRAIGENT_MOCK_LLM", "").lower() in ("1", "true", "yes"):
        model = config.get("model", DEFAULT_MOCK_MODEL) if config else DEFAULT_MOCK_MODEL
        temperature = config.get("temperature") if config else None
        use_cot = config.get("use_cot") if config else None
        return get_mock_accuracy(model, "classification", temperature, use_cot)
    if output is None or expected is None:
        return 0.0
    return 1.0 if str(output).strip().lower() == str(expected).strip().lower() else 0.0


@traigent.optimize(
    eval_dataset=str(DATASETS / "classification.jsonl"),
    objectives=OBJECTIVES,
    scoring_function=mock_accuracy_score,
    configuration_space=CONFIG_SPACE,
    injection_mode="context",  # default, added explicitly for clarity
    execution_mode="edge_analytics",
    mock_mode_config=MOCK_MODE_CONFIG,
)
def classify_text(text: str) -> str:
    """Text classification with multiple objectives."""
    config = traigent.get_config()
    model = config.get("model", DEFAULT_MOCK_MODEL)
    use_cot = config.get("use_cot", False)

    set_mock_model(model)

    # Simulate latency differences
    if "gpt-4o" in model:
        time.sleep(0.05)
    else:
        time.sleep(0.02)
    if use_cot:
        time.sleep(0.01)
    return CLASSIFICATION_LABELS.get(normalize_text(text), "neutral")


async def main() -> None:
    print("Traigent Example 4: Multi-Objective Optimization")
    print("=" * 50)
    configure_mock_notice("04_multi_objective.py")
    print("Balancing accuracy (50%), cost (30%), latency (20%).")
    print_optimization_config(OBJECTIVES, CONFIG_SPACE)

    results = await classify_text.optimize(
        algorithm="random", max_trials=8, random_seed=42
    )

    print_results_table(results, CONFIG_SPACE, OBJECTIVES, is_mock=True, task_type="classification")

    print("\nBest Configuration Found:")
    print(f"  Model: {results.best_config.get('model')}")
    print(f"  Temperature: {results.best_config.get('temperature')}")
    print(f"  Chain-of-Thought: {results.best_config.get('use_cot')}")

    print("\nPerformance:")
    print(f"  Accuracy: {results.best_metrics.get('accuracy', 0):.2%}")
    best_model = results.best_config.get("model", DEFAULT_MOCK_MODEL)
    est_cost = get_mock_cost(best_model, "classification", dataset_size=20)
    est_latency = get_mock_latency(best_model, "classification")
    print(f"  Est. Cost: ${est_cost:.4f} (for 20 examples)")
    print(f"  Est. Latency: {est_latency:.3f}s (per call)")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130)
