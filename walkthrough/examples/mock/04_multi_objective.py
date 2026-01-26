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

from utils.mock_answers import CLASSIFICATION_LABELS, normalize_text, configure_mock_notice
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
MOCK_MODE_CONFIG = {"base_accuracy": SIMULATED_BEST["accuracy"], "variance": 0.0, "random_seed": 42}

OBJECTIVES = ObjectiveSchema.from_objectives([
    ObjectiveDefinition("accuracy", orientation="maximize", weight=0.5),
    ObjectiveDefinition("cost", orientation="minimize", weight=0.3),
    ObjectiveDefinition("latency", orientation="minimize", weight=0.2),
])


def _mock_accuracy() -> float:
    return MOCK_MODE_CONFIG["base_accuracy"]


def mock_accuracy_score(output: str, expected: str, **_) -> float:
    if os.getenv("TRAIGENT_MOCK_LLM", "").lower() in ("1", "true", "yes"):
        return _mock_accuracy()
    if output is None or expected is None:
        return 0.0
    return 1.0 if str(output).strip().lower() == str(expected).strip().lower() else 0.0


@traigent.optimize(
    eval_dataset=str(DATASETS / "classification.jsonl"),
    objectives=OBJECTIVES,
    scoring_function=mock_accuracy_score,
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "temperature": [0.0, 0.3],
        "use_cot": [True, False],
    },
    injection_mode="context",  # default, added explicitly for clarity
    execution_mode="edge_analytics",
    mock_mode_config=MOCK_MODE_CONFIG,
)
def classify_text(text: str) -> str:
    """Text classification with multiple objectives."""
    config = traigent.get_config()
    model = config.get("model", "gpt-3.5-turbo")
    use_cot = config.get("use_cot", False)

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

    # In real mode, use results.best_config and results.best_metrics
    # Example: results.best_config.get("model"), results.best_metrics.get("accuracy")
    _results = await classify_text.optimize(
        algorithm="random", max_trials=8, random_seed=42
    )

    print("\nBest Configuration Found:")
    print(f"  Model: {SIMULATED_BEST['model']}")
    print(f"  Temperature: {SIMULATED_BEST['temperature']}")
    print(f"  Chain-of-Thought: {SIMULATED_BEST['use_cot']}")

    print("\nPerformance:")
    print(f"  Accuracy: {SIMULATED_BEST['accuracy']:.2%}")
    print(f"  Cost: ${SIMULATED_BEST['cost']:.6f}")
    print(f"  Latency: {SIMULATED_BEST['latency']:.3f}s")


if __name__ == "__main__":
    asyncio.run(main())
