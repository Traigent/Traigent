#!/usr/bin/env python3
"""Example 4: Multi-Objective - Balance accuracy, cost, and latency."""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import build_results_table_callback, print_optimization_config
from utils.mock_answers import (
    CLASSIFICATION_LABELS,
    DEFAULT_MOCK_MODEL,
    MOCK_TASK_TOKENS,
    configure_mock_notice,
    get_mock_accuracy,
    get_mock_cost,
    get_mock_latency,
    normalize_text,
    set_mock_model,
)

import traigent
from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema

os.environ.setdefault("TRAIGENT_MOCK_LLM", "true")

traigent.initialize(offline=True, minimal_logging=True)

# Dataset path relative to this file
DATASETS = Path(__file__).parent.parent / "datasets"
SIMULATED_BEST = {
    "model": "gpt-4o",
    "temperature": 0.0,
    "instructions": "CoT",
    "accuracy": 0.95,
    # Cost and latency are not listed here: they are simulated per trial from
    # the shared mock tables (get_mock_cost / get_mock_latency below).
}
MOCK_MODE_CONFIG = {
    "base_accuracy": SIMULATED_BEST["accuracy"],
    "variance": 0.0,
}
CONFIG_SPACE = {
    "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
    "prompt": ["v1", "v2"],
    "temperature": [0.0, 0.3],
    "instructions": ["CoT", "direct"],
}

OBJECTIVES = ObjectiveSchema.from_objectives(
    [
        ObjectiveDefinition("accuracy", orientation="maximize", weight=0.5),
        ObjectiveDefinition("cost", orientation="minimize", weight=0.3),
        ObjectiveDefinition("latency", orientation="minimize", weight=0.2),
    ]
)


def mock_accuracy_score(
    output: Any, expected: str, config: dict | None = None, **_
) -> float:
    """Scoring function with config-dependent mock accuracy."""
    if os.getenv("TRAIGENT_MOCK_LLM", "").lower() in ("1", "true", "yes"):
        model = (
            config.get("model", DEFAULT_MOCK_MODEL) if config else DEFAULT_MOCK_MODEL
        )
        temperature = config.get("temperature") if config else None
        instructions = config.get("instructions") if config else None
        use_cot = instructions == "CoT" if instructions else None
        return float(get_mock_accuracy(model, "classification", temperature, use_cot))
    # traigent.with_usage() hands scorers the wrapper dict, not the bare text.
    text = output.get("text") if isinstance(output, dict) else output
    if text is None or expected is None:
        return 0.0
    return 1.0 if str(text).strip().lower() == str(expected).strip().lower() else 0.0


def mock_latency_ms(
    output: Any, expected: Any, config: dict | None = None, **_
) -> float:
    """Per-example simulated latency in MILLISECONDS (the SDK's `latency` unit)."""
    model = config.get("model", DEFAULT_MOCK_MODEL) if config else DEFAULT_MOCK_MODEL
    return float(get_mock_latency(model, "classification")) * 1000.0


@traigent.optimize(
    eval_dataset=str(DATASETS / "classification.jsonl"),
    objectives=OBJECTIVES,
    scoring_function=mock_accuracy_score,
    metric_functions={"latency": mock_latency_ms},
    configuration_space=CONFIG_SPACE,
    injection_mode="context",  # default, added explicitly for clarity
    offline=True,
    mock_mode_config=MOCK_MODE_CONFIG,
)
def ai_agent_classify_text_sentiment(text: str) -> str | dict[str, Any]:
    """Text classification with multiple objectives."""
    config = traigent.get_config()
    model = config.get("model", DEFAULT_MOCK_MODEL)

    set_mock_model(model)

    label: str = CLASSIFICATION_LABELS.get(normalize_text(text), "neutral")
    # Report simulated per-example usage so the `cost` objective actually varies.
    # NOTE: the SDK treats this like real spend - an execution cost budget would
    # debit these simulated dollars. This example declares no budget.
    response: str | dict[str, Any] = traigent.with_usage(
        label,
        total_cost=get_mock_cost(model, "classification", dataset_size=1),
        input_tokens=MOCK_TASK_TOKENS["classification"]["input"],
        output_tokens=MOCK_TASK_TOKENS["classification"]["output"],
    )
    return response


async def main() -> None:
    print("Traigent Example 4: Multi-Objective Optimization")
    print("=" * 50)
    configure_mock_notice("04_multi_objective.py")
    print("Balancing accuracy (50%), cost (30%), latency (20%).")
    print(
        "Cost and latency are simulated from the static mock pricing/latency tables "
        "(no real API spend); see walkthrough/real/04_multi_objective.py for measured "
        "values."
    )
    print_optimization_config(OBJECTIVES, CONFIG_SPACE)

    results = await ai_agent_classify_text_sentiment.optimize(
        algorithm="random",
        max_trials=8,
        random_seed=42,
        show_progress=False,
        callbacks=[
            build_results_table_callback(
                is_mock=True,
                task_type="classification",
                # This example reports both from the mock tables itself, so the
                # table must show the recorded numbers that drove selection.
                reported_metrics=("cost", "latency"),
            )
        ],
    )

    print("\nBest Configuration Found:")
    print(f"  Model: {results.best_config.get('model')}")
    print(f"  Temperature: {results.best_config.get('temperature')}")
    print(f"  Instructions: {results.best_config.get('instructions')}")

    print("\nPerformance:")
    print(f"  Accuracy: {results.best_metrics.get('accuracy', 0):.2%}")
    print(
        f"  Cost:     ${results.best_metrics.get('cost', 0):.5f} (simulated, 20 examples)"
    )
    print(
        f"  Latency:  {results.best_metrics.get('latency', 0):.0f}ms (simulated, per call)"
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130) from None
