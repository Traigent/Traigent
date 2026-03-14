#!/usr/bin/env python3
"""Example 3: Parameter Mode - Explicit configuration control."""

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
SIMULATED_BEST = {
    "model": "gpt-3.5-turbo",
    "temperature": 0.0,
    "max_tokens": 50,
    "use_system_prompt": True,
    "accuracy": 0.9,
}
MOCK_MODE_CONFIG = {
    "base_accuracy": SIMULATED_BEST["accuracy"],
    "variance": 0.0,
    "random_seed": 42,
}
OBJECTIVES = ["accuracy", "cost"]
CONFIG_SPACE = {
    "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
    "temperature": [0.0, 0.5, 1.0],
    "max_tokens": [50, 150, 300],
    "use_system_prompt": [True, False],
}
SHOW_DETAIL_LOGS = os.getenv("TRAIGENT_SHOW_DETAIL_LOGS", "").lower() in (
    "1",
    "true",
    "yes",
)


def results_match_score(output: str, expected: str, config: dict | None = None, **_) -> float:
    """Simple scoring: 1.0 if expected answer appears in output, else 0.0.

    In mock mode, returns model-dependent accuracy.
    """
    if os.getenv("TRAIGENT_MOCK_LLM", "").lower() in ("1", "true", "yes"):
        model = config.get("model", DEFAULT_MOCK_MODEL) if config else DEFAULT_MOCK_MODEL
        return get_mock_accuracy(model, "simple_qa")
    if output is None or expected is None:
        return 0.0
    return 1.0 if str(expected).strip().lower() in str(output).lower() else 0.0


@traigent.optimize(
    eval_dataset=str(DATASETS / "simple_questions.jsonl"),
    objectives=OBJECTIVES,
    injection_mode="parameter",
    config_param="config",  # Explicit parameter name for clarity
    scoring_function=results_match_score,
    configuration_space=CONFIG_SPACE,
    execution_mode="edge_analytics",
    mock_mode_config=MOCK_MODE_CONFIG,
)
def answer_with_control(question: str, config: dict) -> str:
    """Function with explicit configuration parameter."""
    model = config.get("model", DEFAULT_MOCK_MODEL)
    temperature = config.get("temperature", 0.5)
    max_tokens = config.get("max_tokens", 150)

    set_mock_model(model)

    if SHOW_DETAIL_LOGS:
        print(f"  Using: {model}, temp={temperature}, tokens={max_tokens}")

    return ANSWERS.get(normalize_text(question), "I don't know")


async def main() -> None:
    print("Traigent Example 3: Parameter Mode")
    print("=" * 50)
    configure_mock_notice("03_parameter_mode.py")
    print("Full control with explicit configuration parameter.")
    print_optimization_config(OBJECTIVES, CONFIG_SPACE)

    results = await answer_with_control.optimize(
        algorithm="random", max_trials=4, random_seed=42
    )

    print_results_table(results, CONFIG_SPACE, OBJECTIVES, is_mock=True, task_type="simple_qa")

    print("\nBest Configuration Found:")
    print(f"  Model: {results.best_config.get('model')}")
    print(f"  Temperature: {results.best_config.get('temperature')}")
    print(f"  Max Tokens: {results.best_config.get('max_tokens')}")
    use_sys = "yes" if results.best_config.get('use_system_prompt') else "no"
    print(f"  System Prompt: {use_sys}")
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
