#!/usr/bin/env python3
"""Example 9: RAG Multi-Objective - Balance accuracy, cost, and latency.

Demonstrates how retrieving too little context, using CoT instructions that
consume the context window, or picking the wrong temperature can tank
accuracy - and how Traigent finds the sweet spot.

Mock accuracy values are calibrated from a real run of
walkthrough/real/09_rag_multi_objective.py (random, seed=42, 18 trials).

Usage (run in a terminal from repo root):
    .venv/bin/python walkthrough/mock/09_rag_multi_objective.py
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import build_results_table_callback, print_optimization_config
from utils.mock_answers import (
    DEFAULT_MOCK_MODEL,
    MOCK_TASK_TOKENS,
    RAG_ANSWERS,
    configure_mock_notice,
    get_mock_cost,
    get_mock_latency,
    normalize_text,
    set_mock_model,
)

import traigent
from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema

os.environ.setdefault("TRAIGENT_MOCK_LLM", "true")
os.environ.setdefault("TRAIGENT_OFFLINE_MODE", "true")

traigent.initialize(offline=True, minimal_logging=True)

DATASETS = Path(__file__).parent.parent / "datasets"

OBJECTIVES = ObjectiveSchema.from_objectives(
    [
        ObjectiveDefinition("accuracy", orientation="maximize", weight=0.5),
        ObjectiveDefinition("cost", orientation="minimize", weight=0.2),
        ObjectiveDefinition("latency", orientation="minimize", weight=0.3),
    ]
)

CONFIG_SPACE = {
    "model": [
        "gpt-3.5-turbo",
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-5.2",
        "gpt-5-nano",
        "gpt-5.1",
    ],
    "prompt": ["minimal", "role_based"],
    "temperature": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "instructions": ["CoT", "direct"],
    "max_tokens": [50, 100, 200],
}

# Accuracy values from a real run of walkthrough/real/09_rag_multi_objective.py
# (algorithm=random, seed=42, 18 trials). Keyed by (model, prompt, temperature,
# instructions, max_tokens). Trials 17-18 were not captured, so they fall back
# to the formula below.
_REAL_ACCURACY: dict[tuple, float] = {
    ("gpt-5.1", "minimal", 0.0, "direct", 50): 0.180,
    ("gpt-4o-mini", "minimal", 0.1, "CoT", 200): 0.771,
    ("gpt-5.2", "minimal", 0.0, "CoT", 50): 0.128,
    ("gpt-4o-mini", "minimal", 0.8, "CoT", 200): 0.745,
    ("gpt-5.1", "role_based", 0.3, "direct", 200): 0.811,
    ("gpt-4o", "minimal", 0.2, "direct", 100): 0.741,
    ("gpt-4o", "minimal", 0.3, "direct", 50): 0.710,
    ("gpt-3.5-turbo", "role_based", 0.1, "direct", 100): 0.874,
    ("gpt-5-nano", "role_based", 0.0, "direct", 200): 0.050,
    ("gpt-3.5-turbo", "role_based", 0.1, "direct", 200): 0.868,
    ("gpt-5-nano", "role_based", 0.9, "CoT", 200): 0.000,
    ("gpt-3.5-turbo", "minimal", 0.3, "direct", 50): 0.776,
    ("gpt-4o-mini", "minimal", 0.6, "direct", 100): 0.738,
    ("gpt-5.1", "role_based", 0.2, "direct", 100): 0.711,
    ("gpt-4o-mini", "role_based", 0.1, "CoT", 200): 0.864,
    ("gpt-5.1", "minimal", 0.2, "direct", 100): 0.335,
}

_RAG_MODEL_BASE = {
    "gpt-5.1": 0.75,
    "gpt-5.2": 0.70,
    "gpt-4o-mini": 0.75,
    "gpt-3.5-turbo": 0.72,
    "gpt-4o": 0.68,
    "gpt-5-nano": 0.03,
}
_COT_SENSITIVITY = {
    "gpt-5.1": 0.27,
    "gpt-5.2": 0.02,
    "gpt-4o": 0.05,
    "gpt-4o-mini": 0.02,
    "gpt-3.5-turbo": 0.01,
}


def rag_accuracy_scorer(
    output: Any, expected: str, config: dict | None = None, **_
) -> float:
    """Mock scorer calibrated from a real experiment run (config-only, ignores output)."""
    if config is None:
        return 0.5

    model = config.get("model", DEFAULT_MOCK_MODEL)
    prompt = config.get("prompt", "minimal")
    temperature = config.get("temperature", 0.5)
    instructions = config.get("instructions", "direct")
    max_tokens = config.get("max_tokens", 100)

    key = (model, prompt, temperature, instructions, max_tokens)
    if key in _REAL_ACCURACY:
        return _REAL_ACCURACY[key]

    base = _RAG_MODEL_BASE.get(model, 0.65)
    temp_penalty = max(0.0, temperature - 0.3) * 0.25
    cot_penalty = 0.0
    if instructions == "CoT":
        sensitivity = _COT_SENSITIVITY.get(model, 0.03)
        token_factor = 1.0 + max(0.0, (100 - max_tokens) / 200.0)
        cot_penalty = sensitivity * token_factor
    prompt_penalty = 0.02 if prompt == "role_based" else 0.0
    score = base - temp_penalty - cot_penalty - prompt_penalty
    return float(max(0.0, min(round(score * 20) / 20, 1.0)))


def mock_rag_latency_ms(
    output: Any, expected: Any, config: dict | None = None, **_
) -> float:
    """Per-example simulated latency in MILLISECONDS (the SDK's `latency` unit)."""
    cfg = config or {}
    model = cfg.get("model", DEFAULT_MOCK_MODEL)
    max_tokens = float(cfg.get("max_tokens", 100))
    # Two signals: per-model inference time plus a retrieval penalty for the
    # larger context that a bigger max_tokens budget pulls in.
    return (float(get_mock_latency(model, "rag_qa")) + 0.001 * max_tokens) * 1000.0


@traigent.optimize(
    eval_dataset=str(DATASETS / "rag_questions.jsonl"),
    objectives=OBJECTIVES,
    scoring_function=rag_accuracy_scorer,
    metric_functions={"latency": mock_rag_latency_ms},
    configuration_space=CONFIG_SPACE,
    injection_mode="context",
    offline=True,
)
def rag_agent(question: str) -> str | dict[str, Any]:
    """RAG agent: retrieves context up to max_tokens budget, then answers."""
    config = traigent.get_config()
    model = config.get("model", DEFAULT_MOCK_MODEL)

    set_mock_model(model)

    answer: str = RAG_ANSWERS.get(normalize_text(question), "I don't know.")
    # Report simulated per-example usage so the `cost` objective actually varies.
    # NOTE: the SDK treats this like real spend - an execution cost budget would
    # debit these simulated dollars. This example declares no budget.
    response: str | dict[str, Any] = traigent.with_usage(
        answer,
        total_cost=get_mock_cost(model, "rag_qa", dataset_size=1),
        input_tokens=MOCK_TASK_TOKENS["rag_qa"]["input"],
        output_tokens=MOCK_TASK_TOKENS["rag_qa"]["output"],
    )
    return response


async def main() -> None:
    print("Traigent Example 9: RAG Multi-Objective Optimization")
    print("=" * 55)
    configure_mock_notice("09_rag_multi_objective.py")
    print("Balancing accuracy (50%), cost (20%), latency (30%).")
    print(
        "Cost and latency are simulated from the static mock pricing/latency tables "
        "(no real API spend)."
    )
    print_optimization_config(OBJECTIVES, CONFIG_SPACE)

    results = await rag_agent.optimize(
        algorithm="random",
        max_trials=18,
        random_seed=42,
        show_progress=False,
        callbacks=[
            build_results_table_callback(
                is_mock=True,
                task_type="rag_qa",
                # This example reports both from the mock tables itself, so the
                # table must show the recorded numbers that drove selection -
                # including the max_tokens retrieval penalty in the latency.
                reported_metrics=("cost", "latency"),
            )
        ],
    )

    print("\nBest Configuration Found:")
    print(f"  Model:        {results.best_config.get('model')}")
    print(f"  Prompt:       {results.best_config.get('prompt')}")
    print(f"  Temperature:  {results.best_config.get('temperature')}")
    print(f"  Instructions: {results.best_config.get('instructions')}")
    print(f"  Max Tokens:   {results.best_config.get('max_tokens')}")

    print("\nPerformance:")
    print(f"  Accuracy: {results.best_metrics.get('accuracy', 0):.2%}")
    print(f"  Cost:     ${results.best_metrics.get('cost', 0):.6f} (simulated)")
    print(f"  Latency:  {results.best_metrics.get('latency', 0):.0f}ms (simulated)")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130) from None
