#!/usr/bin/env python3
"""Example 9: RAG Multi-Objective — Balance accuracy, cost, and latency
across retrieval context sizes, prompts, and models.

Demonstrates how retrieving too little context (small max_tokens budget),
using CoT instructions that consume the context window, or picking the
wrong temperature can tank accuracy — and how Traigent finds the sweet spot.

Usage (run in a terminal from repo root):
    .venv/bin/python walkthrough/mock/09_rag_multi_objective.py
"""

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import print_optimization_config, print_results_table
from utils.mock_answers import (
    DEFAULT_MOCK_MODEL,
    RAG_ANSWERS,
    configure_mock_notice,
    normalize_text,
    set_mock_model,
)

import traigent
from traigent import TraigentConfig
from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema

os.environ.setdefault("TRAIGENT_MOCK_LLM", "true")

traigent.initialize(
    config=TraigentConfig(execution_mode="edge_analytics", minimal_logging=True)
)

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

# Base accuracy for each model on RAG QA (factual retrieval task).
# Reflects that even strong models vary widely based on how they use context.
_RAG_MODEL_BASE = {
    "gpt-5.1": 0.90,       # Top reasoning, excellent at using context precisely
    "gpt-5.2": 0.80,       # Strong but less sensitive to instruction style
    "gpt-4o-mini": 0.75,   # Fast and solid at focused retrieval
    "gpt-3.5-turbo": 0.72, # Reliable for factual RAG with direct prompts
    "gpt-4o": 0.68,        # Larger model, but verbose — struggles with tight budgets
    "gpt-5-nano": 0.00,    # Cannot reliably reason over retrieved context
}

# How much CoT hurts each model on this factual RAG task.
# Strong models are optimised for direct instructions — CoT adds noise
# and uses up the context budget before the answer is reached.
_COT_SENSITIVITY = {
    "gpt-5.1": 0.27,
    "gpt-5.2": 0.02,
    "gpt-4o": 0.05,
    "gpt-4o-mini": 0.02,
    "gpt-3.5-turbo": 0.01,
}


def rag_accuracy_scorer(output: str, expected: str, config: dict | None = None, **_) -> float:
    """Mock scorer for RAG QA.

    Accuracy degrades when:
    - The model is too small to reason over retrieved context (gpt-5-nano → 0%)
    - Temperature is high (factual retrieval needs precision)
    - CoT is used with a small max_tokens budget
      (reasoning fills the context window before the answer is written)
    - Role-based prompt wastes tokens on preamble

    The result is rounded to the nearest 5% to match realistic LLM output variance.
    """
    if config is None:
        return 0.5

    model = config.get("model", DEFAULT_MOCK_MODEL)
    base = _RAG_MODEL_BASE.get(model, 0.65)

    if base == 0.0:
        return 0.0

    temperature = config.get("temperature", 0.5)
    instructions = config.get("instructions", "direct")
    max_tokens = config.get("max_tokens", 100)
    prompt = config.get("prompt", "minimal")

    # High temperature hurts factual retrieval (more hallucination)
    temp_penalty = max(0.0, temperature - 0.3) * 0.25

    # CoT penalty: reasoning tokens consume the retrieval budget
    # Small max_tokens amplifies this — there's no room left for the answer
    cot_penalty = 0.0
    if instructions == "CoT":
        sensitivity = _COT_SENSITIVITY.get(model, 0.03)
        token_factor = 1.0 + max(0.0, (100 - max_tokens) / 200.0)
        cot_penalty = sensitivity * token_factor

    # Role-based prompt adds a preamble that competes with retrieved context
    prompt_penalty = 0.02 if prompt == "role_based" else 0.0

    score = base - temp_penalty - cot_penalty - prompt_penalty
    # Round to nearest 5% — realistic for LLM evaluation variance
    return max(0.0, min(round(score * 20) / 20, 1.0))


@traigent.optimize(
    eval_dataset=str(DATASETS / "rag_questions.jsonl"),
    objectives=OBJECTIVES,
    scoring_function=rag_accuracy_scorer,
    configuration_space=CONFIG_SPACE,
    injection_mode="context",
    execution_mode="edge_analytics",
)
def rag_agent(question: str) -> str:
    """RAG agent: retrieves context up to max_tokens budget, then answers."""
    config = traigent.get_config()
    model = config.get("model", DEFAULT_MOCK_MODEL)

    set_mock_model(model)

    # Simulate retrieval: larger context budget takes more time
    max_tokens = config.get("max_tokens", 100)
    time.sleep(0.001 * max_tokens)

    # Simulate inference latency per model
    inference_latency = {
        "gpt-5-nano": 0.005,
        "gpt-4o-mini": 0.008,
        "gpt-3.5-turbo": 0.010,
        "gpt-4o": 0.015,
        "gpt-5.2": 0.020,
        "gpt-5.1": 0.018,
    }.get(model, 0.012)
    time.sleep(inference_latency)

    return RAG_ANSWERS.get(normalize_text(question), "I don't know.")


async def main() -> None:
    print("Traigent Example 9: RAG Multi-Objective Optimization")
    print("=" * 55)
    configure_mock_notice("09_rag_multi_objective.py")
    print("Balancing accuracy (50%), cost (20%), latency (30%).")
    print_optimization_config(OBJECTIVES, CONFIG_SPACE)

    results = await rag_agent.optimize(
        algorithm="random",
        max_trials=18,
        random_seed=42,
    )

    print_results_table(
        results, CONFIG_SPACE, OBJECTIVES, is_mock=True, task_type="rag_qa"
    )

    print("\nBest Configuration Found:")
    print(f"  Model:        {results.best_config.get('model')}")
    print(f"  Prompt:       {results.best_config.get('prompt')}")
    print(f"  Temperature:  {results.best_config.get('temperature')}")
    print(f"  Instructions: {results.best_config.get('instructions')}")
    print(f"  Max Tokens:   {results.best_config.get('max_tokens')}")

    print("\nPerformance:")
    print(f"  Accuracy: {results.best_metrics.get('accuracy', 0):.2%}")
    print(f"  Cost:     ${results.best_metrics.get('cost', 0):.6f}")
    print(f"  Latency:  {results.best_metrics.get('latency', 0):.3f}s")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130)
