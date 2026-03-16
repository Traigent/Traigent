#!/usr/bin/env python3
"""Example 4: Multi-Objective - Balance accuracy, cost, and latency.

Usage (run in a terminal from repo root, works without activating venv):
    export OPENAI_API_KEY="your-key"  # pragma: allowlist secret
    .venv/bin/python walkthrough/real/04_multi_objective.py
"""

import asyncio
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI
from utils.helpers import (
    configure_logging,
    print_cost_estimate,
    print_estimated_time,
    print_optimization_config,
    print_results_table,
    require_openai_key,
    sanitize_traigent_api_key,
)

import traigent
from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema

require_openai_key("04_multi_objective.py")
sanitize_traigent_api_key()
configure_logging()

os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")

traigent.initialize(execution_mode="edge_analytics")

# Dataset path relative to this file
DATASETS = Path(__file__).parent.parent / "datasets"

OBJECTIVES = ObjectiveSchema.from_objectives(
    [
        ObjectiveDefinition("accuracy", orientation="maximize", weight=0.3),
        ObjectiveDefinition("cost", orientation="minimize", weight=0.2),
        ObjectiveDefinition("latency", orientation="minimize", weight=0.5),
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
    "prompt": ["v1", "v2"],
    "temperature": [0.0, 0.3],
    "instructions": ["CoT", "direct"],
}

_LABEL_PATTERN = re.compile(r"\b(positive|negative|neutral)\b", re.IGNORECASE)


def extract_label(response: str) -> str:
    """Extract the first valid sentiment label from the response."""
    if response is None:
        return ""
    match = _LABEL_PATTERN.search(str(response))
    if match:
        return match.group(1).lower()
    return str(response).strip().lower()


@traigent.optimize(
    eval_dataset=str(DATASETS / "classification.jsonl"),
    objectives=OBJECTIVES,
    configuration_space=CONFIG_SPACE,
    injection_mode="context",  # default injection mode, added explicitly for clarity
    execution_mode="edge_analytics",
)
def ai_agent_classify_text_sentiment(text: str) -> str:
    """Text classification with multiple objectives."""
    config = traigent.get_config()

    llm = ChatOpenAI(
        model=config.get("model"),
        temperature=config.get("temperature"),
    )

    instructions = config.get("instructions", "direct")
    prompt_ver = config.get("prompt", "v1")

    # v1: minimal, v2: explicit role
    if prompt_ver == "v2":
        base = f"You are a sentiment classifier.\nText: {text}\nLabel:"
    else:  # v1
        base = f"Classify sentiment:\nText: {text}\nLabel:"

    if instructions == "CoT":
        prompt = f"Think step by step.\n{base}"
    else:
        prompt = base

    try:
        response = llm.invoke(prompt)
        return extract_label(response.content)
    except Exception as exc:
        print(f"LLM call failed: {type(exc).__name__}: {exc}")
        return f"Error: {type(exc).__name__}: {exc}"


async def main() -> None:
    print("Traigent Example 4: Multi-Objective Optimization")
    print("=" * 50)
    print("Balancing accuracy (30%), cost (20%), latency (50%).")
    print_optimization_config(OBJECTIVES, CONFIG_SPACE)
    print_cost_estimate(
        models=CONFIG_SPACE["model"],
        dataset_size=20,
        task_type="classification",
        num_trials=10,
    )

    print_estimated_time("04_multi_objective.py")
    results = await ai_agent_classify_text_sentiment.optimize(
        algorithm="random",
        max_trials=10,
        show_progress=True,
        random_seed=42,
        timeout=300,  # 5 minute timeout
    )

    print_results_table(results, CONFIG_SPACE, OBJECTIVES, is_mock=False)

    print("\nBest Configuration Found:")
    print(f"  Model: {results.best_config.get('model')}")
    print(f"  Temperature: {results.best_config.get('temperature')}")
    print(f"  Instructions: {results.best_config.get('instructions')}")

    print("\nPerformance:")
    print(f"  Accuracy: {results.best_metrics.get('accuracy', 0):.2%}")
    print(f"  Cost: ${results.best_metrics.get('cost', 0):.6f}")
    print(f"  Latency: {results.best_metrics.get('latency', 0):.3f}s")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130)
