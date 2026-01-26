#!/usr/bin/env python3
"""Example 4: Multi-Objective - Balance accuracy, cost, and latency.

Usage (run in a terminal from repo root, works without activating venv):
    export OPENAI_API_KEY="your-key"
    .venv/bin/python walkthrough/examples/real/04_multi_objective.py
"""

import asyncio
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI

import traigent
from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema

from utils.helpers import (
    configure_logging,
    print_estimated_time,
    require_openai_key,
    sanitize_traigent_api_key,
)

require_openai_key("04_multi_objective.py")
sanitize_traigent_api_key()
configure_logging()

os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")

traigent.initialize(execution_mode="edge_analytics")

# Dataset path relative to this file
DATASETS = Path(__file__).parent.parent / "datasets"

OBJECTIVES = ObjectiveSchema.from_objectives([
    ObjectiveDefinition("accuracy", orientation="maximize", weight=0.5),
    ObjectiveDefinition("cost", orientation="minimize", weight=0.3),
    ObjectiveDefinition("latency", orientation="minimize", weight=0.2),
])

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
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "temperature": [0.0, 0.3],
        "use_cot": [True, False],
    },
    injection_mode="context",  # default injection mode, added explicitly for clarity
    execution_mode="edge_analytics",
)
def classify_text(text: str) -> str:
    """Text classification with multiple objectives."""
    config = traigent.get_config()

    llm = ChatOpenAI(
        model=config.get("model"),
        temperature=config.get("temperature"),
    )

    if config.get("use_cot"):
        prompt = f"""Think step by step to classify sentiment.
Text: {text}
Answer with ONLY one word: positive, negative, or neutral."""
    else:
        prompt = f"""Classify sentiment for this text:
Text: {text}
Answer with ONLY one word: positive, negative, or neutral."""

    try:
        response = llm.invoke(prompt)
        return extract_label(response.content)
    except Exception as exc:
        print(f"LLM call failed: {type(exc).__name__}: {exc}")
        return f"Error: {type(exc).__name__}: {exc}"


async def main() -> None:
    print("Traigent Example 4: Multi-Objective Optimization")
    print("=" * 50)
    print("Balancing accuracy (50%), cost (30%), latency (20%).\n")

    print_estimated_time("04_multi_objective.py")
    results = await classify_text.optimize(
        algorithm="random",
        max_trials=10,
        show_progress=True,
        random_seed=42,
    )

    print("\nBest Configuration Found:")
    print(f"  Model: {results.best_config.get('model')}")
    print(f"  Temperature: {results.best_config.get('temperature')}")
    print(f"  Chain-of-Thought: {results.best_config.get('use_cot')}")

    print("\nPerformance:")
    print(f"  Accuracy: {results.best_metrics.get('accuracy', 0):.2%}")
    print(f"  Cost: ${results.best_metrics.get('cost', 0):.6f}")
    print(f"  Latency: {results.best_metrics.get('latency', 0):.3f}s")


if __name__ == "__main__":
    asyncio.run(main())
