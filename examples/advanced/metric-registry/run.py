#!/usr/bin/env python3
"""Demonstrate custom metric aggregation and overrides with Traigent's registry."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

# Ensure the example runs fully locally without attempting cloud coordination.
os.environ.setdefault("TRAIGENT_FORCE_LOCAL", "true")
os.environ.setdefault("TRAIGENT_MOCK_LLM", "true")

import traigent  # noqa: E402
from traigent.metrics import MetricSpec, register_metric, reset_registry  # noqa: E402

os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")


BASE = Path(__file__).parent
DATASET = str(BASE / "evaluation_set.jsonl")


def _normalized_tokens(text: str) -> set[str]:
    return {token for token in text.lower().split() if token}


def accuracy_metric(
    output: str, expected: str, _: dict[str, Any] | None = None
) -> float:
    """Function-scoped accuracy override: exact match ignoring case."""
    return 1.0 if output.strip().lower() == expected.strip().lower() else 0.0


def partial_credit_metric(
    output: str, expected: str, _: dict[str, Any] | None = None
) -> float:
    """Award fractional credit based on token overlap."""
    expected_tokens = _normalized_tokens(expected)
    if not expected_tokens:
        return 0.0
    overlap = _normalized_tokens(output) & expected_tokens
    return len(overlap) / len(expected_tokens)


_RESPONSES: dict[str, dict[str, str]] = {
    "baseline": {
        "list the colors of the us flag": "red blue",
        "give three synonyms for fast": "fast speedy",
        "what is 2+2?": "four",
    },
    "lookup": {
        "list the colors of the us flag": "red white blue",
        "give three synonyms for fast": "quick rapid swift",
        "what is 2+2?": "4",
    },
}


@traigent.optimize(
    eval_dataset=DATASET,
    objectives=["accuracy", "partial_credit", "total_cost"],
    configuration_space={"strategy": list(_RESPONSES.keys())},
    metric_functions={
        "accuracy": accuracy_metric,
        "partial_credit": partial_credit_metric,
    },
    execution_mode="edge_analytics",
    algorithm="grid",
)
def answer_prompt(prompt: str) -> str:
    """Simple rule-based function whose behaviour depends on the strategy."""
    cfg = traigent.get_trial_config()
    strategy = str(cfg.get("strategy", "baseline"))
    lookup = _RESPONSES.get(strategy, _RESPONSES["baseline"])
    return lookup.get(prompt.lower(), "i do not know")


def main() -> None:
    reset_registry()
    # Override the built-in "accuracy" to report the last trial's score instead of averaging
    # across all trials. For custom metrics we can pick the aggregation strategy we prefer;
    # here we sum fractional partial-credit scores so they accumulate across trials.
    register_metric(MetricSpec(name="accuracy", aggregator="last"))
    register_metric(MetricSpec(name="partial_credit", aggregator="sum"))

    try:
        result = asyncio.run(answer_prompt.optimize(max_trials=len(_RESPONSES)))

        print("Best strategy:", result.best_config)
        print("Aggregated metrics:")
        for name, value in sorted(result.metrics.items()):
            print(f"  {name}: {value}")
    finally:
        # Reset global registry so other scripts/tests see the defaults.
        reset_registry()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130)
