#!/usr/bin/env python3
"""Evaluate simple deterministic answers with RAGAS metrics."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

try:  # pragma: no cover - allow running via python path
    import traigent
except ImportError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[3]))
    import traigent

from traigent.metrics import configure_ragas_defaults
from traigent.metrics.ragas_metrics import POPULAR_RAGAS_METRICS

BASE = Path(__file__).parent
DATASET = str(BASE / "evaluation_set.jsonl")

# Keep everything local/offline
os.environ.setdefault("TRAIGENT_FORCE_LOCAL", "true")
os.environ.setdefault("TRAIGENT_MOCK_MODE", "true")
os.environ.setdefault("RAGAS_DISABLE_ANALYTICS", "true")

SUPPORTED_RAGAS_METRICS = {
    name
    for name in ("context_precision", "context_recall", "answer_similarity")
    if name in POPULAR_RAGAS_METRICS
}

if not SUPPORTED_RAGAS_METRICS:
    raise SystemExit(
        "RAGAS metrics unavailable. Install ragas>=0.3.6 to run this example."
    )


_GROUNDED_RESPONSES = {
    "Who wrote Hamlet?": "William Shakespeare",
    "What is the tallest mountain?": "Mount Everest",
}

_GENERIC_RESPONSES = {
    "Who wrote Hamlet?": "A playwright from centuries ago.",
    "What is the tallest mountain?": "One of the Himalayan peaks.",
}


@traigent.optimize(
    eval_dataset=DATASET,
    objectives=list(SUPPORTED_RAGAS_METRICS),
    configuration_space={
        "strategy": ["grounded_lookup", "vague_guess"],
        "tone": ["direct", "rambling"],
    },
    execution_mode="edge_analytics",
)
def answer_question(question: str) -> str:
    cfg = traigent.get_trial_config()
    strategy = cfg.get("strategy", "grounded_lookup")
    tone = cfg.get("tone", "direct")

    if strategy == "grounded_lookup":
        base_answer = _GROUNDED_RESPONSES.get(question, "I am not sure.")
    else:
        base_answer = _GENERIC_RESPONSES.get(question, "I am not sure.")

    if tone == "rambling":
        base_answer = (
            f"{base_answer} I might be overthinking it, but that's what I recall."
        )
    else:
        base_answer = base_answer.rstrip(".")

    return base_answer


async def main() -> None:
    # Ensure we use default column mapping for ragas (dataset already matches expected fields).
    configure_ragas_defaults()

    result = await answer_question.optimize(max_trials=6)

    raw = result.to_dataframe()

    print("Optimization summary:")
    print(f"  trials: {len(result.trials)}")
    print(f"  best_config: {result.best_config}")
    print(f"  metrics: {result.metrics}")

    if raw is not None and not raw.empty:
        print("\n=== Raw trial results ===")
        print(raw.to_string(index=False))

    print("Aggregated RAGAS metrics:")
    for metric, value in sorted(result.metrics.items()):
        print(f"  {metric}: {value:.3f}")

    # Show metrics for the best configuration only.
    best_trial = next(
        (trial for trial in result.trials if trial.config == result.best_config),
        None,
    )
    if (
        best_trial
        and best_trial.metadata
        and best_trial.metadata.get("evaluation_result")
    ):
        evaluation_result = best_trial.metadata["evaluation_result"]
        print("Best trial metrics:")
        for metric, value in sorted(evaluation_result.metrics.items()):
            print(f"  {metric}: {value:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
