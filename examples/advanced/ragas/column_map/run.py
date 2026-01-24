#!/usr/bin/env python3
"""Show how to remap custom metadata fields to RAGAS column expectations."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

try:  # pragma: no cover - allow direct execution
    import traigent
except ImportError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[3]))
    import traigent

from traigent.metrics import configure_ragas_defaults

BASE = Path(__file__).parent
DATASET = str(BASE / "evaluation_set.jsonl")

os.environ.setdefault("TRAIGENT_FORCE_LOCAL", "true")
os.environ.setdefault("TRAIGENT_MOCK_LLM", "true")
os.environ.setdefault("RAGAS_DISABLE_ANALYTICS", "true")

_RESPONSES = {
    "Summarize the benefits of RAG.": "RAG lets language models ground answers in retrieved knowledge bases.",
}


@traigent.optimize(
    eval_dataset=DATASET,
    objectives=["context_recall", "answer_similarity"],
    configuration_space={
        "strategy": ["lookup", "summary_report"],
        "tone": ["concise", "storytelling"],
    },
    execution_mode="edge_analytics",
)
def answer(prompt: str) -> str:
    cfg = traigent.get_trial_config()
    strategy = cfg.get("strategy", "lookup")
    tone = cfg.get("tone", "concise")

    if strategy == "lookup":
        base = _RESPONSES.get(prompt, "I am not sure.")
    else:
        base = "Retrieval augmented generation blends search and generation to use knowledge bases."

    if tone == "storytelling":
        base = f"{base} Imagine a librarian fetching facts before an LLM replies."
    else:
        base = base.rstrip(".") + "."

    return base


async def main() -> None:
    # Remap custom metadata keys so RAGAS can find the relevant fields.
    configure_ragas_defaults(
        column_map={
            "retrieved_contexts": "gold_contexts",
            "reference_contexts": "gold_contexts",
            "reference": "reference_answer",
            "user_input": "prompt",
        }
    )

    result = await answer.optimize(max_trials=6)

    raw = result.to_dataframe()

    print("Optimization summary:")
    print(f"  trials: {len(result.trials)}")
    print(f"  best_config: {result.best_config}")
    print(f"  metrics: {result.metrics}")

    if raw is not None and not raw.empty:
        print("\n=== Raw trial results ===")
        print(raw.to_string(index=False))

    print("Metrics after column remapping:")
    for name in ("context_recall", "answer_similarity"):
        value = result.metrics.get(name)
        if value is not None:
            print(f"  {name}: {value:.3f}")

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
