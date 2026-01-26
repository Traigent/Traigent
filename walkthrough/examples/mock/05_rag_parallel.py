#!/usr/bin/env python3
"""Example 5: RAG Optimization - Parallel evaluation (default on)."""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import traigent
from traigent.config.parallel import ParallelConfig

from utils.mock_answers import RAG_ANSWERS, normalize_text, configure_mock_notice
from traigent import TraigentConfig

os.environ.setdefault("TRAIGENT_MOCK_LLM", "true")
os.environ.setdefault("TRAIGENT_OFFLINE_MODE", "true")

traigent.initialize(config=TraigentConfig(execution_mode="edge_analytics", minimal_logging=True))

# Dataset path relative to this file
DATASETS = Path(__file__).parent.parent / "datasets"
SIMULATED_BEST = {
    "k": 3,
    "retrieval_method": "keyword",
    "model": "gpt-3.5-turbo",
    "temperature": 0.7,
    "accuracy": 0.8344,
}
MOCK_MODE_CONFIG = {"base_accuracy": SIMULATED_BEST["accuracy"], "variance": 0.0, "random_seed": 42}
SHOW_DETAIL_LOGS = os.getenv("TRAIGENT_SHOW_DETAIL_LOGS", "").lower() in ("1", "true", "yes")

KNOWLEDGE_BASE = [
    "Traigent optimizes AI applications without code changes.",
    "You can use seamless mode or parameter mode for configuration.",
    "Local execution mode keeps your data completely private.",
]


def _mock_accuracy() -> float:
    return MOCK_MODE_CONFIG["base_accuracy"]


def semantic_similarity_score(output: str, expected: str, **_) -> float:
    """Score based on key-term overlap. Mock mode returns configured accuracy."""
    if os.getenv("TRAIGENT_MOCK_LLM", "").lower() in ("1", "true", "yes"):
        return _mock_accuracy()
    # Real mode: simple contains check
    if output is None or expected is None:
        return 0.0
    return 1.0 if str(expected).strip().lower() in str(output).lower() else 0.0


@traigent.optimize(
    eval_dataset=str(DATASETS / "rag_questions.jsonl"),
    objectives=["accuracy", "cost"],
    scoring_function=semantic_similarity_score,
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
        "temperature": [0.0, 0.3, 0.7],
        "k": [1, 3, 5],
        "retrieval_method": ["similarity", "keyword"],
    },
    injection_mode="context",  # default, added explicitly for clarity
    execution_mode="edge_analytics",
    mock_mode_config=MOCK_MODE_CONFIG,
)
def rag_qa(question: str) -> str:
    """RAG question answering with configurable retrieval."""
    config = traigent.get_config()
    k = config.get("k", 3)
    method = config.get("retrieval_method", "similarity")

    if SHOW_DETAIL_LOGS:
        print(f"  RAG: k={k}, method={method}")

    answer = RAG_ANSWERS.get(normalize_text(question), "Unknown")
    if answer == "Unknown":
        return answer
    if method == "keyword" and k < 3:
        return "Limited context answer"
    if k < 3:
        return answer.split(".")[0]
    return answer


async def main() -> None:
    print("Traigent Example 5: RAG Optimization (parallel eval on by default)")
    print("=" * 50)
    configure_mock_notice("05_rag_parallel.py")
    print("Optimizing retrieval (k/method) and generation (model/temp) against rag_questions.jsonl.")

    parallel_enabled = os.getenv("TRAIGENT_PARALLEL", "1").lower() not in (
        "0",
        "false",
        "no",
    )
    parallel_config = None
    if parallel_enabled:
        parallel_config = ParallelConfig(mode="parallel", example_concurrency=2)
        print("Parallel eval enabled (example_concurrency=2).")
        print("To disable: set TRAIGENT_PARALLEL=0")
    else:
        print("Parallel eval disabled. To enable: set TRAIGENT_PARALLEL=1")

    # In real mode, use results.best_config and results.best_metrics
    # Example: results.best_config.get("k"), results.best_metrics.get("accuracy")
    _results = await rag_qa.optimize(
        algorithm="random",
        max_trials=8,
        random_seed=42,
        parallel_config=parallel_config,
    )

    print("\nBest Configuration Found:")
    print(f"  Retrieval k: {SIMULATED_BEST['k']}")
    print(f"  Method: {SIMULATED_BEST['retrieval_method']}")
    print(f"  Model: {SIMULATED_BEST['model']}")
    print(f"  Temperature: {SIMULATED_BEST['temperature']}")
    print("\nPerformance:")
    print(f"  Accuracy: {SIMULATED_BEST['accuracy']:.2%}")


if __name__ == "__main__":
    asyncio.run(main())
