#!/usr/bin/env python3
"""Example 5: RAG Optimization - Parallel evaluation (default on)."""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import traigent
from traigent.config.parallel import ParallelConfig

from utils.helpers import print_optimization_config, print_results_table
from utils.mock_answers import (
    DEFAULT_MOCK_MODEL,
    RAG_ANSWERS,
    configure_mock_notice,
    get_mock_accuracy,
    get_mock_cost,
    normalize_text,
    set_mock_model,
)
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
MOCK_MODE_CONFIG = {
    "base_accuracy": SIMULATED_BEST["accuracy"],
    "variance": 0.0,
    "random_seed": 42,
}
OBJECTIVES = ["accuracy", "cost"]
CONFIG_SPACE = {
    "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
    "temperature": [0.0, 0.3, 0.7],
    "k": [1, 3, 5],
    "retrieval_method": ["similarity", "keyword"],
}
SHOW_DETAIL_LOGS = os.getenv("TRAIGENT_SHOW_DETAIL_LOGS", "").lower() in (
    "1",
    "true",
    "yes",
)

KNOWLEDGE_BASE = [
    "Traigent optimizes AI applications without code changes.",
    "You can use seamless mode or parameter mode for configuration.",
    "Local execution mode keeps your data completely private.",
]


def semantic_similarity_score(output: str, expected: str, config: dict | None = None, **_) -> float:
    """Score based on key-term overlap. Mock mode returns model-dependent accuracy."""
    if os.getenv("TRAIGENT_MOCK_LLM", "").lower() in ("1", "true", "yes"):
        model = config.get("model", DEFAULT_MOCK_MODEL) if config else DEFAULT_MOCK_MODEL
        return get_mock_accuracy(model, "rag_qa")
    # Real mode: simple contains check
    if output is None or expected is None:
        return 0.0
    return 1.0 if str(expected).strip().lower() in str(output).lower() else 0.0


@traigent.optimize(
    eval_dataset=str(DATASETS / "rag_questions.jsonl"),
    objectives=OBJECTIVES,
    scoring_function=semantic_similarity_score,
    configuration_space=CONFIG_SPACE,
    injection_mode="context",  # default, added explicitly for clarity
    execution_mode="edge_analytics",
    mock_mode_config=MOCK_MODE_CONFIG,
)
def rag_qa(question: str) -> str:
    """RAG question answering with configurable retrieval."""
    config = traigent.get_config()
    model = config.get("model", DEFAULT_MOCK_MODEL)
    k = config.get("k", 3)
    method = config.get("retrieval_method", "similarity")

    set_mock_model(model)

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
    print("Optimizing retrieval (k/method) and generation (model/temp).")
    print_optimization_config(OBJECTIVES, CONFIG_SPACE)

    parallel_enabled = os.getenv("TRAIGENT_PARALLEL", "1").lower() not in (
        "0",
        "false",
        "no",
    )
    parallel_config = None
    if parallel_enabled:
        parallel_config = ParallelConfig(mode="parallel", example_concurrency=2)
        print("Parallel eval enabled (example_concurrency=2).")
        print("Pause-on-error prompts require sequential trials (parallel eval off).")
        print("To disable parallel eval: set TRAIGENT_PARALLEL=0")
    else:
        print("Parallel eval disabled. To enable: set TRAIGENT_PARALLEL=1")

    results = await rag_qa.optimize(
        algorithm="random",
        max_trials=8,
        random_seed=42,
        parallel_config=parallel_config,
    )

    print_results_table(results, CONFIG_SPACE, OBJECTIVES, is_mock=True, task_type="rag_qa")

    print("\nBest Configuration Found:")
    print(f"  Retrieval k: {results.best_config.get('k')}")
    print(f"  Method: {results.best_config.get('retrieval_method')}")
    print(f"  Model: {results.best_config.get('model')}")
    print(f"  Temperature: {results.best_config.get('temperature')}")
    print("\nPerformance:")
    print(f"  Accuracy: {results.best_metrics.get('accuracy', 0):.2%}")
    best_model = results.best_config.get("model", DEFAULT_MOCK_MODEL)
    est_cost = get_mock_cost(best_model, "rag_qa", dataset_size=20)
    print(f"  Est. Cost: ${est_cost:.4f} (for 20 examples)")


if __name__ == "__main__":
    asyncio.run(main())
