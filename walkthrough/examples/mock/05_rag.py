#!/usr/bin/env python3
"""Example 5: RAG Optimization - Tune retrieval and generation together."""

import asyncio

import traigent

traigent.initialize(execution_mode="edge_analytics")

KNOWLEDGE_BASE = [
    "Traigent optimizes AI applications without code changes.",
    "You can use seamless mode or parameter mode for configuration.",
    "Local execution mode keeps your data completely private.",
]


@traigent.optimize(
    eval_dataset="../datasets/rag_questions.jsonl",
    objectives=["accuracy", "cost"],
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
        "temperature": [0.0, 0.3, 0.7],
        "k": [1, 3, 5],
        "retrieval_method": ["similarity", "keyword"],
    },
    execution_mode="edge_analytics",
)
def rag_qa(question: str) -> str:
    """RAG question answering with configurable retrieval."""
    config = traigent.get_config()
    k = config.get("k", 3)
    method = config.get("retrieval_method", "similarity")

    print(f"  RAG: k={k}, method={method}")

    if "optimize" in question.lower() or "traigent" in question.lower():
        return "Traigent optimizes AI applications" if k >= 3 else "AI tool"
    elif "mode" in question.lower():
        return "seamless and parameter modes"
    return "Unknown"


async def main() -> None:
    print("Traigent Example 5: RAG Optimization")
    print("=" * 50)
    print("Optimizing retrieval (k, method) and generation (model, temp).\n")

    results = await rag_qa.optimize(algorithm="random", max_trials=8)

    print("\nOptimal RAG Configuration:")
    print(f"  Retrieval k: {results.best_config.get('k')}")
    print(f"  Method: {results.best_config.get('retrieval_method')}")
    print(f"  Model: {results.best_config.get('model')}")
    print(f"  Temperature: {results.best_config.get('temperature')}")

    print(f"\nAccuracy: {results.best_metrics.get('accuracy', 0):.2%}")


if __name__ == "__main__":
    asyncio.run(main())
