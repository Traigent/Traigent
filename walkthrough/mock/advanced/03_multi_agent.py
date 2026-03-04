#!/usr/bin/env python3
"""Example: Multi-Agent Optimization - Mock Version

Demonstrates two methods for multi-agent parameter grouping:
1. Per-parameter agent= assignment (on Range/Choices directly)
2. Explicit AgentDefinition

This mock version uses hardcoded responses - no API keys needed.
Run with: TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true python 03_multi_agent.py
"""

import asyncio
from pathlib import Path

import traigent
from traigent.api.parameter_ranges import Choices, IntRange, Range
from traigent.api.types import AgentDefinition

# Compute dataset path relative to this script
SCRIPT_DIR = Path(__file__).parent
DATASET_PATH = str((SCRIPT_DIR / ".." / ".." / "datasets" / "simple_questions.jsonl").resolve())

# Initialize Traigent in mock mode
traigent.initialize(execution_mode="edge_analytics")

# Constants for mock responses
ML_RESPONSE = "Machine learning enables computers to learn from data."


def mock_retrieve(k: int, method: str) -> list[str]:
    """Mock retrieval - returns k documents based on method."""
    all_docs = [
        "The capital of France is Paris.",
        "Machine learning enables computers to learn from data.",
        "2+2 equals 4.",
        "Python is a programming language.",
        "The Eiffel Tower is in Paris.",
    ]
    # MMR would diversify, similarity would rank by relevance
    if method == "mmr":
        # Simulate MMR by spacing out selection
        return all_docs[::2][:k]
    return all_docs[:k]


def mock_generate(
    question: str, context: list[str], temperature: float, model: str
) -> str:
    """Mock generation - uses context and params to generate answer."""
    q = question.lower()
    ctx = " ".join(context).lower()

    # Base answer from context
    if "capital" in q and "france" in q and "paris" in ctx:
        answer = "Paris"
    elif "machine learning" in q and "learn" in ctx:
        answer = ML_RESPONSE
    elif "2+2" in q and "4" in ctx:
        answer = "4"
    else:
        answer = "I can help with that."

    # Model affects response style (mock effect)
    if model == "gpt-4o":
        answer = f"{answer} [gpt-4o]"

    # Temperature affects verbosity (mock effect)
    if temperature > 0.8:
        answer = f"{answer} (creative mode)"

    return answer


# =============================================================================
# Method 1: Per-parameter agent= assignment
# =============================================================================


async def method_1_per_parameter_agent() -> None:
    """Assign agents using the agent= parameter on each Range/Choices."""
    print("\n" + "=" * 60)
    print("Method 1: Per-parameter agent= assignment")
    print("=" * 60)

    @traigent.optimize(
        # Retriever agent parameters - use agent= on Range/Choices directly
        k=IntRange(1, 10, default=3, name="k", agent="retriever"),
        retrieval_method=Choices(
            ["similarity", "mmr", "bm25"], name="retrieval_method", agent="retriever"
        ),
        # Generator agent parameters
        temperature=Range(0.0, 1.0, default=0.7, name="temperature", agent="generator"),
        model=Choices(["gpt-4o-mini", "gpt-4o"], name="model", agent="generator"),
        objectives=["accuracy", "cost"],
        eval_dataset=DATASET_PATH,
        execution_mode="edge_analytics",
    )
    def rag_pipeline(question: str) -> str:
        """Mock RAG pipeline using all configured parameters."""
        config = traigent.get_config()

        # Retrieval phase - uses k and retrieval_method
        docs = mock_retrieve(k=config["k"], method=config["retrieval_method"])

        # Generation phase - uses temperature and model
        return mock_generate(
            question=question,
            context=docs,
            temperature=config["temperature"],
            model=config["model"],
        )

    print("\nParameter-to-agent mapping:")
    print("  Retriever: k, retrieval_method")
    print("  Generator: temperature, model")

    results = await rag_pipeline.optimize(
        algorithm="random", max_trials=8, random_seed=42
    )

    print(f"\nBest config: {results.best_config}")
    print(f"Accuracy: {results.best_metrics.get('accuracy', 0):.2%}")


# =============================================================================
# Method 2: Prefix-based automatic grouping
# =============================================================================


async def method_2_prefix_grouping() -> None:
    """Group parameters automatically by prefix pattern."""
    print("\n" + "=" * 60)
    print("Method 2: Prefix-based automatic grouping")
    print("=" * 60)

    @traigent.optimize(
        # Parameters grouped by prefix (retriever_, generator_)
        retriever_k=IntRange(1, 10, default=3, name="retriever_k"),
        retriever_method=Choices(["dense", "sparse"], name="retriever_method"),
        generator_temp=Range(0.0, 1.0, default=0.7, name="generator_temp"),
        generator_model=Choices(["gpt-4o-mini", "gpt-4o"], name="generator_model"),
        # Enable prefix-based grouping
        agent_prefixes=["retriever_", "generator_"],
        objectives=["accuracy"],
        eval_dataset=DATASET_PATH,
        execution_mode="edge_analytics",
    )
    def rag_pipeline(question: str) -> str:
        """Mock RAG pipeline with prefix-based params."""
        config = traigent.get_config()

        # Retrieval phase - uses retriever_k and retriever_method
        docs = mock_retrieve(k=config["retriever_k"], method=config["retriever_method"])

        # Generation phase - uses generator_temp and generator_model
        return mock_generate(
            question=question,
            context=docs,
            temperature=config["generator_temp"],
            model=config["generator_model"],
        )

    print("\nPrefix-based grouping:")
    print("  retriever_* -> retriever agent")
    print("  generator_* -> generator agent")

    results = await rag_pipeline.optimize(
        algorithm="random", max_trials=8, random_seed=42
    )

    print(f"\nBest config: {results.best_config}")
    print(f"Accuracy: {results.best_metrics.get('accuracy', 0):.2%}")


# =============================================================================
# Method 3: Explicit AgentDefinition
# =============================================================================


async def method_3_explicit_agents() -> None:
    """Define agents explicitly with full control over parameters and measures."""
    print("\n" + "=" * 60)
    print("Method 3: Explicit AgentDefinition")
    print("=" * 60)

    # Define agents explicitly
    retriever_agent = AgentDefinition(
        display_name="Retriever Agent",
        parameter_keys=["k", "retrieval_strategy"],
        measure_ids=["retrieval_precision", "retrieval_latency"],
    )
    generator_agent = AgentDefinition(
        display_name="Generator Agent",
        parameter_keys=["temperature", "model"],
        measure_ids=["generation_quality", "generation_cost"],
    )

    @traigent.optimize(
        # Define parameters (without agent= since we use explicit agents dict)
        k=IntRange(1, 10, default=3, name="k"),
        retrieval_strategy=Choices(
            ["similarity", "mmr", "hybrid"], name="retrieval_strategy"
        ),
        temperature=Range(0.0, 1.0, default=0.7, name="temperature"),
        model=Choices(["gpt-4o-mini", "gpt-4o"], name="model"),
        # Pass explicit agent definitions
        agents={
            "retriever": retriever_agent,
            "generator": generator_agent,
        },
        objectives=["accuracy", "total_cost"],
        eval_dataset=DATASET_PATH,
        execution_mode="edge_analytics",
    )
    def rag_pipeline(question: str) -> str:
        """Mock RAG pipeline with explicit agent definitions."""
        config = traigent.get_config()

        # Retrieval phase - uses k and retrieval_strategy
        docs = mock_retrieve(k=config["k"], method=config["retrieval_strategy"])

        # Generation phase - uses temperature and model
        return mock_generate(
            question=question,
            context=docs,
            temperature=config["temperature"],
            model=config["model"],
        )

    print("\nExplicit agent definitions:")
    print(
        f"  Retriever: params={retriever_agent.parameter_keys}, "
        f"measures={retriever_agent.measure_ids}"
    )
    print(
        f"  Generator: params={generator_agent.parameter_keys}, "
        f"measures={generator_agent.measure_ids}"
    )

    results = await rag_pipeline.optimize(
        algorithm="random", max_trials=12, random_seed=42
    )

    print(f"\nBest config: {results.best_config}")
    print(f"Accuracy: {results.best_metrics.get('accuracy', 0):.2%}")


async def main() -> None:
    print("Traigent Example: Multi-Agent Optimization")
    print("Demonstrates three methods for grouping parameters by agent")

    await method_1_per_parameter_agent()
    # Method 2 (prefix-based) is commented out due to SDK validation requirements
    # await method_2_prefix_grouping()
    await method_3_explicit_agents()

    print("\n" + "=" * 60)
    print("Summary: Both methods achieve the same goal -")
    print("grouping parameters by agent for multi-agent experiments.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
