#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportGeneralTypeIssues=false, reportUndefinedVariable=false, reportArgumentType=false
# mypy: ignore-errors
# flake8: noqa
# ruff: noqa
# pylint: disable=all

# Example of a simple optimization using TraiGent and Stack
from _shared import add_repo_root_to_sys_path, dataset_path, ensure_dataset, init_mock_mode

add_repo_root_to_sys_path(__file__)

from traigent import optimize  # illustrative

SNIPPET_DATASET = dataset_path(__file__, "rag_qa.jsonl")
ensure_dataset(
    SNIPPET_DATASET,
    [
        {
            "input": {"question": "How do I improve RAG quality?"},
            "expected_output": "Tune retriever and generator",
        },
        {
            "input": {"question": "What metrics should I track?"},
            "expected_output": "quality and latency",
        },
    ],
)

try:  # Stack SDK is optional in this illustrative snippet
    from stack import ChatModel, retrieve
except ImportError:  # pragma: no cover - example fallback
    class ChatModel:
        def __init__(self, model: str, temperature: float) -> None:
            self.model = model
            self.temperature = temperature

        def invoke(self, question: str, context: str) -> str:
            return f"[{self.model} @ {self.temperature}] {question} -> {context[:40]}..."

    def retrieve(query: str, k: int):
        return [f"Doc {i+1} for {query}" for i in range(k)]


@optimize(
    eval_dataset=str(SNIPPET_DATASET),
    configuration_space={
        "model": ["fast", "accurate"],
        "temperature": [0.1, 0.4, 0.7],
        "retriever_k": [3, 5, 8],
    },
    objectives=["quality", "cost_per_call_usd", "latency_p95_ms"],
    execution_mode="edge_analytics",
)
def answer(question: str) -> str:
    cfg = answer.current_config()
    llm = ChatModel(model=cfg["model"], temperature=cfg["temperature"])
    ctx = retrieve(query=question, k=cfg["retriever_k"])
    return llm.invoke(question, context=ctx)



"""Example 1: Simple Optimization - Your first TraiGent optimization."""

import asyncio

SIMPLE_DATASET = dataset_path(__file__, "simple_questions.jsonl")
ensure_dataset(
    SIMPLE_DATASET,
    [
        {"input": {"question": "What is 2+2?"}, "expected_output": "4"},
        {
            "input": {"question": "What is the capital of France?"},
            "expected_output": "Paris",
        },
        {
            "input": {"question": "What is machine learning?"},
            "expected_output": "A method where computers learn from data",
        },
        {
            "input": {"question": "Name a primary color."},
            "expected_output": "red",
        },
    ],
)

import traigent  # noqa: E402

# Enable mock mode if no API keys
MOCK = init_mock_mode()


@traigent.optimize(
    eval_dataset=str(SIMPLE_DATASET),
    objectives=["accuracy", "cost"],
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
        "temperature": [0.1, 0.5, 0.9],
    },
    execution_mode="edge_analytics",
)
def answer_question(question: str) -> str:
    """Simple Q&A function to optimize."""

    # In mock mode, return deterministic answers
    if MOCK:
        q = question.lower()
        if "2+2" in q:
            return "4"
        elif "capital" in q and "france" in q:
            return "Paris"
        elif "machine learning" in q:
            return "A method where computers learn from data"
        else:
            return "I don't know"

    # Real implementation would use LLM here
    from langchain_openai import ChatOpenAI

    config = traigent.get_current_config()
    llm = ChatOpenAI(
        model=config.get("model", "gpt-3.5-turbo"),
        temperature=config.get("temperature", 0.5),
    )

    response = llm.invoke(f"Answer concisely: {question}")
    return response.content


async def main():
    print("🎯 TraiGent Example 1: Simple Optimization")
    print("=" * 50)
    print("This example shows basic optimization of model and temperature.\n")

    # Run optimization
    print("🔍 Testing different configurations...\n")
    results = await answer_question.optimize(
        algorithm="grid", max_trials=6 if not MOCK else 3  # Test all combinations
    )

    # Display results
    print("\n" + "=" * 50)
    print("📊 OPTIMIZATION RESULTS:")
    print("=" * 50)
    print("\n🏆 Best Configuration:")
    print(f"   Model: {results.best_config.get('model')}")
    print(f"   Temperature: {results.best_config.get('temperature')}")
    print("\n📈 Performance:")
    print(f"   Accuracy: {results.best_metrics.get('accuracy', 0):.2%}")
    print(f"   Cost per call: ${results.best_metrics.get('cost', 0):.6f}")

    if MOCK:
        print(
            "\n💡 Note: These are mock results. Use real API keys for actual optimization."
        )


if __name__ == "__main__":
    asyncio.run(main())
