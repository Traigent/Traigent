#!/usr/bin/env python3
"""Example 5: RAG Optimization - Tune retrieval and generation together.

Usage:
    export OPENAI_API_KEY="your-key"
    python 05_rag.py
"""

import asyncio

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import traigent

KNOWLEDGE_BASE = [
    "Traigent optimizes AI applications without code changes.",
    "The SDK supports multiple objectives like accuracy, cost, and latency.",
    "You can use seamless mode or parameter mode for configuration.",
    "Local execution mode keeps your data completely private.",
    "Cloud mode provides advanced Bayesian optimization.",
]

_vectorstore = None


def get_vectorstore() -> FAISS:
    """Build vectorstore once."""
    global _vectorstore
    if _vectorstore is None:
        embeddings = OpenAIEmbeddings()
        _vectorstore = FAISS.from_texts(KNOWLEDGE_BASE, embeddings)
    return _vectorstore


@traigent.optimize(
    eval_dataset="./rag_questions.jsonl",
    objectives=["accuracy", "cost"],
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
        "temperature": [0.0, 0.3, 0.7],
        "k": [1, 3, 5],
    },
    execution_mode="edge_analytics",
)
def rag_qa(question: str) -> str:
    """RAG question answering."""
    config = traigent.get_config()
    k = config.get("k", 3)

    # Retrieve documents
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(question, k=k)
    context = "\n".join([d.page_content for d in docs])

    # Generate answer
    llm = ChatOpenAI(
        model=config.get("model", "gpt-3.5-turbo"),
        temperature=config.get("temperature", 0.3),
    )

    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    response = llm.invoke(prompt)
    return str(response.content)


async def main() -> None:
    print("Traigent Example 5: RAG Optimization")
    print("=" * 50)
    print("Optimizing retrieval (k) and generation (model, temp).\n")

    results = await rag_qa.optimize(algorithm="random", max_trials=10, random_seed=42)

    print("\nOptimal RAG Configuration:")
    print(f"  Retrieval k: {results.best_config.get('k')}")
    print(f"  Model: {results.best_config.get('model')}")
    print(f"  Temperature: {results.best_config.get('temperature')}")

    print(f"\nAccuracy: {results.best_metrics.get('accuracy', 0):.2%}")


if __name__ == "__main__":
    asyncio.run(main())
