#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportGeneralTypeIssues=false, reportUndefinedVariable=false, reportArgumentType=false
# mypy: ignore-errors
# flake8: noqa
# ruff: noqa
# pylint: disable=all

"""Example 5: RAG Optimization - Optimize retrieval and generation together."""

import asyncio

from _shared import add_repo_root_to_sys_path, dataset_path, ensure_dataset, init_mock_mode

add_repo_root_to_sys_path(__file__)

RAG_DATASET = dataset_path(__file__, "rag_test.jsonl")
ensure_dataset(
    RAG_DATASET,
    [
        {
            "input": {"question": "What does TraiGent optimize?"},
            "expected_output": "AI applications",
        },
        {
            "input": {"question": "What modes does TraiGent support?"},
            "expected_output": "seamless and parameter modes",
        },
        {
            "input": {"question": "How does Edge Analytics mode work?"},
            "expected_output": "keeps data private",
        },
        {
            "input": {"question": "Name a benefit of RAG optimization."},
            "expected_output": "better answers",
        },
    ],
)

import traigent

MOCK = init_mock_mode()

# Sample knowledge base
KNOWLEDGE_BASE = [
    "TraiGent optimizes AI applications without code changes.",
    "The SDK supports multiple objectives like accuracy, cost, and latency.",
    "You can use seamless mode or parameter mode for configuration.",
    "Local execution mode keeps your data completely private.",
    "Cloud mode provides advanced Bayesian optimization.",
    "TraiGent works with LangChain, OpenAI SDK, and other frameworks.",
    "Custom evaluators let you define your own success metrics.",
    "The playground provides an interactive UI for experimentation.",
]

_VECTORSTORE = None


def _get_vectorstore():
    """Build the FAISS vector store once to avoid recomputing embeddings."""
    global _VECTORSTORE
    if _VECTORSTORE is None:
        from langchain_community.vectorstores import FAISS
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings()
        _VECTORSTORE = FAISS.from_texts(KNOWLEDGE_BASE, embeddings)
    return _VECTORSTORE


@traigent.optimize(
    eval_dataset=str(RAG_DATASET),
    objectives=["accuracy", "cost"],
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
        "temperature": [0.0, 0.3, 0.7],
        "k": [1, 3, 5],  # Number of documents to retrieve
        "retrieval_method": ["similarity", "keyword"],  # RAG strategy
        "use_reranking": [True, False],
    },
    execution_mode="edge_analytics",
)
def rag_qa_system(question: str) -> str:
    """Question answering with RAG optimization."""

    config = traigent.get_current_config()

    # RAG parameters from configuration
    k = config.get("k", 3)
    method = config.get("retrieval_method", "similarity")
    use_reranking = config.get("use_reranking", False)

    print(f"  RAG Config: k={k}, method={method}, rerank={use_reranking}")

    if MOCK:
        # Simulate RAG behavior
        if "optimize" in question.lower() or "traigent" in question.lower():
            if k >= 3:
                return "TraiGent optimizes AI applications"
            else:
                return "AI optimization tool"
        elif "mode" in question.lower():
            return "seamless and parameter modes"
        else:
            return "Unknown answer"

    # Real RAG implementation
    from langchain_openai import ChatOpenAI

    # Retrieve documents
    if method == "similarity":
        vectorstore = _get_vectorstore()
        docs = vectorstore.similarity_search(question, k=k)
    else:
        # Keyword-based retrieval (simplified)
        keywords = question.lower().split()
        docs = [
            doc for doc in KNOWLEDGE_BASE if any(kw in doc.lower() for kw in keywords)
        ][:k]

    # Optional reranking
    if use_reranking and len(docs) > 1:
        # Simple reranking by relevance score (mock)
        docs = sorted(docs, key=lambda d: len(d))[:k]

    # Build context
    context = "\n".join(
        [d.page_content if hasattr(d, "page_content") else d for d in docs]
    )

    # Generate answer
    llm = ChatOpenAI(
        model=config.get("model", "gpt-3.5-turbo"),
        temperature=config.get("temperature", 0.3),
    )

    prompt = f"""Context:
{context}

Question: {question}

Answer based on the context:"""

    response = llm.invoke(prompt)
    return response.content


async def main():
    print("🎯 TraiGent Example 5: RAG Optimization")
    print("=" * 50)
    print("🔍 Optimizing both retrieval and generation\n")

    print("RAG Components to Optimize:")
    print("  📚 Retrieval:")
    print("    • Number of documents (k)")
    print("    • Retrieval method (similarity vs keyword)")
    print("    • Reranking strategy")
    print("  🤖 Generation:")
    print("    • LLM model selection")
    print("    • Temperature for creativity")
    print("\nFinding the best combination...\n")

    # Run optimization
    results = await rag_qa_system.optimize(
        algorithm="random", max_trials=15 if not MOCK else 8
    )

    # Display results
    print("\n" + "=" * 50)
    print("🏆 OPTIMAL RAG CONFIGURATION:")
    print("=" * 50)

    best = results.best_config
    print("\n📚 Retrieval Settings:")
    print(f"  Documents to retrieve (k): {best.get('k')}")
    print(f"  Retrieval method: {best.get('retrieval_method')}")
    print(f"  Use reranking: {best.get('use_reranking')}")

    print("\n🤖 Generation Settings:")
    print(f"  Model: {best.get('model')}")
    print(f"  Temperature: {best.get('temperature')}")

    print("\n📊 Performance:")
    print(f"  Accuracy: {results.best_metrics.get('accuracy', 0):.2%}")
    print(f"  Cost per query: ${results.best_metrics.get('cost', 0):.6f}")

    print("\n💡 Insights:")
    if best.get("k") <= 2:
        print("  ✅ Fewer documents = faster and cheaper")
    else:
        print("  ✅ More context = better accuracy")

    if best.get("use_reranking"):
        print("  ✅ Reranking improves relevance")

    print("\n🎯 RAG optimization balances retrieval quality with generation cost!")


if __name__ == "__main__":
    asyncio.run(main())
