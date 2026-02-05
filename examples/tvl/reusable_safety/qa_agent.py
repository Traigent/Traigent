"""Q&A Agent Example with Inherited Safety Constraints.

This example demonstrates how to create a Q&A agent that inherits
safety constraints from a base specification. The agent focuses on
accuracy for knowledge base queries.

Key Features:
- Inherits 3 safety constraints from base_safety.tvl.yml
- Accuracy-focused optimization (weighted 3x)
- RAGAS faithfulness evaluation
- Task-specific Q&A dataset (50 examples)

Usage:
    # With mock LLM for testing
    TRAIGENT_MOCK_LLM=true python qa_agent.py

    # With real LLM
    python qa_agent.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Import Traigent SDK
from traigent.api import optimize
from traigent.api.safety import faithfulness, hallucination_rate, toxicity_score

# Get the directory containing this script
SCRIPT_DIR = Path(__file__).parent

# Path to the TVL spec (demonstrates inheritance)
TVL_SPEC_PATH = SCRIPT_DIR / "qa_agent.tvl.yml"


def retrieve_context(query: str, k: int = 5) -> str:
    """Simulate document retrieval for the query.

    In production, this would call a vector database or search API.

    Args:
        query: The user's question.
        k: Number of documents to retrieve.

    Returns:
        Retrieved context as a string.
    """
    # Mock retrieval - in production, use RAG pipeline
    return f"[Retrieved {k} relevant documents for: {query}]"


@optimize(
    spec=str(TVL_SPEC_PATH),
    objectives=["accuracy", "latency_p95"],
    # Safety constraints inherited from base_safety.tvl.yml:
    # - hallucination_rate <= 10%
    # - toxicity_score <= 5%
    # - bias_score <= 10%
    # Plus additional safety constraint for this agent:
    safety_constraints=[
        faithfulness.above(0.85, confidence=0.95),  # RAGAS faithfulness
        hallucination_rate().below(0.1),
        toxicity_score().below(0.05),
    ],
)
def qa_agent(
    question: str,
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.1,
    max_tokens: int = 1024,
    retrieval_k: int = 5,
    chunk_size: int = 512,
    use_reranking: bool = True,
) -> dict[str, Any]:
    """Answer questions using retrieved context.

    This agent retrieves relevant documents and generates accurate
    answers based on the context. It's optimized for accuracy with
    inherited safety constraints.

    Args:
        question: The user's question.
        model: LLM model to use.
        temperature: Sampling temperature (lower = more deterministic).
        max_tokens: Maximum tokens in response.
        retrieval_k: Number of documents to retrieve.
        chunk_size: Document chunk size for retrieval.
        use_reranking: Whether to rerank retrieved documents.

    Returns:
        Dict containing the answer and metadata.
    """
    # Step 1: Retrieve relevant context
    context = retrieve_context(question, k=retrieval_k)

    # Step 2: Generate answer using LLM
    # In production, this would call the actual LLM with this prompt
    # prompt = f"Based on context: {context}\nQuestion: {question}\nAnswer:"

    # Mock response for demonstration
    if os.environ.get("TRAIGENT_MOCK_LLM"):
        answer = f"Based on the context, the answer to '{question}' is [mock answer]."
        latency_ms = 150.0
    else:
        # In production: call LLM API
        answer = "Based on the context, here is the answer to your question..."
        latency_ms = 200.0

    return {
        "answer": answer,
        "context": context,
        "question": question,
        "model": model,
        "latency_ms": latency_ms,
        "retrieval_k": retrieval_k,
        "metadata": {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "use_reranking": use_reranking,
            "chunk_size": chunk_size,
        },
    }


def main() -> None:
    """Run the Q&A agent example."""
    print("=" * 60)
    print("Q&A Agent with Inherited Safety Constraints")
    print("=" * 60)
    print()

    # Show TVL spec path
    print(f"TVL Spec: {TVL_SPEC_PATH}")
    print()

    # Demonstrate inheritance by loading the spec
    from traigent.tvl.spec_loader import load_tvl_spec

    spec = load_tvl_spec(spec_path=TVL_SPEC_PATH)
    print("Loaded TVL Spec:")
    print(f"  - Module: {spec.tvl_header.module if spec.tvl_header else 'N/A'}")
    print(f"  - Config space params: {list(spec.configuration_space.keys())}")
    print(f"  - Objectives: {[o.name for o in spec.objective_schema.objectives]}")
    print(f"  - Constraints: {len(spec.constraints)} total (includes inherited)")
    print()

    # Example queries
    queries = [
        "What are your office hours?",
        "How do I reset my password?",
        "What is the return policy?",
    ]

    print("Running Q&A Agent:")
    print("-" * 40)

    for query in queries:
        print(f"\nQ: {query}")
        result = qa_agent(query)
        print(f"A: {result['answer'][:100]}...")
        print(f"   Latency: {result['latency_ms']:.1f}ms")

    print()
    print("=" * 60)
    print("Safety constraints from base_safety.tvl.yml are enforced!")
    print("=" * 60)


if __name__ == "__main__":
    main()
