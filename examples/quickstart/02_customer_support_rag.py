#!/usr/bin/env python
"""
Traigent Quickstart Example 2: Customer Support with RAG

This example demonstrates RAG (Retrieval Augmented Generation) optimization.
Based on the README.md customer support example.

Run with (from repo root):
    TRAIGENT_MOCK_LLM=true python examples/quickstart/02_customer_support_rag.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Ensure mock mode for testing without API keys
os.environ.setdefault("TRAIGENT_MOCK_LLM", "true")

# Set results folder to local directory
os.environ.setdefault(
    "TRAIGENT_RESULTS_FOLDER", str(Path(__file__).parent / ".traigent_results")
)

ROOT_DIR = Path(__file__).resolve().parents[2]
os.environ.setdefault("TRAIGENT_DATASET_ROOT", str(ROOT_DIR))

# Allow running from anywhere: set TRAIGENT_SDK_PATH to the SDK repo root,
# or run from within the repo tree for automatic detection.
try:
    import traigent
except ImportError:
    sys.path.insert(0, os.environ.get("TRAIGENT_SDK_PATH", str(ROOT_DIR)))
    import traigent

from traigent.api.decorators import EvaluationOptions, ExecutionOptions  # noqa: E402

os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")


# Create a simple RAG dataset for customer support
RAG_DATASET_PATH = Path(__file__).parent / "rag_feedback.jsonl"

# Create the dataset if it doesn't exist
if not RAG_DATASET_PATH.exists():
    rag_data = [
        '{"input": {"query": "What is your return policy?"}, "output": "Returns accepted within 30 days"}',
        '{"input": {"query": "Do you offer free shipping?"}, "output": "Free shipping on orders over $50"}',
        '{"input": {"query": "How can I track my order?"}, "output": "Use the tracking link in your confirmation email"}',
        '{"input": {"query": "What payment methods do you accept?"}, "output": "We accept credit cards, PayPal, and Apple Pay"}',
        '{"input": {"query": "How do I contact support?"}, "output": "Email support@example.com or call 1-800-SUPPORT"}',
    ]
    RAG_DATASET_PATH.write_text("\n".join(rag_data) + "\n")


# Simulated knowledge base
KNOWLEDGE_BASE = [
    "Returns accepted within 30 days with original receipt",
    "Free shipping on orders over $50",
    "Track your order using the link in confirmation email",
    "We accept Visa, Mastercard, PayPal, and Apple Pay",
    "Contact support at support@example.com or 1-800-SUPPORT",
    "Business hours: Monday-Friday 9am-5pm EST",
    "Gift cards never expire",
    "Price match guarantee within 14 days of purchase",
]


def simple_retriever(query: str, k: int = 3) -> list[str]:
    """Simple keyword-based retriever for demo purposes.

    In production, you would use a vector store like Chroma or FAISS.
    """
    query_lower = query.lower()
    scores = []
    for doc in KNOWLEDGE_BASE:
        # Simple keyword matching score
        score = sum(1 for word in query_lower.split() if word in doc.lower())
        scores.append((score, doc))

    # Return top-k documents
    scores.sort(reverse=True, key=lambda x: x[0])
    return [doc for _, doc in scores[:k]]


@traigent.optimize(
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "temperature": [0.1, 0.5, 0.9],
        "k": [2, 3, 5],  # RAG retrieval depth
    },
    evaluation=EvaluationOptions(eval_dataset=str(RAG_DATASET_PATH)),
    execution=ExecutionOptions(execution_mode="edge_analytics"),
    max_trials=6,
)
def customer_support_agent(query: str) -> str:
    """Answer customer questions using RAG.

    This demonstrates how Traigent can optimize RAG parameters like:
    - Which model to use
    - Temperature setting
    - Number of retrieved documents (k)
    """
    # Read the tuned config injected by Traigent
    cfg = traigent.get_config()
    k = cfg.get("k", 3)

    # Retrieve relevant documents
    docs = simple_retriever(query, k=k)
    context = "\n".join(docs)  # noqa: F841 - used in real LangChain path below

    # In real usage with LangChain:
    # from langchain_openai import ChatOpenAI
    # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    # response = llm.invoke(f"Context: {context}\nQuestion: {query}\nAnswer:")
    # return response.content

    # For demo, return the most relevant document
    if docs:
        return docs[0]
    return "I'm sorry, I couldn't find information about that."


async def main():
    print("=" * 60)
    print("Traigent Quickstart: Customer Support RAG Optimization")
    print("=" * 60)
    print()

    print("Knowledge Base:")
    for i, doc in enumerate(KNOWLEDGE_BASE[:3], 1):
        print(f"  {i}. {doc[:50]}...")
    print(f"  ... and {len(KNOWLEDGE_BASE) - 3} more documents")
    print()

    print(f"Dataset: {RAG_DATASET_PATH}")
    print(f"Mock mode: {os.environ.get('TRAIGENT_MOCK_LLM', 'false')}")
    print()

    # Test the retriever
    print("Testing retriever:")
    test_query = "What is your return policy?"
    retrieved = simple_retriever(test_query, k=2)
    print(f"  Query: '{test_query}'")
    print(f"  Retrieved: {retrieved}")
    print()

    # Run optimization
    print("Starting RAG optimization...")
    print("(Optimizing model, temperature, and retrieval depth)")
    print()

    results = await customer_support_agent.optimize()

    print()
    print("=" * 60)
    print("Optimization Complete!")
    print("=" * 60)
    print()
    print(f"Best Score: {results.best_score}")
    print(f"Best Configuration: {results.best_config}")
    print()

    # Demonstrate using the optimized agent
    print("Using optimized agent:")
    print("-" * 40)
    test_questions = [
        "What is your return policy?",
        "Do you offer free shipping?",
    ]
    for question in test_questions:
        answer = customer_support_agent(question)
        print(f"Q: {question}")
        print(f"A: {answer}")
        print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130) from None
