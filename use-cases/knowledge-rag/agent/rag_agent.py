#!/usr/bin/env python3
"""
Knowledge & RAG Agent - Document Q&A System

This agent answers questions about CloudStack API documentation using
retrieval-augmented generation. It optimizes for answer accuracy,
grounding (faithfulness to sources), and appropriate abstention.

Usage:
    export TRAIGENT_MOCK_MODE=true
    python use-cases/knowledge-rag/agent/rag_agent.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import evaluator from sibling directory
import importlib.util

import traigent
from traigent.api.decorators import EvaluationOptions, ExecutionOptions

_evaluator_path = Path(__file__).parent.parent / "eval" / "evaluator.py"
_spec = importlib.util.spec_from_file_location("rag_evaluator", _evaluator_path)
_evaluator_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_evaluator_module)
RAGEvaluator = _evaluator_module.RAGEvaluator


def load_knowledge_base() -> list[dict]:
    """Load the CloudStack documentation knowledge base."""
    kb_path = (
        Path(__file__).parent.parent
        / "datasets"
        / "knowledge_base"
        / "cloudstack_docs.json"
    )
    try:
        with open(kb_path) as f:
            data = json.load(f)
            return data.get("documents", [])
    except FileNotFoundError:
        # Return minimal mock data for testing
        return [
            {
                "id": "mock_doc_1",
                "title": "Mock Documentation",
                "section": "General",
                "content": "This is mock documentation content for testing.",
            }
        ]


def simple_retrieval(
    question: str,
    documents: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Simple keyword-based retrieval for demonstration.
    In production, use vector embeddings for semantic search.
    """
    # Tokenize question into keywords
    question_lower = question.lower()
    keywords = [w for w in question_lower.split() if len(w) > 3]

    # Score documents by keyword overlap
    scored_docs = []
    for doc in documents:
        content_lower = doc["content"].lower()
        title_lower = doc["title"].lower()

        # Count keyword matches
        score = sum(1 for kw in keywords if kw in content_lower)
        # Boost for title matches
        score += sum(2 for kw in keywords if kw in title_lower)

        if score > 0:
            scored_docs.append((score, doc))

    # Sort by score and return top_k
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored_docs[:top_k]]


RAG_PROMPT_TEMPLATE = """You are a helpful technical documentation assistant for CloudStack API.
Your job is to answer questions accurately based ONLY on the provided documentation.

IMPORTANT RULES:
1. Only use information from the provided documents
2. If the documents don't contain the answer, say "I don't have information about that in the documentation"
3. Cite the relevant document sections when answering
4. Be concise but complete
5. If you're uncertain, express your uncertainty

Confidence Threshold: {confidence_threshold}
- If your confidence in the answer is below this threshold, abstain from answering

Retrieved Documents:
{context}

Question: {question}

Provide a helpful answer based on the documentation above. If you cannot find the answer in the provided documents, clearly state that."""


@traigent.optimize(
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "temperature": [0.0, 0.1, 0.3],
        "top_k": [3, 5, 7, 10],
        "confidence_threshold": [0.5, 0.7, 0.85],
    },
    objectives=[
        "grounded_accuracy",
        "retrieval_quality",
        "abstention_accuracy",
        "cost",
    ],
    evaluation=EvaluationOptions(
        eval_dataset="use-cases/knowledge-rag/datasets/qa_dataset.jsonl",
        # RAGEvaluator has scoring_function interface: (prediction, expected, input_data) -> dict
        scoring_function=RAGEvaluator(),
    ),
    execution=ExecutionOptions(execution_mode="edge_analytics"),
)
def rag_qa_agent(question: str) -> dict[str, Any]:
    """
    Answer questions about CloudStack API using RAG.

    Args:
        question: User's question about CloudStack API

    Returns:
        Dictionary with 'answer', 'sources', and 'confidence'
    """
    # Get current configuration
    config = traigent.get_config()

    # Extract tuned variables with defaults
    model = config.get("model", "gpt-4o-mini")
    temperature = config.get("temperature", 0.1)
    top_k = config.get("top_k", 5)
    confidence_threshold = config.get("confidence_threshold", 0.7)

    # Load knowledge base
    documents = load_knowledge_base()

    # Retrieve relevant documents
    retrieved_docs = simple_retrieval(question, documents, top_k)

    # Format context from retrieved documents
    if retrieved_docs:
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(
                f"[Document {i}: {doc['title']} ({doc['section']})]\n{doc['content']}"
            )
        context = "\n\n".join(context_parts)
        source_ids = [doc["id"] for doc in retrieved_docs]
    else:
        context = "No relevant documents found."
        source_ids = []

    # Build the prompt
    prompt = RAG_PROMPT_TEMPLATE.format(
        confidence_threshold=confidence_threshold,
        context=context,
        question=question,
    )

    # Use LangChain for LLM call
    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
        )
        response = llm.invoke(prompt)
        answer = response.content

        # Determine if this is an abstention
        abstention_phrases = [
            "don't have information",
            "not mentioned",
            "cannot find",
            "no information",
            "not in the documentation",
            "unable to find",
        ]
        is_abstention = any(phrase in answer.lower() for phrase in abstention_phrases)

        # Estimate confidence based on retrieval quality
        confidence = min(len(retrieved_docs) / top_k, 1.0) if not is_abstention else 0.3

        return {
            "answer": answer,
            "sources": source_ids,
            "confidence": confidence,
            "retrieved_count": len(retrieved_docs),
            "is_abstention": is_abstention,
        }

    except ImportError:
        # Fallback for mock mode without LangChain
        return generate_mock_response(question, retrieved_docs, source_ids)


def generate_mock_response(
    question: str,
    retrieved_docs: list[dict],
    source_ids: list[str],
) -> dict[str, Any]:
    """Generate a mock response for testing without LLM."""
    if not retrieved_docs:
        return {
            "answer": "I don't have information about that in the documentation.",
            "sources": [],
            "confidence": 0.3,
            "retrieved_count": 0,
            "is_abstention": True,
        }

    # Generate answer from first retrieved document
    first_doc = retrieved_docs[0]
    answer = f"Based on the {first_doc['title']} documentation: {first_doc['content'][:200]}..."

    return {
        "answer": answer,
        "sources": source_ids,
        "confidence": 0.8,
        "retrieved_count": len(retrieved_docs),
        "is_abstention": False,
    }


async def run_optimization():
    """Run the RAG agent optimization."""
    print("=" * 60)
    print("Knowledge & RAG Agent - Traigent Optimization")
    print("=" * 60)

    # Check if mock mode is enabled
    mock_mode = os.environ.get("TRAIGENT_MOCK_MODE", "false").lower() == "true"
    print(f"\nMock Mode: {'Enabled' if mock_mode else 'Disabled'}")

    if not mock_mode:
        print("\nWARNING: Running without mock mode will incur API costs!")
        print("Set TRAIGENT_MOCK_MODE=true for testing.\n")

    print("\nStarting optimization...")
    print("Configuration Space:")
    print("  - Models: gpt-3.5-turbo, gpt-4o-mini, gpt-4o")
    print("  - Temperature: 0.0, 0.1, 0.3")
    print("  - Top-K Retrieval: 3, 5, 7, 10")
    print("  - Confidence Threshold: 0.5, 0.7, 0.85")
    print(
        "\nObjectives: grounded_accuracy, retrieval_quality, abstention_accuracy, cost"
    )
    print("-" * 60)

    # Run optimization
    results = await rag_qa_agent.optimize(
        algorithm="random",
        max_trials=20,
    )

    # Display results
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print("\nBest Configuration:")
    for key, value in results.best_config.items():
        print(f"  {key}: {value}")
    print(f"\nBest Score: {results.best_score:.4f}")

    # Apply best config
    rag_qa_agent.apply_best_config(results)
    print("\nBest configuration applied!")

    # Test with sample questions
    print("\n" + "-" * 60)
    print("Testing optimized agent with sample questions...")
    print("-" * 60)

    test_questions = [
        "What is the rate limit for the CloudStack API?",
        "How do I authenticate with the API?",
        "Does CloudStack support GraphQL?",  # Should abstain
    ]

    for question in test_questions:
        print(f"\nQ: {question}")
        result = rag_qa_agent(question)
        print(f"A: {result['answer'][:200]}...")
        print(f"   Sources: {result['sources'][:3]}")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Abstention: {result['is_abstention']}")

    return results


def main():
    """Main entry point."""
    asyncio.run(run_optimization())


if __name__ == "__main__":
    main()
