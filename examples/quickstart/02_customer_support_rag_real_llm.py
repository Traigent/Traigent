#!/usr/bin/env python
"""
TraiGent Quickstart: Customer Support RAG with FAISS and Real LLM Calls

This example demonstrates real RAG (Retrieval Augmented Generation) optimization
with FAISS vector store, OpenAI embeddings, and actual LLM calls, showcasing:
- Semantic similarity search with FAISS
- OpenAI embeddings (text-embedding-3-small)
- RAG retrieval depth optimization (k parameter)
- Model and temperature tuning
- Multi-objective optimization (accuracy, cost, latency)

Run with:
    export OPENAI_API_KEY="your-key-here"
    pip install faiss-cpu  # if not already installed
    python examples/quickstart/02_customer_support_rag_real_llm.py
"""

import asyncio
import csv
import json
import os
import sys
from pathlib import Path

# Check for API key first
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY environment variable not set.")
    print("Please run: export OPENAI_API_KEY='your-key-here'")
    sys.exit(1)

# Check for FAISS
try:
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    print("Please install: pip install faiss-cpu langchain-community langchain-openai")
    sys.exit(1)

# Set results folder to local directory
os.environ.setdefault(
    "TRAIGENT_RESULTS_FOLDER", str(Path(__file__).parent / ".traigent_results")
)

import traigent
from traigent.api.decorators import (
    EvaluationOptions,
    ExecutionOptions,
    InjectionOptions,
)

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
        '{"input": {"query": "What are your business hours?"}, "output": "Monday-Friday 9am-5pm EST"}',
        '{"input": {"query": "Do gift cards expire?"}, "output": "Gift cards never expire"}',
        '{"input": {"query": "Do you price match?"}, "output": "Price match guarantee within 14 days of purchase"}',
    ]
    RAG_DATASET_PATH.write_text("\n".join(rag_data) + "\n")


# Knowledge base documents for the vector store
KNOWLEDGE_BASE_DOCS = [
    {"content": "Returns accepted within 30 days with original receipt. Items must be unused and in original packaging.", "category": "returns"},
    {"content": "Free shipping on orders over $50. Standard shipping takes 5-7 business days.", "category": "shipping"},
    {"content": "Track your order using the tracking link in your confirmation email. You can also check status on our website.", "category": "orders"},
    {"content": "We accept Visa, Mastercard, American Express, PayPal, and Apple Pay. All transactions are secure.", "category": "payments"},
    {"content": "Contact support at support@example.com or call 1-800-SUPPORT. Live chat available on our website.", "category": "support"},
    {"content": "Business hours: Monday-Friday 9am-5pm EST. Weekend support available via email only.", "category": "hours"},
    {"content": "Gift cards never expire and can be used online or in-store. Check balance at giftcards.example.com.", "category": "gift_cards"},
    {"content": "Price match guarantee within 14 days of purchase. Must be identical item from authorized retailer.", "category": "pricing"},
]


def create_vector_store() -> FAISS:
    """Create FAISS vector store with OpenAI embeddings.

    Uses text-embedding-3-small for cost-effective semantic embeddings.
    Cost: ~$0.00002 per 1K tokens (very cheap for small knowledge bases).
    """
    print("Creating FAISS vector store with OpenAI embeddings...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Convert to LangChain Documents with metadata
    documents = [
        Document(
            page_content=doc["content"],
            metadata={"category": doc["category"]}
        )
        for doc in KNOWLEDGE_BASE_DOCS
    ]

    vector_store = FAISS.from_documents(documents, embeddings)
    print(f"  Created vector store with {len(documents)} documents")
    return vector_store


def semantic_retriever(vector_store: FAISS, query: str, k: int = 3) -> list[str]:
    """Retrieve documents using semantic similarity search.

    Unlike keyword matching, this finds semantically similar content
    even if the exact words don't match.
    """
    docs = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]


def rag_accuracy_scorer(output: str, expected: str) -> float:
    """Score RAG response quality.

    Checks if key information from expected output appears in actual output.

    Args:
        output: The actual LLM output (TraiGent passes this automatically)
        expected: The expected output from dataset (TraiGent passes this automatically)
    """
    if not output or not expected:
        return 0.0

    actual_lower = output.lower()
    expected_lower = expected.lower()

    # Extract key terms from expected output
    key_terms = [word for word in expected_lower.split() if len(word) > 3]
    if not key_terms:
        return 1.0 if expected_lower in actual_lower else 0.0

    # Score based on key term matches
    matches = sum(1 for term in key_terms if term in actual_lower)
    return matches / len(key_terms)


# Configuration space - single source of truth for all config values
CONFIGURATION_SPACE = {
    "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
    "temperature": [0.1, 0.3, 0.5, 0.7],
    "k": [2, 3, 5],  # RAG retrieval depth
}

DEFAULT_CONFIG = {
    "model": "gpt-3.5-turbo",
    "temperature": 0.3,
    "k": 3,
}

# Constraints as descriptive strings for printing
CONSTRAINTS_DESCRIPTIONS = [
    "GPT-4o: temperature <= 0.3 (expensive model, keep focused)",
    "GPT-3.5-turbo: k >= 3 (needs more context)",
]

# Global vector store (initialized once)
_vector_store: FAISS | None = None


def get_vector_store() -> FAISS:
    """Get or create the global vector store."""
    global _vector_store
    if _vector_store is None:
        _vector_store = create_vector_store()
    return _vector_store


@traigent.optimize(
    # Configuration space: define the search space for optimization
    configuration_space=CONFIGURATION_SPACE,
    # Default configuration: starting point before optimization
    default_config=DEFAULT_CONFIG,
    # Multi-objective optimization
    objectives=["accuracy", "cost", "latency"],
    # Constraints: skip configurations that don't make sense
    constraints=[
        # Don't use high temperature with GPT-4o (expensive + unpredictable)
        lambda cfg: cfg["temperature"] <= 0.3 if cfg["model"] == "gpt-4o" else True,
        # GPT-3.5-turbo needs more context documents
        lambda cfg: cfg["k"] >= 3 if cfg["model"] == "gpt-3.5-turbo" else True,
    ],
    # Custom metric functions for accuracy
    metric_functions={"accuracy": rag_accuracy_scorer},
    # Evaluation dataset
    evaluation=EvaluationOptions(eval_dataset=str(RAG_DATASET_PATH)),
    # Execution options with statistical stability
    execution=ExecutionOptions(
        execution_mode="edge_analytics",
        minimal_logging=False,
        reps_per_trial=2,  # Run each config twice for stability
        reps_aggregation="mean",
    ),
    # Injection options: how configs are applied
    injection=InjectionOptions(
        auto_override_frameworks=True,
    ),
    # Runtime controls
    max_trials=6,
    timeout=600,  # Override default 60s to allow all trials to complete
    cost_limit=5.00,
    cost_approved=True,
)
def customer_support_agent(
    query: str,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.3,
    k: int = 3,
) -> str:
    """Answer customer questions using RAG with real LLM calls.

    This function uses:
    - FAISS for semantic similarity search
    - OpenAI embeddings for document vectors
    - LangChain ChatOpenAI for LLM responses

    TraiGent automatically injects optimized parameters into this function.
    The model, temperature, and k (retrieval depth) will be varied during optimization.
    """
    # Get vector store and retrieve relevant documents
    vector_store = get_vector_store()
    docs = semantic_retriever(vector_store, query, k=k)
    context = "\n".join(f"- {doc}" for doc in docs)

    # Build prompt with retrieved context
    prompt = f"""You are a helpful customer support agent. Answer the customer's question
based ONLY on the following knowledge base information.

Knowledge Base:
{context}

Customer Question: {query}

Provide a concise, helpful answer based on the information above. If the information
doesn't fully answer the question, say what you can based on what's available."""

    # Make real LLM call
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=150,
    )
    response = llm.invoke(prompt)
    return str(response.content)


def save_results_to_csv(results, dataset_path: Path, output_path: Path) -> None:
    """Save optimization results to a CSV file."""
    # Load questions from dataset
    questions = []
    expected_answers = []
    with open(dataset_path) as f:
        for line in f:
            data = json.loads(line)
            questions.append(data["input"]["query"])
            expected_answers.append(data["output"])

    # Build CSV data structure
    headers = ["Query", "Expected"]
    trial_configs = []

    for i, trial in enumerate(results.trials, 1):
        config = getattr(trial, "config", getattr(trial, "configuration", {}))
        config_str = f"T{i}: {config.get('model', 'N/A')}, k={config.get('k', 'N/A')}"
        headers.append(f"{config_str} Answer")
        headers.append(f"{config_str} Pass")
        trial_configs.append(config)

    # Initialize rows with questions and expected answers
    rows = []
    for q_idx, (question, expected) in enumerate(zip(questions, expected_answers)):
        question_clean = question.replace("\n", " | ").replace("\r", "")
        expected_clean = expected.replace("\n", " | ").replace("\r", "")
        rows.append([question_clean, expected_clean])

    # Fill in trial answers and pass/fail
    trial_pass_counts = []

    for trial in results.trials:
        example_results = trial.metadata.get("example_results", [])

        results_by_id = {}
        for ex_result in example_results:
            ex_idx = int(ex_result.example_id.split("_")[1])
            results_by_id[ex_idx] = ex_result

        pass_count = 0
        total_count = 0
        for q_idx in range(len(questions)):
            ex_result = results_by_id.get(q_idx)
            if ex_result:
                answer = str(ex_result.actual_output) if ex_result.actual_output else ""
                answer = answer.replace("\n", " | ").replace("\r", "")
                metrics = getattr(ex_result, "metrics", {}) or {}
                score = metrics.get("accuracy", metrics.get("score", 0))
                passed = score >= 0.5 if isinstance(score, (int, float)) else False
                rows[q_idx].append(answer)
                rows[q_idx].append("PASS" if passed else "FAIL")
                total_count += 1
                if passed:
                    pass_count += 1
            else:
                rows[q_idx].append("N/A")
                rows[q_idx].append("")

        trial_pass_counts.append((pass_count, total_count))

    # Add summary row
    summary_row = ["SUMMARY", "Pass Rate"]
    for pass_count, total_count in trial_pass_counts:
        if total_count > 0:
            ratio = pass_count / total_count
            summary_row.append(f"{pass_count}/{total_count}")
            summary_row.append(f"{ratio:.1%}")
        else:
            summary_row.append("N/A")
            summary_row.append("N/A")

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(headers)
        writer.writerows(rows)
        writer.writerow(summary_row)

    print(f"Results saved to: {output_path}")


def print_model_summary(results) -> None:
    """Print accuracy breakdown by model and k value."""
    model_results = {}
    k_results = {}
    total_cost = 0.0

    for trial in results.trials:
        config = getattr(trial, "config", getattr(trial, "configuration", {}))
        model = config.get("model", "unknown")
        k_val = config.get("k", "unknown")
        example_results = trial.metadata.get("example_results", [])

        trial_cost = trial.metadata.get("total_example_cost", 0)
        if trial_cost:
            total_cost += float(trial_cost)

        for ex_result in example_results:
            metrics = getattr(ex_result, "metrics", {}) or {}
            score = metrics.get("accuracy", metrics.get("score", 0))
            passed = score >= 0.5 if isinstance(score, (int, float)) else False

            if model not in model_results:
                model_results[model] = []
            model_results[model].append(passed)

            if k_val not in k_results:
                k_results[k_val] = []
            k_results[k_val].append(passed)

    print()
    print("=" * 60)
    print("Results Summary")
    print("=" * 60)
    print()

    print("Accuracy by Model:")
    for model in sorted(model_results.keys()):
        passes = model_results[model]
        accuracy = sum(passes) / len(passes) * 100 if passes else 0
        print(f"  {model}: {accuracy:.1f}% ({sum(passes)}/{len(passes)})")

    print()
    print("Accuracy by Retrieval Depth (k):")
    for k_val in sorted(k_results.keys()):
        passes = k_results[k_val]
        accuracy = sum(passes) / len(passes) * 100 if passes else 0
        print(f"  k={k_val}: {accuracy:.1f}% ({sum(passes)}/{len(passes)})")

    print()
    print(f"Total Cost: ${total_cost:.4f}")
    print()


async def main():
    print("=" * 60)
    print("TraiGent Quickstart: Customer Support RAG (FAISS + Real LLM)")
    print("=" * 60)
    print()

    print("Knowledge Base:")
    for i, doc in enumerate(KNOWLEDGE_BASE_DOCS[:3], 1):
        print(f"  {i}. [{doc['category']}] {doc['content'][:50]}...")
    print(f"  ... and {len(KNOWLEDGE_BASE_DOCS) - 3} more documents")
    print()

    print(f"Dataset: {RAG_DATASET_PATH}")
    print("Using FAISS vector store with OpenAI embeddings")
    print()

    print("Configuration Space:")
    print(f"  - Models: {', '.join(CONFIGURATION_SPACE['model'])}")
    print(f"  - Temperature: {', '.join(str(t) for t in CONFIGURATION_SPACE['temperature'])}")
    print(f"  - Retrieval Depth (k): {', '.join(str(k) for k in CONFIGURATION_SPACE['k'])}")
    print()

    print("Constraints Applied:")
    for constraint in CONSTRAINTS_DESCRIPTIONS:
        print(f"  - {constraint}")
    print()

    # Initialize vector store before optimization
    vector_store = get_vector_store()
    print()

    # Test the semantic retriever
    print("Testing semantic retriever:")
    test_query = "How do I get my money back?"  # Doesn't contain "return" but semantically related
    retrieved = semantic_retriever(vector_store, test_query, k=2)
    print(f"  Query: '{test_query}'")
    print(f"  Retrieved (semantic match):")
    for doc in retrieved:
        print(f"    - {doc[:60]}...")
    print()

    # Run optimization
    print("Starting RAG optimization...")
    print("-" * 40)
    results = await customer_support_agent.optimize()

    print()
    print("=" * 60)
    print("Optimization Complete!")
    print("=" * 60)
    print()
    print(f"Best Score: {results.best_score}")
    print(f"Best Configuration: {results.best_config}")
    print()

    # Show all trial results
    if hasattr(results, "trials") and results.trials:
        print("All Trials:")
        print("-" * 40)
        for i, trial in enumerate(results.trials, 1):
            score = getattr(trial, "score", None) or getattr(trial, "metrics", {}).get(
                "score", "N/A"
            )
            config = getattr(trial, "config", getattr(trial, "configuration", {}))
            print(f"  Trial {i}: {config} -> score={score}")

    # Save results to CSV
    csv_output_path = Path(__file__).parent / "results" / "rag_optimization_results.csv"
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results_to_csv(results, RAG_DATASET_PATH, csv_output_path)

    # Save best config to JSON
    best_config_path = Path(__file__).parent / "results" / "rag_best_config.json"
    with open(best_config_path, "w") as f:
        json.dump({
            "best_score": results.best_score,
            "best_config": results.best_config,
        }, f, indent=2)
    print(f"Best config saved to: {best_config_path}")

    # Print summary
    print_model_summary(results)

    # Demonstrate using the optimized agent
    print("=" * 60)
    print("Using Optimized Configuration")
    print("=" * 60)
    print()

    best_config = customer_support_agent.get_best_config()
    if best_config:
        print(f"Applying best config: {best_config}")
        customer_support_agent.set_config(best_config)

        test_questions = [
            "What is your return policy?",
            "Do you offer free shipping?",
            "How do I contact support?",
        ]
        for question in test_questions:
            print(f"\nQ: {question}")
            answer = customer_support_agent(question)
            print(f"A: {answer}")

    print()
    print("RAG optimization complete with FAISS and real LLM calls!")


if __name__ == "__main__":
    asyncio.run(main())
