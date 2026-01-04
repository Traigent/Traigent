#!/usr/bin/env python
"""
TraiGent Quickstart: Customer Support RAG with FAISS and Real LLM Calls

This example demonstrates production-ready RAG (Retrieval Augmented Generation)
optimization with FAISS vector store, OpenAI embeddings, and document chunking:

- Document chunking for large texts (RecursiveCharacterTextSplitter)
- Semantic similarity search with FAISS
- OpenAI embeddings (text-embedding-3-small)
- Chunk size and overlap optimization (chunk_size, chunk_overlap parameters)
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

# Check for FAISS and text splitter
try:
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    print(
        "Please install: pip install faiss-cpu langchain-community langchain-openai langchain-text-splitters"
    )
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


# =============================================================================
# KNOWLEDGE BASE - Load documents from files
# =============================================================================
KNOWLEDGE_BASE_PATH = Path(__file__).parent / "knowledge_base"


def load_documents_from_directory(base_path: Path) -> list[dict]:
    """Load documents from directory structure.

    Supports: .txt, .md files
    Category inferred from subfolder name.

    Directory structure:
        knowledge_base/
        ├── returns/
        │   └── returns_policy.md
        ├── shipping/
        │   └── shipping_info.md
        └── ...
    """
    documents = []
    if not base_path.exists():
        return documents

    for category_dir in sorted(base_path.iterdir()):
        if category_dir.is_dir():
            category = category_dir.name
            for file_path in sorted(category_dir.glob("*")):
                if file_path.suffix in (".txt", ".md"):
                    documents.append(
                        {
                            "source": file_path.name,
                            "category": category,
                            "content": file_path.read_text(),
                            "file_path": str(file_path),
                        }
                    )
    return documents


# Load documents from knowledge_base directory
KNOWLEDGE_BASE_DOCUMENTS = load_documents_from_directory(KNOWLEDGE_BASE_PATH)

if not KNOWLEDGE_BASE_DOCUMENTS:
    print(f"ERROR: No documents found in {KNOWLEDGE_BASE_PATH}")
    print("Please ensure the knowledge_base/ directory exists with .md or .txt files.")
    sys.exit(1)


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Document]:
    """Split long documents into smaller chunks for better retrieval.

    Args:
        documents: List of document dicts with 'content', 'source', 'category'
        chunk_size: Maximum characters per chunk (default 500)
        chunk_overlap: Overlap between chunks to preserve context (default 50)

    Returns:
        List of LangChain Document objects with metadata
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
    )

    all_chunks = []
    for doc in documents:
        chunks = text_splitter.split_text(doc["content"])
        for i, chunk in enumerate(chunks):
            all_chunks.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": doc["source"],
                        "category": doc["category"],
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    },
                )
            )

    return all_chunks


# Cache for vector stores by (chunk_size, chunk_overlap) tuple
_vector_store_cache: dict[tuple[int, int], FAISS] = {}


def create_vector_store(chunk_size: int = 500, chunk_overlap: int = 50) -> FAISS:
    """Create FAISS vector store with OpenAI embeddings.

    Args:
        chunk_size: Size of document chunks (affects retrieval quality)
        chunk_overlap: Overlap between chunks (sliding window for context preservation)

    Uses text-embedding-3-small for cost-effective semantic embeddings.
    Cost: ~$0.00002 per 1K tokens (very cheap for small knowledge bases).
    """
    print(
        f"Creating FAISS vector store (chunk_size={chunk_size}, overlap={chunk_overlap})..."
    )
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Chunk documents
    documents = chunk_documents(
        KNOWLEDGE_BASE_DOCUMENTS,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    vector_store = FAISS.from_documents(documents, embeddings)
    print(
        f"  Created vector store with {len(documents)} chunks from {len(KNOWLEDGE_BASE_DOCUMENTS)} documents"
    )
    return vector_store


def get_vector_store(chunk_size: int = 500, chunk_overlap: int = 50) -> FAISS:
    """Get or create vector store for given chunk_size and chunk_overlap."""
    cache_key = (chunk_size, chunk_overlap)
    if cache_key not in _vector_store_cache:
        _vector_store_cache[cache_key] = create_vector_store(chunk_size, chunk_overlap)
    return _vector_store_cache[cache_key]


def semantic_retriever(vector_store: FAISS, query: str, k: int = 3) -> list[str]:
    """Retrieve documents using semantic similarity search.

    Unlike keyword matching, this finds semantically similar content
    even if the exact words don't match.
    """
    docs = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]


# LLM-as-Judge for accurate scoring
_judge_llm = None


def get_judge_llm():
    """Get or create the judge LLM (cached for efficiency)."""
    global _judge_llm
    if _judge_llm is None:
        # Use a cheap, fast model for judging
        _judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=10)
    return _judge_llm


def rag_accuracy_scorer(output: str, expected: str) -> float:
    """Score RAG response quality using LLM-as-Judge.

    Uses a separate LLM call to determine if the generated answer
    correctly conveys the same information as the expected answer.

    Args:
        output: The actual LLM output (TraiGent passes this automatically)
        expected: The expected output from dataset (TraiGent passes this automatically)

    Returns:
        1.0 if correct, 0.0 if incorrect
    """
    if not output or not expected:
        return 0.0

    judge = get_judge_llm()

    prompt = f"""You are evaluating if an AI assistant's answer is correct.

Expected answer: {expected}
Actual answer: {output}

Does the actual answer correctly convey the same key information as the expected answer?
- Minor wording differences are OK
- The actual answer may include extra details, that's OK
- The core facts must match

Reply with only "YES" or "NO"."""

    try:
        response = judge.invoke(prompt)
        result = str(response.content).strip().upper()
        return 1.0 if result.startswith("YES") else 0.0
    except Exception:
        # Fallback to keyword matching if judge fails
        actual_lower = output.lower()
        expected_lower = expected.lower()
        key_terms = [word for word in expected_lower.split() if len(word) > 3]
        if not key_terms:
            return 1.0 if expected_lower in actual_lower else 0.0
        matches = sum(1 for term in key_terms if term in actual_lower)
        return matches / len(key_terms)


# =============================================================================
# Configuration Space - including chunk_size and chunk_overlap for RAG optimization
# =============================================================================
CONFIGURATION_SPACE = {
    "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
    "temperature": [0.1, 0.3, 0.5, 0.7],
    "k": [2, 3, 5],  # Number of chunks to retrieve
    "chunk_size": [300, 500, 800],  # Characters per chunk
    "chunk_overlap": [25, 50, 100],  # Sliding window overlap (context preservation)
}

DEFAULT_CONFIG = {
    "model": "gpt-3.5-turbo",
    "temperature": 0.3,
    "k": 3,
    "chunk_size": 500,
    "chunk_overlap": 50,
}

# Constraints as descriptive strings for printing
CONSTRAINTS_DESCRIPTIONS = [
    "GPT-4o: temperature <= 0.3 (expensive model, keep focused)",
    "GPT-3.5-turbo: k >= 3 (needs more context)",
    "chunk_overlap < chunk_size (overlap must be smaller than chunk)",
]


@traigent.optimize(
    configuration_space=CONFIGURATION_SPACE,
    default_config=DEFAULT_CONFIG,
    objectives=["accuracy", "cost", "latency"],
    constraints=[
        # Don't use high temperature with GPT-4o (expensive + unpredictable)
        lambda cfg: cfg["temperature"] <= 0.3 if cfg["model"] == "gpt-4o" else True,
        # GPT-3.5-turbo needs more context documents
        lambda cfg: cfg["k"] >= 3 if cfg["model"] == "gpt-3.5-turbo" else True,
        # Overlap must be smaller than chunk size
        lambda cfg: cfg["chunk_overlap"] < cfg["chunk_size"],
    ],
    metric_functions={"accuracy": rag_accuracy_scorer},
    evaluation=EvaluationOptions(eval_dataset=str(RAG_DATASET_PATH)),
    execution=ExecutionOptions(
        execution_mode="edge_analytics",
        minimal_logging=False,
        # reps_per_trial/reps_aggregation removed - Enterprise only
    ),
    injection=InjectionOptions(
        auto_override_frameworks=True,
    ),
    max_trials=20,
    # Note: timeout/cost_limit/cost_approved must be set via env vars:
    # TRAIGENT_RUN_COST_LIMIT=5.00 TRAIGENT_COST_APPROVED=true timeout 600 python ...
)
def customer_support_agent(
    query: str,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.3,
    k: int = 3,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> str:
    """Answer customer questions using RAG with real LLM calls.

    This function uses:
    - Document chunking with configurable chunk_size and chunk_overlap
    - FAISS for semantic similarity search
    - OpenAI embeddings for document vectors
    - LangChain ChatOpenAI for LLM responses

    TraiGent automatically injects optimized parameters:
    - model: Which LLM to use
    - temperature: Creativity vs consistency
    - k: How many chunks to retrieve
    - chunk_size: Size of document chunks (affects retrieval precision)
    - chunk_overlap: Sliding window overlap (preserves context at boundaries)
    """
    # Get vector store for this chunk_size/overlap and retrieve relevant chunks
    vector_store = get_vector_store(chunk_size, chunk_overlap)
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
    questions = []
    expected_answers = []
    with open(dataset_path) as f:
        for line in f:
            data = json.loads(line)
            questions.append(data["input"]["query"])
            expected_answers.append(data["output"])

    headers = ["Query", "Expected"]
    trial_configs = []

    for i, trial in enumerate(results.trials, 1):
        config = getattr(trial, "config", getattr(trial, "configuration", {}))
        config_str = f"T{i}: {config.get('model', 'N/A')}, k={config.get('k', 'N/A')}, chunk={config.get('chunk_size', 'N/A')}"
        headers.append(f"{config_str} Answer")
        headers.append(f"{config_str} Pass")
        trial_configs.append(config)

    rows = []
    for q_idx, (question, expected) in enumerate(zip(questions, expected_answers)):
        question_clean = question.replace("\n", " | ").replace("\r", "")
        expected_clean = expected.replace("\n", " | ").replace("\r", "")
        rows.append([question_clean, expected_clean])

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

    summary_row = ["SUMMARY", "Pass Rate"]
    for pass_count, total_count in trial_pass_counts:
        if total_count > 0:
            ratio = pass_count / total_count
            summary_row.append(f"{pass_count}/{total_count}")
            summary_row.append(f"{ratio:.1%}")
        else:
            summary_row.append("N/A")
            summary_row.append("N/A")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(headers)
        writer.writerows(rows)
        writer.writerow(summary_row)

    print(f"Results saved to: {output_path}")


def print_model_summary(results) -> None:
    """Print accuracy breakdown by model, k value, chunk_size, and chunk_overlap."""
    model_results = {}
    k_results = {}
    chunk_size_results = {}
    chunk_overlap_results = {}
    total_cost = 0.0

    for trial in results.trials:
        config = getattr(trial, "config", getattr(trial, "configuration", {}))
        model = config.get("model", "unknown")
        k_val = config.get("k", "unknown")
        chunk_size_val = config.get("chunk_size", "unknown")
        chunk_overlap_val = config.get("chunk_overlap", "unknown")
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

            if chunk_size_val not in chunk_size_results:
                chunk_size_results[chunk_size_val] = []
            chunk_size_results[chunk_size_val].append(passed)

            if chunk_overlap_val not in chunk_overlap_results:
                chunk_overlap_results[chunk_overlap_val] = []
            chunk_overlap_results[chunk_overlap_val].append(passed)

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
    print("Accuracy by Chunk Size:")
    for chunk_val in sorted(chunk_size_results.keys()):
        passes = chunk_size_results[chunk_val]
        accuracy = sum(passes) / len(passes) * 100 if passes else 0
        print(
            f"  chunk_size={chunk_val}: {accuracy:.1f}% ({sum(passes)}/{len(passes)})"
        )

    print()
    print("Accuracy by Chunk Overlap:")
    for overlap_val in sorted(chunk_overlap_results.keys()):
        passes = chunk_overlap_results[overlap_val]
        accuracy = sum(passes) / len(passes) * 100 if passes else 0
        print(
            f"  chunk_overlap={overlap_val}: {accuracy:.1f}% ({sum(passes)}/{len(passes)})"
        )

    print()
    print(f"Total Cost: ${total_cost:.4f}")
    print()


async def main():
    print("=" * 60)
    print("TraiGent Quickstart: Customer Support RAG (FAISS + Chunking)")
    print("=" * 60)
    print()

    # Show knowledge base info
    total_chars = sum(len(doc["content"]) for doc in KNOWLEDGE_BASE_DOCUMENTS)
    loaded_from = (
        "files"
        if KNOWLEDGE_BASE_DOCUMENTS and "file_path" in KNOWLEDGE_BASE_DOCUMENTS[0]
        else "fallback"
    )
    print(
        f"Knowledge Base: {len(KNOWLEDGE_BASE_DOCUMENTS)} documents, {total_chars:,} total characters (loaded from {loaded_from})"
    )
    if loaded_from == "files":
        print(f"  Source: {KNOWLEDGE_BASE_PATH}")
    for i, doc in enumerate(KNOWLEDGE_BASE_DOCUMENTS[:3], 1):
        print(
            f"  {i}. [{doc['category']}] {doc['source']} ({len(doc['content']):,} chars)"
        )
    if len(KNOWLEDGE_BASE_DOCUMENTS) > 3:
        print(f"  ... and {len(KNOWLEDGE_BASE_DOCUMENTS) - 3} more documents")
    print()

    print(f"Dataset: {RAG_DATASET_PATH}")
    print("Using FAISS vector store with OpenAI embeddings + document chunking")
    print()

    print("Configuration Space:")
    print(f"  - Models: {', '.join(CONFIGURATION_SPACE['model'])}")
    print(
        f"  - Temperature: {', '.join(str(t) for t in CONFIGURATION_SPACE['temperature'])}"
    )
    print(
        f"  - Retrieval Depth (k): {', '.join(str(k) for k in CONFIGURATION_SPACE['k'])}"
    )
    print(
        f"  - Chunk Size: {', '.join(str(c) for c in CONFIGURATION_SPACE['chunk_size'])}"
    )
    print(
        f"  - Chunk Overlap: {', '.join(str(o) for o in CONFIGURATION_SPACE['chunk_overlap'])}"
    )
    print()

    print("Constraints Applied:")
    for constraint in CONSTRAINTS_DESCRIPTIONS:
        print(f"  - {constraint}")
    print()

    # Show chunking example
    print("Document Chunking Example (chunk_size=500):")
    example_chunks = chunk_documents(KNOWLEDGE_BASE_DOCUMENTS[:1], chunk_size=500)
    print(
        f"  '{KNOWLEDGE_BASE_DOCUMENTS[0]['source']}' ({len(KNOWLEDGE_BASE_DOCUMENTS[0]['content']):,} chars) -> {len(example_chunks)} chunks"
    )
    for i, chunk in enumerate(example_chunks[:2]):
        preview = chunk.page_content[:80].replace("\n", " ")
        print(f'    Chunk {i+1}: "{preview}..."')
    if len(example_chunks) > 2:
        print(f"    ... and {len(example_chunks) - 2} more chunks")
    print()

    # Initialize default vector store
    vector_store = get_vector_store(chunk_size=500, chunk_overlap=50)
    print()

    # Test semantic retriever
    print("Testing semantic retriever:")
    test_query = "How do I get my money back?"
    retrieved = semantic_retriever(vector_store, test_query, k=2)
    print(f"  Query: '{test_query}'")
    print(f"  Retrieved (semantic match):")
    for doc in retrieved:
        preview = doc[:80].replace("\n", " ")
        print(f'    - "{preview}..."')
    print()

    # Run optimization
    print("Starting RAG optimization...")
    print("-" * 40)
    results = await customer_support_agent.optimize(timeout=600)

    print()
    print("=" * 60)
    print("Optimization Complete!")
    print("=" * 60)
    print()
    print(f"Best Score: {results.best_score}")
    print(f"Best Configuration: {results.best_config}")
    print()

    if hasattr(results, "trials") and results.trials:
        print("All Trials:")
        print("-" * 40)
        for i, trial in enumerate(results.trials, 1):
            score = getattr(trial, "score", None) or getattr(trial, "metrics", {}).get(
                "score", "N/A"
            )
            config = getattr(trial, "config", getattr(trial, "configuration", {}))
            print(f"  Trial {i}: {config} -> score={score}")

    # Save results
    csv_output_path = Path(__file__).parent / "results" / "rag_optimization_results.csv"
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results_to_csv(results, RAG_DATASET_PATH, csv_output_path)

    best_config_path = Path(__file__).parent / "results" / "rag_best_config.json"
    with open(best_config_path, "w") as f:
        json.dump(
            {
                "best_score": results.best_score,
                "best_config": results.best_config,
            },
            f,
            indent=2,
        )
    print(f"Best config saved to: {best_config_path}")

    print_model_summary(results)

    # Demo with optimized config
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
    print("RAG optimization complete with FAISS, chunking, and real LLM calls!")


if __name__ == "__main__":
    asyncio.run(main())
