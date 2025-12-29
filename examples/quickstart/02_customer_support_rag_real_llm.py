#!/usr/bin/env python
"""
TraiGent Quickstart: Customer Support RAG with REAL LLM Calls

This example demonstrates RAG (Retrieval Augmented Generation) optimization
with actual OpenAI API calls, showcasing:
- RAG retrieval depth optimization (k parameter)
- Model and temperature tuning
- Multi-objective optimization (accuracy, cost, latency)
- Custom scoring for customer support quality

Run with:
    export OPENAI_API_KEY="your-key-here"
    python examples/quickstart/02_customer_support_rag_real_llm.py
"""

import asyncio
import csv
import json
import os
import sys
from pathlib import Path

from langchain_openai import ChatOpenAI

# Check for API key
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY environment variable not set.")
    print("Please run: export OPENAI_API_KEY='your-key-here'")
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


# Debug wrappers to log constraint checks (same logic, just adds print)
_constraint_results = {}  # Track constraint results by config tuple
_current_config_key = [None]  # Track current config being checked
_trial_counter = [0]  # Count trials that passed all constraints

def make_logged_constraint(name: str, constraint_fn, is_last: bool = False):
    """Wrap a constraint function with logging.

    Args:
        name: Name of the constraint for logging
        constraint_fn: The actual constraint function
        is_last: Set True for the last constraint - it will print trial assignment
    """
    def logged(cfg):
        cfg_key = (cfg.get("model"), cfg.get("temperature"), cfg.get("k"))

        # If this is a new config, start fresh
        if cfg_key != _current_config_key[0]:
            _current_config_key[0] = cfg_key
            if cfg_key not in _constraint_results:
                _constraint_results[cfg_key] = {}

        result = constraint_fn(cfg)
        _constraint_results[cfg_key][name] = result

        # Check if all constraints for this config have been evaluated
        all_checked = len(_constraint_results[cfg_key]) == 2  # We have 2 constraints
        all_passed = all(_constraint_results[cfg_key].values()) if all_checked else None

        # Always increment trial counter - TraiGent assigns trial numbers regardless of constraint result
        if is_last or not result:
            _trial_counter[0] += 1
            trial_num = _trial_counter[0]
            if not result:
                # Constraint failed - trial will be skipped (score=N/A)
                print(f"  [Trial {trial_num}] {cfg} -> REJECTED by '{name}' (will NOT run, score=N/A)")
            elif all_passed:
                # All constraints passed - trial will run
                print(f"  [Trial {trial_num}] {cfg} -> OK (will run)")

        return result
    return logged


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
        make_logged_constraint("gpt4o_temp<=0.3",
            lambda cfg: cfg["temperature"] <= 0.3 if cfg["model"] == "gpt-4o" else True),
        # GPT-3.5-turbo needs more context documents (is_last=True to log trial assignment)
        make_logged_constraint("gpt35_k>=3",
            lambda cfg: cfg["k"] >= 3 if cfg["model"] == "gpt-3.5-turbo" else True,
            is_last=True),
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
    max_trials=3,  # Testing: reduced to verify constraint slot consumption
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

    TraiGent automatically injects optimized parameters into this function.
    The model, temperature, and k (retrieval depth) will be varied during optimization.
    """
    # Retrieve relevant documents
    docs = simple_retriever(query, k=k)
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
    print("TraiGent Quickstart: Customer Support RAG (Real LLM)")
    print("=" * 60)
    print()

    print("Knowledge Base:")
    for i, doc in enumerate(KNOWLEDGE_BASE[:3], 1):
        print(f"  {i}. {doc[:50]}...")
    print(f"  ... and {len(KNOWLEDGE_BASE) - 3} more documents")
    print()

    print(f"Dataset: {RAG_DATASET_PATH}")
    print("Using real OpenAI API calls")
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

    # Test the retriever first
    print("Testing retriever:")
    test_query = "What is your return policy?"
    retrieved = simple_retriever(test_query, k=2)
    print(f"  Query: '{test_query}'")
    print(f"  Retrieved: {retrieved[:2]}...")
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

    # Show all trial results with constraint status
    if hasattr(results, "trials") and results.trials:
        print("All Trials (with constraint status):")
        print("-" * 40)
        for i, trial in enumerate(results.trials, 1):
            score = getattr(trial, "score", None) or getattr(trial, "metrics", {}).get(
                "score", "N/A"
            )
            config = getattr(trial, "config", getattr(trial, "configuration", {}))
            # Check constraint status for this config
            cfg_key = (config.get("model"), config.get("temperature"), config.get("k"))
            constraint_info = _constraint_results.get(cfg_key, {})
            failed_constraints = [k for k, v in constraint_info.items() if not v]
            if failed_constraints:
                constraint_status = f"VIOLATED: {', '.join(failed_constraints)}"
            else:
                constraint_status = "OK"
            print(f"  Trial {i}: {config} -> score={score} | constraints: {constraint_status}")

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
    print("RAG optimization complete with real LLM calls!")


if __name__ == "__main__":
    asyncio.run(main())
