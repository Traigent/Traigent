#!/usr/bin/env python
"""
TraiGent Quickstart: Simple Q&A Agent with REAL LLM Calls

This example demonstrates TraiGent optimization with actual OpenAI API calls,
showcasing various SDK capabilities including:
- Multi-objective optimization (accuracy, cost, latency)
- Configuration constraints
- Multiple repetitions per trial for statistical stability
- Custom scoring functions
- Cost limits and safeguards

Run with:
    export OPENAI_API_KEY="your-key-here"
    python examples/quickstart/01_simple_qa_real_llm.py
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
    ExecutionOptions,
    InjectionOptions,
)

from scorers import custom_accuracy_scorer

# Path to dataset (relative to this file)
# Use hle_20.jsonl for quick HLE test, hle.jsonl for full HLE run
# Use llm_eval_mixed_20.jsonl for quick test, llm_eval_mixed.jsonl for full run
DATASET_PATH = (
    Path(__file__).parent.parent / "datasets" / "quickstart" / "hle_20.jsonl"
    # Path(__file__).parent.parent / "datasets" / "quickstart" / "hle.jsonl"
    # Path(__file__).parent.parent / "datasets" / "quickstart" / "llm_eval_mixed_20.jsonl"
    # Path(__file__).parent.parent / "datasets" / "quickstart" / "llm_eval_mixed.jsonl"
)


# Configuration space - single source of truth for all config values
CONFIGURATION_SPACE = {
    "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"], # try 5.2
    "temperature": [0.1, 0.3, 0.5, 0.7, 0.9],
    "max_tokens": [50, 100, 200],
}

DEFAULT_CONFIG = {
    "model": "gpt-3.5-turbo",
    "temperature": 0.7,
    "max_tokens": 100,
}

# Constraints as descriptive strings for printing
CONSTRAINTS_DESCRIPTIONS = [
    "GPT-4o: temperature <= 0.5",
    "GPT-3.5-turbo: max_tokens <= 100",
]


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
        lambda cfg: cfg["temperature"] <= 0.5 if cfg["model"] == "gpt-4o" else True,
        # Ensure max_tokens is reasonable for the model tier
        lambda cfg: (
            cfg["max_tokens"] <= 100 if cfg["model"] == "gpt-3.5-turbo" else True
        ),
    ],
    # Custom metric functions for accuracy (takes output, expected)
    metric_functions={"accuracy": custom_accuracy_scorer},
    # Evaluation dataset
    eval_dataset=str(DATASET_PATH),
    # Execution options with statistical stability
    execution=ExecutionOptions(
        execution_mode="edge_analytics",
        minimal_logging=False,  # Show detailed logs
        reps_per_trial=2,  # Run each config twice for stability
        reps_aggregation="mean",  # Average the results
    ),
    # Injection options: how configs are applied
    injection=InjectionOptions(
        auto_override_frameworks=True,  # Auto-detect LangChain/OpenAI calls
    ),
    # Runtime controls
    max_trials=5,  # Limit trials for quick demo
    cost_limit=10.00,  # Max $1.00 spend per optimization run
    cost_approved=True,  # Skip cost approval prompt for demo
)
def simple_qa_agent(
    question: str,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.7,
    max_tokens: int = 100,
) -> str:
    """Simple Q&A agent with real LLM calls.

    TraiGent automatically injects optimized parameters into this function.
    The model, temperature, and max_tokens will be varied during optimization.
    """
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    response = llm.invoke(f"Question: {question}\nAnswer concisely:")
    return str(response.content)


def save_results_to_csv(results, dataset_path: Path, output_path: Path) -> None:
    """Save optimization results to a CSV file.

    Creates a CSV with questions as rows and trials as columns.
    Each trial has two columns: Answer and Pass (True/False).

    Args:
        results: OptimizationResult from the optimization run
        dataset_path: Path to the JSONL dataset file
        output_path: Path where the CSV file will be saved
    """
    # Load questions from dataset
    questions = []
    expected_answers = []
    with open(dataset_path) as f:
        for line in f:
            data = json.loads(line)
            questions.append(data["input"]["question"])
            expected_answers.append(data["output"])

    # Build CSV data structure
    # Columns: Question, Expected, Trial1 Answer, Trial1 Pass, Trial2 Answer, Trial2 Pass, ...
    headers = ["Question", "Expected"]
    trial_configs = []

    for i, trial in enumerate(results.trials, 1):
        config = getattr(trial, "config", getattr(trial, "configuration", {}))
        config_str = f"T{i}: {config.get('model', 'N/A')}, t={config.get('temperature', 'N/A')}"
        headers.append(f"{config_str} Answer")
        headers.append(f"{config_str} Pass")
        trial_configs.append(config)

    # Initialize rows with questions and expected answers
    rows = []
    for q_idx, (question, expected) in enumerate(zip(questions, expected_answers)):
        # Sanitize newlines for CSV compatibility with LibreOffice/Excel
        question_clean = question.replace("\n", " | ").replace("\r", "")
        expected_clean = expected.replace("\n", " | ").replace("\r", "")
        rows.append([question_clean, expected_clean])

    # Fill in trial answers and pass/fail, tracking pass counts per trial
    trial_pass_counts = []  # List of (pass_count, total_count) per trial

    for trial in results.trials:
        example_results = trial.metadata.get("example_results", [])

        # Create lookup by example_id
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
                # Sanitize newlines for CSV compatibility
                answer = answer.replace("\n", " | ").replace("\r", "")
                # Use evaluator score for pass/fail, not just "success" (which means call succeeded)
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

    # Add summary row with pass ratios
    summary_row = ["SUMMARY", "Pass Rate"]
    for pass_count, total_count in trial_pass_counts:
        if total_count > 0:
            ratio = pass_count / total_count
            summary_row.append(f"{pass_count}/{total_count}")
            summary_row.append(f"{ratio:.1%}")
        else:
            summary_row.append("N/A")
            summary_row.append("N/A")

    # Write CSV with quoting to handle semicolons and other special chars
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(headers)
        writer.writerows(rows)
        writer.writerow(summary_row)

    print(f"Results saved to: {output_path}")


def save_detailed_results_csv(results, dataset_path: Path, output_path: Path) -> None:
    """Save detailed optimization results with one row per question × trial.

    Includes all available metrics: pass/fail, score, source, category,
    latency, tokens, and cost.

    Args:
        results: OptimizationResult from the optimization run
        dataset_path: Path to the JSONL dataset file
        output_path: Path where the CSV file will be saved
    """
    # Load dataset metadata (source, category)
    dataset_metadata = []
    with open(dataset_path) as f:
        for line in f:
            data = json.loads(line)
            dataset_metadata.append(
                {
                    "question": data["input"]["question"],
                    "expected": data["output"],
                    "source": data.get("source", "unknown"),
                    "category": data.get("category", "unknown"),
                }
            )

    # Build rows: one per (question, trial) combination
    headers = [
        "question_id",
        "question",
        "expected",
        "actual",
        "pass",
        "score",
        "source",
        "category",
        "trial_id",
        "model",
        "temperature",
        "max_tokens",
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "total_cost",
    ]
    rows = []

    for trial_idx, trial in enumerate(results.trials, 1):
        config = getattr(trial, "config", getattr(trial, "configuration", {}))
        example_results = trial.metadata.get("example_results", [])

        # Create a lookup by example_id for faster access
        results_by_id = {}
        for ex_result in example_results:
            ex_idx = int(ex_result.example_id.split("_")[1])
            results_by_id[ex_idx] = ex_result

        for q_idx, meta in enumerate(dataset_metadata):
            ex_result = results_by_id.get(q_idx)
            if ex_result:
                metrics = getattr(ex_result, "metrics", {}) or {}
                # Sanitize newlines for CSV compatibility
                actual_output = str(ex_result.actual_output) if ex_result.actual_output else ""
                actual_output = actual_output.replace("\n", " | ").replace("\r", "")
                rows.append(
                    [
                        f"example_{q_idx}",
                        meta["question"].replace("\n", " | ").replace("\r", ""),
                        meta["expected"].replace("\n", " | ").replace("\r", ""),
                        actual_output,
                        ex_result.success if hasattr(ex_result, "success") else "",
                        metrics.get("accuracy", metrics.get("score", "")),
                        meta["source"],
                        meta["category"],
                        trial_idx,
                        config.get("model", ""),
                        config.get("temperature", ""),
                        config.get("max_tokens", ""),
                        getattr(ex_result, "execution_time", "")
                        or metrics.get("latency_ms", ""),
                        metrics.get("input_tokens", ""),
                        metrics.get("output_tokens", ""),
                        metrics.get("total_cost", ""),
                    ]
                )
            else:
                # No result for this question in this trial
                rows.append(
                    [
                        f"example_{q_idx}",
                        meta["question"].replace("\n", " | ").replace("\r", ""),
                        meta["expected"].replace("\n", " | ").replace("\r", ""),
                        "N/A",
                        "",
                        "",
                        meta["source"],
                        meta["category"],
                        trial_idx,
                        config.get("model", ""),
                        config.get("temperature", ""),
                        config.get("max_tokens", ""),
                        "",
                        "",
                        "",
                        "",
                    ]
                )

    # Write CSV with quoting to handle semicolons and other special chars
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Detailed results saved to: {output_path}")


def print_category_summary(results, dataset_path: Path) -> None:
    """Print accuracy breakdown by source category and by model.

    Args:
        results: OptimizationResult from the optimization run
        dataset_path: Path to the JSONL dataset file
    """
    # Load dataset metadata
    dataset_metadata = []
    with open(dataset_path) as f:
        for line in f:
            data = json.loads(line)
            dataset_metadata.append(
                {
                    "source": data.get("source", "unknown"),
                    "category": data.get("category", "unknown"),
                }
            )

    # Collect results by source and by model
    source_results = {}  # source -> [pass/fail bools]
    model_results = {}  # model -> [pass/fail bools]
    total_cost = 0.0

    for trial in results.trials:
        config = getattr(trial, "config", getattr(trial, "configuration", {}))
        model = config.get("model", "unknown")
        example_results = trial.metadata.get("example_results", [])

        # Track trial cost
        trial_cost = trial.metadata.get("total_example_cost", 0)
        if trial_cost:
            total_cost += float(trial_cost)

        for ex_result in example_results:
            ex_idx = int(ex_result.example_id.split("_")[1])
            if ex_idx < len(dataset_metadata):
                source = dataset_metadata[ex_idx]["source"]
                # Use evaluator score for pass/fail, not just "success" (which means call succeeded)
                metrics = getattr(ex_result, "metrics", {}) or {}
                score = metrics.get("accuracy", metrics.get("score", 0))
                passed = score >= 0.5 if isinstance(score, (int, float)) else False

                # Aggregate by source
                if source not in source_results:
                    source_results[source] = []
                source_results[source].append(passed)

                # Aggregate by model
                if model not in model_results:
                    model_results[model] = []
                model_results[model].append(passed)

    # Print summary
    print()
    print("=" * 60)
    print("Results Summary by Category")
    print("=" * 60)
    print()

    print("Accuracy by Source:")
    for source in sorted(source_results.keys()):
        passes = source_results[source]
        accuracy = sum(passes) / len(passes) * 100 if passes else 0
        print(f"  {source} ({len(passes)} examples): {accuracy:.1f}%")

    print()
    print("Accuracy by Model:")
    for model in sorted(model_results.keys()):
        passes = model_results[model]
        accuracy = sum(passes) / len(passes) * 100 if passes else 0
        print(f"  {model} ({len(passes)} examples): {accuracy:.1f}%")

    print()
    print(f"Total Cost: ${total_cost:.4f}")
    print()


async def main():
    print("=" * 60)
    print("TraiGent Quickstart: Simple Q&A Agent Optimization (Real LLM)")
    print("=" * 60)
    print()

    print(f"Dataset: {DATASET_PATH}")
    print("Using real OpenAI API calls")
    print()

    print("Configuration Space:")
    print(f"  - Models: {', '.join(CONFIGURATION_SPACE['model'])}")
    print(f"  - Temperature: {', '.join(str(t) for t in CONFIGURATION_SPACE['temperature'])}")
    print(f"  - Max Tokens: {', '.join(str(t) for t in CONFIGURATION_SPACE['max_tokens'])}")
    print()

    print("Constraints Applied:")
    for constraint in CONSTRAINTS_DESCRIPTIONS:
        print(f"  - {constraint}")
    print()

    # Run optimization
    print("Starting optimization...")
    print("-" * 40)
    results = await simple_qa_agent.optimize()

    print()
    print("=" * 60)
    print("Optimization Complete!")
    print("=" * 60)
    print()
    print(f"Best Score: {results.best_score}")
    print(f"Best Configuration: {results.best_config}")
    print()

    # Show all trial results if available
    if hasattr(results, "trials") and results.trials:
        print("All Trials:")
        print("-" * 40)
        for i, trial in enumerate(results.trials, 1):
            # Handle different trial result formats
            score = getattr(trial, "score", None) or getattr(trial, "metrics", {}).get(
                "score", "N/A"
            )
            config = getattr(trial, "config", getattr(trial, "configuration", {}))
            print(f"  Trial {i}: {config} -> score={score}")

    print()

    # Save results to CSV
    csv_output_path = Path(__file__).parent / "results" / "optimization_results.csv"
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results_to_csv(results, DATASET_PATH, csv_output_path)

    # Save detailed results CSV
    detailed_csv_path = (
        Path(__file__).parent / "results" / "optimization_results_detailed.csv"
    )
    save_detailed_results_csv(results, DATASET_PATH, detailed_csv_path)

    # Save best config to JSON for proof of selection
    best_config_path = Path(__file__).parent / "results" / "best_config.json"
    with open(best_config_path, "w") as f:
        json.dump({
            "best_score": results.best_score,
            "best_config": results.best_config,
        }, f, indent=2)
    print(f"Best config saved to: {best_config_path}")

    # Print category summary
    print_category_summary(results, DATASET_PATH)

    print()

    # Demonstrate using the optimized function
    print("=" * 60)
    print("Using Optimized Configuration")
    print("=" * 60)
    print()

    # Get the best config and apply it
    best_config = simple_qa_agent.get_best_config()
    if best_config:
        print(f"Applying best config: {best_config}")
        simple_qa_agent.set_config(best_config)

        # Test with a new question
        test_question = "What is the largest planet in our solar system?"
        print(f"\nTest question: {test_question}")
        answer = simple_qa_agent(test_question)
        print(f"Answer: {answer}")

    print()
    print("Optimization complete with real LLM calls!")
    print()
    print("Next steps:")
    print("  1. Adjust configuration_space to explore more options")
    print("  2. Add more constraints to prune the search space")
    print("  3. Increase max_trials for more thorough optimization")
    print("  4. Try different objectives (accuracy, cost, latency, safety)")


if __name__ == "__main__":
    asyncio.run(main())
