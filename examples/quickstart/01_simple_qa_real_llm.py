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

# Path to dataset (relative to this file)
DATASET_PATH = (
    Path(__file__).parent.parent / "datasets" / "quickstart" / "qa_samples.jsonl"
)


def custom_accuracy_scorer(output: str, expected, llm_metrics: dict = None) -> float:
    """Custom scoring function that checks if expected answer is in output.

    Args:
        output: The LLM's response
        expected: The expected answer from the dataset (str or dict with 'output' key)
        llm_metrics: Additional metrics from the LLM call (tokens, latency, etc.)

    Returns:
        Score between 0 and 1
    """
    # Handle case where expected is a dict (full dataset row)
    if isinstance(expected, dict):
        expected = expected.get("output", "")

    if not output or not expected:
        return 0.0
    # Case-insensitive containment check
    return 1.0 if expected.lower() in output.lower() else 0.0


@traigent.optimize(
    # Configuration space: define the search space for optimization
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "temperature": [0.1, 0.3, 0.5, 0.7, 0.9],
        "max_tokens": [50, 100, 200],
    },
    # Default configuration: starting point before optimization
    default_config={
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 100,
    },
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
    # Custom evaluator for scoring
    evaluator=custom_accuracy_scorer,
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
    cost_limit=1.00,  # Max $1.00 spend per optimization run
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
    Each cell contains the LLM's answer for that question in that trial.

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
    # Columns: Question, Expected, Trial1, Trial2, ...
    headers = ["Question", "Expected"]
    trial_configs = []

    for i, trial in enumerate(results.trials, 1):
        config = getattr(trial, "config", getattr(trial, "configuration", {}))
        config_str = f"Trial {i}: {config.get('model', 'N/A')}, t={config.get('temperature', 'N/A')}"
        headers.append(config_str)
        trial_configs.append(config)

    # Initialize rows with questions and expected answers
    rows = []
    for q_idx, (question, expected) in enumerate(zip(questions, expected_answers)):
        rows.append([question, expected])

    # Fill in trial answers
    for trial in results.trials:
        example_results = trial.metadata.get("example_results", [])

        for q_idx in range(len(questions)):
            # Find the example result for this question
            answer = "N/A"
            for ex_result in example_results:
                # example_id is like "example_0", "example_1", etc.
                ex_idx = int(ex_result.example_id.split("_")[1])
                if ex_idx == q_idx:
                    answer = str(ex_result.actual_output)
                    break
            rows[q_idx].append(answer)

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Results saved to: {output_path}")


async def main():
    print("=" * 60)
    print("TraiGent Quickstart: Simple Q&A Agent Optimization (Real LLM)")
    print("=" * 60)
    print()

    print(f"Dataset: {DATASET_PATH}")
    print("Using real OpenAI API calls")
    print()

    print("Configuration Space:")
    print("  - Models: gpt-3.5-turbo, gpt-4o-mini, gpt-4o")
    print("  - Temperature: 0.1, 0.3, 0.5, 0.7, 0.9")
    print("  - Max Tokens: 50, 100, 200")
    print()

    print("Constraints Applied:")
    print("  - GPT-4o: temperature <= 0.5")
    print("  - GPT-3.5-turbo: max_tokens <= 100")
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
    csv_output_path = (
        Path(__file__).parent / "results" / "optimization_results.csv"
    )
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results_to_csv(results, DATASET_PATH, csv_output_path)

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
