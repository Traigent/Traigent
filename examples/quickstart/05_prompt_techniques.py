#!/usr/bin/env python
"""
Traigent Quickstart Example 5: Prompt Technique Optimization

This example shows how to use Traigent to select the best prompting technique
(Chain-of-Thought, step-by-step, direct, etc.) that gets prepended to prompts.

Run with:
    export TRAIGENT_MOCK_LLM=true
    python examples/quickstart/05_prompt_techniques.py
"""

import asyncio
import os
from pathlib import Path

# Ensure mock mode for testing without API keys
os.environ.setdefault("TRAIGENT_MOCK_LLM", "true")

# Set results folder to local directory
os.environ.setdefault(
    "TRAIGENT_RESULTS_FOLDER", str(Path(__file__).parent / ".traigent_results")
)

import traigent

# Path to dataset (reuse existing QA samples)
DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "datasets" / "quickstart" / "qa_samples.jsonl"
)

# Define prompting techniques to test
PROMPT_TECHNIQUES = {
    "direct": "",  # No prefix, just ask directly
    "cot": "Let's think step by step. ",  # Chain of Thought
    "step_by_step": "Break this down into steps:\n1. Understand the question\n2. Recall relevant facts\n3. Provide the answer\n\n",
    "concise": "Answer in one word or short phrase: ",
}


def accuracy_metric(output: str, expected: str) -> float:
    """Simple exact match accuracy (case-insensitive, trimmed)."""
    output_clean = output.strip().lower()
    expected_clean = expected.strip().lower()
    # Check if expected answer is contained in output (handles CoT reasoning)
    return 1.0 if expected_clean in output_clean else 0.0


@traigent.optimize(
    eval_dataset=str(DATASET_PATH),
    objectives=["accuracy"],
    configuration_space={
        "prompt_technique": list(PROMPT_TECHNIQUES.keys()),
        "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
    },
    metric_functions={"accuracy": accuracy_metric},
    execution_mode="edge_analytics",
    injection_mode="seamless",
    max_trials=8,  # 4 techniques x 2 models
)
def answer_question(question: str) -> str:
    """Answer questions with different prompting techniques.

    The prompt_technique parameter controls what prefix is added to the prompt:
    - direct: No prefix, just the question
    - cot: "Let's think step by step."
    - step_by_step: Structured breakdown instructions
    - concise: Request for brief answers
    """
    # Get current configuration
    cfg = traigent.get_config()
    technique = cfg.get("prompt_technique", "direct")

    # Get the technique prefix
    prefix = PROMPT_TECHNIQUES.get(technique, "")

    # Build the full prompt
    full_prompt = f"{prefix}Question: {question}\nAnswer:"

    # In mock mode, return simulated responses
    if os.environ.get("TRAIGENT_MOCK_LLM", "").lower() in {"true", "1", "yes"}:
        return _mock_response(question, technique)

    # Real mode would use LLM here
    # from langchain_openai import ChatOpenAI
    # llm = ChatOpenAI(model=cfg.get("model", "gpt-3.5-turbo"))
    # return llm.invoke(full_prompt).content

    return f"[Would call LLM with: {full_prompt[:50]}...]"


def _mock_response(question: str, technique: str) -> str:
    """Generate mock responses based on technique."""
    answers = {
        "What is the capital of France?": "Paris",
        "What is 2 + 2?": "4",
        "Who wrote Romeo and Juliet?": "William Shakespeare",
        "What is the largest planet in our solar system?": "Jupiter",
        "What year did World War II end?": "1945",
        "What is the chemical symbol for water?": "H2O",
        "Who painted the Mona Lisa?": "Leonardo da Vinci",
        "What is the speed of light?": "299,792,458 meters per second",
    }

    answer = answers.get(question, "I don't know")

    # Simulate different response styles based on technique
    if technique == "cot":
        return f"Let me think about this... The answer is {answer}."
    elif technique == "step_by_step":
        return f"Step 1: Analyzing question.\nStep 2: {answer}"
    elif technique == "concise":
        return answer
    else:  # direct
        return f"The answer is {answer}."


async def main():
    print("=" * 60)
    print("Traigent: Prompt Technique Optimization")
    print("=" * 60)
    print()

    print("Testing these prompting techniques:")
    print("-" * 40)
    for name, prefix in PROMPT_TECHNIQUES.items():
        display = prefix[:40] + "..." if len(prefix) > 40 else prefix or "(no prefix)"
        print(f"  {name}: {display}")
    print()

    print(f"Dataset: {DATASET_PATH}")
    print(f"Mock mode: {os.environ.get('TRAIGENT_MOCK_LLM', 'false')}")
    print()

    # Run optimization
    print("Starting optimization to find best prompt technique...")
    results = await answer_question.optimize()

    print()
    print("=" * 60)
    print("Optimization Complete!")
    print("=" * 60)
    print()
    print(f"Best Score: {results.best_score}")
    print(f"Best Configuration: {results.best_config}")
    print()

    # Extract best technique
    best_technique = results.best_config.get("prompt_technique", "unknown")
    print(f"Winner: '{best_technique}' prompting technique")
    print()

    # Show all trial results if available
    if hasattr(results, "trials") and results.trials:
        print("All Trials:")
        print("-" * 40)
        for i, trial in enumerate(results.trials, 1):
            score = getattr(trial, "score", None) or getattr(trial, "metrics", {}).get(
                "score", "N/A"
            )
            config = getattr(trial, "config", getattr(trial, "configuration", {}))
            technique = config.get("prompt_technique", "?")
            model = config.get("model", "?")
            print(f"  Trial {i}: {technique} + {model} -> score={score}")

    print()
    print("Use cases for prompt technique optimization:")
    print("  - Find which reasoning style works best for your task")
    print("  - Balance verbosity vs accuracy")
    print("  - Optimize for specific domains (math, factual, creative)")


if __name__ == "__main__":
    asyncio.run(main())
