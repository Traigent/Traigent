#!/usr/bin/env python
"""
Traigent Quickstart Example 4: LLM-as-Judge Custom Evaluator

This example shows how to use an LLM (like GPT-4) to evaluate the quality
of your agent's outputs instead of relying on exact match or semantic similarity.

LLM-as-judge is useful when:
- Outputs are subjective (e.g., writing quality, helpfulness)
- Multiple valid answers exist
- You need nuanced evaluation (e.g., tone, completeness, accuracy)

Key concepts:
- metric_functions: Define custom scoring functions like LLM-as-judge
- custom_metrics: Return multiple scores (correctness, friendliness, etc.)
- objectives: Use custom_metrics as optimization objectives!

Run with:
    export TRAIGENT_MOCK_LLM=true
    python examples/quickstart/04_llm_as_judge.py

For real LLM evaluation (costs money):
    export OPENAI_API_KEY=your-key
    python examples/quickstart/04_llm_as_judge.py
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
from traigent.api.decorators import EvaluationOptions, ExecutionOptions
from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema

# Path to dataset
DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "datasets" / "quickstart" / "qa_samples.jsonl"
)


def llm_as_judge_accuracy(output: str, expected: str) -> float:
    """
    Use an LLM to evaluate the accuracy/correctness of the output.

    This evaluator asks an LLM to score the output on correctness.
    The return value becomes the "accuracy" metric.

    In mock mode, this returns simulated scores.
    With real API keys, it calls the LLM for evaluation.
    """
    is_mock = os.environ.get("TRAIGENT_MOCK_LLM", "").lower() == "true"

    if is_mock:
        # Simulate LLM-as-judge scoring in mock mode
        correctness = 0.9 if expected.lower() in output.lower() else 0.4
        completeness = 0.8
        clarity = 0.85
        # Weighted combination for overall accuracy
        return 0.5 * correctness + 0.3 * completeness + 0.2 * clarity

    # Real LLM-as-judge implementation
    try:
        from openai import OpenAI

        client = OpenAI()

        evaluation_prompt = f"""You are an expert evaluator. Score the following response on a scale of 0.0 to 1.0.

Expected answer: {expected}
Actual response: {output}

Score how correct and complete the answer is (0.0 = completely wrong, 1.0 = perfect).
Respond with ONLY a number between 0.0 and 1.0."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": evaluation_prompt}],
            temperature=0.0,
            max_tokens=10,
        )

        try:
            return float(response.choices[0].message.content.strip())
        except ValueError:
            return 0.5

    except ImportError:
        return 1.0 if expected.lower() in output.lower() else 0.0
    except Exception as e:
        print(f"Warning: LLM evaluation failed: {e}")
        return 1.0 if expected.lower() in output.lower() else 0.0


def llm_as_judge_friendliness(output: str, expected: str) -> float:
    """
    Use an LLM to evaluate the friendliness/tone of the output.

    This is a CUSTOM METRIC that can be used as an optimization objective!
    """
    is_mock = os.environ.get("TRAIGENT_MOCK_LLM", "").lower() == "true"

    if is_mock:
        # Simulate friendliness scoring
        # Friendly responses often have exclamation marks and warm language
        friendliness = 0.7 if "!" in output or "great" in output.lower() else 0.5
        return friendliness

    # Real LLM-as-judge implementation
    try:
        from openai import OpenAI

        client = OpenAI()

        evaluation_prompt = f"""You are an expert evaluator. Score the following response on FRIENDLINESS on a scale of 0.0 to 1.0.

Response: {output}

Score how warm, helpful, and approachable the tone is (0.0 = cold/robotic, 1.0 = very friendly).
Respond with ONLY a number between 0.0 and 1.0."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": evaluation_prompt}],
            temperature=0.0,
            max_tokens=10,
        )

        try:
            return float(response.choices[0].message.content.strip())
        except ValueError:
            return 0.5

    except ImportError:
        return 0.5
    except Exception as e:
        print(f"Warning: LLM evaluation failed: {e}")
        return 0.5


# Define custom objectives including our custom metric "friendliness"
# This tells Traigent to optimize for friendliness alongside accuracy and cost!
custom_objectives = ObjectiveSchema.from_objectives(
    [
        ObjectiveDefinition("accuracy", orientation="maximize", weight=0.4),
        ObjectiveDefinition(
            "friendliness", orientation="maximize", weight=0.3
        ),  # Custom metric!
        ObjectiveDefinition("cost", orientation="minimize", weight=0.3),
    ]
)


@traigent.optimize(
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "temperature": [0.1, 0.5, 0.9],
        "response_style": ["concise", "detailed", "friendly"],
    },
    objectives=custom_objectives,  # Uses accuracy, friendliness, and cost!
    evaluation=EvaluationOptions(
        eval_dataset=str(DATASET_PATH),
        # Use metric_functions to define custom evaluation metrics
        # Each function receives (output, expected) and returns a float score
        metric_functions={
            "accuracy": llm_as_judge_accuracy,  # Replaces default accuracy
            "friendliness": llm_as_judge_friendliness,  # Custom metric for optimization!
        },
    ),
    execution=ExecutionOptions(execution_mode="edge_analytics"),
    max_trials=5,
)
def qa_agent_with_style(question: str) -> str:
    """Q&A agent with configurable response style.

    The LLM-as-judge evaluators will score responses on:
    - Accuracy: factual correctness (via llm_as_judge_accuracy)
    - Friendliness: tone warmth (via llm_as_judge_friendliness)
    """
    config = traigent.get_config()
    style = config.get("response_style", "concise")

    # Mock responses demonstrating different styles
    base_answers = {
        "What is the capital of France?": {
            "concise": "Paris",
            "detailed": "Paris is the capital and largest city of France, located in the north-central part of the country.",
            "friendly": "Great question! The capital of France is Paris - a beautiful city known for the Eiffel Tower!",
        },
        "What is 2 + 2?": {
            "concise": "4",
            "detailed": "The sum of 2 + 2 equals 4. This is a basic arithmetic addition operation.",
            "friendly": "Easy one! 2 + 2 = 4. Math is fun!",
        },
        "Who wrote Romeo and Juliet?": {
            "concise": "William Shakespeare",
            "detailed": "Romeo and Juliet was written by William Shakespeare, believed to have been composed between 1591 and 1596.",
            "friendly": "That would be the legendary William Shakespeare! It's one of his most famous plays.",
        },
    }

    if question in base_answers:
        return base_answers[question].get(style, base_answers[question]["concise"])
    return "I don't know"


async def main():
    print("=" * 60)
    print("Traigent Quickstart: LLM-as-Judge Evaluation")
    print("=" * 60)
    print()

    print("Optimization Objectives:")
    print("-" * 40)
    print("  accuracy (40%):    maximize - from LLM judge scores")
    print("  friendliness (30%): maximize - CUSTOM METRIC from LLM judge!")
    print("  cost (30%):        minimize - API costs")
    print()

    print("LLM-as-Judge Metrics (via metric_functions):")
    print("-" * 40)
    print("  - accuracy: llm_as_judge_accuracy(output, expected) -> float")
    print("  - friendliness: llm_as_judge_friendliness(output, expected) -> float")
    print()

    print("Configuration Space:")
    print("-" * 40)
    print("  model:          [gpt-3.5-turbo, gpt-4o-mini, gpt-4o]")
    print("  temperature:    [0.1, 0.5, 0.9]")
    print("  response_style: [concise, detailed, friendly]")
    print()

    print(f"Dataset: {DATASET_PATH}")
    print(f"Mock mode: {os.environ.get('TRAIGENT_MOCK_LLM', 'false')}")
    print()

    if os.environ.get("TRAIGENT_MOCK_LLM", "").lower() == "true":
        print("Note: Running in mock mode - LLM evaluation is simulated.")
        print(
            "      Set TRAIGENT_MOCK_LLM=false and OPENAI_API_KEY for real evaluation."
        )
        print()

    # Run optimization
    print("Starting optimization with LLM-as-judge evaluation...")
    results = await qa_agent_with_style.optimize()

    print()
    print("=" * 60)
    print("Optimization Complete!")
    print("=" * 60)
    print()
    print(f"Best Score: {results.best_score}")
    print(f"Best Configuration: {results.best_config}")
    print()

    # Show custom metrics if available
    if hasattr(results, "best_metrics") and results.best_metrics:
        print("Best Trial Metrics (from LLM-as-Judge):")
        print("-" * 40)
        for key, value in results.best_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
    print()

    print("Key Takeaways:")
    print("-" * 40)
    print("  1. Use metric_functions to define custom evaluation metrics")
    print("  2. Each metric function receives (output, expected) -> float")
    print("  3. Custom metrics can be used as optimization objectives!")
    print("  4. Use ObjectiveSchema to weight your custom metrics")
    print()
    print("Cost tip: Use a cheaper model (gpt-4o-mini) as the judge")
    print("          while testing expensive models (gpt-4o) as the agent.")


if __name__ == "__main__":
    asyncio.run(main())
