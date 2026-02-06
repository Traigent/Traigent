#!/usr/bin/env python3
"""Example: Prompt Optimization - Mock Version

Demonstrates:
- System prompt tuning as a Choices parameter
- Combining prompt selection with other parameters
- Using factory methods with prompt tuning

This mock version uses hardcoded responses - no API keys needed.
Run with: TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true python 02_prompt_optimization.py
"""

import asyncio
from pathlib import Path

import traigent
from traigent.api.parameter_ranges import Choices, Range

# Compute dataset path relative to this script
SCRIPT_DIR = Path(__file__).parent
DATASET_PATH = str((SCRIPT_DIR / ".." / "simple_questions.jsonl").resolve())

# Initialize Traigent in mock mode
traigent.initialize(execution_mode="edge_analytics")

# Define system prompt variants to optimize
SYSTEM_PROMPTS = [
    "You are a helpful assistant.",
    "You are an expert who provides detailed, accurate answers.",
    "You are a friendly assistant who explains things simply.",
    "Answer questions step by step with clear reasoning.",
]

# Create Choices for system prompts
system_prompt = Choices(SYSTEM_PROMPTS, name="system_prompt", default=SYSTEM_PROMPTS[0])

# Other parameters to tune alongside prompts
temperature = Range.temperature()  # [0.0, 1.0]
top_p = Range.top_p()  # [0.1, 1.0]


@traigent.optimize(
    system_prompt=system_prompt,
    temperature=temperature,
    top_p=top_p,
    objectives=["accuracy"],
    eval_dataset=DATASET_PATH,
    execution_mode="edge_analytics",
)
def qa_agent(question: str) -> str:
    """Mock QA agent with tunable system prompt and parameters."""
    config = traigent.get_config()

    # Get all parameters from config
    prompt = config["system_prompt"]
    temp = config["temperature"]
    p_value = config["top_p"]

    q = question.lower()

    # Simulate response quality based on prompt type
    # "step by step" prompt tends to be more accurate for reasoning
    step_by_step_bonus = "step by step" in prompt.lower()
    # "expert" prompt is better for factual questions
    expert_bonus = "expert" in prompt.lower()

    # Base response
    if "2+2" in q:
        answer = "4"
    elif "capital" in q and "france" in q:
        answer = "Paris"
    elif "machine learning" in q:
        if step_by_step_bonus:
            answer = (
                "Step 1: Machine learning is a subset of AI. "
                "Step 2: It enables computers to learn from data. "
                "Step 3: The goal is to make predictions or decisions."
            )
        elif expert_bonus:
            answer = (
                "Machine learning is a computational approach that enables systems "
                "to learn patterns from data and improve performance without "
                "explicit programming."
            )
        else:
            answer = "Machine learning is when computers learn from data."
    elif "color" in q and "sky" in q:
        answer = "blue"
    else:
        answer = "I can help you with that question."

    # Temperature affects response style
    if temp > 0.8:
        answer = f"{answer} (creative, temp={temp:.2f})"

    # Low top_p = more focused responses
    if p_value < 0.5:
        answer = f"{answer} (focused, top_p={p_value:.2f})"

    return answer


async def main() -> None:
    print("Traigent Example: Prompt Optimization")
    print("=" * 60)

    print("\nSystem Prompt Variants:")
    for i, prompt in enumerate(SYSTEM_PROMPTS, 1):
        print(f"  {i}. {prompt[:50]}...")

    print("\nOther Parameters:")
    print(f"  temperature: {temperature.low} - {temperature.high}")
    print(f"  top_p: {top_p.low} - {top_p.high}")

    print("\nRunning optimization to find best prompt + parameters...")
    results = await qa_agent.optimize(algorithm="random", max_trials=16, random_seed=42)

    print("\nBest Configuration:")
    best_prompt = results.best_config.get("system_prompt", "N/A")
    print(f"  System Prompt: {best_prompt[:50]}...")
    print(f"  Temperature: {results.best_config.get('temperature')}")
    print(f"  Top P: {results.best_config.get('top_p')}")

    print("\nPerformance:")
    print(f"  Accuracy: {results.best_metrics.get('accuracy', 0):.2%}")

    # Analyze which prompts performed best
    print("\nPrompt Performance Analysis:")
    prompt_scores: dict[str, list[float]] = {}
    for trial in results.trials:
        prompt = trial.config.get("system_prompt", "unknown")
        prompt_key = prompt[:30] + "..."
        if prompt_key not in prompt_scores:
            prompt_scores[prompt_key] = []
        prompt_scores[prompt_key].append(trial.metrics.get("accuracy", 0))

    for prompt_key, scores in sorted(
        prompt_scores.items(), key=lambda x: sum(x[1]) / len(x[1]), reverse=True
    ):
        avg_score = sum(scores) / len(scores)
        print(f"  {prompt_key}: avg accuracy = {avg_score:.2%}")


if __name__ == "__main__":
    asyncio.run(main())
