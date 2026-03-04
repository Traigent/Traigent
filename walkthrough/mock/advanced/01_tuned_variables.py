#!/usr/bin/env python3
"""Example: Tuned Variables with Factory Methods and Constraints - Mock Version

Demonstrates:
- Factory methods (Range.temperature(), IntRange.max_tokens(), Choices.model())
- Constraints (implies, when().then(), operator-based)
- Domain presets for LLM optimization

This mock version uses hardcoded responses - no API keys needed.
Run with: TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true python 01_tuned_variables.py
"""

import asyncio
from pathlib import Path

import traigent
from traigent.api.constraints import implies
from traigent.api.parameter_ranges import Choices, IntRange, Range

# Compute dataset path relative to this script
SCRIPT_DIR = Path(__file__).parent
DATASET_PATH = str((SCRIPT_DIR / ".." / ".." / "datasets" / "simple_questions.jsonl").resolve())

# Initialize Traigent in mock mode
traigent.initialize(execution_mode="edge_analytics")

# Define parameter ranges using factory methods
temperature = Range.temperature(creative=True)  # [0.7, 1.5]
max_tokens = IntRange.max_tokens(task="medium")  # [256, 1024]
model = Choices.model(provider="openai", tier="balanced")  # [gpt-4o-mini, gpt-4o]
top_p = Range.top_p()  # [0.1, 1.0]

# Define constraints
constraints = [
    # If model is gpt-4o, temperature must be <= 1.0
    implies(model.equals("gpt-4o"), temperature.lte(1.0)),
    # If creative mode (high temp), max_tokens should be >= 512
    implies(temperature.gte(1.0), max_tokens.gte(512)),
    # Constraint using operator syntax (>> means implication)
    (top_p.lte(0.5)) >> (temperature.lte(0.9)),
]


@traigent.optimize(
    temperature=temperature,
    max_tokens=max_tokens,
    model=model,
    top_p=top_p,
    constraints=constraints,
    objectives=["accuracy", "cost"],
    eval_dataset=DATASET_PATH,
    execution_mode="edge_analytics",
)
def creative_writer(question: str) -> str:
    """Mock creative writing function using all configured parameters."""
    config = traigent.get_config()

    # Get all parameters from config
    temp = config["temperature"]
    tokens = config["max_tokens"]
    model_name = config["model"]
    p_value = config["top_p"]

    q = question.lower()

    # Base response based on question type
    if "2+2" in q:
        answer = "4"
    elif "capital" in q and "france" in q:
        answer = "Paris"
    elif "machine learning" in q:
        answer = "Machine learning is a method where computers learn patterns from data"
    elif "color" in q and "sky" in q:
        answer = "blue"
    elif "creative" in q or "story" in q:
        answer = "Once upon a time, in a land of endless possibilities..."
    else:
        answer = "I can help you with that question."

    # Simulate how parameters affect the response
    # Higher temperature = more creative/verbose
    if temp > 1.0:
        answer = f"{answer} (creative mode, temp={temp:.2f})"

    # More tokens = more detail
    if tokens > 512:
        answer = f"{answer} [detailed response with {tokens} tokens]"

    # Model affects quality simulation
    if model_name == "gpt-4o":
        answer = f"{answer} [gpt-4o]"

    # Low top_p = more focused
    if p_value < 0.5:
        answer = f"{answer} (focused, top_p={p_value:.2f})"

    return answer


async def main() -> None:
    print("Traigent Example: Tuned Variables with Factory Methods")
    print("=" * 60)

    print("\nParameter Ranges (from factory methods):")
    print(f"  temperature: {temperature.low} - {temperature.high} (creative mode)")
    print(f"  max_tokens: {max_tokens.low} - {max_tokens.high}")
    print(f"  model: {model.values}")
    print(f"  top_p: {top_p.low} - {top_p.high}")

    print("\nConstraints:")
    print("  1. gpt-4o -> temperature <= 1.0")
    print("  2. temperature >= 1.0 -> max_tokens >= 512")
    print("  3. top_p <= 0.5 -> temperature <= 0.9")

    # Use optuna algorithm which handles all parameter types correctly
    print("\nRunning optimization...")
    results = await creative_writer.optimize(
        algorithm="optuna", max_trials=12, random_seed=42
    )

    print("\nBest Configuration:")
    print(f"  Model: {results.best_config.get('model')}")
    print(f"  Temperature: {results.best_config.get('temperature')}")
    print(f"  Max Tokens: {results.best_config.get('max_tokens')}")
    print(f"  Top P: {results.best_config.get('top_p')}")

    print("\nPerformance:")
    print(f"  Accuracy: {results.best_metrics.get('accuracy', 0):.2%}")
    print(f"  Cost: ${results.best_metrics.get('cost', 0):.6f}")

    print("\nTrial Summary (first 5):")
    for i, trial in enumerate(results.trials[:5]):
        print(
            f"  Trial {i+1}: model={trial.config.get('model', 'N/A')}, "
            f"temp={trial.config.get('temperature', 'N/A'):.2f}, "
            f"accuracy={trial.metrics.get('accuracy', 0):.2%}"
        )


if __name__ == "__main__":
    asyncio.run(main())
