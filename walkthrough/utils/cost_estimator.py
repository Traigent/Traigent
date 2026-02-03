#!/usr/bin/env python3
"""Cost estimator for Traigent optimization runs."""

from typing import Any

# Approximate costs per 1K tokens (as of Jan 2026)
# Synced with mock_answers.py MOCK_MODEL_COSTS
MODEL_COSTS = {
    # --- OpenAI models ---
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4": {"input": 0.03, "output": 0.06},  # Legacy (8k context)
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    # Future/Hypothetical Models (2026 Context)
    "gpt-4.1-nano": {"input": 0.0001, "output": 0.0004},  # Released Apr 2025
    "gpt-5-nano": {"input": 0.00008, "output": 0.0003},  # Released Aug 2025
    "gpt-5.1": {"input": 0.002, "output": 0.008},  # Released Nov 2025
    "gpt-5.2": {"input": 0.003, "output": 0.012},  # Released Dec 2025
    # --- Anthropic models ---
    # Corrected: 3.5 Haiku pricing ($0.80/$4.00 per 1M tokens)
    "claude-3-5-haiku-20241022": {"input": 0.0008, "output": 0.004},
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
    "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    # --- Google Gemini models ---
    # Standard tier prices (Prompts <= 128k tokens)
    # Note: Costs double if prompts > 128k tokens
    "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
    "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
    "gemini-2.0-flash-exp": {"input": 0.000075, "output": 0.0003},
}

# Updated Average token estimates
AVERAGE_TOKENS = {
    "simple_qa": {"input": 50, "output": 30},
    "classification": {"input": 80, "output": 10},
    "generation": {"input": 100, "output": 200},
    # INCREASED: 500 -> 2000 to account for retrieved context chunks
    "rag_qa": {"input": 2000, "output": 100},
    "code_generation": {"input": 150, "output": 300},
    "summarization": {"input": 1000, "output": 200},
}


def estimate_cost(
    models: list[str],
    dataset_size: int,
    task_type: str = "simple_qa",
    num_trials: int = None,
) -> dict[str, Any]:
    """
    Estimate the cost of running Traigent optimization.

    Args:
        models: List of model names to test
        dataset_size: Number of examples in evaluation dataset
        task_type: Type of task (affects token estimates)
        num_trials: Number of optimization trials (if None, assumes testing all models)

    Returns:
        Dictionary with cost breakdown and total estimate
    """

    if num_trials is None:
        num_trials = len(models)

    # Get token estimates for task type
    tokens = AVERAGE_TOKENS.get(task_type, AVERAGE_TOKENS["simple_qa"])
    input_tokens = tokens["input"]
    output_tokens = tokens["output"]

    # Calculate costs per model
    model_costs = {}
    total_cost = 0.0

    for model in models:
        if model in MODEL_COSTS:
            costs = MODEL_COSTS[model]
            # Cost for one example
            example_cost = (input_tokens / 1000) * costs["input"] + (
                output_tokens / 1000
            ) * costs["output"]
            # Cost for full dataset
            dataset_cost = example_cost * dataset_size
            model_costs[model] = dataset_cost
            total_cost += dataset_cost
        else:
            # Unknown model - use conservative estimate (similar to gpt-3.5-turbo)
            example_cost = (input_tokens / 1000) * 0.0005 + (
                output_tokens / 1000
            ) * 0.0015
            model_costs[model] = example_cost * dataset_size
            total_cost += model_costs[model]

    # Scale by number of trials if different from number of models
    if num_trials != len(models):
        scale_factor = num_trials / len(models)
        total_cost *= scale_factor

    return {
        "models": model_costs,
        "dataset_size": dataset_size,
        "task_type": task_type,
        "num_trials": num_trials,
        "estimated_tokens": {
            "input": input_tokens * dataset_size * num_trials,
            "output": output_tokens * dataset_size * num_trials,
        },
        "total_cost": total_cost,
        "cost_range": {
            "min": total_cost * 0.7,  # 30% lower
            "max": total_cost * 1.5,  # 50% higher
        },
        "notes": [
            "Costs are estimates based on average token usage",
            "Actual costs may vary based on prompt complexity",
            "Prices as of 2026 - check latest pricing",
        ],
    }


def format_cost_estimate(estimate: dict[str, Any]) -> str:
    """Format cost estimate for display."""

    output = []
    output.append("=" * 50)
    output.append("💰 COST ESTIMATE")
    output.append("=" * 50)
    output.append("")

    output.append(f"Dataset size: {estimate['dataset_size']} examples")
    output.append(f"Task type: {estimate['task_type']}")
    output.append(f"Number of trials: {estimate['num_trials']}")
    output.append("")

    output.append("Cost by model:")
    for model, cost in estimate["models"].items():
        output.append(f"  • {model}: ${cost:.4f}")
    output.append("")

    output.append("Estimated tokens:")
    output.append(f"  Input: {estimate['estimated_tokens']['input']:,}")
    output.append(f"  Output: {estimate['estimated_tokens']['output']:,}")
    output.append("")

    total = estimate["total_cost"]
    output.append(f"TOTAL ESTIMATED COST: ${total:.2f}")
    output.append(
        f"Range: ${estimate['cost_range']['min']:.2f} - ${estimate['cost_range']['max']:.2f}"
    )
    output.append("")

    for note in estimate["notes"]:
        output.append(f"⚠️ {note}")

    return "\n".join(output)


def main():
    """Example usage and testing."""

    # Example 1: Simple Q&A optimization
    print("Example 1: Simple Q&A Optimization")
    estimate = estimate_cost(
        models=["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        dataset_size=20,
        task_type="simple_qa",
        num_trials=9,  # 3 models x 3 temperatures
    )
    print(format_cost_estimate(estimate))
    print()

    # Example 2: RAG optimization
    print("\nExample 2: RAG Optimization")
    estimate = estimate_cost(
        models=["gpt-3.5-turbo", "gpt-4o-mini"],
        dataset_size=15,
        task_type="rag_qa",
        num_trials=12,  # 2 models x 3 temps x 2 retrieval settings
    )
    print(format_cost_estimate(estimate))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130)
