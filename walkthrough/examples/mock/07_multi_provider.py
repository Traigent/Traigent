#!/usr/bin/env python3
"""Example 7: Multi-Provider LLM Support - Use any LLM vendor with Traigent.

This example demonstrates how to use different LLM providers (OpenAI, Anthropic,
Google Gemini) within the same Traigent optimization workflow. You can optimize
across providers to find the best model for your use case.

Key concepts:
- Define a configuration space that includes models from multiple providers
- Use conditional logic in your function to call the appropriate provider
- Traigent optimizes across all providers equally using your scoring function

The mock version simulates responses with provider-specific accuracy patterns.
For real API calls, see walkthrough/examples/real/07_multi_provider.py.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import traigent
from traigent import TraigentConfig

from utils.helpers import print_optimization_config, print_results_table
from utils.mock_answers import (
    ANSWERS,
    DEFAULT_MOCK_MODEL,
    configure_mock_notice,
    get_mock_accuracy,
    get_mock_cost,
    get_mock_latency,
    normalize_text,
    set_mock_model,
)

# -----------------------------------------------------------------------------
# Environment Setup (Mock Mode)
# -----------------------------------------------------------------------------
os.environ.setdefault("TRAIGENT_MOCK_LLM", "true")
os.environ.setdefault("TRAIGENT_OFFLINE_MODE", "true")

traigent.initialize(
    config=TraigentConfig(execution_mode="edge_analytics", minimal_logging=True)
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATASETS = Path(__file__).parent.parent / "datasets"

# Models from three different providers: OpenAI, Anthropic, and Google
# In real mode, each requires its respective API key
PROVIDER_MODELS = {
    # OpenAI models (requires OPENAI_API_KEY)
    "gpt-4o-mini": "openai",
    "gpt-4o": "openai",
    # Anthropic models (requires ANTHROPIC_API_KEY)
    "claude-3-haiku-20240307": "anthropic",
    "claude-3-5-sonnet-20241022": "anthropic",
    # Google Gemini models (requires GOOGLE_API_KEY) - includes free tier
    "gemini-1.5-flash": "google",
    "gemini-1.5-pro": "google",
}

OBJECTIVES = ["accuracy", "cost", "latency"]
CONFIG_SPACE = {
    "model": list(PROVIDER_MODELS.keys()),
    "temperature": [0.1, 0.5],
}

MOCK_MODE_CONFIG = {
    "base_accuracy": 0.85,
    "variance": 0.0,
    "random_seed": 42,
}


def get_provider_for_model(model_name: str) -> str:
    """Return the provider name for a given model.

    Args:
        model_name: The model identifier (e.g., 'gpt-4o-mini', 'claude-3-haiku')

    Returns:
        Provider name: 'openai', 'anthropic', or 'google'
    """
    return PROVIDER_MODELS.get(model_name, "openai")


def results_match_score(
    output: str, expected: str, config: dict | None = None, **_
) -> float:
    """Score based on expected answer containment.

    In mock mode: Returns provider/model-dependent accuracy scores.
    In real mode: Returns 1.0 if expected answer appears in output, else 0.0.
    """
    if os.getenv("TRAIGENT_MOCK_LLM", "").lower() in ("1", "true", "yes"):
        model = config.get("model", DEFAULT_MOCK_MODEL) if config else DEFAULT_MOCK_MODEL
        temperature = config.get("temperature") if config else None
        return get_mock_accuracy(model, "simple_qa", temperature)
    if output is None or expected is None:
        return 0.0
    return 1.0 if str(expected).strip().lower() in str(output).lower() else 0.0


@traigent.optimize(
    eval_dataset=str(DATASETS / "simple_questions.jsonl"),
    objectives=OBJECTIVES,
    scoring_function=results_match_score,
    configuration_space=CONFIG_SPACE,
    injection_mode="context",
    execution_mode="edge_analytics",
    mock_mode_config=MOCK_MODE_CONFIG,
)
def answer_with_any_provider(question: str) -> str:
    """Answer a question using the configured LLM provider and model.

    This function demonstrates how to route requests to different providers
    based on the model name. Traigent injects the configuration, and your
    code decides which provider SDK to call.

    In production, you would import and use:
    - langchain_openai.ChatOpenAI for OpenAI models
    - langchain_anthropic.ChatAnthropic for Claude models
    - langchain_google_genai.ChatGoogleGenerativeAI for Gemini models
    """
    config = traigent.get_config()
    model = config.get("model", DEFAULT_MOCK_MODEL)
    # In mock mode, we return pre-defined answers
    # In real mode, this would call the appropriate provider SDK
    set_mock_model(model)

    # This is where you would add provider-specific logic:
    # if provider == "openai":
    #     llm = ChatOpenAI(model=model, temperature=config.get("temperature"))
    # elif provider == "anthropic":
    #     llm = ChatAnthropic(model=model, temperature=config.get("temperature"))
    # elif provider == "google":
    #     llm = ChatGoogleGenerativeAI(model=model, temperature=config.get("temperature"))

    time.sleep(get_mock_latency(model, "simple_qa") * 0.01)
    return ANSWERS.get(normalize_text(question), "I don't know")


async def main() -> None:
    """Run the multi-provider optimization example."""
    print("Traigent Example 7: Multi-Provider LLM Support")
    print("=" * 55)
    configure_mock_notice("07_multi_provider.py")

    # Show the provider breakdown
    print("\nProvider Breakdown:")
    providers_summary = {}
    for model, provider in PROVIDER_MODELS.items():
        providers_summary.setdefault(provider, []).append(model)
    for provider, models in providers_summary.items():
        print(f"  {provider.capitalize()}: {', '.join(models)}")

    print_optimization_config(OBJECTIVES, CONFIG_SPACE)

    # Run optimization across all providers
    # max_trials=12 covers all 6 models x 2 temperatures
    results = await answer_with_any_provider.optimize(
        algorithm="grid",
        max_trials=12,
        show_progress=True,
        random_seed=42,
    )

    print_results_table(
        results, CONFIG_SPACE, OBJECTIVES, is_mock=True, task_type="simple_qa"
    )

    # Show best configuration with provider info
    best_model = results.best_config.get("model", DEFAULT_MOCK_MODEL)
    best_provider = get_provider_for_model(best_model)
    best_temp = results.best_config.get("temperature")

    print("\nBest Configuration Found:")
    print(f"  Provider: {best_provider.capitalize()}")
    print(f"  Model: {best_model}")
    print(f"  Temperature: {best_temp}")
    print("\nPerformance:")
    print(f"  Accuracy: {results.best_metrics.get('accuracy', 0):.2%}")
    estimated_cost = get_mock_cost(best_model, "simple_qa", dataset_size=20)
    print(f"  Est. Cost: ${estimated_cost:.4f} (for 20 examples)")
    estimated_latency = get_mock_latency(best_model, "simple_qa")
    print(f"  Est. Latency: {estimated_latency:.3f}s (per call)")

    print("\n" + "-" * 55)
    print("Optimization finished.")
    print("To use this with real API calls:")
    print("  1. Set the required API key(s):")
    print("     export OPENAI_API_KEY='your-key'      # For GPT models")
    print("     export ANTHROPIC_API_KEY='your-key'  # For Claude models")
    print("     export GOOGLE_API_KEY='your-key'     # For Gemini models")
    print("  2. Run: python walkthrough/examples/real/07_multi_provider.py")
    print("\nNote: Gemini offers a free tier - great for testing!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130)
