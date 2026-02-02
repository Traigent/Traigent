#!/usr/bin/env python3
"""Example 7: Multi-Provider LLM Support - Use any LLM vendor with Traigent.

This example demonstrates how to use different LLM providers (OpenAI, Anthropic,
Google Gemini) within the same Traigent optimization workflow. You can optimize
across providers to find the best model for your use case.

Key concepts:
- Define a configuration space that includes models from multiple providers
- Use conditional logic in your function to call the appropriate provider
- Traigent optimizes across all providers equally using your scoring function
- Uses core provider validation (traigent.providers) to check API keys

Usage (run from repo root):
    # Set at least one API key (set all three to test all providers)
    export OPENAI_API_KEY="your-openai-key"        # For GPT models
    export ANTHROPIC_API_KEY="your-anthropic-key"  # For Claude models
    export GOOGLE_API_KEY="your-google-key"        # For Gemini models (free tier!)

    .venv/bin/python walkthrough/examples/real/07_multi_provider.py

Note: This example validates API keys using Traigent's core provider validation
and skips any invalid providers. Gemini offers a generous free tier!
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import traigent
from traigent import TraigentConfig
from traigent.providers import (
    get_provider_for_model,
    print_provider_status,
    validate_providers,
)

from utils.helpers import (
    configure_logging,
    print_cost_estimate,
    print_estimated_time,
    print_optimization_config,
    print_results_table,
    setup_example_logger,
)
from utils.scoring import token_match_score

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
configure_logging()
logger = setup_example_logger("07_multi_provider")

os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")

traigent.initialize(
    config=TraigentConfig(execution_mode="edge_analytics", minimal_logging=True)
)

# -----------------------------------------------------------------------------
# Provider Detection - Check which API keys are available and valid
# Uses Traigent core provider validation (traigent.providers.validation)
# -----------------------------------------------------------------------------

# Models organized by provider
OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o"]
ANTHROPIC_MODELS = ["claude-3-haiku-20240307", "claude-3-5-sonnet-20241022"]
GOOGLE_MODELS = ["gemini-1.5-flash", "gemini-1.5-pro"]

# All models we want to potentially use
ALL_MODELS = OPENAI_MODELS + ANTHROPIC_MODELS + GOOGLE_MODELS

# Validate all providers using core validation
PROVIDER_STATUS = validate_providers(ALL_MODELS)

# Build available models list based on which providers are valid
AVAILABLE_MODELS: list[str] = []
for provider, status in PROVIDER_STATUS.items():
    if status.valid and status.models:
        AVAILABLE_MODELS.extend(status.models)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATASETS = Path(__file__).parent.parent / "datasets"
OBJECTIVES = ["accuracy", "cost", "latency"]

# Parallel trials - Traigent uses LangChain which enables automatic token tracking
# across parallel execution via the LangChain interceptor (keyed by example_id)
PARALLEL_TRIALS = 4

# Configuration space - only includes models for available providers
CONFIG_SPACE = {
    "model": AVAILABLE_MODELS if AVAILABLE_MODELS else ["gpt-4o-mini"],
    "temperature": [0.1, 0.5],
}


def create_llm_client(model: str, temperature: float) -> Any:
    """Create the appropriate LLM client based on the model's provider.

    Args:
        model: Model name (determines which provider SDK to use)
        temperature: Temperature setting for the model

    Returns:
        LangChain chat model instance for the appropriate provider

    Raises:
        ValueError: If the provider SDK is not available
    """
    provider = get_provider_for_model(model)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature)

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, temperature=temperature)

    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, temperature=temperature)

    else:
        raise ValueError(f"Unknown provider for model: {model}")


@traigent.optimize(
    eval_dataset=str(DATASETS / "simple_questions.jsonl"),
    objectives=OBJECTIVES,
    scoring_function=token_match_score,
    configuration_space=CONFIG_SPACE,
    injection_mode="context",
    execution_mode="edge_analytics",
    # Skip core validation - this example validates upfront and filters invalid
    # providers for graceful degradation (run with available providers only).
    # Core validation would fail-fast on any invalid provider instead.
    validate_providers=False,
)
def answer_with_any_provider(question: str) -> str:
    """Answer a question using the configured LLM provider and model.

    Traigent injects the configuration (model, temperature) and this function
    routes the request to the appropriate provider SDK based on the model name.
    """
    config = traigent.get_config()
    model = config.get("model", "gpt-4o-mini")
    temperature = config.get("temperature", 0.1)

    llm = create_llm_client(model, temperature)

    try:
        response = llm.invoke(
            "Answer with the final answer only. Do not ask questions. "
            "Keep it concise and use the question's terminology. "
            "If the answer is numeric, use digits only (no commas), no units, "
            "and follow the units implied by the question. Do not round. "
            f"{question}"
        )
        return str(response.content)
    except Exception as exc:
        logger.warning("LLM call failed (%s): %s: %s", model, type(exc).__name__, exc)
        return f"Error: {type(exc).__name__}: {exc}"


async def main() -> None:
    """Run the multi-provider optimization example."""
    print("Traigent Example 7: Multi-Provider LLM Support")
    print("=" * 55)
    print("This example makes LLM API calls to any available provider.")

    print_provider_status(PROVIDER_STATUS)

    if not AVAILABLE_MODELS:
        print("\nERROR: No API keys found!")
        print("Set at least one of these environment variables:")
        print("  export OPENAI_API_KEY='your-key'")
        print("  export ANTHROPIC_API_KEY='your-key'")
        print("  export GOOGLE_API_KEY='your-key'  # Free tier available!")
        print("\nTip: Try Gemini first - it has a generous free tier.")
        sys.exit(1)

    print_optimization_config(OBJECTIVES, CONFIG_SPACE)
    print_cost_estimate(
        models=CONFIG_SPACE["model"],
        dataset_size=20,
        task_type="simple_qa",
        num_trials=len(CONFIG_SPACE["model"]) * len(CONFIG_SPACE["temperature"]),
    )

    print_estimated_time("07_multi_provider.py")

    # Calculate total trials dynamically based on configuration space
    # This ensures grid search covers all combinations: models × temperatures
    # Example: 6 models × 2 temperatures = 12 total trials
    total_trials = len(CONFIG_SPACE["model"]) * len(CONFIG_SPACE["temperature"])

    record_runtime = os.getenv("TRAIGENT_RECORD_RUNTIME", "").lower() in (
        "1",
        "true",
        "yes",
    )
    start_time = time.perf_counter() if record_runtime else 0.0
    results = await answer_with_any_provider.optimize(
        algorithm="grid",
        max_trials=total_trials,  # Covers all config combinations for grid search
        parallel_trials=PARALLEL_TRIALS,
        timeout=600,  # 10 min for 12 trials
        show_progress=True,
        random_seed=42,
    )
    if record_runtime:
        runtime_seconds = time.perf_counter() - start_time
        print(f"\nRecorded runtime (for estimate update): {runtime_seconds:.1f}s")

    print_results_table(results, CONFIG_SPACE, OBJECTIVES, is_mock=False)

    # Show best configuration with provider info
    best_model = results.best_config.get("model", "unknown")
    best_provider = get_provider_for_model(best_model)
    best_temp = results.best_config.get("temperature")

    print("\nBest Configuration Found:")
    print(f"  Provider: {best_provider.capitalize()}")
    print(f"  Model: {best_model}")
    print(f"  Temperature: {best_temp}")
    print("\nPerformance:")
    print(f"  Accuracy: {results.best_metrics.get('accuracy', 0):.2%}")
    print(f"  Cost: ${results.best_metrics.get('cost', 0):.6f}")
    if "latency" in results.best_metrics:
        print(f"  Latency: {results.best_metrics.get('latency', 0):.3f}s")

    print("\n" + "-" * 55)
    print("Summary: Traigent optimized across all available providers")
    print(f"and found {best_provider.capitalize()}'s {best_model} to be the best")
    print("configuration for this task and dataset.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130)
