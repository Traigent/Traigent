#!/usr/bin/env python3
"""Example: Configuration Spaces - Defining Parameters to Optimize."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# --- Setup for running from repo without installation ---
# Add repo root to path so we can import examples.utils and traigent
_module_path = Path(__file__).resolve()
for _depth in range(1, 7):
    try:
        _repo_root = _module_path.parents[_depth]
        if (_repo_root / "traigent").is_dir() and (_repo_root / "examples").is_dir():
            if str(_repo_root) not in sys.path:
                sys.path.insert(0, str(_repo_root))
            break
    except IndexError:
        continue
from examples.utils.langchain_compat import ChatOpenAI, HumanMessage

try:
    import traigent
except ImportError:  # pragma: no cover - support IDE execution paths
    import importlib

    module_path = Path(__file__).resolve()
    for depth in (2, 3):
        try:
            sys.path.append(str(module_path.parents[depth]))
        except IndexError:
            continue
    traigent = importlib.import_module("traigent")

# Create dataset file
DATASET_FILE = os.path.join(os.path.dirname(__file__), "test_queries.jsonl")


def create_sample_dataset() -> str:
    """Create sample summarization dataset with expected outputs."""
    dataset = [
        {
            "input": {"query": "What is machine learning?"},
            "output": "A brief explanation of machine learning and its purpose.",
        },
        {
            "input": {"query": "How do I bake chocolate chip cookies?"},
            "output": "Simple steps to bake chocolate chip cookies at home.",
        },
        {
            "input": {"query": "Explain quantum computing"},
            "output": "A high-level explanation of quantum computing basics.",
        },
    ]

    # Write to JSONL file
    with open(DATASET_FILE, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

    return DATASET_FILE


# Create the dataset file
create_sample_dataset()


def _summary_f1(output: str | None, expected: str | None, llm_metrics=None) -> float:
    """Token-overlap F1 for free-form summaries."""
    if not output or not expected:
        return 0.0
    import re
    from collections import Counter

    def tokens(s: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9]+", s.lower())

    p = Counter(tokens(output))
    r = Counter(tokens(expected))
    overlap = sum((p & r).values())
    if overlap == 0:
        return 0.0
    p_total = sum(p.values()) or 1
    r_total = sum(r.values()) or 1
    precision = overlap / p_total
    recall = overlap / r_total
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@traigent.optimize(
    configuration_space={
        # Keep space small (<=8 combos) for quick runs in examples
        "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
        "temperature": [0.1, 0.5],
        "max_tokens": [100, 200],
        "response_style": ["concise", "balanced"],
        "use_examples": [False],
    },
    eval_dataset=DATASET_FILE,
    # Report summary_f1 as 'accuracy' for a meaningful single objective
    metric_functions={"accuracy": _summary_f1},
    objectives=["accuracy"],
    execution_mode="edge_analytics",
    max_trials=10,
)
def intelligent_assistant(query: str) -> str:
    """An intelligent assistant that answers queries with optimal parameters."""

    # Check for mock mode
    if os.environ.get("TRAIGENT_MOCK_MODE", "false").lower() == "true":
        return "This is a mock response for testing purposes."

    # Get the current configuration from Traigent
    current = traigent.get_config()
    config = current if isinstance(current, dict) else {}

    # Map response style to system prompt
    style_prompts = {
        "concise": "Answer very briefly in 1-2 sentences.",
        "detailed": "Provide a comprehensive answer with explanations.",
        "balanced": "Give a clear, moderate-length answer.",
    }

    # Build the prompt
    style_key = str(config.get("response_style", "balanced"))
    system_prompt = style_prompts.get(style_key, style_prompts["balanced"])

    if config.get("use_examples", False):
        system_prompt = f"{system_prompt} Include examples when helpful."

    # Create LLM with optimized parameters
    llm = ChatOpenAI(
        model=config.get("model", "gpt-3.5-turbo"),
        temperature=config.get("temperature", 0.7),
        model_kwargs={"max_tokens": config.get("max_tokens", 200)},
        top_p=config.get("top_p", 0.9),
    )

    # Construct the full prompt
    full_prompt = f"{system_prompt}\n\nUser Question: {query}"

    # Get response
    response = llm.invoke([HumanMessage(content=full_prompt)])
    return getattr(response, "content", str(response))


def demonstrate_configuration_types() -> None:
    """Show different types of configuration parameters."""

    # Example 1: Discrete choices (categorical)
    discrete_config = {
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "output_format": ["json", "text", "markdown"],
        "language": ["en", "es", "fr", "de"],
    }

    # Example 2: Numerical ranges (discrete values)
    numerical_config = {
        "temperature": [0.0, 0.25, 0.5, 0.75, 1.0],
        "max_tokens": [50, 100, 200, 500, 1000],
        "frequency_penalty": [-2.0, -1.0, 0.0, 1.0, 2.0],
    }

    # Example 3: Boolean flags
    boolean_config = {
        "use_cache": [True, False],
        "stream_response": [True, False],
        "include_metadata": [True, False],
    }

    # Example 4: Mixed configuration
    mixed_config = {
        **discrete_config,
        **numerical_config,
        **boolean_config,
        # Custom business logic parameters
        "verbosity_level": ["low", "medium", "high"],
        "retry_strategy": ["none", "exponential", "linear"],
        "timeout_seconds": [10, 30, 60, 120],
    }

    print("Configuration Space Examples:")
    print(f"Total discrete choices: {discrete_config}")
    print(f"Total numerical values: {numerical_config}")
    print(f"Total boolean flags: {boolean_config}")
    print(f"Mixed configuration has {len(mixed_config)} parameters")

    # Calculate total combinations
    total_combinations = 1
    for values in mixed_config.values():
        total_combinations *= len(values)
    print(f"Total possible configurations: {total_combinations:,}")


if __name__ == "__main__":
    import asyncio

    async def main():
        print("=" * 60)
        print("Traigent Core Concepts: Configuration Spaces")
        print("=" * 60)

        # Show configuration types
        demonstrate_configuration_types()

        print("\nRunning optimization with configuration space...")

        # Run the optimization
        results = await intelligent_assistant.optimize()

        if results:
            print("\nBest configuration found:")
            print(f"  Model: {results.best_config.get('model')}")
            print(f"  Temperature: {results.best_config.get('temperature')}")
            print(f"  Max Tokens: {results.best_config.get('max_tokens')}")
            style = results.best_config.get("response_style")
            print(f"  Response Style: {style}")

            # Test with the optimized configuration
            test_response = intelligent_assistant("What is machine learning?")
            print(f"\nTest response: {test_response[:200]}...")

    asyncio.run(main())
