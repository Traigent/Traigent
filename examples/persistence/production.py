"""Production script: Load optimized config and run.

This script demonstrates the production workflow:
1. Load the exported config on decoration (no optimization)
2. Use the optimized configuration for all calls
3. Optionally override via TRAIGENT_CONFIG_PATH env var

Usage:
    # Using load_from parameter
    python examples/persistence/production.py

    # Using environment variable override
    TRAIGENT_CONFIG_PATH=examples/persistence/configs/prod.json python examples/persistence/production.py
"""

import asyncio
from pathlib import Path

import traigent

# Determine config path relative to this file
CONFIG_PATH = Path(__file__).parent / "configs" / "prod.json"


@traigent.optimize(
    # Load config from exported file - no optimization runs!
    load_from=str(CONFIG_PATH) if CONFIG_PATH.exists() else None,
    # Still define configuration_space for validation and potential re-optimization
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4"],
        "temperature": [0.0, 0.3, 0.7],
        "max_tokens": [50, 100, 200],
    },
    objectives=["accuracy"],
    execution_mode="mock",
)
async def sentiment_agent(text: str) -> str:
    """Analyze sentiment of the given text.

    In production, the config is already loaded from the exported file.
    No optimization runs - just uses the pre-optimized configuration.
    """
    _config = traigent.get_config()  # noqa: F841 - shows how to access config

    # Mock implementation for demo
    if "love" in text.lower() or "best" in text.lower():
        return "positive"
    elif "terrible" in text.lower() or "not recommend" in text.lower():
        return "negative"
    return "neutral"


async def main():
    print("=" * 60)
    print("Config Persistence Demo - Production Phase")
    print("=" * 60)

    # Check if config was loaded
    if sentiment_agent.best_config:
        print("\n1. Config loaded successfully!")
        print(f"   Loaded from: {CONFIG_PATH}")
        print(f"   Config: {sentiment_agent.best_config}")
    else:
        print("\n1. No config loaded.")
        print(f"   Expected config at: {CONFIG_PATH}")
        print("   Run optimize.py first to generate the config.")
        return

    # Run inference with loaded config
    print("\n2. Running inference with optimized config...")

    test_inputs = [
        "I absolutely love this new feature!",
        "This is the worst service I've experienced.",
        "It works as expected, nothing more.",
    ]

    for text in test_inputs:
        result = await sentiment_agent(text)
        print(f"   '{text[:40]}...' -> {result}")

    # Show that config can be accessed
    print("\n3. Accessing config programmatically:")
    print(f"   sentiment_agent.best_config = {sentiment_agent.best_config}")
    print(f"   sentiment_agent.current_config = {sentiment_agent.current_config}")

    print("\n" + "=" * 60)
    print("Production run complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
