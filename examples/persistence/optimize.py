"""Development script: Run optimization and export best config.

This script demonstrates the development workflow:
1. Define an optimizable function with configuration space
2. Run optimization against an evaluation dataset
3. Export the best configuration for production deployment

Usage:
    TRAIGENT_MOCK_LLM=true python examples/persistence/optimize.py
"""

import asyncio
from pathlib import Path

import traigent

# Sample evaluation data (in practice, load from a JSONL file)
EVAL_DATA = [
    {"input": "I love this product!", "expected": "positive"},
    {"input": "Terrible experience.", "expected": "negative"},
    {"input": "It's okay, nothing special.", "expected": "neutral"},
    {"input": "Best purchase ever!", "expected": "positive"},
    {"input": "Would not recommend.", "expected": "negative"},
]


@traigent.optimize(
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4"],
        "temperature": [0.0, 0.3, 0.7],
        "max_tokens": [50, 100, 200],
    },
    objectives=["accuracy"],
    eval_dataset=EVAL_DATA,
    max_trials=5,  # Keep low for demo
    execution_mode="mock",  # Use mock mode for demo
)
async def sentiment_agent(text: str) -> str:
    """Analyze sentiment of the given text.

    In production, this would call an actual LLM.
    The @traigent.optimize decorator injects the optimal config.
    """
    _config = traigent.get_config()  # noqa: F841 - shows how to access config

    # Mock implementation for demo
    # In production: call OpenAI/Anthropic with config["model"], etc.
    if "love" in text.lower() or "best" in text.lower():
        return "positive"
    elif "terrible" in text.lower() or "not recommend" in text.lower():
        return "negative"
    return "neutral"


async def main():
    print("=" * 60)
    print("Config Persistence Demo - Development Phase")
    print("=" * 60)

    # Run optimization
    print("\n1. Running optimization...")
    results = await sentiment_agent.optimize()

    print(f"\n   Completed {results.total_trials} trials")
    print(f"   Best score: {results.best_score:.3f}")
    print(f"   Best config: {results.best_config}")

    # Export for production
    output_dir = Path(__file__).parent / "configs"
    output_path = output_dir / "prod.json"

    print(f"\n2. Exporting config to {output_path}...")
    sentiment_agent.export_config(output_path)

    print("\n3. Next steps:")
    print(f"   - Commit {output_path} to version control")
    print("   - In production, use: load_from='configs/prod.json'")
    print("   - Or set TRAIGENT_CONFIG_PATH environment variable")

    print("\n" + "=" * 60)
    print("Optimization complete! Config exported for deployment.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
