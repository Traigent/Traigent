#!/usr/bin/env python3
"""
TVL 0.9 Tutorial: Getting Started with Traigent Optimization

This script demonstrates how to use a TVL spec with the Traigent decorator
to optimize an LLM application.

Run with (from repo root): .venv/bin/python examples/tvl/tutorials/01_getting_started/run_optimization.py
"""

import random
from pathlib import Path

import traigent
from traigent.tvl import load_tvl_spec

# Load the TVL spec to see what we're optimizing
SPEC_PATH = Path(__file__).parent / "chatbot_optimization.tvl.yml"


def simulate_chatbot_response(model: str, temperature: float, max_tokens: int) -> dict:
    """Simulate a chatbot response with quality and latency metrics.

    In a real application, this would call an LLM API.
    """
    # Simulate latency based on model
    base_latency = 100 if model == "gpt-3.5-turbo" else 150
    latency = base_latency + random.gauss(0, 10) + (max_tokens * 0.5)

    # Simulate quality (higher temperature = more creative but less consistent)
    base_quality = 0.85 if model == "gpt-4o-mini" else 0.75
    quality = base_quality - (temperature * 0.1) + random.gauss(0, 0.05)
    quality = max(0.0, min(1.0, quality))  # Clamp to [0, 1]

    return {
        "response_quality": quality,
        "latency_ms": latency,
    }


# Option 1: Use the @optimize decorator with TVL spec
# Note: Set TRAIGENT_MOCK_LLM=true environment variable for tutorials
@traigent.optimize(
    tvl_spec=str(SPEC_PATH),
)
def optimized_chatbot(
    query: str,
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 256,
) -> dict:
    """A chatbot function that will be optimized by Traigent.

    The decorator injects optimized parameter values for model,
    temperature, and max_tokens based on the TVL spec.
    """
    # Simulate the response
    metrics = simulate_chatbot_response(model, temperature, max_tokens)

    return {
        "answer": f"[Simulated response from {model}]",
        "metrics": metrics,
    }


def main():
    """Demonstrate TVL spec loading and optimization."""
    print("=" * 60)
    print("TVL 0.9 Tutorial: Getting Started")
    print("=" * 60)

    # First, let's inspect the TVL spec
    print("\n1. Loading TVL Spec...")
    spec = load_tvl_spec(spec_path=SPEC_PATH)

    print(f"   Spec path: {spec.path}")
    print(f"   TVars defined: {len(spec.tvars) if spec.tvars else 0}")

    if spec.tvars:
        print("\n   Tuned Variables:")
        for tvar in spec.tvars:
            print(f"   - {tvar.name} ({tvar.type}): {tvar.domain}")

    print(f"\n   Objectives: {len(spec.objective_schema.objectives)}")
    for obj in spec.objective_schema.objectives:
        print(f"   - {obj.name} ({obj.orientation})")

    print(f"\n   Constraints: {len(spec.constraints)}")
    print(f"   Budget: {spec.budget.max_trials} trials")

    # Now run the optimization
    print("\n2. Running Optimization (mock mode)...")
    print("   This would normally search for optimal parameters.")

    # Call the decorated function
    result = optimized_chatbot("What is machine learning?")
    print(f"\n   Response: {result['answer']}")
    print(f"   Quality: {result['metrics']['response_quality']:.3f}")
    print(f"   Latency: {result['metrics']['latency_ms']:.1f}ms")

    print("\n" + "=" * 60)
    print("Tutorial complete! Next: 02_typed_tvars")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130)
