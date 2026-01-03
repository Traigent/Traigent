#!/usr/bin/env python3
"""
Test script for Traigent optimization validation system.
This file contains functions decorated with @traigent.optimize for testing.

NOTE: This test file uses dataset paths that require the 'data/' directory
to exist in the repository root. The data/ directory is in .gitignore,
so these tests are skipped if the datasets don't exist.
"""

import os
from pathlib import Path
from typing import Any

import pytest

# Enable mock mode for testing
os.environ["TRAIGENT_MOCK_MODE"] = "true"

# Get project root to resolve dataset paths
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"

# Skip this module if the data directory doesn't exist
if not _DATA_DIR.exists():
    pytest.skip(
        "Skipping test_optimization_validation: data/ directory not found",
        allow_module_level=True,
    )

import traigent

# Use absolute paths to avoid working directory issues
_TEST_DATASET = str(_DATA_DIR / "test_dataset.jsonl")
_QA_DATASET = str(_DATA_DIR / "qa_dataset.jsonl")
_SIMPLE_DATASET = str(_DATA_DIR / "simple_dataset.jsonl")


@traigent.optimize(
    eval_dataset=_TEST_DATASET,
    objectives=["accuracy", "cost"],
    configuration_space={
        "temperature": [0.0, 0.5, 1.0],
        "model": ["gpt-3.5-turbo", "gpt-4"],
    },
)
def analyze_sentiment(
    text: str, temperature: float = 0.7, model: str = "gpt-3.5-turbo"
) -> dict[str, Any]:
    """Analyze sentiment with configurable parameters."""
    # Simulate sentiment analysis
    import time

    time.sleep(0.1)  # Simulate processing time

    # Simple mock logic based on parameters
    confidence = 0.8 if temperature < 0.5 else 0.7
    if model == "gpt-4":
        confidence += 0.1

    sentiment = (
        "positive"
        if "good" in text.lower()
        else "negative" if "bad" in text.lower() else "neutral"
    )

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "model_used": model,
        "temperature_used": temperature,
    }


@traigent.optimize(
    eval_dataset=_QA_DATASET,
    objectives=["accuracy"],
    configuration_space={
        "max_tokens": [50, 100, 200],
        "temperature": [0.0, 0.3, 0.7],
    },
)
def answer_question(
    question: str, max_tokens: int = 100, temperature: float = 0.0
) -> str:
    """Answer questions with configurable parameters."""
    # Simulate question answering
    import random

    answers = [
        f"Based on my analysis with {max_tokens} tokens at temp {temperature}, the answer is...",
        f"After processing (tokens={max_tokens}, temp={temperature}), I believe...",
        f"Using configuration max_tokens={max_tokens}, temperature={temperature}: The answer is...",
    ]

    return random.choice(answers)


def unoptimized_function(text: str) -> str:
    """This function is not decorated and should not be discovered."""
    return f"Unoptimized: {text}"


# Function with no default parameters
@traigent.optimize(
    eval_dataset=_SIMPLE_DATASET,
    objectives=["speed"],
    configuration_space={"method": ["fast", "slow"]},
)
def process_data(data: str, method: str) -> str:
    """Process data with no default parameters."""
    return f"Processed {data} using {method} method"


if __name__ == "__main__":
    print("Test functions defined successfully!")
    print("\nFunctions with @traigent.optimize:")
    print("1. analyze_sentiment - has defaults")
    print("2. answer_question - has defaults")
    print("3. process_data - no defaults")
    print("\nUndecorated functions:")
    print("4. unoptimized_function - should be ignored")
