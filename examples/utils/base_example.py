#!/usr/bin/env python3
"""
Shared utilities for Traigent examples.

This module provides common functionality used across examples:
- Mock mode setup
- Dataset creation helpers
- Result display utilities
- Error handling patterns
"""

import json
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

# Always enable mock mode for examples
os.environ["TRAIGENT_MOCK_LLM"] = "true"


class BaseExample(ABC):
    """Base class for Traigent examples."""

    def __init__(self, name: str):
        self.name = name
        self._temp_files: list[str] = []

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """Return Traigent optimization configuration."""
        pass

    def create_dataset(self, samples: list[dict]) -> str:
        """Create a temporary JSONL dataset file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for sample in samples:
                json.dump(sample, f)
                f.write("\n")
            dataset_path = f.name

        self._temp_files.append(dataset_path)
        return dataset_path

    def cleanup(self):
        """Clean up temporary files."""
        for temp_file in self._temp_files:
            try:
                Path(temp_file).unlink(missing_ok=True)
            except Exception:
                pass  # Ignore cleanup errors

    def __del__(self):
        """Ensure cleanup on destruction."""
        self.cleanup()


def create_sample_dataset(task_type: str, num_samples: int = 5) -> list[dict]:
    """Create sample dataset for common tasks."""

    datasets = {
        "sentiment": [
            {"input": {"text": "This product is amazing!"}, "output": "positive"},
            {"input": {"text": "Terrible service"}, "output": "negative"},
            {"input": {"text": "It's okay"}, "output": "neutral"},
        ],
        "qa": [
            {"input": {"question": "What is AI?"}, "output": "Artificial Intelligence"},
            {
                "input": {"question": "How does ML work?"},
                "output": "Machine Learning uses data",
            },
        ],
        "classification": [
            {"input": {"text": "Buy now! Limited offer!"}, "output": "spam"},
            {"input": {"text": "Meeting at 3 PM"}, "output": "ham"},
        ],
    }

    base_samples = datasets.get(task_type, datasets["sentiment"])
    # Repeat samples to reach desired count
    samples = []
    while len(samples) < num_samples:
        samples.extend(base_samples[: num_samples - len(samples)])

    return samples[:num_samples]


def display_optimization_results(result) -> None:
    """Display optimization results in a consistent format."""
    if not result:
        print("❌ No optimization results available")
        return

    print(f"\n🎉 Optimization Complete: {result.get('function_name', 'Unknown')}")
    print("=" * 60)

    if "best_config" in result:
        print("📊 Best Configuration:")
        for key, value in result["best_config"].items():
            print(f"  • {key}: {value}")

    if "best_score" in result:
        print(f"🎯 Best Score: {result['best_score']:.1%}")

    if "total_cost" in result and result["total_cost"] is not None:
        print(f"💰 Total Cost: ${result['total_cost']:.4f}")

    print("✅ Optimization successful!")


def safe_run_example(example_func, *args, **kwargs):
    """Run an example function with error handling."""
    try:
        return example_func(*args, **kwargs)
    except Exception as e:
        print(f"⚠️ Example encountered an issue: {e}")
        print("💡 Try running with mock mode enabled:")
        print("   export TRAIGENT_MOCK_LLM=true")
        return None


# Common dataset templates
SENTIMENT_DATASET = [
    {"input": {"text": "This product is amazing!"}, "output": "positive"},
    {"input": {"text": "Terrible service, very disappointed"}, "output": "negative"},
    {"input": {"text": "It's okay, nothing special"}, "output": "neutral"},
    {"input": {"text": "Best purchase ever!"}, "output": "positive"},
    {"input": {"text": "Waste of money"}, "output": "negative"},
]

QA_DATASET = [
    {
        "input": {"question": "What is AI?"},
        "output": "Artificial Intelligence is technology",
    },
    {
        "input": {"question": "How does ML work?"},
        "output": "Machine Learning uses algorithms",
    },
    {
        "input": {"question": "What is deep learning?"},
        "output": "Deep learning uses neural networks",
    },
]

CLASSIFICATION_DATASET = [
    {"input": {"text": "Buy now! Limited time offer!"}, "output": "spam"},
    {"input": {"text": "Meeting rescheduled to 3 PM"}, "output": "ham"},
    {"input": {"text": "Congratulations! You won!"}, "output": "spam"},
    {"input": {"text": "Project deadline approaching"}, "output": "ham"},
]

if __name__ == "__main__":
    try:
        print("🔧 Traigent Examples Shared Utilities")
        print("This module provides common functionality for examples.")
        print("\nAvailable utilities:")
        print("  • BaseExample class for consistent example structure")
        print("  • Dataset creation helpers")
        print("  • Result display utilities")
        print("  • Error handling patterns")
        print("  • Sample datasets for common tasks")
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130) from None
