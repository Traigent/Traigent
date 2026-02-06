#!/usr/bin/env python3
"""Runner for evaluation datasets example."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import asdict, is_dataclass
from typing import Any

# Add parent directory to path to import the module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation_datasets import (
    DynamicDatasetGenerator,
    analyze_dataset_characteristics,
    create_python_dataset,
    demonstrate_dataset_formats,
)


def _example_to_dict(example: Any) -> dict[str, Any]:
    """Convert dataset samples to JSON-serialisable dictionaries."""

    if hasattr(example, "input_data"):
        return {
            "input": example.input_data,
            "expected_output": example.expected_output,
            "metadata": getattr(example, "metadata", {}) or {},
        }
    if is_dataclass(example):
        return asdict(example)
    if isinstance(example, dict):
        return example
    return {"value": str(example)}


async def main() -> None:
    """Run the evaluation datasets example."""
    results_file = "results.json"

    print("Running Traigent Evaluation Datasets Example")
    print("=" * 50)

    # Demonstrate dataset formats
    demonstrate_dataset_formats()

    # Analyze dataset characteristics
    analyze_dataset_characteristics()

    # Create sample datasets for display
    python_dataset = create_python_dataset()
    math_dataset = DynamicDatasetGenerator.generate_math_problems(5)
    translation_dataset = DynamicDatasetGenerator.generate_translation_pairs(5)

    # Create results for display
    results_data = {
        "concept": "Evaluation Datasets",
        "description": "Providing test data for Traigent optimization",
        "dataset_formats": {
            "jsonl_file": {
                "example": "eval_dataset='data.jsonl'",
                "description": "File-based dataset, one JSON object per line",
                "advantages": [
                    "Easy to version control",
                    "Can handle large datasets",
                    "Human-readable format",
                ],
            },
            "python_list": {
                "example": "eval_dataset=[{...}, {...}]",
                "description": "In-memory Python list of dictionaries",
                "advantages": [
                    "Dynamic generation possible",
                    "No file I/O overhead",
                    "Programmatic creation",
                ],
            },
            "dynamic_generation": {
                "example": "eval_dataset=generate_dataset()",
                "description": "Function that returns dataset",
                "advantages": [
                    "Fresh data each run",
                    "Parameterizable generation",
                    "Infinite variations possible",
                ],
            },
        },
        "required_structure": {
            "input": "Dictionary with input parameters for your function",
            "expected_output": "Dictionary with expected results",
            "metadata": "(Optional) Additional information for evaluation",
        },
        "example_datasets": {
            "classification": {
                "size": len(python_dataset),
                "sample": _example_to_dict(python_dataset[0]),
            },
            "math_problems": {
                "size": len(math_dataset),
                "sample": _example_to_dict(math_dataset[0]),
            },
            "translation": {
                "size": len(translation_dataset),
                "sample": _example_to_dict(translation_dataset[0]),
            },
        },
        "best_practices": {
            "size_recommendations": {
                "minimum": "5-10 examples",
                "recommended": "20-50 examples",
                "complex_scenarios": "100+ examples",
            },
            "quality_guidelines": [
                "Include edge cases and typical cases",
                "Ensure balanced representation",
                "Verify correct expected outputs",
                "Keep consistent format",
                "Use representative production data",
            ],
        },
        "dataset_impact": {
            "small_dataset": "Quick optimization, may miss edge cases",
            "medium_dataset": "Balanced optimization time and coverage",
            "large_dataset": "Comprehensive optimization, longer runtime",
        },
        "key_insights": [
            "Dataset quality directly impacts optimization results",
            "More diverse examples lead to more robust configurations",
            "Dynamic datasets enable testing different scenarios",
            "JSONL format is best for large, static datasets",
            "Python lists are ideal for small, dynamic datasets",
        ],
    }

    # Save results
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\nWrote {os.path.abspath(results_file)}")
    print("\nDataset examples created successfully!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130) from None
