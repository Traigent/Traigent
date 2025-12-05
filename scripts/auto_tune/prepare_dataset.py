#!/usr/bin/env python3
"""
Prepare dataset for auto-tuning optimization.
Loads data, splits it, and prepares it for TraiGent optimization.
"""

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_params() -> Dict[str, Any]:
    """Load parameters from params.yaml."""
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def generate_sample_dataset() -> List[Dict[str, Any]]:
    """
    Generate a sample dataset for testing.
    In production, this would load real data.
    """
    # Sample tasks for different problem types
    tasks = [
        {
            "input": "Classify the sentiment of: 'This product exceeded my expectations!'",
            "expected": "positive",
            "type": "classification",
        },
        {
            "input": "Summarize: 'AI is transforming industries through automation and intelligent decision-making.'",
            "expected": "AI transforms industries via automation and smart decisions.",
            "type": "summarization",
        },
        {
            "input": "Extract entities from: 'Apple Inc. announced new products in Cupertino.'",
            "expected": ["Apple Inc.", "Cupertino"],
            "type": "extraction",
        },
        {
            "input": "Translate to Spanish: 'Hello, how are you?'",
            "expected": "Hola, ¿cómo estás?",
            "type": "generation",
        },
        {
            "input": "Answer: What is the capital of France?",
            "expected": "Paris",
            "type": "qa",
        },
    ]

    # Expand dataset with variations
    expanded_dataset = []
    for task in tasks:
        for i in range(5):  # Create 5 variations of each task
            variant = task.copy()
            variant["id"] = f"{task['type']}_{i}"
            expanded_dataset.append(variant)

    return expanded_dataset


def split_dataset(
    dataset: List[Dict[str, Any]], split_ratio: float, seed: int
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split dataset into train and test sets."""
    random.seed(seed)
    shuffled = dataset.copy()
    random.shuffle(shuffled)

    split_point = int(len(shuffled) * split_ratio)
    train_set = shuffled[:split_point]
    test_set = shuffled[split_point:]

    return train_set, test_set


def prepare_for_traigent(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Prepare dataset in TraiGent-compatible format.
    """
    return {
        "version": "1.0",
        "problem_type": "mixed",  # Multiple problem types
        "dataset": dataset,
        "metadata": {
            "total_samples": len(dataset),
            "problem_types": list({d.get("type", "unknown") for d in dataset}),
            "prepared_at": os.environ.get("CI_COMMIT_SHA", "local"),
        },
    }


def main():
    """Main preparation pipeline."""
    print("📊 Preparing dataset for auto-tuning...")

    # Load parameters
    params = load_params()
    split_ratio = params["prepare"]["split_ratio"]
    seed = params["prepare"]["seed"]

    # Create directories
    raw_dir = Path("data/raw")
    prepared_dir = Path("data/prepared")
    prepared_dir.mkdir(parents=True, exist_ok=True)

    # Load or generate dataset
    raw_data_file = raw_dir / "dataset.json"
    if raw_data_file.exists():
        print(f"Loading dataset from {raw_data_file}")
        with open(raw_data_file) as f:
            dataset = json.load(f)
    else:
        print("Generating sample dataset...")
        dataset = generate_sample_dataset()
        # Save for reproducibility
        raw_dir.mkdir(parents=True, exist_ok=True)
        with open(raw_data_file, "w") as f:
            json.dump(dataset, f, indent=2)

    # Split dataset
    train_set, test_set = split_dataset(dataset, split_ratio, seed)
    print(f"Split dataset: {len(train_set)} train, {len(test_set)} test")

    # Prepare TraiGent-compatible format
    train_data = prepare_for_traigent(train_set)
    test_data = prepare_for_traigent(test_set)

    # Save prepared data
    with open(prepared_dir / "train.json", "w") as f:
        json.dump(train_data, f, indent=2)

    with open(prepared_dir / "test.json", "w") as f:
        json.dump(test_data, f, indent=2)

    print(f"✅ Dataset prepared and saved to {prepared_dir}")

    # Output statistics for DVC metrics
    stats = {
        "train_samples": len(train_set),
        "test_samples": len(test_set),
        "total_samples": len(dataset),
        "split_ratio": split_ratio,
    }

    with open("data/prepared/stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return 0


if __name__ == "__main__":
    exit(main())
