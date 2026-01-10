#!/usr/bin/env python3
"""Download HotPotQA dataset for DSPy + Traigent example.

This script downloads the HotPotQA dataset using DSPy's built-in data loader
and saves it locally in JSONL format for offline use.

HotPotQA is a multi-hop question answering dataset from:
- Carnegie Mellon University
- Stanford University
- Universite de Montreal

Usage:
    python download_data.py

Requirements:
    pip install dspy-ai>=2.5.0

License:
    HotPotQA is released under CC BY-SA 4.0.
    See: https://hotpotqa.github.io/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def download_hotpotqa(
    train_size: int = 200,
    dev_size: int = 100,
    test_size: int = 50,
) -> dict[str, list[dict]]:
    """Download HotPotQA dataset using DSPy's built-in loader.

    Args:
        train_size: Number of training examples (for DSPy prompt optimization)
        dev_size: Number of dev examples (for Traigent hyperparameter optimization)
        test_size: Number of test examples (for final validation)

    Returns:
        Dict with train, dev, test splits
    """
    try:
        from dspy.datasets import HotPotQA
    except ImportError:
        print("ERROR: DSPy not installed. Install with:")
        print("  pip install dspy-ai>=2.5.0")
        sys.exit(1)

    print(f"Downloading HotPotQA dataset...")
    print(f"  Train size: {train_size}")
    print(f"  Dev size: {dev_size}")
    print(f"  Test size: {test_size}")

    # Load dataset with specific seeds for reproducibility
    dataset = HotPotQA(
        train_seed=42,
        train_size=train_size,
        eval_seed=2024,
        dev_size=dev_size,
        test_size=test_size,
    )

    # Convert to serializable format
    def example_to_dict(example) -> dict:
        """Convert DSPy Example to dict."""
        return {
            "question": example.question,
            "answer": example.answer,
            # Include any additional fields if present
            **{k: v for k, v in example.items() if k not in ("question", "answer")},
        }

    splits = {
        "train": [example_to_dict(x) for x in dataset.train],
        "dev": [example_to_dict(x) for x in dataset.dev],
        "test": [example_to_dict(x) for x in dataset.test] if dataset.test else [],
    }

    print(f"  Downloaded {len(splits['train'])} train examples")
    print(f"  Downloaded {len(splits['dev'])} dev examples")
    print(f"  Downloaded {len(splits['test'])} test examples")

    return splits


def save_to_jsonl(data: list[dict], filepath: Path) -> None:
    """Save data to JSONL format."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"  Saved: {filepath}")


def main():
    """Download and save HotPotQA dataset."""
    print("=" * 60)
    print("HotPotQA Dataset Downloader")
    print("=" * 60)
    print()

    # Download
    splits = download_hotpotqa(
        train_size=200,  # For DSPy: 200 examples recommended to avoid overfitting
        dev_size=100,  # For Traigent: validation set
        test_size=50,  # Held-out for final evaluation
    )

    # Save to JSONL files
    print("\nSaving to JSONL files...")
    save_to_jsonl(splits["train"], DATA_DIR / "hotpotqa_train.jsonl")
    save_to_jsonl(splits["dev"], DATA_DIR / "hotpotqa_dev.jsonl")
    save_to_jsonl(splits["test"], DATA_DIR / "hotpotqa_test.jsonl")

    # Create combined file for reference
    all_data = splits["train"] + splits["dev"] + splits["test"]
    save_to_jsonl(all_data, DATA_DIR / "hotpotqa_all.jsonl")

    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    print(f"\nData saved to: {DATA_DIR}")
    print("\nDataset splits:")
    print(f"  hotpotqa_train.jsonl : {len(splits['train'])} examples (DSPy training)")
    print(
        f"  hotpotqa_dev.jsonl   : {len(splits['dev'])} examples (Traigent validation)"
    )
    print(f"  hotpotqa_test.jsonl  : {len(splits['test'])} examples (Final evaluation)")
    print("\nNext step: Run the optimization example:")
    print("  python run.py")


if __name__ == "__main__":
    main()
