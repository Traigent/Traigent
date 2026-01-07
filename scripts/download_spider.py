#!/usr/bin/env python3
"""Download Spider dataset from HuggingFace.

This script downloads the Spider text-to-SQL dataset and caches it locally.
The HuggingFace version includes questions and SQL queries but NOT the SQLite
database files needed for execution evaluation.

For execution-based evaluation, download the full dataset from:
https://yale-lily.github.io/spider

Usage:
    python scripts/download_spider.py

The dataset will be cached by HuggingFace in ~/.cache/huggingface/datasets/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' package not installed.")
    print("Install with: pip install datasets")
    sys.exit(1)


def download_spider(output_dir: Path | None = None, force: bool = False) -> dict:
    """Download Spider dataset from HuggingFace.

    Args:
        output_dir: Optional directory to save dataset as JSON files.
                   If None, only downloads to HuggingFace cache.
        force: If True, re-download even if cached.

    Returns:
        Dict with dataset info and statistics.
    """
    print("Downloading Spider dataset from HuggingFace...")
    print("Source: https://huggingface.co/datasets/xlangai/spider")
    print()

    # Load dataset (will use cache if available)
    dataset = load_dataset("xlangai/spider")

    train_data = dataset["train"]
    val_data = dataset["validation"]

    stats = {
        "train_size": len(train_data),
        "validation_size": len(val_data),
        "total_size": len(train_data) + len(val_data),
        "columns": list(train_data.features.keys()),
    }

    print(f"Train examples: {stats['train_size']}")
    print(f"Validation examples: {stats['validation_size']}")
    print(f"Total: {stats['total_size']}")
    print(f"Columns: {stats['columns']}")
    print()

    # Count unique databases
    train_dbs = set(train_data["db_id"])
    val_dbs = set(val_data["db_id"])
    all_dbs = train_dbs | val_dbs

    stats["unique_databases"] = len(all_dbs)
    stats["train_databases"] = len(train_dbs)
    stats["validation_databases"] = len(val_dbs)

    print(f"Unique databases: {stats['unique_databases']}")
    print(f"  Train: {stats['train_databases']}")
    print(f"  Validation: {stats['validation_databases']}")
    print()

    # Optionally save to JSON files
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_file = output_dir / "train.json"
        val_file = output_dir / "validation.json"

        if not force and train_file.exists() and val_file.exists():
            print(f"JSON files already exist in {output_dir}")
            print("Use --force to overwrite.")
        else:
            print(f"Saving to {output_dir}...")

            # Convert to list of dicts
            train_list = [dict(row) for row in train_data]
            val_list = [dict(row) for row in val_data]

            with open(train_file, "w") as f:
                json.dump(train_list, f, indent=2)
            print(f"  Saved {train_file}")

            with open(val_file, "w") as f:
                json.dump(val_list, f, indent=2)
            print(f"  Saved {val_file}")

        stats["output_dir"] = str(output_dir)

    print()
    print("Download complete!")
    print()
    print("Note: This dataset does NOT include SQLite database files.")
    print("For execution-based evaluation, download the full dataset from:")
    print("  https://yale-lily.github.io/spider")

    return stats


def show_sample(n: int = 3) -> None:
    """Show sample examples from the dataset."""
    print(f"\n{'='*60}")
    print(f"Sample examples (first {n} from train split):")
    print("=" * 60)

    dataset = load_dataset("xlangai/spider", split="train")

    for i, example in enumerate(dataset):
        if i >= n:
            break
        print(f"\n--- Example {i+1} ---")
        print(f"DB: {example['db_id']}")
        print(f"Question: {example['question']}")
        print(f"SQL: {example['query']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Spider text-to-SQL dataset from HuggingFace"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Directory to save JSON files (optional)",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-download even if cached",
    )
    parser.add_argument(
        "--show-samples",
        "-s",
        action="store_true",
        help="Show sample examples after download",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        default=3,
        help="Number of samples to show (default: 3)",
    )

    args = parser.parse_args()

    download_spider(output_dir=args.output_dir, force=args.force)

    if args.show_samples:
        show_sample(args.num_samples)


if __name__ == "__main__":
    main()
