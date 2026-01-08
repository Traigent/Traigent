#!/usr/bin/env python3
"""Download Spider dataset from HuggingFace and/or with SQLite databases.

This script downloads the Spider text-to-SQL dataset. Two modes:

1. HuggingFace only (default): Downloads questions/SQL from HuggingFace.
   Fast but no SQLite databases for execution evaluation.

2. Full with databases (--with-databases): Downloads the full Spider dataset
   from Google Drive including all 140 SQLite database files needed for
   execution-based evaluation.

Usage:
    # HuggingFace only (fast, no execution eval)
    python scripts/download_spider.py

    # Full with SQLite databases (slower, enables execution eval)
    python scripts/download_spider.py --with-databases

The HuggingFace data is cached in ~/.cache/huggingface/datasets/
The full dataset with databases is extracted to data/spider/
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

# Google Drive file ID for official Spider dataset
SPIDER_GDRIVE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"
DATA_DIR = Path(__file__).parent.parent / "data" / "spider"

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


def download_spider_databases(force: bool = False) -> dict:
    """Download full Spider dataset with SQLite databases from Google Drive.

    Args:
        force: If True, re-download even if already exists.

    Returns:
        Dict with download info and statistics.
    """
    try:
        import gdown
    except ImportError:
        print("Error: 'gdown' package not installed.")
        print("Install with: pip install gdown")
        sys.exit(1)

    database_dir = DATA_DIR / "database"

    if database_dir.exists() and not force:
        # Check if databases are already there
        db_count = len(list(database_dir.glob("*/*.sqlite")))
        if db_count > 0:
            print(f"Spider databases already exist at {database_dir}")
            print(f"Found {db_count} SQLite databases")
            print("Use --force to re-download.")
            return {"database_dir": str(database_dir), "database_count": db_count}

    print("Downloading Spider dataset with SQLite databases...")
    print(f"Source: Google Drive (file ID: {SPIDER_GDRIVE_ID})")
    print(f"Destination: {DATA_DIR}")
    print()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "spider.zip"

    # Download from Google Drive
    url = f"https://drive.google.com/uc?id={SPIDER_GDRIVE_ID}"
    print(f"Downloading from {url}...")
    gdown.download(url, str(zip_path), quiet=False)

    if not zip_path.exists():
        print("Error: Download failed!")
        sys.exit(1)

    print(f"\nExtracting to {DATA_DIR}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR.parent)  # Extract to data/ (zip contains spider_data/ folder)

    # Clean up zip file
    zip_path.unlink()
    print("Cleaned up zip file.")

    # The zip contains a folder named "spider_data" - move contents to "spider"
    spider_data_dir = DATA_DIR.parent / "spider_data"
    if spider_data_dir.exists():
        import shutil

        for item in spider_data_dir.iterdir():
            dest = DATA_DIR / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(item), str(dest))
        spider_data_dir.rmdir()
        print(f"Moved contents from spider_data/ to spider/")

    # Clean up macOS artifacts
    macosx_dir = DATA_DIR.parent / "__MACOSX"
    if macosx_dir.exists():
        shutil.rmtree(macosx_dir)

    # Count databases
    database_dir = DATA_DIR / "database"
    if database_dir.exists():
        db_count = len(list(database_dir.glob("*/*.sqlite")))
    else:
        db_count = 0

    stats = {
        "database_dir": str(database_dir),
        "database_count": db_count,
        "data_dir": str(DATA_DIR),
    }

    print()
    print(f"Download complete!")
    print(f"  Database directory: {database_dir}")
    print(f"  SQLite databases: {db_count}")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Spider text-to-SQL dataset"
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
    parser.add_argument(
        "--with-databases",
        "-d",
        action="store_true",
        help="Download full Spider with SQLite databases for execution evaluation",
    )

    args = parser.parse_args()

    # Download HuggingFace data
    download_spider(output_dir=args.output_dir, force=args.force)

    # Optionally download full dataset with databases
    if args.with_databases:
        print()
        print("=" * 60)
        print()
        download_spider_databases(force=args.force)

    if args.show_samples:
        show_sample(args.num_samples)


if __name__ == "__main__":
    main()
