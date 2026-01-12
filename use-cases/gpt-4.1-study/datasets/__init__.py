"""Datasets for GPT-4.1 replication study experiments."""

from pathlib import Path

DATASETS_DIR = Path(__file__).parent

CODING_DATASET = DATASETS_DIR / "coding_dataset.jsonl"
INSTRUCTION_FOLLOWING_DATASET = DATASETS_DIR / "instruction_following_dataset.jsonl"
LONG_CONTEXT_DATASET = DATASETS_DIR / "long_context_dataset.jsonl"
FUNCTION_CALLING_DATASET = DATASETS_DIR / "function_calling_dataset.jsonl"

__all__ = [
    "DATASETS_DIR",
    "CODING_DATASET",
    "INSTRUCTION_FOLLOWING_DATASET",
    "LONG_CONTEXT_DATASET",
    "FUNCTION_CALLING_DATASET",
]
