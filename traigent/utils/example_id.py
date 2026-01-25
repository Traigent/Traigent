"""Stable example ID generation for dataset examples.

This module provides utilities to generate stable, deterministic example IDs
that remain consistent across multiple optimization runs using the same dataset.

Example IDs follow the format: ex_{hash}_{index}
- hash: 8-12 character hash of the dataset identifier
- index: Zero-based index of the example in the dataset

Thread Safety: All functions are thread-safe and stateless.
"""

import hashlib


def generate_stable_example_id(dataset_hash: str, example_index: int) -> str:
    """Generate stable, deterministic example ID.

    Creates a unique ID that combines a dataset hash with an example index,
    ensuring the same dataset will always generate the same IDs across runs.

    Args:
        dataset_hash: Hash of the dataset name/path (from compute_dataset_hash)
        example_index: Zero-based index of example in dataset

    Returns:
        Stable example ID like "ex_a3f4b2c8_0"

    Example:
        >>> dataset_hash = compute_dataset_hash("my_dataset")
        >>> generate_stable_example_id(dataset_hash, 0)
        'ex_12345678_0'
        >>> generate_stable_example_id(dataset_hash, 42)
        'ex_12345678_42'

    Thread Safety: Safe - pure function with no shared state
    """
    combined = f"{dataset_hash}:{example_index}"
    id_hash = hashlib.sha256(combined.encode()).hexdigest()[:8]
    return f"ex_{id_hash}_{example_index}"


def compute_dataset_hash(dataset_name: str) -> str:
    """Compute hash of dataset identifier.

    Creates a 12-character hash from a dataset name that serves as a stable
    identifier for the dataset across optimization runs.

    Args:
        dataset_name: Name or path of the dataset

    Returns:
        12-character hexadecimal hash like "a3f4b2c891de"

    Example:
        >>> compute_dataset_hash("customer_support_qa")
        'a3f4b2c891de'

    Thread Safety: Safe - pure function with no shared state
    """
    return hashlib.sha256(dataset_name.encode()).hexdigest()[:12]
