"""Deterministic train/selection/test splitting for skill training."""

from __future__ import annotations

import random

from traigent.evaluators.base import Dataset
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


def _subset(dataset: Dataset, indices: list[int], name: str) -> Dataset:
    return Dataset(
        examples=[dataset.examples[i] for i in indices],
        name=name,
        description=dataset.description,
        metadata=dict(dataset.metadata or {}),
    )


def split_dataset(
    dataset: Dataset,
    selection_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[Dataset, Dataset, Dataset | None]:
    """Split a Dataset deterministically with a held-out selection gate."""

    total = len(dataset.examples)
    indices = list(range(total))
    rng = random.Random(seed)  # NOSONAR - deterministic statistical split, no crypto
    rng.shuffle(indices)

    selection_count = int(total * selection_fraction)
    test_count = int(total * test_fraction)
    if selection_count < 5:
        raise ValueError(
            f"selection split requires at least 5 examples, got {selection_count}"
        )
    if selection_count < 10:
        logger.warning(
            "Skill training selection split has only %d examples; gate may be noisy",
            selection_count,
        )
    if total - selection_count - test_count < 1:
        raise ValueError("skill training split leaves no training examples")

    selection_indices = indices[:selection_count]
    test_indices = indices[selection_count : selection_count + test_count]
    train_indices = indices[selection_count + test_count :]

    base_name = dataset.name or "dataset"
    train = _subset(dataset, train_indices, f"{base_name}__train")
    selection = _subset(dataset, selection_indices, f"{base_name}__selection")
    test = (
        _subset(dataset, test_indices, f"{base_name}__test") if test_indices else None
    )
    return train, selection, test


__all__ = ["split_dataset"]
