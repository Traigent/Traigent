"""Deterministic index ordering for skill-training dataset partitions."""

from __future__ import annotations

import hashlib


def deterministic_index_order(total: int, seed: int) -> list[int]:
    """Return all indices in a stable seed-derived order without a PRNG."""
    if total < 0:
        raise ValueError("total must be non-negative")

    seed_bytes = str(seed).encode("utf-8")
    return sorted(
        range(total), key=lambda index: (_index_digest(seed_bytes, index), index)
    )


def deterministic_sample_indices(total: int, count: int, seed: int) -> list[int]:
    """Return the first count indices from the stable seed-derived ordering."""
    if count < 0:
        raise ValueError("count must be non-negative")
    if count > total:
        raise ValueError("count cannot exceed total")
    return deterministic_index_order(total, seed)[:count]


def _index_digest(seed_bytes: bytes, index: int) -> bytes:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(seed_bytes)
    digest.update(b":")
    digest.update(str(index).encode("ascii"))
    return digest.digest()
