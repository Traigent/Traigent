"""Stable example ID generation for dataset examples.

This module provides utilities to generate stable, deterministic example IDs
that remain consistent across multiple optimization runs using the same dataset.

Example IDs follow the format: ex_{hash}_{index}
- hash: 8-12 character hash of the dataset identifier
- index: Zero-based index of the example in the dataset

Thread Safety: All functions are thread-safe and stateless.
"""

import hashlib
from typing import Any


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


#: The id the SDK's own evaluators mint for the example at position ``idx``.
#: It encodes the index and nothing else.
SDK_MINTED_EXAMPLE_ID = "example_{idx}"


def sdk_minted_example_id(example_index: int) -> str:
    """Return the SDK-minted ``example_<index>`` id for a row position."""
    return SDK_MINTED_EXAMPLE_ID.format(idx=example_index)


def coerce_candidate_example_id(value: Any) -> str | None:
    """Normalize a recorded example id the way the wire does (strip, str)."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    if isinstance(value, (int, float)):
        return str(value)
    return None


def resolve_wire_example_id(
    candidate: Any, dataset_hash: str, example_index: int
) -> str:
    """Return the example id the SDK sends on the wire for a row.

    Single source of truth shared by the measures builder and the example
    content map so both key an example identically. Only the SDK's own
    ``example_<index>`` mint for this exact index is passed through; anything
    else (a customer-authored id, another shape, a mismatched index) falls
    back to the opaque ``ex_<hash>_<index>`` id. Customer ids never cross.
    """
    candidate = coerce_candidate_example_id(candidate)
    if candidate is not None and candidate == sdk_minted_example_id(example_index):
        return candidate
    return generate_stable_example_id(dataset_hash, example_index)


#: Reserved nested key for LOCAL-only provenance in a trial's metadata
#: (``{"subset_indices": [...], "privacy_mode": True}``). A dict value is never
#: numeric, so sync's numeric-metric filtering skips it, and nothing flattens
#: metrics under a dict; it is never sent anywhere.
LOCAL_MARKER_KEY = "_traigent_local"


#: Private (non-field) attribute naming the dataset row an ``ExampleResult`` was
#: produced for. Set by the built-in evaluators where they create each result and
#: read when the trial result is assembled. It is NOT a dataclass field, so
#: ``ExampleResult.to_dict`` / ``dataclasses.asdict`` / JSON paths never see it:
#: it is local provenance only and never reaches any payload.
ROW_INDEX_ATTR = "_traigent_row_index"


#: Private attribute on a ``TrialResult`` holding its results' true dataset rows
#: (``list[int]``, result order). Not in ``metadata`` (that feeds the backend
#: payload); local provenance consumed only when the trial is persisted.
TRIAL_ROW_INDICES_ATTR = "_traigent_row_indices"


def stamp_row_index(result: Any, index: int) -> Any:
    """Record the true dataset row on a freshly created example result."""
    try:
        setattr(result, ROW_INDEX_ATTR, int(index))
    except Exception:  # a __slots__/frozen result simply carries no provenance
        pass
    return result


def row_indices_of(example_results: Any) -> list[int] | None:
    """The stamped dataset row of each result, in list order, or ``None``.

    ``None`` unless EVERY result is stamped with a distinct non-negative int
    (custom evaluators that build their own results, gaps and duplicates all
    mean "no provenance").
    """
    if example_results is None:
        return None
    if not isinstance(example_results, (list, tuple)):
        return None
    rows: list[int] = []
    for result in example_results:
        row = getattr(result, ROW_INDEX_ATTR, None)
        if type(row) is not int or row < 0:
            return None
        rows.append(row)
    if len(set(rows)) != len(rows):
        return None
    return rows


def rows_from_indices(
    row_indices: Any, n_results: int, subset: list[int] | None
) -> Any:
    """Compose stamped row indices with a trial's subset into ``rows``.

    ``"full"`` when the results are exactly rows 0..n-1 of the whole dataset,
    else the list of dataset rows (``subset[i]`` per index when a subset ran),
    else ``"unordered"`` (no provenance, a length mismatch, or an index outside
    the subset).
    """
    if (
        not isinstance(row_indices, list)
        or len(row_indices) != n_results
        or not all(type(j) is int and j >= 0 for j in row_indices)
    ):
        return "unordered"
    if subset is None:
        return "full" if row_indices == list(range(n_results)) else list(row_indices)
    if any(j >= len(subset) for j in row_indices):
        return "unordered"
    return [subset[j] for j in row_indices]
