"""Dataset registry utilities.

This module provides helpers to resolve dataset identifiers to concrete file paths
while attaching optional metadata such as human-readable names and descriptions.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability CONC-Quality-Maintainability FUNC-EVAL-METRICS REQ-EVAL-005 SYNC-OptimizationFlow

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from traigent.utils.exceptions import ValidationError

DATASET_REGISTRY_ENV = "TRAIGENT_DATASET_REGISTRY"


@dataclass(slots=True)
class DatasetRegistryEntry:
    """Represents a single dataset entry registered for evaluation."""

    name: str
    path: str
    description: str | None = None
    metadata: dict[str, Any] | None = None


_CACHE: dict[str, DatasetRegistryEntry] | None = None
_CACHE_SOURCE: Path | None = None
_CACHE_MTIME: float | None = None


def clear_dataset_registry_cache() -> None:
    """Clear any cached registry contents (useful for tests)."""

    global _CACHE, _CACHE_SOURCE, _CACHE_MTIME
    _CACHE = None
    _CACHE_SOURCE = None
    _CACHE_MTIME = None


def _load_registry() -> Mapping[str, DatasetRegistryEntry]:
    """Load the dataset registry from disk if configured."""

    global _CACHE, _CACHE_SOURCE, _CACHE_MTIME

    registry_path_env = os.getenv(DATASET_REGISTRY_ENV)
    if not registry_path_env:
        _CACHE = {}
        _CACHE_SOURCE = None
        _CACHE_MTIME = None
        return {}

    registry_path = Path(registry_path_env).expanduser()
    try:
        resolved_path = registry_path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValidationError(
            f"Dataset registry file not found: {registry_path}"
        ) from exc

    try:
        mtime = resolved_path.stat().st_mtime
    except OSError as exc:  # pragma: no cover - unexpected filesystem issue
        raise ValidationError(
            f"Unable to read dataset registry metadata: {exc}"
        ) from exc

    if _CACHE is not None and _CACHE_SOURCE == resolved_path and _CACHE_MTIME == mtime:
        return _CACHE

    try:
        with resolved_path.open(encoding="utf-8") as handle:
            raw_data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid JSON in dataset registry: {exc}") from exc
    except OSError as exc:
        raise ValidationError(f"Failed to read dataset registry: {exc}") from exc

    if not isinstance(raw_data, dict):
        raise ValidationError("Dataset registry must be a JSON object at the top level")

    datasets_obj = raw_data.get("datasets", raw_data)
    if not isinstance(datasets_obj, dict):
        raise ValidationError(
            "Dataset registry must map dataset identifiers to entries"
        )

    parsed: dict[str, DatasetRegistryEntry] = {}

    for key, value in datasets_obj.items():
        if not isinstance(key, str) or not key:
            raise ValidationError("Dataset identifiers must be non-empty strings")

        if isinstance(value, str):
            entry = DatasetRegistryEntry(name=key, path=value)
            parsed[key] = entry
            continue

        if not isinstance(value, dict):
            raise ValidationError(
                f"Dataset '{key}' entry must be either a string path or an object"
            )

        path_value = value.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise ValidationError(f"Dataset '{key}' must define a non-empty 'path'")

        display_name = value.get("name")
        if display_name is not None and not isinstance(display_name, str):
            raise ValidationError(f"Dataset '{key}' field 'name' must be a string")

        description = value.get("description")
        if description is not None and not isinstance(description, str):
            raise ValidationError(
                f"Dataset '{key}' field 'description' must be a string"
            )

        metadata = value.get("metadata")
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise ValidationError(
                    f"Dataset '{key}' field 'metadata' must be an object"
                )
            metadata_dict = dict(metadata)
        else:
            metadata_dict = None

        parsed[key] = DatasetRegistryEntry(
            name=display_name or key,
            path=path_value,
            description=description,
            metadata=metadata_dict,
        )

    _CACHE = parsed
    _CACHE_SOURCE = resolved_path
    _CACHE_MTIME = mtime
    return parsed


def resolve_dataset_reference(
    reference: str,
) -> tuple[str, DatasetRegistryEntry | None]:
    """Resolve a dataset identifier or path using the registry.

    Args:
        reference: Dataset identifier or file path provided by the user.

    Returns:
        A tuple containing the resolved file path string and the registry entry (if any).
    """

    registry = _load_registry()
    entry = registry.get(reference)
    if entry is not None:
        return entry.path, entry

    return reference, None


def list_registered_datasets() -> Mapping[str, DatasetRegistryEntry]:
    """Expose the currently registered datasets (mostly for debugging)."""

    return _load_registry()
