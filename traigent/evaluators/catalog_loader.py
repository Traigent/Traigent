"""Metric/evaluator recommendation catalog helpers."""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

__all__ = [
    "catalog_entries",
    "catalog_version",
    "load_catalog",
]

_CATALOG_DIR = Path(__file__).resolve().with_name("catalog")
_CATALOG_PATH = _CATALOG_DIR / "metric_eval_catalog.v1.json"
_CATALOG_ENTRY_SCHEMA_PATH = (
    _CATALOG_DIR / "schemas" / "metric_eval_catalog_entry_schema.json"
)

_REQUIRED_ENTRY_FIELDS = frozenset(
    {
        "entry_id",
        "schema_version",
        "status",
        "version",
        "task_types",
        "measure_type",
        "metric",
        "evaluation_method",
        "evaluator_binding",
        "provenance",
        "impact_estimate",
        "confidence",
        "limitations",
    }
)
_VALID_TASK_TYPES = frozenset({"code_gen", "rag", "general"})
_VALID_MEASURE_TYPES = frozenset(
    {
        "sanity_check",
        "accuracy",
        "quality",
        "latency",
        "safety",
        "efficiency",
        "reliability",
    }
)
_VALID_EVALUATION_METHODS = frozenset(
    {"deterministic", "llm_based", "statistical", "hybrid"}
)
_VALID_OUTPUT_TYPES = frozenset({"binary", "discrete", "continuous", "ranking"})


def load_catalog() -> list[dict[str, Any]]:
    """Load and validate all metric/evaluator catalog entries."""
    return copy.deepcopy(list(_load_catalog_cached()))


def catalog_version() -> str:
    """Return the version declared by the catalog file's entries."""
    versions = {
        str(entry.get("version", "")).strip() for entry in _load_catalog_cached()
    }
    versions.discard("")
    if not versions:
        raise ValueError("Metric/evaluator catalog entries must declare a version")
    if len(versions) > 1:
        raise ValueError(
            "Metric/evaluator catalog entries declare multiple versions: "
            + ", ".join(sorted(versions))
        )
    return next(iter(versions))


def catalog_entries(task_type: str | None = None) -> list[dict[str, Any]]:
    """Return catalog entries, optionally filtered by task type."""
    entries = load_catalog()
    if task_type is None:
        return entries
    return [
        entry
        for entry in entries
        if task_type in {str(value) for value in entry.get("task_types", [])}
    ]


@lru_cache(maxsize=1)
def _load_catalog_cached() -> tuple[dict[str, Any], ...]:
    with _CATALOG_PATH.open(encoding="utf-8") as catalog_file:
        payload = json.load(catalog_file)
    if not isinstance(payload, list):
        raise ValueError("Metric/evaluator catalog must be a JSON array of entries")

    schema = _load_catalog_entry_schema()
    entries: list[dict[str, Any]] = []
    for index, entry in enumerate(payload):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Metric/evaluator catalog entry {index} must be an object"
            )
        _validate_catalog_entry(entry, schema)
        entries.append(dict(entry))
    return tuple(entries)


def _load_catalog_entry_schema() -> dict[str, Any]:
    with _CATALOG_ENTRY_SCHEMA_PATH.open(encoding="utf-8") as schema_file:
        schema = json.load(schema_file)
    if not isinstance(schema, dict):
        raise ValueError("Metric/evaluator catalog entry schema must be an object")
    return schema


def _validate_catalog_entry(entry: dict[str, Any], schema: dict[str, Any]) -> None:
    try:
        from jsonschema import validate
        from jsonschema.exceptions import ValidationError
    except Exception:
        _minimal_validate_catalog_entry(entry)
        return

    try:
        validate(instance=entry, schema=schema)
    except ValidationError as exc:
        entry_id = entry.get("entry_id", "<unknown>")
        raise ValueError(
            f"Invalid metric/evaluator catalog entry {entry_id}: {exc.message}"
        ) from exc


def _minimal_validate_catalog_entry(entry: dict[str, Any]) -> None:
    missing = sorted(_REQUIRED_ENTRY_FIELDS - set(entry))
    if missing:
        raise ValueError(f"Metric/evaluator catalog entry missing fields: {missing}")
    if entry["status"] != "active":
        raise ValueError(f"Unsupported status: {entry['status']}")
    task_types = {str(value) for value in entry["task_types"]}
    if not task_types or not task_types <= _VALID_TASK_TYPES:
        raise ValueError(f"Unsupported task_types: {sorted(task_types)}")
    if entry["measure_type"] not in _VALID_MEASURE_TYPES:
        raise ValueError(f"Unsupported measure_type: {entry['measure_type']}")
    if entry["evaluation_method"] not in _VALID_EVALUATION_METHODS:
        raise ValueError(f"Unsupported evaluation_method: {entry['evaluation_method']}")
    metric = entry.get("metric")
    if not isinstance(metric, dict):
        raise ValueError("Metric/evaluator catalog entry metric must be an object")
    if metric.get("output_type") not in _VALID_OUTPUT_TYPES:
        raise ValueError(f"Unsupported output_type: {metric.get('output_type')}")
    if entry["evaluation_method"] == "llm_based" and not entry.get("cost_note"):
        raise ValueError("llm_based metric/evaluator catalog entries need cost_note")
