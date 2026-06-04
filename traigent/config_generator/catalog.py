"""Governed TVAR recommendation catalog helpers."""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from traigent.config_generator.types import EvidenceRef, TVarRecommendation

_CATALOG_DIR = Path(__file__).resolve().with_name("catalog")
_CATALOG_PATH = _CATALOG_DIR / "tvar_catalog.v1.json"
_CATALOG_ENTRY_SCHEMA_PATH = _CATALOG_DIR / "schemas" / "tvar_catalog_entry_schema.json"

_REQUIRED_ENTRY_FIELDS = frozenset(
    {
        "entry_id",
        "schema_version",
        "name",
        "range_type",
        "range_kwargs",
        "kind",
        "effectuation_status",
    }
)
_VALID_RANGE_TYPES = frozenset({"Range", "IntRange", "LogRange", "Choices"})
_VALID_KINDS = frozenset({"value", "cardinality", "topology", "policy"})
_VALID_EFFECTUATION = frozenset({"executable", "manual_guidance", "advisory"})


def load_catalog() -> list[dict[str, Any]]:
    """Load and validate all governed TVAR catalog entries."""
    return copy.deepcopy(list(_load_catalog_cached()))


def catalog_entries(agent_type: str | None = None) -> list[dict[str, Any]]:
    """Return catalog entries, optionally filtered by agent type."""
    entries = load_catalog()
    if agent_type is None:
        return entries
    return [
        entry
        for entry in entries
        if agent_type in {str(value) for value in entry.get("agent_types", [])}
    ]


def entry_to_recommendation(entry: dict[str, Any]) -> TVarRecommendation:
    """Project a catalog entry back into the public TVarRecommendation shape."""
    evidence_refs = tuple(
        EvidenceRef(
            scope=str(ref["scope"]),
            metric=str(ref["metric"]),
            n=int(ref["n"]),
            model=str(ref["model"]),
            baseline=_restore_evidence_value(ref.get("baseline")),
            candidate=_restore_evidence_value(ref.get("candidate")),
            delta=ref.get("delta"),
            limitations=tuple(str(item) for item in ref.get("limitations", ())),
        )
        for ref in entry.get("evidence_refs", ())
    )
    return TVarRecommendation(
        name=str(entry["name"]),
        range_type=str(entry["range_type"]),
        range_kwargs=dict(entry.get("range_kwargs", {})),
        category=str(entry.get("category", "")),
        reasoning=str(entry.get("reasoning", "")),
        impact_estimate=str(entry.get("impact_estimate", "medium")),
        entry_id=str(entry.get("entry_id", "")),
        catalog_entry_id=str(entry.get("entry_id", "")),
        kind=str(entry.get("kind", "")),
        effectuation_status=str(entry.get("effectuation_status", "")),
        effectuation_strategy=str(entry.get("effectuation_strategy", "")),
        evidence_refs=evidence_refs,
        apply_guidance=str(entry.get("apply_guidance", "")),
        recommended_values=tuple(entry.get("recommended_values", ())),
    )


@lru_cache(maxsize=1)
def _load_catalog_cached() -> tuple[dict[str, Any], ...]:
    with _CATALOG_PATH.open(encoding="utf-8") as catalog_file:
        payload = json.load(catalog_file)
    if not isinstance(payload, list):
        raise ValueError("TVAR catalog must be a JSON array of entries")

    schema = _load_catalog_entry_schema()
    entries: list[dict[str, Any]] = []
    for index, entry in enumerate(payload):
        if not isinstance(entry, dict):
            raise ValueError(f"TVAR catalog entry {index} must be an object")
        _validate_catalog_entry(entry, schema)
        entries.append(dict(entry))
    return tuple(entries)


def _load_catalog_entry_schema() -> dict[str, Any]:
    with _CATALOG_ENTRY_SCHEMA_PATH.open(encoding="utf-8") as schema_file:
        schema = json.load(schema_file)
    if not isinstance(schema, dict):
        raise ValueError("TVAR catalog entry schema must be an object")
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
            f"Invalid TVAR catalog entry {entry_id}: {exc.message}"
        ) from exc


def _minimal_validate_catalog_entry(entry: dict[str, Any]) -> None:
    missing = sorted(_REQUIRED_ENTRY_FIELDS - set(entry))
    if missing:
        raise ValueError(f"TVAR catalog entry missing fields: {missing}")
    if entry["range_type"] not in _VALID_RANGE_TYPES:
        raise ValueError(f"Unsupported range_type: {entry['range_type']}")
    if entry["kind"] not in _VALID_KINDS:
        raise ValueError(f"Unsupported kind: {entry['kind']}")
    if entry["effectuation_status"] not in _VALID_EFFECTUATION:
        raise ValueError(
            f"Unsupported effectuation_status: {entry['effectuation_status']}"
        )


def _restore_evidence_value(value: Any) -> Any:
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    return value
