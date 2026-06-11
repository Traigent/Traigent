"""Tests for the metric/evaluator recommendation catalog."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import validate

from traigent.evaluators.base import BaseEvaluator
from traigent.evaluators.catalog_loader import catalog_entries, catalog_version

_CATALOG_PATH = (
    Path(__file__).resolve().parents[3]
    / "traigent/evaluators/catalog/metric_eval_catalog.v1.json"
)
_SCHEMA_PATH = (
    Path(__file__).resolve().parents[3]
    / "traigent/evaluators/catalog/schemas/metric_eval_catalog_entry_schema.json"
)

# Locked to TraigentSchema/traigent_schema/schemas/measures/measure_schema.json.
_CANONICAL_MEASURE_TYPES = {
    "sanity_check",
    "accuracy",
    "quality",
    "latency",
    "safety",
    "efficiency",
    "reliability",
}
_CANONICAL_EVALUATION_METHODS = {
    "deterministic",
    "llm_based",
    "statistical",
    "hybrid",
}
_CANONICAL_OUTPUT_TYPES = {
    "binary",
    "discrete",
    "continuous",
    "ranking",
}


class _RegistryProbe(BaseEvaluator):
    async def evaluate(self, *args: object, **kwargs: object) -> object:
        raise NotImplementedError


def _raw_catalog_entries() -> list[dict[str, Any]]:
    with _CATALOG_PATH.open(encoding="utf-8") as catalog_file:
        payload = json.load(catalog_file)
    assert isinstance(payload, list)
    return payload


def _entry_schema() -> dict[str, Any]:
    with _SCHEMA_PATH.open(encoding="utf-8") as schema_file:
        payload = json.load(schema_file)
    assert isinstance(payload, dict)
    return payload


def test_metric_eval_catalog_file_validates_against_entry_schema() -> None:
    schema = _entry_schema()

    for entry in _raw_catalog_entries():
        validate(instance=entry, schema=schema)

    assert catalog_version() == "1.0.0"
    assert len(catalog_entries()) == len(_raw_catalog_entries())


def test_metric_eval_catalog_enums_match_measure_schema_sets() -> None:
    entries = catalog_entries()

    assert {entry["measure_type"] for entry in entries} <= _CANONICAL_MEASURE_TYPES
    assert {
        entry["evaluation_method"] for entry in entries
    } <= _CANONICAL_EVALUATION_METHODS
    assert {
        entry["metric"]["output_type"] for entry in entries
    } <= _CANONICAL_OUTPUT_TYPES


def test_metric_eval_catalog_builtin_functions_are_real_registry_keys() -> None:
    builtin_registry = set(_RegistryProbe()._metric_registry)
    builtin_functions = {
        entry["metric"]["builtin_function"]
        for entry in catalog_entries()
        if entry["metric"]["builtin_function"] is not None
    }

    assert builtin_functions
    assert builtin_functions <= builtin_registry


def test_metric_eval_catalog_entries_have_cost_provenance_and_limitations() -> None:
    entries = catalog_entries()

    for entry in entries:
        assert entry["provenance"]
        assert entry["limitations"]
        if entry["evaluation_method"] == "llm_based":
            assert entry["cost_note"].strip()
