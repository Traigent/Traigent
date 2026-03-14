"""Local-only example content map generation for report enrichment."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import jsonschema

from traigent.utils.example_id import compute_dataset_hash, generate_stable_example_id
from traigent.utils.exceptions import ValidationError

EXAMPLE_CONTENT_MAP_SCHEMA_VERSION = "1.0.0"


def build_example_content_map(
    dataset_path: str | Path,
    *,
    dataset_identifier: str | None = None,
) -> dict[str, Any]:
    """Build a local-only example content map from a dataset file."""
    path = Path(dataset_path)
    examples = _load_examples(path)
    identifier = (
        dataset_identifier if dataset_identifier is not None else str(path.resolve())
    )
    dataset_hash = compute_dataset_hash(identifier)
    example_map: dict[str, Any] = {}

    for index, example in enumerate(examples):
        example_id = generate_stable_example_id(dataset_hash, index)
        payload: dict[str, Any] = {
            "example_num": index + 1,
            "input": example["input"],
            "expected_output": example["expected_output"],
            # Keep stored entries aligned with fingerprint canonicalization.
            "metadata": example["metadata"],
        }
        example_map[example_id] = payload

    return {
        "schema_version": EXAMPLE_CONTENT_MAP_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_fingerprint": compute_dataset_fingerprint(examples),
        "example_map": example_map,
    }


def compute_dataset_fingerprint(
    examples: list[dict[str, Any]],
) -> str:
    """Compute deterministic fingerprint over dataset content and order."""
    canonical_examples = [
        {
            "input": ex["input"],
            "expected_output": ex["expected_output"],
            "metadata": ex.get("metadata") or {},
        }
        for ex in examples
    ]
    serialized = json.dumps(
        canonical_examples,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def validate_example_content_map(payload: dict[str, Any]) -> list[str]:
    """Validate a generated example map against schema contract."""
    schema = _load_example_content_map_schema()
    try:
        jsonschema.validate(
            instance=payload,
            schema=schema,
            format_checker=jsonschema.FormatChecker(),
        )
    except jsonschema.SchemaError as exc:
        return [f"Schema definition error: {exc.message}"]
    except jsonschema.ValidationError as exc:
        return [str(exc)]
    return []


def _load_example_content_map_schema() -> dict[str, Any]:
    """Load schema from TraigentSchema package with fallback definition."""
    try:
        from traigent_schema import load_schema as tg_load_schema

        return cast(dict[str, Any], tg_load_schema("example_content_map_schema"))
    except ImportError:
        return _fallback_example_content_map_schema()


def _load_examples(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _load_examples_from_jsonl(path)
    if suffix == ".json":
        return _load_examples_from_json(path)
    raise ValidationError(f"Unsupported dataset format: {suffix}. Use .json or .jsonl")


def _load_examples_from_jsonl(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValidationError(
                    f"Invalid JSON on line {line_no} in {path}: {exc}"
                ) from exc
            examples.append(_normalize_example_item(data, source=f"{path}:{line_no}"))
    if not examples:
        raise ValidationError(f"No examples found in dataset: {path}")
    return examples


def _load_examples_from_json(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid JSON in {path}: {exc}") from exc

    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict) and isinstance(payload.get("examples"), list):
        items = payload["examples"]
    elif isinstance(payload, dict):
        items = [payload]
    else:
        raise ValidationError(f"Unsupported JSON dataset structure in {path}")

    examples = [
        _normalize_example_item(item, source=f"{path}:{index}")
        for index, item in enumerate(items)
    ]
    if not examples:
        raise ValidationError(f"No examples found in dataset: {path}")
    return examples


def _normalize_example_item(item: Any, *, source: str) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValidationError(f"Dataset entry must be an object: {source}")
    if "input" not in item:
        raise ValidationError(f"Dataset entry missing required 'input' field: {source}")
    if "output" not in item and "expected_output" not in item:
        raise ValidationError(
            f"Dataset entry missing required 'output' or 'expected_output' field: {source}"
        )
    has_output = "output" in item
    has_expected_output = "expected_output" in item
    output_value = item.get("output")
    expected_output_value = item.get("expected_output")
    if has_output and has_expected_output and output_value != expected_output_value:
        raise ValidationError(
            "Dataset entry has conflicting 'output' and 'expected_output' values: "
            f"{source}"
        )
    expected_output = output_value if has_output else expected_output_value
    metadata = {
        key: value
        for key, value in item.items()
        if key not in {"input", "output", "expected_output"}
    }
    return {
        "input": item["input"],
        "expected_output": expected_output,
        "metadata": metadata,
    }


def _fallback_example_content_map_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Example Content Map Schema",
        "type": "object",
        "properties": {
            "schema_version": {
                "type": "string",
                "pattern": r"^\d+\.\d+\.\d+$",
            },
            "generated_at": {"type": "string", "format": "date-time"},
            "dataset_fingerprint": {"type": "string"},
            "example_map": {
                "type": "object",
                "patternProperties": {
                    "^.+$": {
                        "type": "object",
                        "properties": {
                            "example_num": {"type": "integer", "minimum": 1},
                            "input": {},
                            "expected_output": {},
                            "metadata": {"type": "object"},
                        },
                        "required": ["example_num", "input", "expected_output"],
                        "additionalProperties": False,
                    }
                },
                "additionalProperties": False,
            },
        },
        "required": [
            "schema_version",
            "generated_at",
            "dataset_fingerprint",
            "example_map",
        ],
        "additionalProperties": False,
    }
