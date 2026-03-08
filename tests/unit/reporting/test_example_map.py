"""Unit tests for local report example-map generation."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

import traigent.reporting.example_map as example_map_module
from traigent.reporting.example_map import (
    build_example_content_map,
    compute_dataset_fingerprint,
    validate_example_content_map,
)
from traigent.utils.example_id import compute_dataset_hash, generate_stable_example_id
from traigent.utils.exceptions import ValidationError


def test_build_example_content_map_from_jsonl(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        '\n'.join(
            [
                json.dumps(
                    {"input": {"q": "Where is Paris?"}, "output": "France", "topic": "geo"}
                ),
                json.dumps(
                    {
                        "input": {"q": "2+2"},
                        "output": 4,
                        "difficulty": "easy",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    payload = build_example_content_map(
        dataset,
        dataset_identifier="qa_dataset_v1",
    )
    errors = validate_example_content_map(payload)
    assert errors == []
    assert payload["schema_version"] == "1.0.0"
    assert payload["dataset_fingerprint"].startswith("sha256:")
    assert len(payload["example_map"]) == 2

    dataset_hash = compute_dataset_hash("qa_dataset_v1")
    ex0 = generate_stable_example_id(dataset_hash, 0)
    ex1 = generate_stable_example_id(dataset_hash, 1)
    assert payload["example_map"][ex0]["example_num"] == 1
    assert payload["example_map"][ex0]["expected_output"] == "France"
    assert payload["example_map"][ex0]["metadata"] == {"topic": "geo"}
    assert payload["example_map"][ex1]["example_num"] == 2
    assert payload["example_map"][ex1]["expected_output"] == 4
    assert payload["example_map"][ex1]["metadata"] == {"difficulty": "easy"}


def test_build_example_content_map_uses_resolved_path_for_default_ids(
    tmp_path, monkeypatch
):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        json.dumps({"input": {"q": "hello"}, "output": "world"}) + "\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    relative_payload = build_example_content_map(Path("dataset.jsonl"))
    absolute_payload = build_example_content_map(dataset)

    assert relative_payload["example_map"] == absolute_payload["example_map"]


def test_build_example_content_map_includes_empty_metadata(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        json.dumps({"input": {"q": "hello"}, "output": "world"}) + "\n",
        encoding="utf-8",
    )

    payload = build_example_content_map(dataset, dataset_identifier="dataset_v1")
    entry = next(iter(payload["example_map"].values()))

    assert entry["metadata"] == {}


def test_dataset_fingerprint_is_deterministic_and_order_sensitive():
    examples = [
        {"input": {"a": 1}, "expected_output": "x", "metadata": {"k": 1}},
        {"input": {"a": 2}, "expected_output": "y", "metadata": {"k": 2}},
    ]
    same_order = [
        {"input": {"a": 1}, "expected_output": "x", "metadata": {"k": 1}},
        {"input": {"a": 2}, "expected_output": "y", "metadata": {"k": 2}},
    ]
    reversed_order = list(reversed(same_order))

    assert compute_dataset_fingerprint(examples) == compute_dataset_fingerprint(same_order)
    assert compute_dataset_fingerprint(examples) != compute_dataset_fingerprint(reversed_order)


def test_validate_example_content_map_rejects_unknown_fields(tmp_path):
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps([{"input": {"x": 1}, "output": {"y": 2}}]),
        encoding="utf-8",
    )
    payload = build_example_content_map(dataset)
    payload["extra_field"] = True
    errors = validate_example_content_map(payload)
    assert errors


def test_validate_example_content_map_handles_schema_errors(
    tmp_path, monkeypatch
):
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps([{"input": {"x": 1}, "output": {"y": 2}}]),
        encoding="utf-8",
    )
    payload = build_example_content_map(dataset)

    def raise_schema_error(*args, **kwargs):
        raise jsonschema.SchemaError("broken schema")

    monkeypatch.setattr(example_map_module.jsonschema, "validate", raise_schema_error)

    errors = validate_example_content_map(payload)

    assert errors == ["Schema definition error: broken schema"]


def test_build_example_content_map_rejects_conflicting_output_fields(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "input": {"q": "test"},
                "output": "answer_a",
                "expected_output": "answer_b",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValidationError,
        match="conflicting 'output' and 'expected_output' values",
    ):
        build_example_content_map(dataset)
