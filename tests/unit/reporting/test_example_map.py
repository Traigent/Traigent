"""Unit tests for local report example-map generation."""

from __future__ import annotations

import json

from traigent.reporting.example_map import (
    build_example_content_map,
    compute_dataset_fingerprint,
    validate_example_content_map,
)
from traigent.utils.example_id import compute_dataset_hash, generate_stable_example_id


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
    assert payload["example_map"][ex1]["example_num"] == 2
    assert payload["example_map"][ex1]["expected_output"] == 4


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
