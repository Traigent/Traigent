"""Unit tests for local report example-map generation."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
from click.testing import CliRunner

import traigent.reporting.example_map as example_map_module
from traigent.cli.main import cli
from traigent.reporting.example_map import (
    build_example_content_map,
    compute_dataset_fingerprint,
    validate_example_content_map,
)
from traigent.utils.exceptions import ValidationError


def test_build_example_content_map_from_jsonl(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "input": {"q": "Where is Paris?"},
                        "output": "France",
                        "topic": "geo",
                    }
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

    # Keys are the ids the SDK sends on the wire (SDK-minted example_<index>).
    ex0, ex1 = "example_0", "example_1"
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

    assert compute_dataset_fingerprint(examples) == compute_dataset_fingerprint(
        same_order
    )
    assert compute_dataset_fingerprint(examples) != compute_dataset_fingerprint(
        reversed_order
    )


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


def test_validate_example_content_map_handles_schema_errors(tmp_path, monkeypatch):
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


def test_report_example_map_uses_canonical_dataset_row_coercion(tmp_path):
    """The customer-facing report command uses runtime row semantics (#3056)."""
    dataset = tmp_path / "dataset.jsonl"
    output = tmp_path / "report.json"
    rows = [
        {"input": {"q": "output"}, "output": "o"},
        {"input_data": {"q": "expected"}, "expected": "e"},
        {"input": {"q": "expected_output"}, "expected_output": "eo"},
        {"input": {"q": "answer"}, "answer": "a"},
        {"input": {"q": "target"}, "target": "t"},
        {"input": {"q": "label"}, "label": "l"},
        {"input_data": {"q": "optional"}},
        {
            "input": {"q": "conflict"},
            "output": "winner",
            "expected": "loser",
            "label": "also-loser",
            "metadata": {"example_id": "customer-row-7", "topic": "nested"},
            "topic": "incidental",
        },
    ]
    dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    result = CliRunner().invoke(
        cli,
        ["report-example-map", "--dataset", str(dataset), "--output", str(output)],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    entries = payload["example_map"]
    assert list(entries) == [f"example_{index}" for index in range(8)]
    assert [entries[f"example_{index}"]["expected_output"] for index in range(7)] == [
        "o",
        "e",
        "eo",
        "a",
        "t",
        "l",
        None,
    ]
    assert entries["example_7"]["input"] == {"q": "conflict"}
    assert entries["example_7"]["expected_output"] == "winner"
    assert entries["example_7"]["metadata"] == {
        "example_id": "customer-row-7",
        "topic": "nested",
    }
    assert payload["dataset_fingerprint"] == compute_dataset_fingerprint(
        [
            {
                "input": row.get("input", row.get("input_data")),
                "expected_output": expected,
                "metadata": metadata,
            }
            for row, expected, metadata in zip(
                rows,
                ["o", "e", "eo", "a", "t", "l", None, "winner"],
                [
                    {},
                    {},
                    {},
                    {},
                    {},
                    {},
                    {},
                    {"example_id": "customer-row-7", "topic": "nested"},
                ],
                strict=True,
            )
        ]
    )


def test_standard_report_fingerprint_and_id_vector_is_unchanged(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        json.dumps({"input": {"q": "hello"}, "output": "world", "topic": "geo"}) + "\n",
        encoding="utf-8",
    )

    payload = build_example_content_map(dataset)

    assert list(payload["example_map"]) == ["example_0"]
    assert payload["example_map"]["example_0"] == {
        "example_num": 1,
        "input": {"q": "hello"},
        "expected_output": "world",
        "metadata": {"topic": "geo"},
    }
    assert payload["dataset_fingerprint"] == (
        "sha256:084619368f591e4778935d134e6deb888f2e58b22b9812c66924770c53f70449"
    )
