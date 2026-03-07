"""CLI tests for `traigent report-example-map` command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from traigent.cli.main import cli
from traigent.reporting.example_map import build_example_content_map


def test_report_example_map_command_generates_output(tmp_path: Path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps({"input": {"q": "hello"}, "output": "world"}),
                json.dumps({"input": {"q": "1+1"}, "output": 2}),
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "example_map.json"

    runner = CliRunner()
    with patch("traigent.cli.main.WORKSPACE_ROOT", tmp_path):
        result = runner.invoke(
            cli,
            [
                "report-example-map",
                "--dataset",
                str(dataset),
                "--output",
                str(output),
                "--dataset-identifier",
                "dataset_for_ids",
            ],
        )

    assert result.exit_code == 0, result.output
    assert output.exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0.0"
    assert payload["dataset_fingerprint"].startswith("sha256:")
    assert len(payload["example_map"]) == 2


def test_report_example_map_command_fails_on_invalid_dataset(tmp_path: Path):
    dataset = tmp_path / "invalid.jsonl"
    dataset.write_text(json.dumps({"not_input": 1}) + "\n", encoding="utf-8")
    output = tmp_path / "example_map.json"

    runner = CliRunner()
    with patch("traigent.cli.main.WORKSPACE_ROOT", tmp_path):
        result = runner.invoke(
            cli,
            [
                "report-example-map",
                "--dataset",
                str(dataset),
                "--output",
                str(output),
            ],
        )

    assert result.exit_code != 0
    assert "missing required 'input' field" in result.output.lower()


def test_report_example_map_command_defaults_to_resolved_dataset_path(
    tmp_path: Path, monkeypatch
):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        json.dumps({"input": {"q": "hello"}, "output": "world"}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "example_map.json"

    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    with patch("traigent.cli.main.WORKSPACE_ROOT", tmp_path):
        result = runner.invoke(
            cli,
            [
                "report-example-map",
                "--dataset",
                "dataset.jsonl",
                "--output",
                "example_map.json",
            ],
        )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    expected_payload = build_example_content_map(dataset)
    assert payload["example_map"] == expected_payload["example_map"]
