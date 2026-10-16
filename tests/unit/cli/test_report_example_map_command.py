"""CLI tests for `traigent report-example-map` command."""

from __future__ import annotations

import json
from pathlib import Path

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


def test_report_example_map_reads_project_files_outside_the_sdk_install(
    tmp_path: Path, monkeypatch
):
    """#2511: on an installed SDK the old workspace root was site-packages, so
    every project dataset was rejected. Paths resolve against the CWD."""
    import traigent

    sdk_root = Path(traigent.__file__).resolve().parents[1]
    assert sdk_root not in tmp_path.resolve().parents  # a genuine outside dir
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "d.jsonl").write_text(
        json.dumps({"input": {"q": "hi"}, "output": "x"}) + "\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        cli,
        [
            "report-example-map",
            "--dataset",
            "data/d.jsonl",
            "--output",
            "results/curation/map.json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "results" / "curation" / "map.json").exists()
