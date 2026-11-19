"""`traigent validate --objectives` accepts custom metric names."""

from __future__ import annotations

import json

from click.testing import CliRunner

from traigent.cli.main import cli
from traigent.utils.validation import Validators


def test_custom_objective_names_are_valid() -> None:
    result = Validators.validate_objectives(["my_metric", "f1.macro", "p95-latency"])
    assert result.is_valid
    assert not result.warnings


def test_malformed_objective_names_are_rejected() -> None:
    for bad in ("", "9lives", "has space", "x" * 200):
        assert not Validators.validate_objectives([bad]).is_valid


def test_non_string_objective_is_rejected() -> None:
    assert not Validators.validate_objectives([3]).is_valid


def test_duplicate_objective_warns_but_passes() -> None:
    result = Validators.validate_objectives(["accuracy", "accuracy"])
    assert result.is_valid
    assert result.warnings


def test_cli_accepts_custom_objective(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))
    (tmp_path / "d.jsonl").write_text(
        json.dumps({"input": {"q": "a"}, "output": "b"}) + "\n"
    )
    ok = CliRunner().invoke(cli, ["validate", "d.jsonl", "-o", "my_custom_metric"])
    assert ok.exit_code == 0, ok.output
    assert "Objectives validation passed" in ok.output

    bad = CliRunner().invoke(cli, ["validate", "d.jsonl", "-o", "bad name"])
    assert bad.exit_code != 0
