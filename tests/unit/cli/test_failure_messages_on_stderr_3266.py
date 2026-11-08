"""Failures of check/optimize/validate/auth whoami go to stderr (Traigent#3266)."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from traigent.cli.auth_commands import auth
from traigent.cli.main import cli


def _runner() -> CliRunner:
    try:
        return CliRunner(mix_stderr=False)  # click < 8.2
    except TypeError:
        return CliRunner()  # click >= 8.2 always separates streams


def test_check_syntax_error_reason_on_stderr(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text("def f(:\n")
    result = _runner().invoke(cli, ["check", str(bad)])
    assert result.exit_code == 1
    assert "Error during validation" in result.stderr
    assert "Error during validation" not in result.stdout


def test_optimize_load_error_reason_on_stderr(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text("def f(:\n")
    result = _runner().invoke(cli, ["optimize", str(bad)])
    assert result.exit_code == 1
    assert "Error loading file" in result.stderr
    assert "Error loading file" not in result.stdout


def test_optimize_no_functions_reason_on_stderr(tmp_path: Path) -> None:
    empty = tmp_path / "empty.py"
    empty.write_text("x = 1\n")
    result = _runner().invoke(cli, ["optimize", str(empty)])
    assert result.exit_code == 1
    assert "No functions with @traigent.optimize" in result.stderr
    assert "No functions with @traigent.optimize" not in result.stdout


def test_validate_bad_dataset_reason_on_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "bad.json").write_text("not json{")
    result = _runner().invoke(cli, ["validate", "bad.json"])
    assert result.exit_code != 0
    assert "validation failed" in result.stderr
    assert "validation failed" not in result.stdout


def test_whoami_invalid_format_on_stderr() -> None:
    result = _runner().invoke(auth, ["whoami", "nonsensekey"])
    assert result.exit_code == 1
    assert "Invalid API key format" in result.stderr
    assert "Invalid API key format" not in result.stdout
