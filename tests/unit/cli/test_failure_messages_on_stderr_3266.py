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


def test_whoami_backend_rejection_on_stderr() -> None:
    import contextlib
    import io

    from traigent.cli.auth_commands import _handle_200_response

    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        assert _handle_200_response({"valid": False}) is False
    assert "Invalid API key" in err.getvalue()
    assert "Invalid API key" not in out.getvalue()


def test_check_no_functions_reason_on_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    empty = tmp_path / "empty.py"
    empty.write_text("x = 1\n")
    result = _runner().invoke(cli, ["check", str(empty)])
    assert result.exit_code == 1
    assert "No optimizable functions found" in result.stderr
    assert "No optimizable functions found" not in result.stdout


def test_check_allow_empty_stays_on_stdout_exit_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    empty = tmp_path / "empty.py"
    empty.write_text("x = 1\n")
    result = _runner().invoke(cli, ["check", str(empty), "--allow-empty"])
    assert result.exit_code == 0
    assert "No optimizable functions found" in result.stdout


def test_check_dry_run_blocking_issues_listed_on_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "m.py").write_text(
        "import traigent\n\n"
        "@traigent.optimize(configuration_space={'t': [0.1, 0.5]}, objectives=['accuracy'])\n"
        "def f(q: str) -> str:\n    return q\n"
    )
    result = _runner().invoke(cli, ["check", "m.py", "--dry-run"])
    if result.exit_code == 0:
        pytest.skip("fixture produced no blocking issue")
    assert "Validation Issues" in result.stderr
    assert "Validation Issues" not in result.stdout
