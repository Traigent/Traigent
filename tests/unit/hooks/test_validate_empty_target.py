"""Regression tests for `hooks validate --exit-code` on empty targets (#2639)."""

# Traceability: CONC-Quality-Reliability REQ-ORCH-003

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from traigent.cli.hooks_commands import hooks


def test_explicit_empty_dir_with_exit_code_fails(tmp_path: Path) -> None:
    res = CliRunner().invoke(hooks, ["validate", str(tmp_path), "--exit-code"])
    assert res.exit_code == 1, res.output


def test_explicit_empty_dir_allow_empty_passes(tmp_path: Path) -> None:
    res = CliRunner().invoke(
        hooks, ["validate", str(tmp_path), "--exit-code", "--allow-empty"]
    )
    assert res.exit_code == 0, res.output


def test_explicit_dot_counts_as_explicit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    res = CliRunner().invoke(hooks, ["validate", ".", "--exit-code"])
    assert res.exit_code == 1, res.output


def test_default_target_empty_cwd_with_exit_code_unchanged(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    res = CliRunner().invoke(hooks, ["validate", "--exit-code"])
    assert res.exit_code == 0, res.output


def test_explicit_empty_dir_without_exit_code_unchanged(tmp_path: Path) -> None:
    res = CliRunner().invoke(hooks, ["validate", str(tmp_path)])
    assert res.exit_code == 0, res.output


def test_explicit_dir_with_function_still_validates(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(
        "import traigent\n"
        "@traigent.optimize(configuration_space={'model': ['gpt-4o-mini']})\n"
        "def f(q):\n    return q\n"
    )
    res = CliRunner().invoke(hooks, ["validate", str(tmp_path), "--exit-code"])
    assert "Found 1 decorated function" in res.output, res.output
