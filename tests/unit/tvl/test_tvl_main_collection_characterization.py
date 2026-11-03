"""Characterization of file collection and reporting in ``traigent.tvl.__main__.main``."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import traigent.tvl.__main__ as tvl_main
from traigent.tvl.__main__ import main


@pytest.fixture
def seen(monkeypatch: pytest.MonkeyPatch):
    """Replace the validator with a recorder; pass/fail by file name."""
    calls: dict[str, object] = {"files": None, "kwargs": None, "finds": []}
    real_find = tvl_main.find_tvl_files

    def fake_find(directory, recursive=True):
        calls["finds"].append((str(directory), recursive))
        return real_find(directory, recursive=recursive)

    def fake_validate(files, *, strict=False, verbose=False):
        calls["files"] = [str(f) for f in files]
        calls["kwargs"] = (strict, verbose)
        failed = sum(1 for f in files if "fail" in f.name)
        return len(files) - failed, failed

    monkeypatch.setattr(tvl_main, "find_tvl_files", fake_find)
    monkeypatch.setattr(tvl_main, "validate_tvl_files", fake_validate)
    return calls


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "d" / "sub").mkdir(parents=True)
    (tmp_path / "d" / "a.tvl.yml").write_text("x: 1\n")
    (tmp_path / "d" / "sub" / "b.tvl.yaml").write_text("x: 1\n")
    (tmp_path / "d" / "c.txt").write_text("x\n")
    (tmp_path / "f.tvl.yml").write_text("x: 1\n")
    (tmp_path / "fail.tvl.yml").write_text("x: 1\n")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["prog", *args])
    return main()


def test_directory_recursive_by_default_and_files_kept_in_arg_order(
    tree, seen, monkeypatch, capsys
):
    assert _run(monkeypatch, "f.tvl.yml", "d", "missing.tvl.yml") == 0
    assert seen["files"] == [
        "f.tvl.yml",
        str(Path("d") / "a.tvl.yml"),
        str(Path("d") / "sub" / "b.tvl.yaml"),
        "missing.tvl.yml",
    ]
    assert seen["finds"] == [("d", True)]
    assert seen["kwargs"] == (False, False)
    assert capsys.readouterr().out == ""


def test_no_recursive_flag(tree, seen, monkeypatch):
    assert _run(monkeypatch, "d", "--no-recursive", "--strict", "-v") == 0
    assert seen["files"] == [str(Path("d") / "a.tvl.yml")]
    assert seen["finds"] == [("d", False)]
    assert seen["kwargs"] == (True, True)


def test_no_args_searches_cwd(tree, seen, monkeypatch):
    _run(monkeypatch)
    assert seen["finds"][0][0] == str(Path.cwd())
    assert seen["finds"][0][1] is True


def test_summary_printed_only_when_verbose_or_failures(tree, seen, monkeypatch, capsys):
    assert _run(monkeypatch, "f.tvl.yml") == 0
    assert capsys.readouterr().out == ""
    assert _run(monkeypatch, "f.tvl.yml", "-v") == 0
    assert capsys.readouterr().out == "\nValidation complete: 1/1 files passed\n"
    assert _run(monkeypatch, "f.tvl.yml", "fail.tvl.yml") == 1
    assert capsys.readouterr().out == "\nValidation complete: 1/2 files passed\n"


def test_empty_set_errors_to_stderr_naming_searched_paths(
    tree, seen, monkeypatch, capsys
):
    (tree / "empty").mkdir()
    assert _run(monkeypatch, "empty", "empty") == 1
    cap = capsys.readouterr()
    assert cap.out == ""
    assert cap.err == (
        "ERROR: no TVL spec files (.tvl.yml, .tvl.yaml) found under: "
        "empty, empty. Pass --allow-empty if an empty set is expected.\n"
    )
    assert seen["files"] is None


def test_allow_empty_message_only_when_verbose(tree, seen, monkeypatch, capsys):
    (tree / "empty").mkdir()
    assert _run(monkeypatch, "empty", "--allow-empty") == 0
    assert capsys.readouterr().out == ""
    assert _run(monkeypatch, "empty", "--allow-empty", "-v") == 0
    assert capsys.readouterr().out == "No TVL files found to validate.\n"
    assert seen["files"] is None


def test_no_args_empty_cwd_message_names_cwd(tree, seen, monkeypatch, capsys):
    empty = tree / "empty_cwd"
    empty.mkdir()
    monkeypatch.chdir(empty)
    assert _run(monkeypatch) == 1
    assert capsys.readouterr().err.endswith(
        f"found under: {Path.cwd()}. Pass --allow-empty if an empty set is expected.\n"
    )


def test_is_dir_probed_in_argument_order(tree, seen, monkeypatch):
    calls = []
    real = Path.is_dir

    def spy(self):
        calls.append(str(self))
        return real(self)

    monkeypatch.setattr(Path, "is_dir", spy)
    _run(monkeypatch, "f.tvl.yml", "d")
    assert calls[:2] == ["f.tvl.yml", "d"]
