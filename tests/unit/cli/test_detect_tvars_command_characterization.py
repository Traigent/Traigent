"""Golden characterization for the detect-tvars command (path/filter handling)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from traigent.cli.detect_tvars_command import detect_tvars

SRC = """
def answer(q):
    temperature = 0.7
    model = "gpt-4o-mini"
    max_tokens = 256
    return call_llm(q, temperature=temperature, model=model, max_tokens=max_tokens)
"""


def _run(*args):
    return CliRunner().invoke(detect_tvars, list(args))


def _entries(res):
    return json.loads(res.stdout)


def test_single_file_json_and_confidence_filter(tmp_path: Path):
    f = tmp_path / "a.py"
    f.write_text(SRC)
    low = _run(str(f), "--json")
    assert low.exit_code == 0
    entries = _entries(low)
    assert entries, "expected candidates"
    assert {e["function"] for e in entries} == {"answer"}
    high = _run(str(f), "--json", "--min-confidence", "high")
    medium = _run(str(f), "--json", "--min-confidence", "MEDIUM")
    assert len(_entries(high)) <= len(_entries(medium)) <= len(entries)
    assert all(e["confidence"] == "high" for e in _entries(high))
    assert all(e["confidence"] in ("high", "medium") for e in _entries(medium))
    assert [e["name"] for e in _entries(high)] == [
        e["name"] for e in entries if e["confidence"] == "high"
    ]


def test_non_py_file_rejected(tmp_path: Path):
    f = tmp_path / "a.txt"
    f.write_text(SRC)
    res = _run(str(f))
    assert res.exit_code != 0
    assert "Expected a .py file" in res.output


def test_single_file_function_not_found(tmp_path: Path):
    f = tmp_path / "a.py"
    f.write_text(SRC)
    res = _run(str(f), "--function", "nope")
    assert res.exit_code != 0
    assert "Function 'nope' not found in source" in res.output


def test_single_file_syntax_error_is_fatal(tmp_path: Path):
    f = tmp_path / "bad.py"
    f.write_text("def (:\n")
    res = _run(str(f))
    assert res.exit_code != 0
    assert "syntax error at line 1" in res.output


def test_directory_without_python_files_exits_zero(tmp_path: Path):
    res = _run(str(tmp_path))
    assert res.exit_code == 0
    assert "No Python files found" in res.output


def test_directory_skips_hidden_private_and_vendor_components(tmp_path: Path):
    for sub in (".hid", "_priv", "venv", "node_modules", "__pycache__", "ok"):
        d = tmp_path / sub
        d.mkdir()
        (d / "m.py").write_text(SRC)
    (tmp_path / "top.py").write_text(SRC.replace("answer", "top_fn"))
    (tmp_path / "_skipme.py").write_text(SRC.replace("answer", "skipme"))
    res = _run(str(tmp_path), "--json")
    assert res.exit_code == 0
    funcs = {e["function"] for e in _entries(res)}
    assert funcs == {"answer", "top_fn"}


def test_directory_target_itself_may_have_dotted_components(tmp_path: Path):
    d = tmp_path / ".work" / "agent"
    d.mkdir(parents=True)
    (d / "m.py").write_text(SRC)
    res = _run(str(d), "--json")
    assert res.exit_code == 0
    assert _entries(res)


def test_directory_all_unparseable_is_fatal(tmp_path: Path):
    (tmp_path / "bad.py").write_text("def (:\n")
    res = _run(str(tmp_path))
    assert res.exit_code != 0
    assert "Could not parse any Python file in the target directory" in res.output


def test_directory_bad_file_skipped_when_another_parses(tmp_path: Path):
    (tmp_path / "bad.py").write_text("def (:\n")
    (tmp_path / "good.py").write_text(SRC)
    res = _run(str(tmp_path), "--json")
    assert res.exit_code == 0
    assert {e["function"] for e in _entries(res)} == {"answer"}


def test_directory_function_not_found_anywhere(tmp_path: Path):
    (tmp_path / "good.py").write_text(SRC)
    res = _run(str(tmp_path), "-f", "nope")
    assert res.exit_code != 0
    assert "Function 'nope' not found in source" in res.output


def test_directory_function_found_in_one_file(tmp_path: Path):
    (tmp_path / "good.py").write_text(SRC)
    (tmp_path / "other.py").write_text(SRC.replace("answer", "other_fn"))
    res = _run(str(tmp_path), "-f", "other_fn", "--json")
    assert res.exit_code == 0
    assert {e["function"] for e in _entries(res)} == {"other_fn"}


def test_directory_only_hidden_files_scanned_zero_is_ok(tmp_path: Path):
    d = tmp_path / ".hid"
    d.mkdir()
    (d / "m.py").write_text(SRC)
    res = _run(str(tmp_path), "--json")
    assert res.exit_code == 0
    assert _entries(res) == []


@pytest.mark.parametrize("level", ["high", "medium", "low"])
def test_filter_never_returns_function_without_candidates(tmp_path: Path, level):
    (tmp_path / "a.py").write_text(SRC + "\ndef empty():\n    return 1\n")
    res = _run(str(tmp_path), "--json", "--min-confidence", level)
    assert "empty" not in {e["function"] for e in _entries(res)}
