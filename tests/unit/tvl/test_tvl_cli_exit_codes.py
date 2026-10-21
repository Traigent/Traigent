"""Exit codes of ``python -m traigent.tvl`` for missing and empty input (#2414).

The CLI is documented as a pre-commit / CI gate, so it must fail when the
input it was pointed at does not exist or holds no spec files; otherwise a
renamed or never-checked-out spec directory makes the gate pass having
validated nothing.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from traigent.tvl.__main__ import main

_REPO_ROOT = Path(__file__).resolve().parents[3]
_HELLO_SPEC = _REPO_ROOT / "examples" / "tvl" / "hello_tvl" / "hello_tvl.tvl.yml"


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "good").mkdir()
    shutil.copy(_HELLO_SPEC, tmp_path / "good" / "hello_tvl.tvl.yml")
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "bad.tvl.yml").write_text("key: [unclosed\n")
    (tmp_path / "tvl_empty").mkdir()
    (tmp_path / "tvl_misnamed").mkdir()
    (tmp_path / "tvl_misnamed" / "promotion-gate.tvl").write_text("x: 1\n")
    (tmp_path / "empty_cwd").mkdir()
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _run(monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    monkeypatch.setattr(sys, "argv", ["python -m traigent.tvl", *args])
    return main()


@pytest.mark.parametrize(
    "args",
    [
        ("tvl_missing/", "--strict"),
        ("tvl_missing/promotion-gate.tvl.yml", "--strict"),
        ("tvl_missing.tvl.yml", "good/hello_tvl.tvl.yml", "--strict"),
        ("tvl_missing/", "--allow-empty"),
    ],
    ids=["missing-dir", "missing-file", "missing-plus-valid", "missing-allow-empty"],
)
def test_missing_path_fails(workspace, monkeypatch, capsys, args):
    assert _run(monkeypatch, *args) == 1
    assert "File not found" in capsys.readouterr().err


@pytest.mark.parametrize(
    "args",
    [("tvl_empty/", "--strict"), ("tvl_misnamed/", "--strict")],
    ids=["empty-dir", "only-tvl-suffix"],
)
def test_empty_discovered_set_fails(workspace, monkeypatch, capsys, args):
    assert _run(monkeypatch, *args) == 1
    err = capsys.readouterr().err
    assert "no TVL spec files (.tvl.yml, .tvl.yaml) found under" in err
    assert args[0].rstrip("/") in err
    assert "--allow-empty" in err


def test_no_args_in_cwd_without_specs_fails(workspace, monkeypatch, capsys):
    monkeypatch.chdir(workspace / "empty_cwd")
    assert _run(monkeypatch, "--strict") == 1
    assert str(workspace / "empty_cwd") in capsys.readouterr().err


def test_allow_empty_restores_exit_zero(workspace, monkeypatch):
    assert _run(monkeypatch, "tvl_empty/", "--strict", "--allow-empty") == 0
    monkeypatch.chdir(workspace / "empty_cwd")
    assert _run(monkeypatch, "--strict", "--allow-empty") == 0


def test_valid_and_invalid_controls_unchanged(workspace, monkeypatch):
    assert _run(monkeypatch, "good/", "--strict") == 0
    assert _run(monkeypatch, "bad/", "--strict") == 1


_MISSPELT = """\
tvars:
  - name: temperature
    type: float
    domain:
      range: [0.0, 1.0]
objectives:
  - name: accuracy
    direction: maximize
contraints:
  - rule: "temperature < 0.5"
"""

_INVERTED_RANGE = _MISSPELT.replace("contraints", "constraints").replace(
    "[0.0, 1.0]", "[1.0, 0.0]"
)


@pytest.mark.parametrize("body", [_MISSPELT, _INVERTED_RANGE], ids=["typo", "range"])
@pytest.mark.parametrize("strict", [False, True], ids=["default", "strict"])
def test_loader_rejected_spec_fails_in_every_mode(
    workspace, monkeypatch, capsys, body, strict
):
    """A spec load_tvl_spec rejects must not exit 0 without --strict (#2678)."""
    (workspace / "bad_schema.tvl.yml").write_text(body)
    args = ("bad_schema.tvl.yml", *(("--strict",) if strict else ()))
    assert _run(monkeypatch, *args) == 1
    err = capsys.readouterr().err
    assert "FAILED: bad_schema.tvl.yml" in err
    assert "Unexpected error" not in err
