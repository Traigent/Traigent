"""Guard (#2981): shipped configs and examples must not name retired models.

Scans every string literal in tracked config/example text files (non-comment
lines) with ``is_retired_model``, so a denylist addition cannot leave a shipped
file telling users to try a model the SDK now rejects.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from traigent.config.retired_models import is_retired_model

REPO_ROOT = Path(__file__).resolve().parents[3]
SCAN_DIRS = ("configs", "examples", "traigent/examples")
SCAN_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".csv", ".tvl", ".toml"}
#: Recorded historical measurements, not guidance to users.
EXCLUDED_NAMES = {"raw_results.json"}
_LITERAL = re.compile(
    r"""["']([A-Za-z0-9_./:@-]{3,80})["']|\b(claude-[A-Za-z0-9._-]+)"""
)
#: Unquoted YAML scalars under model-like keys (``model: x``, ``- x`` lists).
_YAML_MODEL_KEY = re.compile(
    r"^\s*(?:-\s+)?(?:model|model_name|models)\s*:\s*([A-Za-z0-9_./:@-]{3,80})\s*(?:#.*)?$"
)
_YAML_LIST_ITEM = re.compile(r"^\s*-\s+([A-Za-z0-9_./:@-]{3,80})\s*(?:#.*)?$")
_YAML_MODEL_LIST_START = re.compile(r"^\s*(?:model|model_name|models)\s*:\s*(?:#.*)?$")
_COMMENT = re.compile(r"^\s*(#|//)")


def _tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", *SCAN_DIRS],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [
        REPO_ROOT / name
        for name in out.split("\0")
        if name
        and Path(name).suffix in SCAN_SUFFIXES
        and Path(name).name not in EXCLUDED_NAMES
    ]


def _retired_literals(path: Path) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    # An unreadable tracked file must fail the guard, not be skipped.
    lines = path.read_text(encoding="utf-8").splitlines()
    in_model_list = False
    for lineno, line in enumerate(lines, start=1):
        if _COMMENT.match(line):
            continue
        if path.suffix in {".yaml", ".yml", ".tvl"}:
            unquoted = _YAML_MODEL_KEY.match(line)
            if unquoted is None and in_model_list:
                unquoted = _YAML_LIST_ITEM.match(line)
            if unquoted and is_retired_model(unquoted.group(1)):
                hits.append((lineno, unquoted.group(1)))
            if _YAML_MODEL_LIST_START.match(line):
                in_model_list = True
            elif not _YAML_LIST_ITEM.match(line):
                in_model_list = False
        for match in _LITERAL.finditer(line):
            literal = match.group(1) or match.group(2)
            if literal and is_retired_model(literal) and (lineno, literal) not in hits:
                hits.append((lineno, literal))
    return hits


def test_shipped_configs_and_examples_name_no_retired_models() -> None:
    offenders = {
        str(path.relative_to(REPO_ROOT)): hits
        for path in _tracked_files()
        if (hits := _retired_literals(path))
    }
    assert not offenders, offenders


def test_scanner_flags_retired_ids_and_skips_comments(tmp_path: Path) -> None:
    sample = tmp_path / "x.yaml"
    sample.write_text(
        '# model: "claude-3-5-haiku-latest"\n'
        'model: "claude-3-5-haiku-latest"\n'
        'ok: "claude-sonnet-4-6"\n'
    )
    assert _retired_literals(sample) == [(2, "claude-3-5-haiku-latest")]


def test_scanner_flags_unquoted_yaml_model_values(tmp_path: Path) -> None:
    sample = tmp_path / "x.yaml"
    sample.write_text("model: claude-3-5-haiku-latest\nmodel_name: claude-sonnet-4-6\n")
    assert _retired_literals(sample) == [(1, "claude-3-5-haiku-latest")]
    listed = tmp_path / "y.yaml"
    listed.write_text("models:\n  - claude-3-5-haiku-latest\n  - claude-sonnet-4-6\n")
    assert _retired_literals(listed) == [(2, "claude-3-5-haiku-latest")]


def test_scanner_fails_on_unreadable_file(tmp_path: Path) -> None:
    import pytest

    bad = tmp_path / "bad.yaml"
    bad.write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(UnicodeDecodeError):
        _retired_literals(bad)
