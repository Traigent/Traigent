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
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (UnicodeDecodeError, OSError):
        return hits
    for lineno, line in enumerate(lines, start=1):
        if _COMMENT.match(line):
            continue
        for match in _LITERAL.finditer(line):
            literal = match.group(1) or match.group(2)
            if literal and is_retired_model(literal):
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
