"""#3183: ruff format is the one formatter; no workflow/tox/config runs Black."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
BLACK_CMD = re.compile(r"(^|[\s`;&|(])(python\d*\s+-m\s+)?black(\s|$|=)", re.I)


def _lines(path: Path):
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.lstrip().startswith("#"):
            yield i, line


def test_workflows_do_not_run_black():
    hits = [
        f"{p.relative_to(REPO)}:{i}: {line.strip()}"
        for p in sorted((REPO / ".github" / "workflows").glob("*.y*ml"))
        for i, line in _lines(p)
        if BLACK_CMD.search(line) or "pip install" in line and "black" in line.lower()
    ]
    assert not hits, hits


def test_workflows_run_ruff_format_check():
    for rel in (
        ".github/workflows/quality.yml",
        ".github/workflows/examples-smoke.yml",
    ):
        text = (REPO / rel).read_text(encoding="utf-8")
        assert "ruff format --check" in text, rel


def test_tox_and_makefile_do_not_use_black():
    for rel in ("scripts/config/tox.ini", "Makefile"):
        hits = [
            f"{rel}:{i}: {line.strip()}"
            for i, line in _lines(REPO / rel)
            if re.search(r"\bblack\b", line, re.I) and "--profile" not in line
        ]
        assert not hits, hits


def test_no_black_tool_config_or_stale_precommit():
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert "black" not in pyproject.get("tool", {})
    assert not (REPO / "scripts/config/.pre-commit-config.yaml").exists()


def test_contributor_docs_do_not_tell_users_to_run_black():
    for rel in (
        "docs/contributing/CONTRIBUTING.md",
        "docs/contributing/ADDING_NEW_INTEGRATIONS.md",
        "docs/getting-started/testing.md",
        "docs/LOCAL_CI_GATE.md",
    ):
        hits = [
            f"{rel}:{i}: {line.strip()}"
            for i, line in _lines(REPO / rel)
            if re.search(r"(^|\s|`)black(\s|`|$)", line, re.I) and "profile" not in line
        ]
        assert not hits, hits
