"""Pre-commit tool revs must match the versions uv.lock resolves (#2469).

Local hooks and CI used to disagree on isort/mypy/black versions, so the
verdict depended on which copy of a tool ran.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[3]

# pre-commit repo URL fragment -> uv.lock package name
_TOOLS = {
    "pycqa/isort": "isort",
    "pre-commit/mirrors-mypy": "mypy",
}


def _locked_versions() -> dict[str, str]:
    lock = tomllib.loads((_ROOT / "uv.lock").read_text())
    return {p["name"]: p["version"] for p in lock["package"]}


def _precommit_revs() -> dict[str, str]:
    cfg = yaml.safe_load((_ROOT / ".pre-commit-config.yaml").read_text())
    return {
        r["repo"].lower().split("github.com/")[-1]: r["rev"]
        for r in cfg["repos"]
        if "rev" in r
    }


@pytest.mark.parametrize(("repo", "package"), sorted(_TOOLS.items()))
def test_precommit_rev_matches_uv_lock(repo: str, package: str) -> None:
    rev = _precommit_revs()[repo]
    assert re.sub(r"^v", "", rev) == _locked_versions()[package]
