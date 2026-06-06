"""Packaging guard for the Chroma integration dependency scope."""

from __future__ import annotations

import tomllib
from pathlib import Path


def test_chroma_is_explicit_opt_in_extra_only() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text())
    extras = pyproject["project"]["optional-dependencies"]

    assert "chroma" in extras
    assert any(dep.startswith("langchain-chroma") for dep in extras["chroma"])

    broad_extras = {"integrations", "recommended", "all", "enterprise"}
    for extra in broad_extras:
        assert extra in extras
        assert not any("chroma" in dep.lower() for dep in extras[extra])
