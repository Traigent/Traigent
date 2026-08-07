"""Packaging guard for the Chroma integration dependency scope."""

from __future__ import annotations

import tomllib
from pathlib import Path


def test_chroma_extra_is_not_advertised_while_upstream_is_unpatched() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text())
    extras = pyproject["project"]["optional-dependencies"]

    assert "chroma" not in extras

    broad_extras = {"integrations", "recommended", "all", "enterprise"}
    for extra in broad_extras:
        assert extra in extras
        assert not any(
            dependency.lower().startswith(("langchain-chroma", "chromadb"))
            for dependency in extras[extra]
        )
