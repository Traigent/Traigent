"""Regression tests for the bring-your-own DSPy packaging boundary."""

from __future__ import annotations

import tomllib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DSPY_CHAIN = {"diskcache", "dspy", "dspy-ai"}


def test_root_package_does_not_publish_dspy_extra() -> None:
    """The public Traigent wheel must not offer the retired DSPy extra."""
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())

    assert "dspy" not in pyproject["project"]["optional-dependencies"]


def test_shipped_lock_excludes_dspy_dependency_chain() -> None:
    """Lock regeneration must not retain DSPy or vulnerable diskcache transitively."""
    lock = tomllib.loads((_REPO_ROOT / "uv.lock").read_text())
    locked_names = {package["name"] for package in lock["package"]}

    assert locked_names.isdisjoint(_DSPY_CHAIN), (
        "Traigent's shipped lock still contains retired DSPy-chain packages: "
        f"{sorted(locked_names & _DSPY_CHAIN)}"
    )


def test_development_install_does_not_request_retired_dspy_extra() -> None:
    """Development bootstrap surfaces must use only public root extras."""
    bootstrap_surfaces = (
        _REPO_ROOT / "Makefile",
        _REPO_ROOT / "docs" / "contributing" / "README.md",
    )

    for path in bootstrap_surfaces:
        contents = path.read_text()
        assert "[all,dev,dspy,docs]" not in contents, path
        assert "[all,dev,docs]" in contents, path
