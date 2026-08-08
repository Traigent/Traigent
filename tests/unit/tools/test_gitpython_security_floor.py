"""Regression guards for the transitive GitPython security dependency.

GitPython is not a direct Traigent dependency: published optional extras can
reach it through multiple integrations, including ``wandb`` and
``mlflow-skinny``. Keep the resolved artifact at or above the version that
fixes GHSA-jm78-9fvv-mhgr and GHSA-wvpp-8hx9-p66j without changing the
published dependency surface. This floor covers the current Aikido GitPython
issue group: 471288061, 471288073, 471288052, 471288025, 471288041, and
471288082.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import cast

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version


_REPO_ROOT = Path(__file__).resolve().parents[3]
_LOCK_PATH = _REPO_ROOT / "uv.lock"
_PYPROJECT_PATH = _REPO_ROOT / "pyproject.toml"
_SECURITY_FLOOR = Version("3.1.58")


def _resolved_gitpython_version() -> Version:
    lock = tomllib.loads(_LOCK_PATH.read_text())
    matches = [
        package for package in lock["package"] if package["name"].lower() == "gitpython"
    ]
    assert len(matches) == 1, "uv.lock must contain exactly one GitPython package"
    return Version(matches[0]["version"])


def _direct_gitpython_locations(
    core_dependencies: list[str], optional_dependencies: dict[str, list[str]]
) -> list[str]:
    """Return published dependency surfaces that directly declare GitPython."""
    gitpython_name = canonicalize_name("GitPython")
    locations = []
    surfaces = [("project.dependencies", core_dependencies)]
    surfaces.extend(
        (f"project.optional-dependencies.{extra_name}", requirements)
        for extra_name, requirements in optional_dependencies.items()
    )
    for location, requirements in surfaces:
        if any(
            canonicalize_name(Requirement(requirement).name) == gitpython_name
            for requirement in requirements
        ):
            locations.append(location)
    return locations


def _published_dependency_specs() -> tuple[list[str], dict[str, list[str]]]:
    project = tomllib.loads(_PYPROJECT_PATH.read_text())["project"]
    return (
        cast(list[str], project["dependencies"]),
        cast(dict[str, list[str]], project["optional-dependencies"]),
    )


def test_gitpython_resolved_version_meets_security_floor() -> None:
    """The transitive dependency must resolve to the patched release."""
    resolved = _resolved_gitpython_version()
    assert resolved >= _SECURITY_FLOOR, (
        f"uv.lock resolves GitPython {resolved}, below the required "
        f"{_SECURITY_FLOOR} security floor for GHSA-jm78-9fvv-mhgr and "
        "GHSA-wvpp-8hx9-p66j. "
        "Refresh only the GitPython lock artifact; do not add it as a direct "
        "project dependency."
    )


def test_gitpython_is_not_a_direct_published_dependency() -> None:
    """GitPython remains transitive across core and every public extra."""
    core_dependencies, optional_dependencies = _published_dependency_specs()
    direct_locations = _direct_gitpython_locations(
        core_dependencies, optional_dependencies
    )
    assert not direct_locations, (
        "GitPython must remain transitive; remove its direct declaration from: "
        f"{', '.join(direct_locations)}"
    )


def test_direct_gitpython_requirement_probe_is_detected() -> None:
    """PEP 508 parsing detects direct specs with extras, markers, and URLs."""
    direct_locations = _direct_gitpython_locations(
        ["GitPython @ https://example.test/GitPython-3.1.58.whl"],
        {"integrations": ["GitPython[security]>=3.1.58; python_version >= '3.11'"]},
    )
    assert direct_locations == [
        "project.dependencies",
        "project.optional-dependencies.integrations",
    ]
