#!/usr/bin/env python3
"""Guard against dependency floor drift between pyproject.toml and requirements/*.txt.

Motivation: three separate Greptile P1/security findings in 24h on PRs #730,
#731, #732 traced back to the same class of bug — a CVE-fixed dependency
floor was raised in ``pyproject.toml`` but the matching ``requirements/*.txt``
file was left behind. That leaves ``pip install -r requirements.txt`` (Docker
layers, manual installs, some CI paths) able to resolve a known-vulnerable
version even though application code was safe.

This script is the generic guard. It:

1. Reads the minimum version of every package declared in ``pyproject.toml``
   (core dependencies and every optional extra).
2. For each ``requirements/*.txt`` file, parses ``==``, ``>=``, ``~=`` pinned
   packages and asserts their floor is ``>=`` the pyproject floor.
3. Exits non-zero with a readable diff if any drift is detected.

Intentionally does NOT attempt to solve lockfile sync (that is ``uv``'s job).
The scope is spec-file floors only.

Run directly:

    python scripts/ci/check_dep_floor_drift.py

Wired into pre-commit + the release-gate lint_type job.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
REQUIREMENTS_DIR = REPO_ROOT / "requirements"

# Matches either "package>=1.2.3" or "package==1.2.3" etc.
# Captures name, operator, and the minimum-version token.
_SPEC_RE = re.compile(
    r"""
    ^\s*
    (?P<name>[A-Za-z0-9_][A-Za-z0-9_.\-]*)     # package name
    \s*
    (?:\[[^\]]+\])?                             # optional extras block
    \s*
    (?P<op>==|>=|~=)                            # operator we accept as a floor
    \s*
    (?P<version>[0-9][0-9A-Za-z.+!\-]*)         # version string
    """,
    re.VERBOSE,
)


def _parse_version_tuple(version: str) -> tuple[int, ...]:
    """Return a sortable integer tuple for PEP 440-ish version strings.

    Keeps the comparison simple enough to be right for all realistic
    dependency floors: digits before the first non-dot/non-digit char are
    compared numerically, the rest are ignored. For anything more exotic
    (``dev``/``rc``/``post`` suffixes), prefer the numeric prefix — this
    errs on the side of "treat as older" which makes this check strictly
    more conservative, never less.
    """
    parts: list[int] = []
    for token in version.split("."):
        digits = re.match(r"^(\d+)", token)
        if not digits:
            break
        parts.append(int(digits.group(1)))
    return tuple(parts) if parts else (0,)


def _normalize_name(name: str) -> str:
    """PEP 503 normalized name for cross-file comparison."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _extract_floor(spec: str) -> tuple[str, str] | None:
    """Return (normalized_name, minimum_version) for a dependency spec, or None.

    Only considers specs that carry a hard floor (``>=``, ``==``, ``~=``).
    Loose specs (``"package"`` with no pin) contribute no floor to compare.
    """
    match = _SPEC_RE.match(spec)
    if not match:
        return None
    return _normalize_name(match.group("name")), match.group("version")


def _collect_pyproject_floors() -> dict[str, str]:
    """Walk every dependency list in pyproject.toml, return highest floor per package."""
    data = tomllib.loads(PYPROJECT_PATH.read_text())
    project = data.get("project", {})

    candidate_lists: list[list[str]] = []
    if isinstance(project.get("dependencies"), list):
        candidate_lists.append(project["dependencies"])
    for extra_deps in (project.get("optional-dependencies") or {}).values():
        if isinstance(extra_deps, list):
            candidate_lists.append(extra_deps)

    floors: dict[str, str] = {}
    for spec_list in candidate_lists:
        for spec in spec_list:
            if not isinstance(spec, str):
                continue
            pair = _extract_floor(spec)
            if pair is None:
                continue
            name, version = pair
            current = floors.get(name)
            if current is None or _parse_version_tuple(version) > _parse_version_tuple(
                current
            ):
                floors[name] = version
    return floors


def _collect_requirements_floors(path: Path) -> dict[str, str]:
    """Parse a requirements file, return the floor for each package it pins."""
    floors: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            # Skip options like -r base.txt, -e . etc.
            continue
        pair = _extract_floor(line)
        if pair is None:
            continue
        name, version = pair
        # Keep the highest floor if the same package is listed twice.
        current = floors.get(name)
        if current is None or _parse_version_tuple(version) > _parse_version_tuple(
            current
        ):
            floors[name] = version
    return floors


def main() -> int:
    pyproject_floors = _collect_pyproject_floors()

    drifts: list[tuple[Path, str, str, str]] = []
    for req_path in sorted(REQUIREMENTS_DIR.glob("*.txt")):
        req_floors = _collect_requirements_floors(req_path)
        for name, req_version in req_floors.items():
            py_version = pyproject_floors.get(name)
            if py_version is None:
                continue
            if _parse_version_tuple(req_version) < _parse_version_tuple(py_version):
                drifts.append((req_path, name, req_version, py_version))

    if not drifts:
        print("OK: no dependency floor drift between pyproject.toml and requirements/")
        return 0

    print("❌ Dependency floor drift detected:\n", file=sys.stderr)
    for req_path, name, req_version, py_version in drifts:
        rel = req_path.relative_to(REPO_ROOT)
        print(
            f"  {name}: {rel}=>={req_version} lags pyproject.toml=>={py_version}",
            file=sys.stderr,
        )
    print(
        "\nRaise the requirements/*.txt floors to match pyproject.toml. "
        "See CVE-triage PRs #720, #731, #732 for context.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
