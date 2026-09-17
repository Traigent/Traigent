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
3. Asserts every *core* dependency that carries a floor in
   ``[project.dependencies]`` is enforced by ``requirements/requirements.txt``
   -- present, carrying a floor, and not gated behind an environment marker.
   A bare name is not protection, and neither is a conditional floor.

   Known conservative gap: a package floored only via an ``-r other.txt``
   include is reported as absent. That is a false positive, never a false
   clean, and the core mirror is not supposed to delegate its own floors.
4. Exits non-zero with a readable diff if any drift is detected.

Check 3 exists because checks 1-2 iterate the requirements file and look each
name up in pyproject, so a package that is in pyproject but **absent** from the
mirror was invisible to them -- the guard could not fail for the one case that
matters most, a newly-declared security floor. Found on PR #2210, where
``yarl``/``filelock`` were added to ``pyproject.toml`` and this script stayed
green; ``anyio``, ``requests``, ``PyJWT`` and ``pydantic`` turned out to have
been missing from the mirror for far longer. Review round 2 of that PR then
caught the first version of check 3 accepting a bare name as protection --
reproducing the very defect it was written to catch.

Scope note: check 3 covers ``[project.dependencies]``. Check 4 (below) extends the
same "presence alone is not protection" logic to optional extras, but only for the
extras that have a matching ``requirements-<extra>.txt`` mirror file -- extras with
no mirror file are not this script's job to invent (issue #2211); see
``requirements/README.md`` for which extras are mirrored today.

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


def _collect_core_floors() -> dict[str, str]:
    """Floors declared in ``[project.dependencies]`` only (not extras)."""
    data = tomllib.loads(PYPROJECT_PATH.read_text())
    deps = data.get("project", {}).get("dependencies") or []
    floors: dict[str, str] = {}
    for spec in deps:
        if not isinstance(spec, str):
            continue
        pair = _extract_floor(spec)
        if pair is not None:
            floors[pair[0]] = pair[1]
    return floors


def _find_unprotected_floors(
    floors: dict[str, str],
    requirements_path: Path,
    expected_markers: dict[str, str] | None = None,
) -> list[tuple[str, str, str]]:
    """Floors that ``requirements_path`` does not actually enforce.

    Shared by the core check and the per-extra check (issue #2211): given a set
    of ``{name: floor}`` pairs a spec file is expected to protect, and the
    mirror file meant to protect them, return ``(name, floor, reason)`` for
    every one it does not. Three ways a floor goes unenforced, and
    **presence alone is not protection**:

    * ``absent``      -- the package is not in the mirror at all.
    * ``unpinned``    -- listed with no floor (``yarl``).
    * ``conditional`` -- floored only behind an environment marker the
      declaring source did not itself use (``yarl>=1.24.5,<2; python_version <
      "3.11"`` in the mirror, unconditional in the declaring source). A
      marker-gated mirror line protects only the environments the marker
      admits -- and one no supported interpreter satisfies protects nothing at
      all while still looking pinned. ``expected_markers`` (optional,
      ``{name: marker text}``) lets a caller whose *own* declaration is
      already marker-gated (e.g. ``faiss-cpu`` is Linux/macOS-only in
      pyproject's ``integrations`` extra) accept a mirror line carrying that
      identical marker as protecting exactly what the declaration promised.

    Both leave ``pip install -r <requirements_path>`` free to resolve a
    known-vulnerable version. An earlier version of this function accepted a
    bare name as "present", which reproduced the exact defect it was written to
    catch; see PR #2210 review round 2.

    A package that carries no floor in ``floors`` is not reported: there is no
    floor to enforce.
    """
    if not floors:
        return []
    if not requirements_path.exists():
        # Absent input is a finding, never a silent pass.
        return [(name, version, "absent") for name, version in sorted(floors.items())]

    expected_markers = expected_markers or {}

    # A floor only protects unconditionally when its line carries no
    # environment marker (or exactly the marker the declaration itself used).
    # Collect the three states separately rather than reusing
    # _collect_requirements_floors(), which is marker-blind because its own
    # job (comparing floor values) does not depend on applicability.
    unconditional: set[str] = set()
    conditional: set[str] = set()
    listed: set[str] = set()
    for raw_line in requirements_path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        spec, _, marker = line.partition(";")
        name_match = re.match(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]*", spec.strip())
        if not name_match:
            continue
        name = _normalize_name(name_match.group(0))
        listed.add(name)
        if _extract_floor(spec.strip()) is None:
            continue
        marker = marker.strip()
        if not marker or marker == expected_markers.get(name):
            unconditional.add(name)
        else:
            conditional.add(name)

    findings: list[tuple[str, str, str]] = []
    for name, version in sorted(floors.items()):
        if name in unconditional:
            continue
        if name in conditional:
            reason = "conditional"
        elif name in listed:
            reason = "unpinned"
        else:
            reason = "absent"
        findings.append((name, version, reason))
    return findings


def _find_unprotected_core_floors() -> list[tuple[str, str, str]]:
    """Core floors that ``requirements/requirements.txt`` does not actually enforce.

    See ``_find_unprotected_floors`` for the three unenforced reasons returned.
    """
    return _find_unprotected_floors(
        _collect_core_floors(), REQUIREMENTS_DIR / "requirements.txt"
    )


def _collect_requirements_names(path: Path) -> set[str]:
    """Every package name a requirements file lists, floored or bare."""
    if not path.exists():
        return set()
    names: set[str] = set()
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        spec, _, _marker = line.partition(";")
        name_match = re.match(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]*", spec.strip())
        if name_match:
            names.add(_normalize_name(name_match.group(0)))
    return names


def _collect_extra_floors(extra: str) -> tuple[dict[str, str], dict[str, str]]:
    """Floors (and their own environment markers) for one optional-dependencies extra.

    Excludes anything already FLOORED in ``requirements/requirements.txt`` --
    every extra mirror file starts with ``-r requirements.txt``, so a package
    floored there is core's job to enforce (check 3), not this extra's.

    The floor is what earns the exemption, not mere presence. Exempting a BARE
    core name opened a hole: a package declared only under extras (``mcp`` is
    declared under ``hybrid``/``mcp``) could have its core pin replaced by a
    bare name and escape every check -- check 3 never sees it, because it is
    not in pyproject's core dependencies, and this check skipped it because
    the name appeared in the core mirror. Requiring a real floor closes that. Matches the drift issue's own repro (#2211), which likewise
    treats presence in the core mirror as "not this extra's problem" -- e.g.
    ``claude-code-sdk``/``mcp`` are declared under the ``hybrid`` extra in
    pyproject.toml but already floored directly in ``requirements.txt``
    (tracked separately as over-inclusion, not drift).

    Returns ``(floors, markers)``: ``markers[name]`` is the exact marker text
    pyproject itself used for that dependency (``""`` if unconditional), e.g.
    ``faiss-cpu`` is declared ``; sys_platform != 'win32'`` in the
    ``integrations`` extra -- a mirror line carrying that same marker protects
    exactly what pyproject promised and should not be flagged ``conditional``.
    """
    data = tomllib.loads(PYPROJECT_PATH.read_text())
    deps = (data.get("project", {}).get("optional-dependencies") or {}).get(extra, [])
    core_mirror = REQUIREMENTS_DIR / "requirements.txt"
    already_mirrored = (
        set(_collect_requirements_floors(core_mirror))
        if core_mirror.exists()
        else set()
    )
    floors: dict[str, str] = {}
    markers: dict[str, str] = {}
    for spec in deps:
        if not isinstance(spec, str):
            continue
        pair = _extract_floor(spec)
        if pair is None or pair[0] in already_mirrored:
            continue
        name, version = pair
        floors[name] = version
        markers[name] = spec.partition(";")[2].strip()
    return floors, markers


def _find_unprotected_extra_floors() -> dict[str, list[tuple[str, str, str]]]:
    """Extra floors that their ``requirements-<extra>.txt`` mirror does not enforce.

    Only extras that already have a matching ``requirements-<extra>.txt`` file
    are checked -- an extra with no mirror file has nothing this script can
    compare against (issue #2211; `requirements/README.md` documents which
    extras are mirrored). Returns ``{extra: [(name, floor, reason), ...]}`` for
    every extra with at least one unenforced floor.
    """
    data = tomllib.loads(PYPROJECT_PATH.read_text())
    extras = sorted(data.get("project", {}).get("optional-dependencies") or {})

    results: dict[str, list[tuple[str, str, str]]] = {}
    for extra in extras:
        req_path = REQUIREMENTS_DIR / f"requirements-{extra}.txt"
        if not req_path.exists():
            continue
        floors, markers = _collect_extra_floors(extra)
        findings = _find_unprotected_floors(floors, req_path, expected_markers=markers)
        if findings:
            results[extra] = findings
    return results


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

    unprotected_core = _find_unprotected_core_floors()
    unprotected_extras = _find_unprotected_extra_floors()

    if not drifts and not unprotected_core and not unprotected_extras:
        print("OK: no dependency floor drift between pyproject.toml and requirements/")
        return 0

    reason_detail = {
        "absent": "absent from the mirror",
        "unpinned": "listed in the mirror with no floor",
        "conditional": (
            "floored only behind an environment marker, so the floor "
            "does not apply to every supported install"
        ),
    }

    if unprotected_core:
        print(
            "❌ Core dependency floors not enforced by requirements/requirements.txt:\n",
            file=sys.stderr,
        )
        for name, version, reason in unprotected_core:
            print(
                f"  {name}: pyproject.toml=>={version}, {reason_detail[reason]}",
                file=sys.stderr,
            )
        print(
            "\n`pip install -r requirements/requirements.txt` applies no floor for these, so a "
            "resolver can still pick a known-vulnerable version. Add each to the core "
            "requirements file with the same specifier. Listing the bare name is not enough.\n",
            file=sys.stderr,
        )

    if unprotected_extras:
        print(
            "❌ Extra dependency floors not enforced by their requirements-<extra>.txt mirror:\n",
            file=sys.stderr,
        )
        for extra in sorted(unprotected_extras):
            for name, version, reason in unprotected_extras[extra]:
                print(
                    f"  requirements-{extra}.txt: {name}: pyproject.toml=>={version}, "
                    f"{reason_detail[reason]}",
                    file=sys.stderr,
                )
        print(
            "\n`pip install -r requirements/requirements-<extra>.txt` applies no floor for these. "
            "Add each to its extra's requirements file with the same specifier. Listing the bare "
            "name is not enough. (Extras with no requirements-<extra>.txt mirror are not checked "
            "here -- see requirements/README.md.)\n",
            file=sys.stderr,
        )

    if not drifts:
        return 1

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
