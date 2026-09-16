#!/usr/bin/env python3
"""Reject a declared version that still identifies as a published release
once the public API has moved on since that release's tag.

`traigent/_version.py` reads `pyproject.toml`'s `[project].version` as the
single source of truth for `traigent.__version__` in a development checkout.
When a release is tagged (`vX.Y.Z`) but `pyproject.toml` is never bumped
afterward, every later commit keeps identifying as that already-published
version even though its public API has changed -- a source install and the
PyPI wheel then report the same version while exposing different behavior.
That is exactly what happened for #2265: PR #2226 added
`EvaluationOptions.task_type` after `v0.27.0` was tagged, but
`pyproject.toml` still read `0.27.0`, so version guards (in this repo, in
skills, in customer code) could not tell the two apart.

This check enforces the fix directly: once a tag `v<version>` exists, no
later commit may still declare `version == <version>` if any file under a
public-API path (see PUBLIC_API_PATHS) differs from what that tag shipped.
The remedy is always the same -- bump `pyproject.toml` to the next dev
version (e.g. `0.28.0.dev0`) right after tagging a release, before merging
further public API changes.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

# Repo root as invoked, not the script's own location: the CLI is meant to
# run with cwd already at the repo root (that is how the pr-gate.yml and
# release-gate.yml workflow steps call it), and tests exercise it against
# throwaway repos via `cwd=`. Anchoring to `__file__` instead would silently
# check the real Traigent checkout even when invoked against a temp repo.
REPO_ROOT = Path.cwd()

# Paths that make up Traigent's public surface. Kept in sync by hand: a new
# top-level public re-export module belongs here too when it is added.
# traigent/__init__.py is the package's re-export surface; traigent/api/ is
# where the decorator, config-space and constraints DSL public classes live.
PUBLIC_API_PATHS: tuple[str, ...] = ("traigent/__init__.py", "traigent/api/")


def _run(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout


def read_current_version(pyproject_path: Path) -> str:
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    version = data.get("project", {}).get("version")
    if not isinstance(version, str) or not version:
        raise ValueError(f"no [project].version found in {pyproject_path}")
    return version


def matching_release_tag(root: Path, version: str) -> str | None:
    """Return f"v{version}" if that exact tag exists in `root`'s repo, else None."""
    tag = f"v{version}"
    existing = _run(root, "tag", "-l", tag).strip()
    return tag if existing == tag else None


def public_api_changed_since(
    root: Path, tag: str, paths: tuple[str, ...] = PUBLIC_API_PATHS
) -> str:
    """Return the (possibly empty) `git diff --name-only` output for `paths`
    between `tag` and HEAD."""
    return _run(root, "diff", "--name-only", f"{tag}..HEAD", "--", *paths).strip()


def check(root: Path) -> tuple[bool, str]:
    """Run the identity check. Returns (ok, message)."""
    version = read_current_version(root / "pyproject.toml")
    tag = matching_release_tag(root, version)
    if tag is None:
        return True, (
            f"pyproject.toml version {version!r} has no matching tag "
            f"'v{version}'; nothing to check."
        )

    head_sha = _run(root, "rev-parse", "HEAD").strip()
    tag_sha = _run(root, "rev-parse", tag).strip()
    if head_sha == tag_sha:
        return (
            True,
            f"HEAD is exactly {tag}; version {version!r} correctly identifies it.",
        )

    changed = public_api_changed_since(root, tag)
    if not changed:
        return True, (
            f"No public-API files changed since {tag}; version {version!r} is still accurate."
        )

    return False, (
        f"pyproject.toml still declares version {version!r}, matching published tag "
        f"{tag}, but the public API changed since that tag:\n{changed}\n\n"
        "Bump [project].version in pyproject.toml (e.g. to a '*.dev0' pre-release) "
        "before merging further public API changes on top of a released version. "
        "See issue #2265."
    )


def main() -> int:
    ok, message = check(REPO_ROOT)
    if ok:
        print(message)
        return 0
    print(f"::error::{message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
