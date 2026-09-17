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

import ast
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

# Base paths that always make up Traigent's public surface, even when
# traigent/__init__.py cannot be parsed (e.g. a throwaway fixture repo with
# no real export table): traigent/__init__.py is the package's re-export
# surface itself; traigent/api/ is where the decorator, config-space and
# constraints DSL public classes live.
PUBLIC_API_PATHS: tuple[str, ...] = ("traigent/__init__.py", "traigent/api/")


def _run(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout


def _resolve_module_path(root: Path, ref: str, module_name: str) -> str | None:
    """Return the diff pathspec for `module_name` at `ref`: the flat module
    file (`traigent/economics.py`) or, if `module_name` is a package, the
    whole package directory (`traigent/admin/`) so a change anywhere inside
    it counts -- not just to its `__init__.py`. None if neither exists at
    `ref` (e.g. the module has since been renamed or removed)."""
    base = module_name.replace(".", "/")
    for candidate, pathspec in (
        (f"{base}.py", f"{base}.py"),
        (f"{base}/__init__.py", f"{base}/"),
    ):
        try:
            _run(root, "cat-file", "-e", f"{ref}:{candidate}")
        except subprocess.CalledProcessError:
            continue
        return pathspec
    return None


def derive_public_api_paths(
    root: Path, ref: str = "HEAD", base: tuple[str, ...] = PUBLIC_API_PATHS
) -> tuple[str, ...]:
    """Return `base` plus the file path of every module that backs a name
    exported through `traigent/__init__.py`'s `__all__` at `ref` -- via its
    `_LAZY_EXPORTS` table or a module-level `from traigent.x import Name`.

    Most of Traigent's actual root-exported public surface (`ExecutionBudget`,
    `Dataset`, `ObservationDTO`, `ScoreRecordDTO`, ...) is *defined* outside
    `traigent/__init__.py` and `traigent/api/`; a change to one of those
    defining modules changes the public API without touching either base
    path (#2290 review finding). This statically parses `__init__.py`'s
    source at `ref` (no import -- the diff may span refs whose code cannot
    be safely executed) and falls back to `base` alone when the file is
    missing, unparseable, or (as in this script's own test fixtures) does
    not define an `__all__`/`_LAZY_EXPORTS` at all -- so the check never
    drops the two paths it has always protected.
    """
    try:
        source = _run(root, "show", f"{ref}:traigent/__init__.py")
    except subprocess.CalledProcessError:
        return base

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return base

    all_names: set[str] = set()
    name_to_module: dict[str, str] = {}

    def _assign_targets(node: ast.AST) -> tuple[ast.expr, ...]:
        # `__all__ = [...]` is a plain Assign (possibly multiple targets);
        # `_LAZY_EXPORTS: dict[str, tuple[str, str]] = {...}` is an
        # AnnAssign (exactly one target). Both need handling.
        if isinstance(node, ast.Assign):
            return tuple(node.targets)
        if isinstance(node, ast.AnnAssign):
            return (node.target,)
        return ()

    for node in ast.walk(tree):
        value = getattr(node, "value", None)
        for target in _assign_targets(node):
            if not isinstance(target, ast.Name):
                continue
            if target.id == "__all__" and isinstance(value, ast.List):
                all_names.update(
                    elt.value
                    for elt in value.elts
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                )
            elif target.id == "_LAZY_EXPORTS" and isinstance(value, ast.Dict):
                for key, val in zip(value.keys, value.values, strict=True):
                    if (
                        isinstance(key, ast.Constant)
                        and isinstance(key.value, str)
                        and isinstance(val, ast.Tuple)
                        and val.elts
                        and isinstance(val.elts[0], ast.Constant)
                        and isinstance(val.elts[0].value, str)
                    ):
                        name_to_module[key.value] = val.elts[0].value
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                name_to_module.setdefault(alias.asname or alias.name, node.module)

    modules = {
        module
        for name, module in name_to_module.items()
        if name in all_names and module.startswith("traigent.")
    }
    derived = {
        resolved
        for module in modules
        if (resolved := _resolve_module_path(root, ref, module)) is not None
    }
    return tuple(sorted(set(base) | derived))


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

    public_api_paths = derive_public_api_paths(root, "HEAD")
    changed = public_api_changed_since(root, tag, paths=public_api_paths)
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
