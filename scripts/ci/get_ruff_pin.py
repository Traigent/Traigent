#!/usr/bin/env python3
"""Print the exact ruff version pinned in pyproject.toml's ``dev`` extra.

Motivation (issue #1550): the SDK Required PR Gate's ``preflight`` job used
to run a bare ``pip install ruff``, which resolves to whatever ruff released
most recently. A new ruff release can add/strengthen lint rules and start
failing PRs on pre-existing test-file lint debt that has nothing to do with
the PR's own diff — this happened to PR #1540.

``pyproject.toml``'s ``dev`` extra is the single source of truth for "the
ruff version this repo uses locally." This script reads that exact pin so
CI (and, via ``.pre-commit-config.yaml``'s ``rev:``, the local pre-commit
hook) can install the same deterministic version instead of drifting.

Fails loudly (non-zero exit, message on stderr) if the ``dev`` extra doesn't
carry an exact ``ruff==X.Y.Z`` pin — a loose spec like ``ruff>=0.1.0`` would
silently reintroduce the unpinned-latest bug this script exists to prevent.

Run directly:

    python scripts/ci/get_ruff_pin.py

Wired into .github/workflows/pr-gate.yml's ``preflight`` job.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"


def get_ruff_pin() -> str:
    """Return the exact ruff version pinned in the ``dev`` extra.

    Raises ValueError if the ``dev`` extra has no ``ruff`` entry, or if that
    entry is not an exact ``==`` pin.
    """
    data = tomllib.loads(PYPROJECT_PATH.read_text())
    dev_deps = (data.get("project", {}).get("optional-dependencies", {}) or {}).get(
        "dev", []
    )

    for spec in dev_deps:
        if not isinstance(spec, str):
            continue
        name = spec.split("==")[0].split(">=")[0].split("~=")[0].split("[")[0].strip()
        if name.lower() != "ruff":
            continue
        if "==" not in spec:
            raise ValueError(
                f"pyproject.toml dev extra pins ruff with a loose spec ({spec!r}); "
                "expected an exact 'ruff==X.Y.Z' pin (see issue #1550 — a loose "
                "spec lets CI resolve an unpinned, potentially newer ruff)."
            )
        return spec.split("==", 1)[1].strip()

    raise ValueError(
        "pyproject.toml's [project.optional-dependencies] 'dev' extra has no "
        "'ruff' entry to pin against (see issue #1550)."
    )


def main() -> int:
    try:
        print(get_ruff_pin())
    except ValueError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
