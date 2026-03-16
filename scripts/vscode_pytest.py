#!/usr/bin/env python3
"""Run pytest from VS Code tasks/debug configs using the repo root context."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent

    # VS Code launches this script by path, which would otherwise put only
    # `scripts/` on sys.path. Force repo-root execution so local imports match
    # CLI usage from the project root.
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))

    try:
        import pytest
    except ImportError:
        print("pytest is not installed in the selected interpreter.", file=sys.stderr)
        return 2

    return pytest.main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
