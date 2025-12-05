#!/usr/bin/env python3
"""List Python modules for review within a folder.

Skips tests/ and __init__.py files. Outputs one path per line relative to cwd.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def iter_modules(folder: str) -> list[Path]:
    base = Path(folder)
    return sorted(
        p
        for p in base.rglob("*.py")
        if "tests" not in p.parts and p.name != "__init__.py"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    args = ap.parse_args(argv)

    for module in iter_modules(args.folder):
        print(module.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
