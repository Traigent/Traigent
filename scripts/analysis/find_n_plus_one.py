#!/usr/bin/env python3
"""Heuristic N+1 query detector.

Searches for common patterns suggesting N+1 queries:
- DB queries inside loops
- Per-ID fetch loops that could be batched

Outputs file:line with a short snippet for review.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

LOOP_RE = re.compile(r"^\s*for\s+\w+\s+in\s+.+:\s*$")
QUERY_RE = re.compile(r"(session|db\.session)\.(query|execute)\(")
GET_RE = re.compile(r"\.get\(")


def scan_file(path: Path) -> list[tuple[int, str]]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    suspects: list[tuple[int, str]] = []

    in_loop = False
    loop_indent = 0
    for i, ln in enumerate(lines, start=1):
        stripped = ln.lstrip(" ")
        indent = len(ln) - len(stripped)

        if LOOP_RE.match(stripped):
            in_loop = True
            loop_indent = indent
            continue

        if in_loop and indent <= loop_indent and stripped:
            in_loop = False

        if in_loop and (QUERY_RE.search(stripped) or GET_RE.search(stripped)):
            # Capture a short snippet
            snippet = stripped[:140]
            suspects.append((i, snippet))

    return suspects


def main():
    files = [p for p in SRC.rglob("*.py") if p.is_file()]
    total = 0
    for f in files:
        hits = scan_file(f)
        if hits:
            print(f"\n{f}")
            for ln, snippet in hits:
                print(f"  {ln}: {snippet}")
            total += len(hits)
    print(f"\nTotal suspected N+1 sites: {total}")


if __name__ == "__main__":
    main()

