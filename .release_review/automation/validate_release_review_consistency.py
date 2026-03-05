#!/usr/bin/env python3
"""Validate release-review docs for deprecated path/command usage."""

from __future__ import annotations

import re
from pathlib import Path

TARGET_FILES = [
    Path(".release_review/CAPTAIN_PROTOCOL.md"),
    Path(".release_review/PRE_RELEASE_REVIEW_PLAN.md"),
    Path(".release_review/START_REVIEW.md"),
    Path(".release_review/PRE_RELEASE_REVIEW_TRACKING.md"),
    Path(".release_review/RELEASE_QUALITY_PLAYBOOK.md"),
    Path(".release_review/automation/README.md"),
]

DEPRECATED_PATTERNS = [
    (
        re.compile(r"\.release_review/artifacts/"),
        "use `.release_review/runs/<release_id>/...` instead of `.release_review/artifacts/`",
    ),
    (
        re.compile(r"\bpython\s+\.release_review/"),
        "use `python3` instead of `python` for release-review commands",
    ),
]


def main() -> int:
    violations: list[str] = []

    for file in TARGET_FILES:
        if not file.exists():
            violations.append(f"missing required file: {file}")
            continue
        content = file.read_text()
        for pattern, message in DEPRECATED_PATTERNS:
            for match in pattern.finditer(content):
                line = content.count("\n", 0, match.start()) + 1
                violations.append(f"{file}:{line}: {message}")

    if violations:
        print("❌ Release-review consistency check failed")
        for item in violations:
            print(f"  - {item}")
        return 1

    print("✅ Release-review consistency check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
