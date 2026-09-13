"""Guard test: removed/deprecated execution-mode vocabulary must not be
presented as CURRENT in user-facing docs, examples, or walkthrough content.

Traigent#2271 fixed the naming for the three execution modes (``local``,
``cloud``, ``hybrid_api``) and removed/deprecated a handful of older names.
This test scans ``docs/``, ``examples/``, and ``walkthrough/`` (Markdown and
Python files only) and fails if any of the removed/deprecated names appear on
a line that does not read as a migration note.

A line is treated as a migration note (and exempted) if it contains one of
the ``EXEMPT_KEYWORDS`` (case-insensitive) — e.g. "deprecated", "removed",
"legacy", "formerly", "migrat[ion/e/ing]". Genuine historical changelogs that
describe already-merged past commits in period-accurate terminology, without
using any of those keywords on the offending line, are listed explicitly in
``ALLOWLIST_LINES`` with a one-line reason — do not add new entries there to
silence a doc that is actually presenting an old name as current; fix the doc
instead.

See also: docs/user-guide/execution-modes.md (the canonical mode matrix).
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

SCAN_DIRS = ("docs", "examples", "walkthrough")
SCAN_EXTENSIONS = (".md", ".py")

# name -> compiled regex. Matched case-insensitively against each line.
FORBIDDEN_PATTERNS: dict[str, re.Pattern[str]] = {
    "edge_analytics": re.compile(r"edge_analytics", re.IGNORECASE),
    'execution_mode="privacy"': re.compile(
        r'execution_mode\s*=\s*"privacy"', re.IGNORECASE
    ),
    'execution_mode="cloud"': re.compile(
        r'execution_mode\s*=\s*"cloud"', re.IGNORECASE
    ),
    'execution_mode="standard"': re.compile(
        r'execution_mode\s*=\s*"standard"', re.IGNORECASE
    ),
    "hybrid session": re.compile(r"hybrid session", re.IGNORECASE),
    "Hybrid Service": re.compile(r"hybrid service", re.IGNORECASE),
}

# A line containing any of these (case-insensitive) reads as a migration
# note, not a claim that the old name is current, and is exempted.
EXEMPT_KEYWORDS = ("deprecated", "removed", "migrat", "legacy", "formerly")

# (relative_path, line_number) -> reason. Line numbers are 1-based. Use this
# ONLY for genuine historical/changelog text that cannot carry an
# EXEMPT_KEYWORDS on the same line without falsifying the historical record.
ALLOWLIST_LINES: dict[tuple[str, int], str] = {
    (
        "docs/fake-completion-tracking.md",
        17,
    ): "changelog entry describing an already-merged historical commit in "
    "period-accurate terminology (predates the execution-mode consolidation)",
    (
        "docs/fake-completion-tracking.md",
        23,
    ): "changelog entry describing an already-merged historical commit in "
    "period-accurate terminology (predates the execution-mode consolidation)",
}

# Whole files exempted outright (documented reason required). Prefer
# ALLOWLIST_LINES for anything narrower than a full file.
ALLOWLIST_FILES: dict[str, str] = {}


def _iter_scanned_files() -> list[Path]:
    files: list[Path] = []
    for scan_dir in SCAN_DIRS:
        root = PROJECT_ROOT / scan_dir
        if not root.exists():
            continue
        for ext in SCAN_EXTENSIONS:
            files.extend(root.rglob(f"*{ext}"))
    return sorted(files)


def find_violations() -> list[str]:
    """Return a list of human-readable violation descriptions."""
    violations: list[str] = []
    for path in _iter_scanned_files():
        rel_path = path.relative_to(PROJECT_ROOT).as_posix()
        if rel_path in ALLOWLIST_FILES:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        for line_no, line in enumerate(lines, start=1):
            lowered = line.lower()
            if any(keyword in lowered for keyword in EXEMPT_KEYWORDS):
                continue
            if (rel_path, line_no) in ALLOWLIST_LINES:
                continue
            for name, pattern in FORBIDDEN_PATTERNS.items():
                if pattern.search(line):
                    violations.append(
                        f"{rel_path}:{line_no}: presents removed/deprecated "
                        f"name {name!r} as current: {line.strip()!r}"
                    )
    return violations


class TestModeVocabulary(unittest.TestCase):
    """Removed/deprecated execution-mode names must not read as current."""

    def test_no_removed_mode_names_presented_as_current(self) -> None:
        violations = find_violations()
        self.assertEqual(
            violations,
            [],
            "Found removed/deprecated execution-mode vocabulary presented as "
            "current (see docs/user-guide/execution-modes.md for the current "
            "names). Either fix the wording, add an EXEMPT_KEYWORDS word "
            "(deprecated/removed/migrat.../legacy/formerly) to the same "
            "line, or — for genuine historical changelog text only — add a "
            "narrowly-scoped ALLOWLIST_LINES entry with a reason:\n"
            + "\n".join(violations),
        )

    def test_scanner_actually_finds_files(self) -> None:
        """Sanity check that the scanner is not silently scanning nothing."""
        files = _iter_scanned_files()
        self.assertGreater(
            len(files),
            50,
            "Expected to scan more than 50 docs/examples/walkthrough files",
        )


if __name__ == "__main__":
    unittest.main()
