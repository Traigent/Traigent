"""Guard: shipped examples, defaults and probes must not name retired models.

Regression for #2656 (paid walkthrough).
"""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]

# Models the provider has retired (or shuts down within days of this change).
# Walkthrough scripts must not ship them even though some are not in the
# runtime denylist (kept valid for explicit user references).
_WALKTHROUGH_DEAD = (
    "gpt-3.5-turbo",
    "claude-sonnet-4-20250514",
    "gemini-2.0-flash",
    "gemini-1.5-pro-latest",
)


def test_walkthrough_real_has_no_retired_models() -> None:
    files = list((REPO_ROOT / "walkthrough" / "real").rglob("*.py"))
    files.append(REPO_ROOT / "walkthrough" / "demo" / "optimize_and_observe.py")
    files.append(REPO_ROOT / "walkthrough" / "utils" / "helpers.py")
    pattern = re.compile("|".join(re.escape(m) for m in _WALKTHROUGH_DEAD))
    hits = [
        f"{f.relative_to(REPO_ROOT)}:{i}"
        for f in files
        if f.exists()
        for i, line in enumerate(f.read_text().splitlines(), 1)
        if pattern.search(line)
    ]
    assert not hits, hits
