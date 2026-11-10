"""The docs' examples maps list the advanced example folders that exist (#3166).

examples/README.md was corrected, but the two docs pages it links to still
listed examples/advanced/execution-modes/ (gone) and missed the folders added
since.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_PAGES = ["docs/examples/START_HERE.md", "docs/examples/README.md"]
_ENTRY = re.compile(r"^\|\s+[|`]- ([\w.-]+)/")


def _advanced_entries(text: str) -> set[str]:
    entries: set[str] = set()
    inside = False
    for line in text.splitlines():
        if re.match(r"^\|- advanced/", line):
            inside = True
            continue
        if inside:
            match = _ENTRY.match(line)
            if not match:
                break
            entries.add(match.group(1))
    return entries


@pytest.mark.parametrize("page", _PAGES)
def test_advanced_map_matches_the_folders(page: str) -> None:
    real = {p.name for p in (_REPO / "examples" / "advanced").iterdir() if p.is_dir()}
    real.discard("__pycache__")
    text = (_REPO / page).read_text(encoding="utf-8")

    assert _advanced_entries(text) == real
    assert "execution-modes" not in text
