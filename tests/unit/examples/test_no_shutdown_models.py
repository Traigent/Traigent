"""Examples, walkthroughs and the providers manifest avoid announced-shutdown models.

OpenAI announced removal of gpt-4o / gpt-4o-mini on 2027-01-14
(#2646, #2650, #2656). The SDK-wide default is an owner decision and is not
covered here; only example/walkthrough surfaces are.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCAN_ROOTS = ("traigent/examples", "examples", "walkthrough")
SUFFIXES = {".py", ".yml", ".yaml", ".sh", ".json"}
# Legacy price/latency tables keep old entries so old saved results still resolve.
ALLOWED = {
    "walkthrough/utils/cost_estimator.py",
    "walkthrough/utils/mock_answers.py",
}
SKIP_NAMES = {"raw_results.json"}
PATTERN = re.compile(r"gpt-4o(?![\w.-]*(?:audio|search))")


def test_examples_and_manifest_do_not_use_gpt_4o_family() -> None:
    offenders: list[str] = []
    for root in SCAN_ROOTS:
        for path in (REPO / root).rglob("*"):
            if (
                not path.is_file()
                or path.suffix not in SUFFIXES
                or path.name in SKIP_NAMES
            ):
                continue
            rel = path.relative_to(REPO).as_posix()
            if rel in ALLOWED:
                continue
            if PATTERN.search(path.read_text(encoding="utf-8", errors="ignore")):
                offenders.append(rel)
    assert not offenders, offenders
