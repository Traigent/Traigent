"""Examples, walkthroughs and the providers manifest avoid announced-shutdown models.

OpenAI announced removal of gpt-4o / gpt-4o-mini on 2027-01-14 and of
gpt-3.5-turbo on 2026-10-23 (#2646, #2650, #2656). The SDK-wide default is an
owner decision and is not covered here; only example, walkthrough and docs
surfaces are.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCAN_ROOTS = ("traigent/examples", "examples", "walkthrough")
DOCS_ROOTS = ("docs",)
SUFFIXES = {".py", ".yml", ".yaml", ".sh", ".json"}
DOCS_SUFFIXES = SUFFIXES | {".md"}
# Legacy price/latency tables keep old entries so old saved results still resolve.
ALLOWED = {
    "walkthrough/utils/cost_estimator.py",
    "walkthrough/utils/mock_answers.py",
}
SKIP_NAMES = {"raw_results.json"}
PATTERN = re.compile(r"gpt-4o(?![\w.-]*(?:audio|search))")
GPT35_PATTERN = re.compile(r"gpt-3\.5")


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


def test_examples_walkthroughs_and_docs_do_not_use_gpt_3_5() -> None:
    offenders: list[str] = []
    roots = [(r, SUFFIXES) for r in SCAN_ROOTS] + [
        (r, DOCS_SUFFIXES) for r in DOCS_ROOTS
    ]
    for root, suffixes in roots:
        for path in (REPO / root).rglob("*"):
            if (
                not path.is_file()
                or path.suffix not in suffixes
                or path.name in SKIP_NAMES
            ):
                continue
            rel = path.relative_to(REPO).as_posix()
            if rel in ALLOWED:
                continue
            if GPT35_PATTERN.search(path.read_text(encoding="utf-8", errors="ignore")):
                offenders.append(rel)
    readme = (REPO / "README.md").read_text(encoding="utf-8", errors="ignore")
    if GPT35_PATTERN.search(readme):
        offenders.append("README.md")
    assert not offenders, offenders
