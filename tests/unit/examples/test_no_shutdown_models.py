"""Examples, walkthroughs and the providers manifest avoid announced-shutdown models.

OpenAI announced removal of gpt-4o / gpt-4o-mini on 2027-01-14 and of
gpt-3.5-turbo on 2026-10-23 (#2646, #2650, #2656). The SDK-wide default is an
owner decision and is not covered here; only example, walkthrough and docs
surfaces are.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import litellm

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


MODEL_ID = re.compile(
    r"(?<![\w.-])(?:(?:openai|anthropic|gemini)/)?((?:gpt|claude|gemini|o[134])-[A-Za-z0-9][\w.-]*)"
)


def _litellm_table() -> dict:
    path = Path(litellm.__file__).parent / "model_prices_and_context_window_backup.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_guarded_surfaces_mention_no_model_with_announced_shutdown() -> None:
    """Any model id on a guarded surface must have no litellm deprecation_date."""
    table = _litellm_table()
    roots = [(r, SUFFIXES) for r in SCAN_ROOTS] + [
        (r, DOCS_SUFFIXES) for r in DOCS_ROOTS
    ]
    files = [REPO / "README.md"]
    for root, suffixes in roots:
        files += [
            p for p in (REPO / root).rglob("*") if p.is_file() and p.suffix in suffixes
        ]
    offenders: dict[str, set[str]] = {}
    for path in files:
        rel = path.relative_to(REPO).as_posix()
        if path.name in SKIP_NAMES or rel in ALLOWED:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in MODEL_ID.finditer(text):
            model = match.group(1).rstrip(".-")
            for key in (model, f"openai/{model}"):
                if (table.get(key) or {}).get("deprecation_date"):
                    offenders.setdefault(rel, set()).add(
                        f"{model} ({table[key]['deprecation_date']})"
                    )
    assert not offenders, {k: sorted(v) for k, v in offenders.items()}
