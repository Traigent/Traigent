"""Examples, walkthroughs and the providers manifest avoid announced-shutdown models.

OpenAI announced removal of gpt-4o / gpt-4o-mini on 2027-01-14 and of
gpt-3.5-turbo on 2026-10-23 (#2646, #2650, #2656). The SDK-wide default is an
owner decision and is not covered here; only example, walkthrough and docs
surfaces are.
"""

from __future__ import annotations

import json
import re
import subprocess
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
# Recorded historical measurements keep the model they were measured on. Only
# these files may carry a ``# recorded-historical-begin`` ... ``-end`` block,
# which the guard excludes from scanning. Such a file must only display the
# recorded data and never call the recorded model.
RECORDED_HISTORICAL_FILES = {"walkthrough/demo/optimize_rag.py"}
_RECORDED_BLOCK = re.compile(
    r"^# recorded-historical-begin.*?^# recorded-historical-end[^\n]*$",
    re.DOTALL | re.MULTILINE,
)
PATTERN = re.compile(r"gpt-4o(?![\w.-]*(?:audio|search))")
GPT35_PATTERN = re.compile(r"gpt-3\.5")


def _read_guarded(path: Path) -> str:
    """File text with the narrow recorded-historical exemption applied."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    rel = path.relative_to(REPO).as_posix()
    if rel in RECORDED_HISTORICAL_FILES:
        text = _RECORDED_BLOCK.sub("", text)
    elif (
        "recorded-historical-begin" in text
        and rel != "tests/unit/examples/test_no_shutdown_models.py"
    ):
        raise AssertionError(f"{rel} is not allowed a recorded-historical block")
    return text


def _guarded_files(root: str) -> list[Path]:
    """Files under ``root`` that the guard should scan.

    Uses git-tracked files only, so untracked runtime artifacts (for example
    ``.traigent/sessions/**`` written by running an example) are not scanned.
    Falls back to a filesystem walk that skips ``.traigent`` directories when
    git is unavailable.
    """
    try:
        listing = subprocess.run(
            ["git", "ls-files", "-z", "--", root],
            cwd=REPO,
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return [
            p
            for p in (REPO / root).rglob("*")
            if ".traigent" not in p.relative_to(REPO).parts
        ]
    return [REPO / rel for rel in listing.decode("utf-8").split("\0") if rel]


def test_examples_and_manifest_do_not_use_gpt_4o_family() -> None:
    offenders: list[str] = []
    for root in SCAN_ROOTS:
        for path in _guarded_files(root):
            if (
                not path.is_file()
                or path.suffix not in SUFFIXES
                or path.name in SKIP_NAMES
            ):
                continue
            rel = path.relative_to(REPO).as_posix()
            if rel in ALLOWED:
                continue
            if PATTERN.search(_read_guarded(path)):
                offenders.append(rel)
    assert not offenders, offenders


def test_examples_walkthroughs_and_docs_do_not_use_gpt_3_5() -> None:
    offenders: list[str] = []
    roots = [(r, SUFFIXES) for r in SCAN_ROOTS] + [
        (r, DOCS_SUFFIXES) for r in DOCS_ROOTS
    ]
    for root, suffixes in roots:
        for path in _guarded_files(root):
            if (
                not path.is_file()
                or path.suffix not in suffixes
                or path.name in SKIP_NAMES
            ):
                continue
            rel = path.relative_to(REPO).as_posix()
            if rel in ALLOWED:
                continue
            if GPT35_PATTERN.search(_read_guarded(path)):
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
            p for p in _guarded_files(root) if p.is_file() and p.suffix in suffixes
        ]
    offenders: dict[str, set[str]] = {}
    for path in files:
        rel = path.relative_to(REPO).as_posix()
        if path.name in SKIP_NAMES or rel in ALLOWED:
            continue
        text = _read_guarded(path)
        for match in MODEL_ID.finditer(text):
            model = match.group(1).rstrip(".-")
            for key in (model, f"openai/{model}"):
                if (table.get(key) or {}).get("deprecation_date"):
                    offenders.setdefault(rel, set()).add(
                        f"{model} ({table[key]['deprecation_date']})"
                    )
    assert not offenders, {k: sorted(v) for k, v in offenders.items()}


def test_recorded_historical_block_is_narrow_and_present() -> None:
    """The exemption removes only the marked block, not the whole file."""
    path = REPO / "walkthrough/demo/optimize_rag.py"
    raw = path.read_text(encoding="utf-8")
    stripped = _read_guarded(path)
    assert "recorded-historical-begin" in raw
    assert GPT35_PATTERN.search(raw) and not GPT35_PATTERN.search(stripped)
    assert "def main" in stripped
