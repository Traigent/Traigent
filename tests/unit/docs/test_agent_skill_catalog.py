"""Guard docs/agent-skill.md against removed or renamed traigent-skills IDs.

Regression test for the agent-skill guide shipping skill IDs (``traigent``,
``traigent-quickstart``) and a canonical link (``skills/traigent``) that no
longer exist in the traigent-skills catalog after its 2026-07 rename/merge.
Both selective-install ``--skill`` flags and the ``skills/<name>`` GitHub link
are checked against a pinned snapshot of the catalog
(``docs/agent-skill-catalog.json``) so a future rename in traigent-skills is
caught here instead of shipping a dead command/link to users.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOC_PATH = PROJECT_ROOT / "docs" / "agent-skill.md"
CATALOG_PATH = PROJECT_ROOT / "docs" / "agent-skill-catalog.json"

SKILL_FLAG_RE = re.compile(r"--skill\s+([A-Za-z0-9][A-Za-z0-9._*-]*)")
SKILLS_LINK_RE = re.compile(
    r"github\.com/Traigent/traigent-skills/(?:tree|blob)/[^/\s)]+/skills/([A-Za-z0-9][A-Za-z0-9._-]*)"
)


def _load_catalog_names() -> set[str]:
    data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    return set(data["skill_names"])


def _extract_referenced_ids(doc_text: str) -> set[str]:
    ids: set[str] = set()
    for match in SKILL_FLAG_RE.finditer(doc_text):
        name = match.group(1)
        if name == "*":
            continue  # wildcard install, nothing to resolve
        ids.add(name)
    for match in SKILLS_LINK_RE.finditer(doc_text):
        ids.add(match.group(1))
    return ids


def test_agent_skill_doc_references_are_non_empty() -> None:
    """Sanity check: the guide still names at least one skill ID and one link."""
    doc_text = DOC_PATH.read_text(encoding="utf-8")
    referenced = _extract_referenced_ids(doc_text)
    assert referenced, (
        "Expected docs/agent-skill.md to name at least one --skill ID or "
        "skills/<name> link; the extraction regexes may need updating."
    )


def test_agent_skill_doc_ids_resolve_against_pinned_catalog() -> None:
    """Every --skill ID and skills/<name> link in the guide must exist upstream."""
    doc_text = DOC_PATH.read_text(encoding="utf-8")
    referenced = _extract_referenced_ids(doc_text)
    catalog_names = _load_catalog_names()

    missing = sorted(referenced - catalog_names)
    assert not missing, (
        "docs/agent-skill.md names skill ID(s)/link(s) not present in the "
        f"pinned traigent-skills catalog ({CATALOG_PATH.name}): {missing}. "
        "If they were renamed upstream, update docs/agent-skill.md and "
        "refresh docs/agent-skill-catalog.json from traigent-skills' "
        "catalog/skills.json; if they were removed, replace the reference."
    )
