"""Docs and example helpers do not teach spend approval as the dry-run route (#2817).

``TRAIGENT_COST_APPROVED`` / ``cost_approved=True`` approve spend and also cover
a later real run, so pairing them with mock mode as the "free dry run" recipe
points the wrong way. The supported routes are a smaller slice or the scoped
``TRAIGENT_COST_PREFLIGHT_MOCK_EXEMPT`` (with a non-production environment).
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

APPROVAL = re.compile(r"TRAIGENT_COST_APPROVED|cost_approved\s*=\s*True")
MOCK = re.compile(r"mock", re.IGNORECASE)
# A paragraph that mentions approval only to warn against it is fine.
WARNS = re.compile(
    r"do not use|avoid|also approves|also covers?|rather than ``?TRAIGENT_COST_APPROVED|rather than approval",
    re.IGNORECASE,
)

SCANNED = [
    "docs/walkthrough.md",
    "docs/examples/API_PATTERNS.md",
    "docs/features/safety-gates.md",
    "docs/getting-started/testing.md",
    "traigent/examples/providers/_demo.py",
]


def _paragraphs(path: Path) -> list[str]:
    return re.split(r"\n\s*\n", path.read_text(encoding="utf-8"))


def test_no_doc_pairs_mock_mode_with_cost_approval_as_the_dry_run_recipe_2817() -> None:
    offenders = []
    for relative in SCANNED:
        for paragraph in _paragraphs(REPO / relative):
            if (
                APPROVAL.search(paragraph)
                and MOCK.search(paragraph)
                and not WARNS.search(paragraph)
            ):
                offenders.append((relative, paragraph.strip().splitlines()[0][:80]))
    assert not offenders, offenders


def test_the_scanned_files_name_the_exemption_route_2817() -> None:
    for relative in SCANNED[:4]:
        text = (REPO / relative).read_text(encoding="utf-8")
        assert "TRAIGENT_COST_PREFLIGHT_MOCK_EXEMPT" in text, relative


def test_no_unqualified_zero_cost_mock_claims_2817() -> None:
    claims = {
        "docs/agent-skill.md": "Runs a mock optimization (zero cost)",
        "walkthrough/real/advanced/README.md": "costs nothing",
        "traigent/cli/onboard_commands.py": "verify zero-cost local execution",
    }
    for relative, claim in claims.items():
        assert claim not in (REPO / relative).read_text(encoding="utf-8"), relative


def test_mock_adapter_docstrings_match_is_mock_llm_2817() -> None:
    from traigent.integrations.utils.mock_adapter import MockAdapter

    assert "never** auto-enabled via environment variables" not in (
        MockAdapter.__doc__ or ""
    )
    doc = MockAdapter.get_mock_response.__doc__ or ""
    assert "no longer reachable from any production code path" not in doc
    assert "is_mock_llm" in (MockAdapter.__doc__ or "")
