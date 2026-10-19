"""Guard: the SDK's bundled public skills must not leak internal lifecycle vocabulary.

``traigent/skills/**`` ships inside the public SDK package. The canonical
``Traigent/traigent-skills`` repo lints its skills for internal next-run
lifecycle vocabulary (``FORBIDDEN_EXACT_LIFECYCLE_TOKENS`` in its
``tests/contract/test_skill_lints.py``); this is the SDK-side equivalent so a
vendored/synced skill cannot publish those tokens undetected (issue #1564).

Keep ``FORBIDDEN_EXACT_LIFECYCLE_TOKENS`` in step with the canonical set.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SKILLS_DIR = Path(__file__).resolve().parents[3] / "traigent" / "skills"

FORBIDDEN_EXACT_LIFECYCLE_TOKENS = {
    "artifact_states",
    "audit_evaluator",
    "audit_stale",
    "blocker_codes",
    "compare_baseline",
    "LC_V1_DERIVED",
    "ranked_operations",
    "reason_code",
    "result_stale",
    "run_holdout",
    "score-stale",
    "score_examples",
    "score_stale",
    "scored-against",
    "scored_against",
    "scored-needs-tuning",
    "scored_needs_tuning",
    "smartopt_available",
    "synth_harder_examples",
    "target_artifact",
    "tied_with_baseline",
    "trust_label",
    "unaudited",
    "unknown_freshness",
    "validated_on_holdout",
}


def _token_re(token: str) -> re.Pattern[str]:
    """Match a token without catching longer identifiers or file names."""
    return re.compile(
        rf"(?<![A-Za-z0-9_-]){re.escape(token)}(?![A-Za-z0-9_-])",
        re.IGNORECASE,
    )


def _skill_files() -> list[Path]:
    return sorted(p for p in SKILLS_DIR.rglob("*") if p.is_file())


def _leaks(text: str) -> list[tuple[str, int]]:
    found = []
    for token in sorted(FORBIDDEN_EXACT_LIFECYCLE_TOKENS):
        match = _token_re(token).search(text)
        if match:
            found.append((token, text.count("\n", 0, match.start()) + 1))
    return found


def test_bundled_skills_exist() -> None:
    assert _skill_files(), f"no bundled skill files found under {SKILLS_DIR}"


def test_no_leaked_lifecycle_vocab_in_bundled_skills() -> None:
    violations = []
    for path in _skill_files():
        text = path.read_text(encoding="utf-8")
        for token, line in _leaks(text):
            rel = path.relative_to(SKILLS_DIR.parents[1]).as_posix()
            violations.append(f"{rel}:{line}: {token}")
    assert not violations, (
        "internal lifecycle vocabulary leaked into the public SDK skills:\n"
        + "\n".join(violations)
    )


@pytest.mark.parametrize("token", sorted(FORBIDDEN_EXACT_LIFECYCLE_TOKENS))
def test_guard_detects_each_forbidden_token(token: str) -> None:
    """The scanner itself must flag every token (guards against a dead lint)."""
    assert _leaks(f"Next step: see {token} in the output.\n")


def test_guard_ignores_longer_identifiers() -> None:
    assert not _leaks("use audit_evaluator_quality or my-audit_evaluator-x\n")
