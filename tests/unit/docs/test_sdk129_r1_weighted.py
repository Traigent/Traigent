from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_spec_does_not_claim_score_objective_is_preserved() -> None:
    text = (REPO / "docs/api-reference/complete-function-specification.md").read_text(
        encoding="utf-8"
    )
    assert "weighted_selection_score" not in text
    assert "the evaluator's own value is never" not in text
