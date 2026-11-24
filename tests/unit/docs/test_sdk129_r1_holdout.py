from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_guide_does_not_claim_unshipped_synthetic_holdout_guard() -> None:
    text = (REPO / "docs/guides/continuous-tuning.md").read_text(encoding="utf-8")
    assert "holdout_synthetic_rows" not in text
    assert not (
        REPO / "traigent/tuning"
    ).exists() or "holdout_synthetic_rows" not in "".join(
        p.read_text(encoding="utf-8") for p in (REPO / "traigent/tuning").rglob("*.py")
    )
