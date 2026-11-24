from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_guided_readme_offline_claim_is_scoped_to_traigent_tracking() -> None:
    text = (REPO / "examples/core/guided-generation/README.md").read_text(
        encoding="utf-8"
    )
    assert "Use local/offline mode to keep\n  them on your machine" not in text
    assert "it does not stop prompts from" in text
