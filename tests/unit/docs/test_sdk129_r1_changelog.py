from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_changelog_has_no_unlanded_tracing_scrub_entry() -> None:
    text = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "Tracing scrubs session span names" not in text
    assert "traigent-tracing` add-on is 0.1.3" not in text
