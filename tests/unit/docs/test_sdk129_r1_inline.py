from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_client_guide_inline_dataset_uses_input_wrapper() -> None:
    text = (REPO / "docs/hybrid-mode-client-guide.md").read_text(encoding="utf-8")
    assert 'dataset = [{"example_id": "case_001"}' not in text
    assert '{"input": {"example_id": "case_001"}}' in text
