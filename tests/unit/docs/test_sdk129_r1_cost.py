from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_contract_documents_equal_share_over_requested_rows() -> None:
    from traigent.evaluators.hybrid_api import HybridAPIEvaluator

    class Resp:
        def get_reported_total_cost(self) -> float:
            return 0.60

    # The SDK divides the batch total by the number of requested rows,
    # without netting priced rows first.
    assert HybridAPIEvaluator._fallback_example_cost(Resp(), 2) == 0.30
    text = (REPO / "docs/hybrid-mode-api-contract.md").read_text(encoding="utf-8")
    assert "Priced rows are netted out first" not in text
    assert "divided by the number of requested rows" in text
