"""SDK-129 area r3.1 regressions (sol round-3 blocking findings)."""

from __future__ import annotations

from traigent.tvl import ObjectiveSpec, PromotionGate, PromotionPolicy


def test_missing_objective_cannot_yield_superior_under_bonferroni() -> None:
    gate = PromotionGate(
        PromotionPolicy(
            alpha=0.05,
            adjust="bonferroni",
            min_effect={"a": 0.05, "b": 0.05},
        ),
        [ObjectiveSpec("a", "maximize"), ObjectiveSpec("b", "maximize")],
    )
    decision = gate.evaluate({"a": [0, 0]}, {"a": [0.5, 0.65]})
    assert decision.decision == "reject"
    assert all(r.verdict != "superior" for r in decision.objective_results)
