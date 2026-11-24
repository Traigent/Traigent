"""SDK-129 area r1 regressions (sol blocking findings)."""

from __future__ import annotations

import pytest


def test_f4_unsigned_numpy_samples_do_not_wrap() -> None:
    np = pytest.importorskip("numpy")
    from traigent.tvl.promotion_gate import (
        ObjectiveSpec,
        PromotionGate,
        PromotionPolicy,
    )

    def run(inc, cand):
        gate = PromotionGate(
            PromotionPolicy(min_effect={"a": 1.0}), [ObjectiveSpec("a", "maximize")]
        )
        return gate.evaluate({"a": inc}, {"a": cand})

    lists = run([100], [90])
    arrays = run(np.array([100], dtype=np.uint8), np.array([90], dtype=np.uint8))
    assert arrays.decision == lists.decision
    assert arrays.objective_results[0].effect_size == pytest.approx(
        lists.objective_results[0].effect_size
    )
    assert lists.objective_results[0].effect_size == pytest.approx(-10)
