"""A missing min_effect must never be treated as 0.0 (Traigent#2658)."""

from __future__ import annotations

import pytest

from traigent.tvl import ObjectiveSpec, PromotionGate, PromotionPolicy

INC = {"accuracy": [0.80 + 0.001 * (i % 5) for i in range(200)]}
CAND = {"accuracy": [0.82 + 0.001 * (i % 5) for i in range(200)]}


@pytest.mark.parametrize(
    "min_effect",
    [{"latency": 5.0}, {}, {"Accuracy": 0.01}, {"acuracy": 0.01}],
)
def test_missing_misspelt_or_empty_min_effect_rejected_before_evaluation(
    min_effect: dict[str, float],
) -> None:
    with pytest.raises(ValueError, match="accuracy"):
        PromotionGate(
            PromotionPolicy(min_effect=min_effect),
            [ObjectiveSpec("accuracy", "maximize")],
        )


def test_present_key_decides_as_before() -> None:
    objs = [ObjectiveSpec("accuracy", "maximize")]
    huge = PromotionGate(PromotionPolicy(min_effect={"accuracy": 2.0}), objs)
    assert huge.evaluate(INC, CAND).decision == "no_decision"
    small = PromotionGate(PromotionPolicy(min_effect={"accuracy": 0.001}), objs)
    assert small.evaluate(INC, CAND).decision == "promote"
