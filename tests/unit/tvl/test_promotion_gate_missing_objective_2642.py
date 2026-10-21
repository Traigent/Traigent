"""A declared objective with no samples must fail closed (Traigent#2642)."""

from __future__ import annotations

import random

import pytest

from traigent.tvl import ObjectiveSpec, PromotionGate, PromotionPolicy

OBJS = [ObjectiveSpec("accuracy", "maximize"), ObjectiveSpec("harm_rate", "minimize")]
POLICY = PromotionPolicy(min_effect={"accuracy": 0.0, "harm_rate": 0.05})


def _samples() -> tuple[list[float], list[float]]:
    rng = random.Random(0)
    return (
        [0.70 + rng.random() * 0.02 for _ in range(30)],
        [0.80 + rng.random() * 0.02 for _ in range(30)],
    )


@pytest.mark.parametrize(
    "candidate_extra,incumbent_extra",
    [
        ({}, {"harm_rate": [0.0] * 30}),  # candidate missing the objective
        ({"harmrate": [0.9] * 30}, {"harm_rate": [0.0] * 30}),  # typo
        ({"harm_rate": [0.0] * 30}, {}),  # incumbent missing the objective
    ],
)
def test_missing_objective_rejects(candidate_extra, incumbent_extra) -> None:
    inc, cand = _samples()
    gate = PromotionGate(POLICY, OBJS)
    decision = gate.evaluate(
        {"accuracy": inc, **incumbent_extra}, {"accuracy": cand, **candidate_extra}
    )
    assert decision.decision == "reject"
    assert "harm_rate" in decision.reason
    assert "missing" in decision.reason.lower()


def test_all_objectives_present_still_promotes() -> None:
    inc, cand = _samples()
    harm = [0.01 + 0.001 * (i % 5) for i in range(30)]
    gate = PromotionGate(POLICY, OBJS)
    decision = gate.evaluate(
        {"accuracy": inc, "harm_rate": harm},
        {"accuracy": cand, "harm_rate": harm},
    )
    assert decision.decision == "promote"
