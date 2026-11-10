"""A non-finite objective value must not put a trial on the Pareto front (#2949).

NaN differences satisfy neither ``diff < -eps`` nor ``diff > eps``, so a NaN
objective used to compare as "equal" and the point could still dominate on the
other objective, evicting every valid point. A non-finite value is now treated
like a missing one: the trial cannot be ranked.
"""

from __future__ import annotations

import math
from datetime import datetime

import pytest

from traigent.api.types import TrialResult, TrialStatus
from traigent.utils.multi_objective import ParetoFrontCalculator

OBJECTIVES = ["accuracy", "cost"]


def _trial(name: str, accuracy: float, cost: float) -> TrialResult:
    return TrialResult(
        trial_id=name,
        config={"name": name},
        metrics={"accuracy": accuracy, "cost": cost},
        status=TrialStatus.COMPLETED,
        duration=0.1,
        timestamp=datetime.now(),
        metadata={},
    )


def _front(trials: list[TrialResult]) -> list[str]:
    calc = ParetoFrontCalculator(maximize={"accuracy": True, "cost": False})
    return sorted(
        p.config["name"] for p in calc.calculate_pareto_front(trials, OBJECTIVES)
    )


VALID = [_trial("B", 0.9, 0.01), _trial("C", 0.95, 0.02)]


@pytest.mark.parametrize(
    ("accuracy", "cost"),
    [
        (math.nan, 0.001),
        (math.inf, 0.001),
        (-math.inf, 0.001),
        (0.99, math.nan),
        (0.99, -math.inf),
        (0.99, math.inf),
    ],
)
def test_non_finite_point_is_excluded_and_valid_front_survives(
    accuracy: float, cost: float
) -> None:
    trials = [_trial("A", accuracy, cost), *VALID]
    assert _front(trials) == ["B", "C"]


def test_all_non_finite_gives_empty_front() -> None:
    assert _front([_trial("A", math.nan, 0.1), _trial("D", 0.5, math.inf)]) == []


def test_finite_points_unchanged() -> None:
    assert _front([_trial("A", 0.99, 0.001), *VALID]) == ["A"]
