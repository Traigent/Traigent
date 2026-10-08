"""An unmeasured cost must not rank as the best cost (#2446).

#2442 made an unmeasured trial's cost ABSENT instead of ``0.0``. Two ranking
surfaces still gave that absence the best possible cost:

- the Pareto front kept the trial, because a pair with a missing objective is
  incomparable and so the trial is never dominated;
- the batch composite score skipped the missing objective and renormalized over
  the remaining weights, scoring the trial on quality alone.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from traigent.api.types import TrialResult, TrialStatus
from traigent.optimizers.registry import get_optimizer
from traigent.utils.multi_objective import ParetoFrontCalculator


def _trial(trial_id: str, **metrics: float) -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        config={"id": trial_id},
        metrics=dict(metrics),
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime.now(UTC),
    )


def test_pareto_front_excludes_a_trial_with_unmeasured_cost() -> None:
    calc = ParetoFrontCalculator(maximize={"accuracy": True, "cost": False})
    measured = _trial("measured", accuracy=0.80, cost=0.05)
    unmeasured = _trial("unmeasured", accuracy=0.95)  # cost key absent

    front = calc.calculate_pareto_front([measured, unmeasured], ["accuracy", "cost"])

    assert [p.trial.trial_id for p in front] == ["measured"]


def test_pareto_front_is_empty_when_no_trial_measured_every_objective() -> None:
    calc = ParetoFrontCalculator(maximize={"accuracy": True, "cost": False})
    front = calc.calculate_pareto_front(
        [_trial("a", accuracy=0.8), _trial("b", accuracy=0.9)], ["accuracy", "cost"]
    )
    assert front == []


@pytest.mark.parametrize("kind", ["parallel_batch", "adaptive_batch"])
def test_batch_composite_score_never_rewards_a_missing_cost(kind: str) -> None:
    optimizer = get_optimizer(
        kind,
        {"param1": [1, 2]},
        ["accuracy", "cost"],
        objective_weights={"accuracy": 0.5, "cost": 0.5},
    )

    measured = optimizer._calculate_composite_score({"accuracy": 0.8, "cost": 0.05})
    unmeasured = optimizer._calculate_composite_score({"accuracy": 0.95})

    assert measured > float("-inf")
    # Not scored on quality alone: an unknown cost can never win.
    assert unmeasured == float("-inf")
    assert unmeasured < measured
