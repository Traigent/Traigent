"""A trial with unpriced calls (strict accounting off) must not win a cost ranking.

Traigent#2539: ``TRAIGENT_STRICT_COST_ACCOUNTING=false`` records an unpriced
call as ``cost=0.0`` with ``cost_unpriced=1.0``. That zero is unknown spend, so
it must be excluded from cost ranking like ``cost_unmeasured`` trials.
"""

from __future__ import annotations

from datetime import UTC, datetime

from traigent.api.types import OptimizationResult, TrialResult, TrialStatus
from traigent.core.result_selection import select_best_configuration
from traigent.utils.multi_objective import ParetoFrontCalculator
from traigent.utils.trial_costs import is_cost_unranked


def _trial(model: str, **metrics: float) -> TrialResult:
    return TrialResult(
        trial_id=model,
        config={"model": model},
        metrics=dict(metrics),
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime.now(UTC),
    )


PRICED = ("priced", {"accuracy": 1.0, "cost": 0.000225})
UNPRICED = ("unpriced", {"accuracy": 1.0, "cost": 0.0, "cost_unpriced": 1.0})


def test_is_cost_unranked_flags_unpriced_and_unmeasured():
    assert is_cost_unranked({"cost": 0.0, "cost_unpriced": 1.0})
    assert is_cost_unranked({"cost_unmeasured": 1.0})
    assert not is_cost_unranked({"cost": 0.0, "cost_unpriced": 0.0})
    assert not is_cost_unranked({"cost": 0.0})


def test_unpriced_zero_cost_trial_does_not_win_cost_selection():
    trials = [_trial(UNPRICED[0], **UNPRICED[1]), _trial(PRICED[0], **PRICED[1])]
    result = select_best_configuration(
        trials=trials,
        primary_objective="cost",
        config_space_keys={"model"},
        aggregate_configs=False,
        comparability_mode="legacy",
        objective_orientations={"cost": "minimize"},
    )
    assert result.best_config == {"model": "priced"}


def test_unpriced_zero_cost_excluded_from_pareto_front():
    trials = [_trial(UNPRICED[0], **UNPRICED[1]), _trial(PRICED[0], **PRICED[1])]
    front = ParetoFrontCalculator().calculate_pareto_front(trials, ["accuracy", "cost"])
    models = [p.config["model"] for p in front]
    assert "unpriced" not in models


def test_unpriced_zero_cost_does_not_win_weighted_score():
    trials = [_trial(UNPRICED[0], **UNPRICED[1]), _trial(PRICED[0], **PRICED[1])]
    result = OptimizationResult(
        trials=trials,
        best_config={"model": "priced"},
        best_score=1.0,
        optimization_id="o",
        duration=1.0,
        convergence_info={},
        status="completed",
        objectives=["accuracy", "cost"],
        algorithm="grid",
        timestamp=datetime.now(),
    )
    scored = result.calculate_weighted_scores()
    assert scored["best_weighted_config"]["model"] == "priced"
