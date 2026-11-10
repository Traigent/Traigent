"""A plateau tolerance of 0 is treated as 1e-6 by the stop condition itself (#3025).

The cross-SDK stop-rules contract says a tolerance of 0 is treated as 1e-6, so
float noise in the running best still plateaus. The decorator path already
floored it; the stop condition and the manager used 0 literally.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from traigent.api.types import TrialResult, TrialStatus
from traigent.core.objectives import create_default_objectives
from traigent.core.stop_condition_manager import StopConditionManager
from traigent.core.stop_conditions import PlateauAfterNStopCondition


def _trial(i: int, accuracy: float) -> TrialResult:
    return TrialResult(
        trial_id=f"t{i}",
        config={"i": i},
        metrics={"accuracy": accuracy},
        status=TrialStatus.COMPLETED,
        duration=0.1,
        timestamp=datetime.now(),
    )


def _stop_trial(
    condition: PlateauAfterNStopCondition, scores: list[float]
) -> int | None:
    trials: list[TrialResult] = []
    for i, score in enumerate(scores, start=1):
        trials.append(_trial(i, score))
        if condition.should_stop(trials):
            return i
    return None


def _condition(epsilon: float) -> PlateauAfterNStopCondition:
    return PlateauAfterNStopCondition(
        window_size=2,
        epsilon=epsilon,
        objective_schema=create_default_objectives(["accuracy"]),
    )


def test_zero_tolerance_plateaus_on_sub_1e6_gains() -> None:
    assert _stop_trial(_condition(0), [0.5, 0.5000005, 0.500001]) == 2


def test_zero_tolerance_plateaus_on_equal_values() -> None:
    assert _stop_trial(_condition(0.0), [0.5, 0.5]) == 2


def test_zero_tolerance_keeps_running_on_real_gains() -> None:
    assert _stop_trial(_condition(0), [0.5, 0.51, 0.52]) is None


def test_positive_tolerance_is_used_as_given() -> None:
    assert _stop_trial(_condition(1e-9), [0.5, 0.5000005, 0.500001]) is None


def test_manager_zero_tolerance_floors_too() -> None:
    manager = StopConditionManager(
        max_trials=None,
        max_samples=None,
        samples_include_pruned=True,
        plateau_window=2,
        plateau_epsilon=0.0,
        objective_schema=create_default_objectives(["accuracy"]),
        metric_limit=None,
        metric_name=None,
        metric_include_pruned=True,
    )
    trials = [_trial(1, 0.5), _trial(2, 0.5000005)]
    stop, reason = manager.should_stop(trials)
    assert stop is True
    assert reason == "plateau"


def test_negative_tolerance_still_rejected() -> None:
    with pytest.raises(ValueError):
        _condition(-1e-9)


def test_gain_equal_to_tolerance_does_not_move_the_best() -> None:
    # Binary-exact values: 0.75 is not more than 0.5 + 0.25, so the best stays
    # 0.5 until 0.875; the window spread 0.375 exceeds the tolerance.
    condition = PlateauAfterNStopCondition(
        window_size=3,
        epsilon=0.25,
        objective_schema=create_default_objectives(["accuracy"]),
    )
    assert _stop_trial(condition, [0.5, 0.75, 0.875]) is None
