import pytest

from traigent.api.types import TrialStatus
from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema
from traigent.core.stop_condition_manager import StopConditionManager
from traigent.core.stop_conditions import (
    MaxSamplesStopCondition,
    MaxTrialsStopCondition,
    PlateauAfterNStopCondition,
)


class DummyTrial:
    def __init__(self, *, status: TrialStatus = TrialStatus.COMPLETED, metrics=None):
        self.status = status
        self.metrics = metrics or {}
        self.metadata = {}
        self.trial_id = "dummy"


def make_schema() -> ObjectiveSchema:
    return ObjectiveSchema.from_objectives(
        [ObjectiveDefinition(name="accuracy", orientation="maximize", weight=1.0)]
    )


def base_manager(**overrides):
    kwargs = {
        "max_trials": None,
        "max_samples": None,
        "samples_include_pruned": True,
        "plateau_window": None,
        "plateau_epsilon": None,
        "objective_schema": None,
        "budget_limit": None,
        "budget_metric": "total_cost",
        "include_pruned": True,
    }
    kwargs.update(overrides)
    return StopConditionManager(**kwargs)


def test_update_max_trials_creates_and_updates_condition():
    manager = base_manager()
    assert not manager.conditions

    manager.update_max_trials(3)
    assert isinstance(manager.conditions[-1], MaxTrialsStopCondition)

    manager.update_max_trials(None)
    assert all(not isinstance(c, MaxTrialsStopCondition) for c in manager.conditions)


def test_update_max_samples_adjusts_include_pruned_flag():
    manager = base_manager()
    manager.update_max_samples(5)
    condition = next(
        c for c in manager.conditions if isinstance(c, MaxSamplesStopCondition)
    )
    manager.update_samples_include_pruned(False)
    # the call should toggle include_pruned via set_include_pruned, which resets counters
    assert condition._include_pruned is False  # pylint: disable=protected-access


def test_plateau_requires_objective_schema():
    with pytest.raises(ValueError):
        base_manager(plateau_window=3, plateau_epsilon=1e-3)

    manager = base_manager(
        plateau_window=2,
        plateau_epsilon=0.01,
        objective_schema=make_schema(),
    )
    assert any(isinstance(c, PlateauAfterNStopCondition) for c in manager.conditions)


def test_should_stop_returns_reason_when_condition_triggers():
    manager = base_manager(max_trials=1)
    should_stop, reason = manager.should_stop([DummyTrial(), DummyTrial()])
    assert should_stop is True
    assert reason == "max_trials"
