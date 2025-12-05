import pytest

from traigent.api.types import TrialResult, TrialStatus
from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema
from traigent.core.stop_conditions import (
    BudgetStopCondition,
    MaxSamplesStopCondition,
    MaxTrialsStopCondition,
    PlateauAfterNStopCondition,
)


def _make_trial(trial_id: str, metrics: dict[str, float]) -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        config={"param": 1},
        metrics=metrics,
        status=TrialStatus.COMPLETED,
        duration=0.1,
        timestamp=None,
    )


def test_max_trials_stop_condition_triggers_at_threshold():
    stop_condition = MaxTrialsStopCondition(max_trials=3)

    history = [_make_trial("t1", {"accuracy": 0.5})]
    assert not stop_condition.should_stop(history)

    history.append(_make_trial("t2", {"accuracy": 0.6}))
    assert not stop_condition.should_stop(history)

    history.append(_make_trial("t3", {"accuracy": 0.7}))
    assert stop_condition.should_stop(history)


def test_max_trials_stop_condition_invalid():
    with pytest.raises(ValueError):
        MaxTrialsStopCondition(max_trials=0)


def test_plateau_after_n_stop_condition_uses_schema_weights():
    schema = ObjectiveSchema.from_objectives(
        [
            ObjectiveDefinition("accuracy", "maximize", 0.6),
            ObjectiveDefinition("total_cost", "minimize", 0.4),
        ]
    )

    stop_condition = PlateauAfterNStopCondition(
        window_size=3,
        epsilon=0.0,
        objective_schema=schema,
    )

    history: list[TrialResult] = []

    history.append(_make_trial("t1", {"accuracy": 0.80, "total_cost": 0.05}))
    assert not stop_condition.should_stop(history)

    history.append(_make_trial("t2", {"accuracy": 0.80, "total_cost": 0.05}))
    assert not stop_condition.should_stop(history)

    history.append(_make_trial("t3", {"accuracy": 0.80, "total_cost": 0.05}))
    assert stop_condition.should_stop(history)


def test_plateau_after_n_stop_condition_reset():
    schema = ObjectiveSchema.from_objectives(
        [ObjectiveDefinition("accuracy", "maximize", 1.0)]
    )

    stop_condition = PlateauAfterNStopCondition(
        window_size=2,
        epsilon=0.01,
        objective_schema=schema,
    )

    history = [
        _make_trial("t1", {"accuracy": 0.5}),
        _make_trial("t2", {"accuracy": 0.5}),
    ]
    assert stop_condition.should_stop(history)

    stop_condition.reset()
    assert not stop_condition.should_stop(history[:1])


def test_objective_schema_weighted_score_handles_minimize():
    schema = ObjectiveSchema.from_objectives(
        [
            ObjectiveDefinition("accuracy", "maximize", 0.5),
            ObjectiveDefinition("total_cost", "minimize", 0.5),
        ]
    )

    high_cost = schema.compute_weighted_score({"accuracy": 0.8, "total_cost": 0.10})
    low_cost = schema.compute_weighted_score({"accuracy": 0.8, "total_cost": 0.05})

    assert low_cost is not None and high_cost is not None
    assert low_cost > high_cost


def test_plateau_after_n_invalid_parameters():
    schema = ObjectiveSchema.from_objectives(
        [ObjectiveDefinition("accuracy", "maximize", 1.0)]
    )

    with pytest.raises(ValueError):
        PlateauAfterNStopCondition(window_size=0, epsilon=0.0, objective_schema=schema)

    with pytest.raises(ValueError):
        PlateauAfterNStopCondition(window_size=1, epsilon=-0.1, objective_schema=schema)

    with pytest.raises(ValueError):
        PlateauAfterNStopCondition(window_size=1, epsilon=0.0, objective_schema=None)


def test_budget_stop_condition_triggers_on_cost():
    stop_condition = BudgetStopCondition(budget=0.2, metric_name="total_cost")

    history = [
        _make_trial("t1", {"total_cost": 0.05}),
        _make_trial("t2", {"total_cost": 0.10}),
    ]
    assert not stop_condition.should_stop(history)

    history.append(_make_trial("t3", {"total_cost": 0.06}))
    assert stop_condition.should_stop(history)


def test_budget_stop_condition_uses_metadata_fallback():
    trial = _make_trial("t1", {"accuracy": 1.0})
    trial.metadata = {"total_example_cost": 0.3}

    stop_condition = BudgetStopCondition(budget=0.25)
    assert stop_condition.should_stop([trial])


def test_budget_stop_condition_invalid_budget():
    with pytest.raises(ValueError):
        BudgetStopCondition(budget=0.0)

    with pytest.raises(ValueError):
        BudgetStopCondition(budget=-1)


def test_budget_stop_condition_respects_include_pruned():
    trial = _make_trial("t1", {"total_cost": 0.2})
    trial.status = TrialStatus.PRUNED

    stop_condition = BudgetStopCondition(budget=0.1, include_pruned=False)
    assert not stop_condition.should_stop([trial])


def _make_trial_with_examples(
    trial_id: str,
    examples_attempted: int,
    status: TrialStatus = TrialStatus.COMPLETED,
) -> TrialResult:
    trial = _make_trial(trial_id, {"examples_attempted": float(examples_attempted)})
    trial.status = status
    trial.metadata = {"examples_attempted": examples_attempted}
    return trial


def test_max_samples_stop_condition_counts_examples():
    stop_condition = MaxSamplesStopCondition(max_samples=5)

    trial1 = _make_trial_with_examples("t1", 2)
    assert not stop_condition.should_stop([trial1])

    trial2 = _make_trial_with_examples("t2", 3)
    assert stop_condition.should_stop([trial1, trial2])


def test_max_samples_stop_condition_excludes_pruned_when_configured():
    stop_condition = MaxSamplesStopCondition(max_samples=2, include_pruned=False)

    pruned_trial = _make_trial_with_examples("t1", 2, status=TrialStatus.PRUNED)
    assert not stop_condition.should_stop([pruned_trial])

    # Successful trial should now push us over the limit.
    completed_trial = _make_trial_with_examples("t2", 2)
    assert stop_condition.should_stop([pruned_trial, completed_trial])


def test_max_samples_stop_condition_update_limit_and_include_pruned():
    stop_condition = MaxSamplesStopCondition(max_samples=4, include_pruned=False)
    trials = [
        _make_trial_with_examples("t1", 2, status=TrialStatus.PRUNED),
        _make_trial_with_examples("t2", 2),
    ]
    assert not stop_condition.should_stop(trials)

    stop_condition.update_limit(3)
    stop_condition.set_include_pruned(True)
    # Now the pruned trial should contribute toward the limit.
    assert stop_condition.should_stop(trials)
