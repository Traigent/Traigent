import pytest

import traigent.core.stop_conditions as stop_conditions
from traigent.api.types import TrialResult, TrialStatus
from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema
from traigent.core.stop_conditions import (
    BudgetStopCondition,
    HypervolumeConvergenceStopCondition,
    MaxSamplesStopCondition,
    MaxTrialsStopCondition,
    MetricLimitStopCondition,
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


def test_max_trials_stop_condition_ignores_abandoned_trials():
    stop_condition = MaxTrialsStopCondition(max_trials=2)

    abandoned = _make_trial("t-abandoned", {"accuracy": 0.1})
    abandoned.metadata = {"abandoned": True}

    first_executed = _make_trial("t1", {"accuracy": 0.5})
    assert not stop_condition.should_stop([abandoned, first_executed])

    second_executed = _make_trial("t2", {"accuracy": 0.6})
    assert stop_condition.should_stop([abandoned, first_executed, second_executed])


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


def test_metric_limit_stop_condition_triggers_on_named_metric():
    stop_condition = MetricLimitStopCondition(limit=0.2, metric_name="total_cost")

    history = [
        _make_trial("t1", {"total_cost": 0.05}),
        _make_trial("t2", {"total_cost": 0.10}),
    ]
    assert not stop_condition.should_stop(history)

    history.append(_make_trial("t3", {"total_cost": 0.06}))
    assert stop_condition.should_stop(history)


def test_metric_limit_stop_condition_uses_total_cost_metadata_fallback():
    trial = _make_trial("t1", {"accuracy": 1.0})
    trial.metadata = {"total_example_cost": 0.3}

    stop_condition = MetricLimitStopCondition(limit=0.25, metric_name="total_cost")
    assert stop_condition.should_stop([trial])


def test_metric_limit_stop_condition_uses_cost_fallback_for_total_cost():
    trial = _make_trial("t1", {"cost": 0.3})

    stop_condition = MetricLimitStopCondition(limit=0.25, metric_name="total_cost")
    assert stop_condition.should_stop([trial])


def test_metric_limit_stop_condition_invalid_limit():
    with pytest.raises(ValueError):
        MetricLimitStopCondition(limit=0.0, metric_name="tokens")

    with pytest.raises(ValueError):
        MetricLimitStopCondition(limit=-1, metric_name="tokens")

    with pytest.raises(ValueError):
        MetricLimitStopCondition(limit=1.0, metric_name="")


def test_metric_limit_stop_condition_respects_include_pruned():
    trial = _make_trial("t1", {"total_cost": 0.2})
    trial.status = TrialStatus.PRUNED

    stop_condition = MetricLimitStopCondition(
        limit=0.1,
        metric_name="total_cost",
        include_pruned=False,
    )
    assert not stop_condition.should_stop([trial])


def test_metric_limit_stop_condition_requires_mandatory_metric():
    stop_condition = MetricLimitStopCondition(limit=0.1, metric_name="total_cost")

    with pytest.raises(ValueError, match="Mandatory metric 'total_cost' missing"):
        stop_condition.should_stop([_make_trial("t1", {"accuracy": 0.9})])


def test_metric_limit_stop_condition_rejects_non_numeric_metric():
    stop_condition = MetricLimitStopCondition(limit=0.1, metric_name="total_cost")

    with pytest.raises(ValueError, match="is not numeric"):
        stop_condition.should_stop([_make_trial("t1", {"total_cost": "free"})])


def test_budget_stop_condition_alias_warns_and_uses_metric_limit_reason():
    with pytest.warns(DeprecationWarning, match="BudgetStopCondition"):
        stop_condition = BudgetStopCondition(budget=0.1, metric_name="tokens")

    assert stop_condition.reason == "metric_limit"
    assert stop_condition.should_stop([_make_trial("t1", {"tokens": 0.1})])


def test_budget_stop_condition_without_metric_warns_to_use_cost_limit():
    with pytest.warns(DeprecationWarning) as warnings_record:
        stop_condition = BudgetStopCondition(budget=0.1)

    warning_messages = [str(warning.message) for warning in warnings_record]
    assert any("BudgetStopCondition" in msg for msg in warning_messages)
    assert any("money spend control, use cost_limit" in msg for msg in warning_messages)

    assert stop_condition.reason == "metric_limit"
    assert stop_condition.should_stop([_make_trial("t1", {"cost": 0.1})])


def test_stop_conditions_exports_are_explicit():
    assert set(stop_conditions.__all__) == {
        "BudgetStopCondition",
        "CostLimitStopCondition",
        "HypervolumeConvergenceStopCondition",
        "MaxSamplesStopCondition",
        "MaxTrialsStopCondition",
        "MetricLimitStopCondition",
        "PlateauAfterNStopCondition",
        "StopCondition",
    }
    for name in stop_conditions.__all__:
        assert hasattr(stop_conditions, name)


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


def test_max_samples_stop_condition_uses_example_results_fallback():
    stop_condition = MaxSamplesStopCondition(max_samples=3)
    trial = _make_trial("t1", {"accuracy": 1.0})
    trial.example_results = [object(), object(), object()]

    assert stop_condition.should_stop([trial])


def test_hypervolume_convergence_stop_condition_triggers_on_stagnation():
    stop_condition = HypervolumeConvergenceStopCondition(
        window=2,
        threshold=0.01,
        objective_names=["accuracy"],
        directions=["maximize"],
    )

    trials = [
        _make_trial("t1", {"accuracy": 0.5}),
        _make_trial("t2", {"accuracy": 0.5}),
        _make_trial("t3", {"accuracy": 0.5}),
    ]

    assert stop_condition.should_stop(trials)
    assert stop_condition.reason == "convergence"


def test_hypervolume_convergence_stop_condition_reset_clears_state():
    stop_condition = HypervolumeConvergenceStopCondition(
        window=2,
        threshold=0.01,
        objective_names=["accuracy"],
        directions=["maximize"],
    )

    assert stop_condition.should_stop(
        [
            _make_trial("t1", {"accuracy": 0.5}),
            _make_trial("t2", {"accuracy": 0.5}),
            _make_trial("t3", {"accuracy": 0.5}),
        ]
    )

    stop_condition.reset()
    assert not stop_condition.should_stop([_make_trial("t4", {"accuracy": 0.5})])
