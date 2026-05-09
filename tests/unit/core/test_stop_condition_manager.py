from datetime import UTC, datetime

import pytest

from traigent.api.types import TrialResult, TrialStatus
from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema
from traigent.core.stop_condition_manager import StopConditionManager
from traigent.core.stop_conditions import (
    HypervolumeConvergenceStopCondition,
    MaxSamplesStopCondition,
    MaxTrialsStopCondition,
    MetricLimitStopCondition,
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
        "metric_limit": None,
        "metric_name": None,
        "metric_include_pruned": True,
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


def test_metric_limit_requires_metric_name():
    with pytest.raises(ValueError, match="metric_name is required"):
        base_manager(metric_limit=10.0)


def test_metric_limit_condition_is_registered():
    manager = base_manager(metric_limit=10.0, metric_name="total_tokens")
    assert any(isinstance(c, MetricLimitStopCondition) for c in manager.conditions)


def make_trial(
    metrics: dict[str, float], status: TrialStatus = TrialStatus.COMPLETED
) -> TrialResult:
    """Create a TrialResult for testing."""
    return TrialResult(
        trial_id=f"trial_{id(metrics)}",
        config={},
        metrics=metrics,
        status=status,
        duration=1.0,
        timestamp=datetime.now(UTC),
    )


class TestAddConvergenceCondition:
    """Tests for add_convergence_condition method."""

    def test_add_convergence_condition_creates_hypervolume_condition(self) -> None:
        """Test that add_convergence_condition creates and adds a HypervolumeConvergenceStopCondition."""
        manager = base_manager()
        assert len(manager.conditions) == 0

        condition = manager.add_convergence_condition(
            window=5,
            threshold=0.01,
            objective_names=["accuracy", "latency"],
            directions=["maximize", "minimize"],
        )

        assert isinstance(condition, HypervolumeConvergenceStopCondition)
        assert any(
            isinstance(c, HypervolumeConvergenceStopCondition)
            for c in manager.conditions
        )
        assert len(manager.conditions) == 1

    def test_add_convergence_condition_with_reference_point(self) -> None:
        """Test that add_convergence_condition accepts reference_point parameter."""
        manager = base_manager()

        condition = manager.add_convergence_condition(
            window=3,
            threshold=0.001,
            objective_names=["score"],
            directions=["maximize"],
            reference_point=[0.0],
        )

        assert condition._reference_point == [0.0]

    def test_convergence_condition_triggers_stop(self) -> None:
        """Test that convergence condition properly triggers stop."""
        manager = base_manager()
        manager.add_convergence_condition(
            window=2,
            threshold=0.01,
            objective_names=["accuracy"],
            directions=["maximize"],
        )

        # Add trials with no improvement
        trials = [
            make_trial({"accuracy": 0.5}),
            make_trial({"accuracy": 0.5}),
            make_trial({"accuracy": 0.5}),
        ]

        should_stop, reason = manager.should_stop(trials)
        assert should_stop is True
        assert reason == "convergence"

    def test_convergence_condition_does_not_stop_with_improvement(self) -> None:
        """Test that convergence condition doesn't stop when there's improvement."""
        manager = base_manager()
        manager.add_convergence_condition(
            window=3,
            threshold=0.01,
            objective_names=["accuracy"],
            directions=["maximize"],
        )

        # Add trials with improvement
        trials = [
            make_trial({"accuracy": 0.5}),
            make_trial({"accuracy": 0.6}),
            make_trial({"accuracy": 0.7}),
        ]

        should_stop, reason = manager.should_stop(trials)
        assert should_stop is False
        assert reason is None

    def test_multiple_conditions_convergence_and_max_trials(self) -> None:
        """Test that multiple stop conditions work together."""
        manager = base_manager(max_trials=10)
        manager.add_convergence_condition(
            window=2,
            threshold=0.01,
            objective_names=["accuracy"],
            directions=["maximize"],
        )

        # There should be 2 conditions now
        assert len(manager.conditions) == 2
        assert any(isinstance(c, MaxTrialsStopCondition) for c in manager.conditions)
        assert any(
            isinstance(c, HypervolumeConvergenceStopCondition)
            for c in manager.conditions
        )

    def test_reset_clears_convergence_condition_state(self) -> None:
        """Test that reset clears the convergence condition state."""
        manager = base_manager()
        manager.add_convergence_condition(
            window=2,
            threshold=0.01,
            objective_names=["accuracy"],
            directions=["maximize"],
        )

        # Add trials that would trigger stop
        trials = [
            make_trial({"accuracy": 0.5}),
            make_trial({"accuracy": 0.5}),
            make_trial({"accuracy": 0.5}),
        ]
        should_stop, _ = manager.should_stop(trials)
        assert should_stop is True

        # Reset and verify state is cleared
        manager.reset()

        # Now with fewer trials it should not stop
        new_trials = [make_trial({"accuracy": 0.5})]
        should_stop, _ = manager.should_stop(new_trials)
        assert should_stop is False
