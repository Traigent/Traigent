"""Tests for HypervolumeConvergenceStopCondition."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from traigent.api.types import TrialResult, TrialStatus
from traigent.core.stop_conditions import HypervolumeConvergenceStopCondition

_trial_counter = 0


def make_trial(
    metrics: dict[str, float], status: TrialStatus = TrialStatus.COMPLETED
) -> TrialResult:
    """Create a minimal TrialResult for testing."""
    global _trial_counter
    _trial_counter += 1
    return TrialResult(
        trial_id=f"trial_{_trial_counter}",
        config={},
        metrics=metrics,
        status=status,
        duration=1.0,
        timestamp=datetime.now(UTC),
    )


class TestHypervolumeConvergenceStopCondition:
    """Tests for HypervolumeConvergenceStopCondition."""

    def test_initialization(self) -> None:
        """Test valid initialization."""
        condition = HypervolumeConvergenceStopCondition(
            window=5,
            threshold=0.01,
            objective_names=["accuracy", "latency"],
            directions=["maximize", "minimize"],
        )
        assert condition.reason == "convergence"

    def test_invalid_window_raises(self) -> None:
        """Test that invalid window raises ValueError."""
        with pytest.raises(ValueError, match="window must be a positive"):
            HypervolumeConvergenceStopCondition(
                window=0,
                threshold=0.01,
                objective_names=["accuracy"],
                directions=["maximize"],
            )

    def test_negative_threshold_raises(self) -> None:
        """Test that negative threshold raises ValueError."""
        with pytest.raises(ValueError, match="threshold must be non-negative"):
            HypervolumeConvergenceStopCondition(
                window=5,
                threshold=-0.01,
                objective_names=["accuracy"],
                directions=["maximize"],
            )

    def test_mismatched_lengths_raises(self) -> None:
        """Test that mismatched objective_names and directions raises ValueError."""
        with pytest.raises(ValueError, match="must have same length"):
            HypervolumeConvergenceStopCondition(
                window=5,
                threshold=0.01,
                objective_names=["accuracy", "latency"],
                directions=["maximize"],  # Missing one direction
            )

    def test_empty_objectives_raises(self) -> None:
        """Test that empty objectives raises ValueError."""
        with pytest.raises(ValueError, match="At least one objective"):
            HypervolumeConvergenceStopCondition(
                window=5,
                threshold=0.01,
                objective_names=[],
                directions=[],
            )

    def test_does_not_stop_before_window_filled(self) -> None:
        """Test that condition doesn't stop before window is filled."""
        condition = HypervolumeConvergenceStopCondition(
            window=5,
            threshold=0.01,
            objective_names=["score"],
            directions=["maximize"],
        )

        # Add fewer trials than window
        trials = [
            make_trial({"score": 0.5}),
            make_trial({"score": 0.5}),
            make_trial({"score": 0.5}),
        ]

        assert not condition.should_stop(trials)

    def test_stops_when_improvement_below_threshold(self) -> None:
        """Test that condition stops when improvement is consistently low."""
        condition = HypervolumeConvergenceStopCondition(
            window=3,
            threshold=0.01,
            objective_names=["score"],
            directions=["maximize"],
        )

        # Add trials with no improvement
        trials = [
            make_trial({"score": 0.5}),
            make_trial({"score": 0.5}),  # No improvement
            make_trial({"score": 0.5}),  # No improvement
            make_trial({"score": 0.5}),  # No improvement
        ]

        # After window is filled with zero improvements, should stop
        assert condition.should_stop(trials)

    def test_does_not_stop_with_improvement(self) -> None:
        """Test that condition doesn't stop when improvements occur."""
        condition = HypervolumeConvergenceStopCondition(
            window=3,
            threshold=0.01,
            objective_names=["score"],
            directions=["maximize"],
        )

        # Add trials with consistent improvement
        trials = [
            make_trial({"score": 0.5}),
            make_trial({"score": 0.6}),
            make_trial({"score": 0.7}),
            make_trial({"score": 0.8}),
        ]

        assert not condition.should_stop(trials)

    def test_reset_clears_state(self) -> None:
        """Test that reset clears internal state."""
        condition = HypervolumeConvergenceStopCondition(
            window=3,
            threshold=0.01,
            objective_names=["score"],
            directions=["maximize"],
        )

        # Build up some state
        trials = [
            make_trial({"score": 0.5}),
            make_trial({"score": 0.5}),
            make_trial({"score": 0.5}),
            make_trial({"score": 0.5}),
        ]
        condition.should_stop(trials)

        # Reset
        condition.reset()

        # Should not stop immediately after reset
        assert not condition.should_stop([make_trial({"score": 0.5})])

    def test_handles_missing_metrics(self) -> None:
        """Test that trials with missing metrics are skipped."""
        condition = HypervolumeConvergenceStopCondition(
            window=3,
            threshold=0.01,
            objective_names=["accuracy", "latency"],
            directions=["maximize", "minimize"],
        )

        trials = [
            make_trial({"accuracy": 0.5}),  # Missing latency
            make_trial({"latency": 100}),  # Missing accuracy
            make_trial({"accuracy": 0.5, "latency": 100}),
        ]

        # Should not crash and should not stop (not enough valid trials)
        assert not condition.should_stop(trials)

    def test_handles_failed_trials(self) -> None:
        """Test that failed trials are skipped."""
        condition = HypervolumeConvergenceStopCondition(
            window=3,
            threshold=0.01,
            objective_names=["score"],
            directions=["maximize"],
        )

        trials = [
            make_trial({"score": 0.5}),
            make_trial({"score": 0.5}, status=TrialStatus.FAILED),
            make_trial({"score": 0.5}, status=TrialStatus.PRUNED),
            make_trial({"score": 0.5}),
        ]

        # Only 2 valid trials, should not stop
        assert not condition.should_stop(trials)

    def test_two_objectives_maximize_minimize(self) -> None:
        """Test with two objectives: maximize accuracy, minimize latency."""
        condition = HypervolumeConvergenceStopCondition(
            window=3,
            threshold=0.001,
            objective_names=["accuracy", "latency"],
            directions=["maximize", "minimize"],
        )

        # Trials showing Pareto improvement
        trials = [
            make_trial({"accuracy": 0.7, "latency": 100}),
            make_trial({"accuracy": 0.8, "latency": 90}),  # Pareto improvement
            make_trial({"accuracy": 0.85, "latency": 85}),  # Pareto improvement
        ]

        assert not condition.should_stop(trials)

        # Add trials with no improvement
        trials.extend(
            [
                make_trial({"accuracy": 0.7, "latency": 110}),  # Dominated
                make_trial({"accuracy": 0.75, "latency": 95}),  # Dominated
                make_trial({"accuracy": 0.76, "latency": 94}),  # Dominated
            ]
        )

        # Window filled with zero improvements should stop
        assert condition.should_stop(trials)

    def test_with_reference_point(self) -> None:
        """Test with explicit reference point."""
        condition = HypervolumeConvergenceStopCondition(
            window=3,
            threshold=0.01,
            objective_names=["score"],
            directions=["maximize"],
            reference_point=[0.0],  # Explicit reference
        )

        trials = [
            make_trial({"score": 0.5}),
            make_trial({"score": 0.5}),
            make_trial({"score": 0.5}),
            make_trial({"score": 0.5}),
        ]

        assert condition.should_stop(trials)

    def test_incremental_processing(self) -> None:
        """Test that trials are processed incrementally."""
        condition = HypervolumeConvergenceStopCondition(
            window=2,
            threshold=0.01,
            objective_names=["score"],
            directions=["maximize"],
        )

        # First batch
        trials = [make_trial({"score": 0.5})]
        assert not condition.should_stop(trials)

        # Second batch (same list, more trials added)
        trials.append(make_trial({"score": 0.5}))
        assert not condition.should_stop(trials)

        # Third batch
        trials.append(make_trial({"score": 0.5}))
        assert condition.should_stop(trials)

    def test_single_objective_hypervolume(self) -> None:
        """Test hypervolume calculation with single objective (1D case)."""
        condition = HypervolumeConvergenceStopCondition(
            window=3,
            threshold=0.01,
            objective_names=["score"],
            directions=["maximize"],
            reference_point=[0.0],
        )

        # Add trials with improvement
        trials = [
            make_trial({"score": 0.3}),
            make_trial({"score": 0.5}),
            make_trial({"score": 0.7}),
        ]

        # 1D hypervolume should show improvement
        assert not condition.should_stop(trials)

        # Add more trials with no improvement
        trials.extend(
            [
                make_trial({"score": 0.6}),  # Less than best
                make_trial({"score": 0.65}),  # Less than best
                make_trial({"score": 0.68}),  # Less than best
            ]
        )

        # Should eventually converge
        assert condition.should_stop(trials)

    def test_three_objectives_approximate_hypervolume(self) -> None:
        """Test hypervolume calculation with 3+ objectives (approximation)."""
        condition = HypervolumeConvergenceStopCondition(
            window=3,
            threshold=0.001,
            objective_names=["accuracy", "latency", "cost"],
            directions=["maximize", "minimize", "minimize"],
        )

        # Add trials that show improvement
        trials = [
            make_trial({"accuracy": 0.7, "latency": 100, "cost": 10}),
            make_trial({"accuracy": 0.8, "latency": 90, "cost": 8}),
            make_trial({"accuracy": 0.85, "latency": 80, "cost": 7}),
        ]

        # Should not stop with improvement
        assert not condition.should_stop(trials)

        # Add dominated trials
        trials.extend(
            [
                make_trial({"accuracy": 0.6, "latency": 120, "cost": 12}),
                make_trial({"accuracy": 0.65, "latency": 110, "cost": 11}),
                make_trial({"accuracy": 0.68, "latency": 105, "cost": 10}),
            ]
        )

        # Should converge when all new trials are dominated
        assert condition.should_stop(trials)

    def test_non_sequence_iterable(self) -> None:
        """Test that non-sequence iterables are converted to list."""
        condition = HypervolumeConvergenceStopCondition(
            window=2,
            threshold=0.01,
            objective_names=["score"],
            directions=["maximize"],
        )

        # Create a generator (non-sequence iterable)
        def trial_generator():
            yield make_trial({"score": 0.5})
            yield make_trial({"score": 0.5})
            yield make_trial({"score": 0.5})

        # Should handle generator without error
        assert condition.should_stop(trial_generator())

    def test_no_new_trials_returns_current_state(self) -> None:
        """Test that calling should_stop with no new trials checks current state."""
        condition = HypervolumeConvergenceStopCondition(
            window=2,
            threshold=0.01,
            objective_names=["score"],
            directions=["maximize"],
        )

        # First call with some trials
        trials = [
            make_trial({"score": 0.5}),
            make_trial({"score": 0.5}),
            make_trial({"score": 0.5}),
        ]
        result1 = condition.should_stop(trials)

        # Second call with same trials (no new ones)
        result2 = condition.should_stop(trials)

        # Should return same result
        assert result1 == result2
        assert result1 is True  # Window filled with zero improvement

    def test_minimize_single_objective(self) -> None:
        """Test single objective with minimize direction."""
        condition = HypervolumeConvergenceStopCondition(
            window=2,
            threshold=0.01,
            objective_names=["cost"],
            directions=["minimize"],
        )

        # Add trials with improvement (lower is better for minimize)
        trials = [
            make_trial({"cost": 100}),
            make_trial({"cost": 80}),  # Improvement
            make_trial({"cost": 60}),  # Improvement
        ]

        assert not condition.should_stop(trials)

    def test_pareto_front_update_removes_dominated(self) -> None:
        """Test that adding a dominating point removes dominated points."""
        condition = HypervolumeConvergenceStopCondition(
            window=5,
            threshold=0.001,
            objective_names=["accuracy", "speed"],
            directions=["maximize", "maximize"],
        )

        # Add some points to the Pareto front
        trials = [
            make_trial({"accuracy": 0.5, "speed": 0.5}),
            make_trial({"accuracy": 0.6, "speed": 0.4}),
            make_trial({"accuracy": 0.4, "speed": 0.6}),
        ]
        condition.should_stop(trials)

        # Add a dominating point
        trials.append(make_trial({"accuracy": 0.7, "speed": 0.7}))  # Dominates all
        condition.should_stop(trials)

        # The Pareto front should now only have the dominating point
        assert len(condition._pareto_front) == 1
        assert condition._pareto_front[0] == [0.7, 0.7]

    def test_2d_hypervolume_calculation(self) -> None:
        """Test accurate 2D hypervolume calculation."""
        condition = HypervolumeConvergenceStopCondition(
            window=3,
            threshold=0.01,
            objective_names=["x", "y"],
            directions=["maximize", "maximize"],
            reference_point=[0.0, 0.0],
        )

        # Add specific points to test 2D hypervolume
        trials = [
            make_trial({"x": 0.3, "y": 0.7}),
            make_trial({"x": 0.5, "y": 0.5}),
            make_trial({"x": 0.7, "y": 0.3}),
        ]

        condition.should_stop(trials)

        # Verify Pareto front was updated correctly
        assert len(condition._pareto_front) >= 1

    def test_dominated_point_not_added_to_front(self) -> None:
        """Test that dominated points are not added to Pareto front."""
        condition = HypervolumeConvergenceStopCondition(
            window=5,
            threshold=0.01,
            objective_names=["accuracy", "latency"],
            directions=["maximize", "minimize"],
        )

        # Add a good point
        trials = [make_trial({"accuracy": 0.9, "latency": 50})]
        condition.should_stop(trials)

        # Add dominated points
        trials.extend(
            [
                make_trial({"accuracy": 0.7, "latency": 100}),  # Dominated
                make_trial({"accuracy": 0.8, "latency": 80}),  # Dominated
            ]
        )
        condition.should_stop(trials)

        # Pareto front should only have the dominating point
        assert len(condition._pareto_front) == 1
