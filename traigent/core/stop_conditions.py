"""Reusable stop conditions for optimization orchestration."""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from traigent.api.types import TrialResult, TrialStatus
from traigent.core.objectives import ObjectiveSchema
from traigent.core.utils import extract_examples_attempted
from traigent.utils.logging import get_logger

if TYPE_CHECKING:
    from traigent.core.cost_enforcement import CostEnforcer

logger = get_logger(__name__)


class StopCondition(ABC):
    """Interface for reusable stop criteria."""

    reason: str = "condition"

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal tracking state."""

    @abstractmethod
    def should_stop(self, trials: Iterable[TrialResult]) -> bool:
        """Return ``True`` when the condition dictates stopping."""


class MaxTrialsStopCondition(StopCondition):
    """Stop once a maximum number of executed trials has been reached.

    Counts all trials except those explicitly marked as abandoned in metadata.
    This keeps cache/cap-abandoned trials from reducing the execution budget,
    while still counting pruned trials produced during execution.
    """

    reason = "max_trials"

    _ABANDONED_METADATA_KEY = "abandoned"

    def __init__(self, max_trials: int | None) -> None:
        if max_trials is None:
            self._max_trials = None
            return

        if not isinstance(max_trials, int) or max_trials <= 0:
            raise ValueError("max_trials must be a positive integer")

        self._max_trials = max_trials

    def reset(self) -> None:  # noqa: D401 - interface requirement
        return

    def should_stop(self, trials: Iterable[TrialResult]) -> bool:
        if self._max_trials is None:
            return False

        def is_abandoned(trial: TrialResult) -> bool:
            metadata = getattr(trial, "metadata", None) or {}
            return bool(metadata.get(self._ABANDONED_METADATA_KEY, False))

        count = sum(1 for trial in trials if not is_abandoned(trial))
        return count >= self._max_trials


class PlateauAfterNStopCondition(StopCondition):
    """Stop when the best weighted score plateaus for ``window_size`` trials."""

    reason = "plateau"

    def __init__(
        self,
        *,
        window_size: int,
        epsilon: float,
        objective_schema: ObjectiveSchema | None,
    ) -> None:
        if objective_schema is None:
            raise ValueError("objective_schema is required for plateau detection")
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative")

        self._window_size = window_size
        self._epsilon = float(epsilon)
        self._schema = objective_schema
        self._history: deque[float] = deque(maxlen=self._window_size)
        self._best_score: float | None = None
        self._last_index = 0

    def reset(self) -> None:
        self._history.clear()
        self._best_score = None
        self._last_index = 0

    def should_stop(self, trials: Iterable[TrialResult]) -> bool:
        if isinstance(trials, Sequence):
            trial_seq = trials
        else:
            trial_seq = list(trials)

        new_trials = trial_seq[self._last_index :]
        if not new_trials:
            return False

        for trial in new_trials:
            score = self._schema.compute_weighted_score(trial.metrics)
            if score is None:
                continue

            if self._best_score is None or score > self._best_score + self._epsilon:
                self._best_score = score

            if self._best_score is not None:
                self._history.append(self._best_score)

        self._last_index = len(trial_seq)

        if len(self._history) < self._window_size:
            return False

        delta = self._history[-1] - self._history[0]
        return abs(delta) <= self._epsilon


class BudgetStopCondition(StopCondition):
    """Stop when the cumulative metric exceeds a specified budget."""

    reason = "budget"

    def __init__(
        self,
        *,
        budget: float,
        metric_name: str = "total_cost",
        include_pruned: bool = True,
    ) -> None:
        if budget is None or float(budget) <= 0:
            raise ValueError("budget must be a positive number")
        if not metric_name:
            raise ValueError("metric_name must be provided")

        self._budget = float(budget)
        self._metric = metric_name
        self._include_pruned = include_pruned
        self._running_total = 0.0
        self._last_index = 0

    def reset(self) -> None:  # noqa: D401 - interface requirement
        self._running_total = 0.0
        self._last_index = 0

    def should_stop(self, trials: Iterable[TrialResult]) -> bool:
        if isinstance(trials, Sequence):
            trial_seq = trials
        else:
            trial_seq = list(trials)

        new_trials = trial_seq[self._last_index :]
        if not new_trials:
            return self._running_total >= self._budget

        for trial in new_trials:
            if not self._include_pruned and trial.status == TrialStatus.PRUNED:
                continue

            metrics = trial.metrics or {}
            value = metrics.get(self._metric)

            if value is None and self._metric == "total_cost":
                # Default to historical per-trial totals recorded in metadata.
                value = (trial.metadata or {}).get("total_example_cost")

            if value is None:
                raise ValueError(
                    f"Mandatory metric '{self._metric}' missing for trial {trial.trial_id}"
                )

            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Metric '{self._metric}' for trial {trial.trial_id} is not numeric: {value!r}"
                ) from exc

            self._running_total += numeric_value

            if self._running_total >= self._budget:
                self._last_index = len(trial_seq)
                return True

        self._last_index = len(trial_seq)
        return self._running_total >= self._budget


class MaxSamplesStopCondition(StopCondition):
    """Stop once cumulative ``examples_attempted`` reaches a threshold."""

    reason = "max_samples"

    def __init__(
        self,
        *,
        max_samples: int | None,
        include_pruned: bool = True,
    ) -> None:
        if max_samples is None:
            self._max_samples = None
        else:
            if not isinstance(max_samples, int) or max_samples <= 0:
                raise ValueError("max_samples must be a positive integer or None")
            self._max_samples = max_samples
        self._include_pruned = include_pruned
        self._total_attempted = 0
        self._last_index = 0

    def reset(self) -> None:  # noqa: D401
        self._total_attempted = 0
        self._last_index = 0

    def update_limit(self, value: int | None) -> None:
        if value is None:
            self._max_samples = None
        else:
            if not isinstance(value, int) or value <= 0:
                raise ValueError("max_samples must be a positive integer or None")
            self._max_samples = value
        self._total_attempted = 0
        self._last_index = 0

    def set_include_pruned(self, include_pruned: bool) -> None:
        self._include_pruned = include_pruned
        self._total_attempted = 0
        self._last_index = 0

    def should_stop(self, trials: Iterable[TrialResult]) -> bool:
        if self._max_samples is None:
            return False

        if isinstance(trials, Sequence):
            trial_seq = trials
        else:
            trial_seq = list(trials)

        new_trials = trial_seq[self._last_index :]
        if not new_trials:
            return self._total_attempted >= self._max_samples

        for trial in new_trials:
            if not self._include_pruned and trial.status == TrialStatus.PRUNED:
                continue

            attempted = extract_examples_attempted(
                trial, default=None, check_example_results=True
            )
            if attempted is not None:
                self._total_attempted += attempted

            if self._total_attempted >= self._max_samples:
                self._last_index = len(trial_seq)
                return True

        self._last_index = len(trial_seq)
        return self._total_attempted >= self._max_samples


class CostLimitStopCondition(StopCondition):
    """Stop when cost limit reached using shared CostEnforcer.

    This stop condition uses a shared CostEnforcer instance to avoid double
    counting costs. The enforcer tracks actual costs; this condition just
    checks the enforcer's state.

    Note:
        The CostEnforcer must be passed in and is expected to be tracking
        costs already (via track_cost() calls from the orchestrator).
    """

    reason = "cost_limit"

    def __init__(self, cost_enforcer: CostEnforcer) -> None:
        """Initialize with shared cost enforcer.

        Args:
            cost_enforcer: Shared CostEnforcer instance that tracks costs.
        """
        self._cost_enforcer = cost_enforcer

    def reset(self) -> None:
        """Reset is handled by the shared CostEnforcer, not here."""
        # Note: We don't reset the enforcer here because it's shared
        # and may be used across multiple stop condition checks
        pass

    def should_stop(self, trials: Iterable[TrialResult]) -> bool:
        """Check if cost limit has been reached.

        Args:
            trials: Trial results (not used - we check the enforcer directly).

        Returns:
            True if the cost limit has been reached.
        """
        # Cost is already tracked by the orchestrator - just check status
        return self._cost_enforcer.is_limit_reached

    def get_reason(self) -> str:
        """Get a descriptive reason for stopping.

        Returns:
            Human-readable description of why optimization stopped.
        """
        status = self._cost_enforcer.get_status()
        if status.unknown_cost_mode:
            return (
                f"Trial limit reached: {status.trial_count} trials "
                f"(cost unknown, fallback mode)"
            )
        return (
            f"Cost limit reached: ${status.accumulated_cost_usd:.2f} "
            f">= ${status.limit_usd:.2f} USD"
        )


class HypervolumeConvergenceStopCondition(StopCondition):
    """Stop when hypervolume improvement falls below threshold.

    This stop condition implements convergence detection based on hypervolume
    improvement as specified in TVL 0.9. It monitors the hypervolume indicator
    over a sliding window and triggers early stopping when improvement stagnates.

    The hypervolume indicator measures the volume of objective space dominated
    by the Pareto front. When this improvement falls below a threshold over
    multiple consecutive trials, optimization is considered converged.

    Example:
        ```python
        from traigent.core.stop_conditions import HypervolumeConvergenceStopCondition

        condition = HypervolumeConvergenceStopCondition(
            window=10,
            threshold=0.001,
            objective_names=["accuracy", "latency"],
            directions=["maximize", "minimize"],
            reference_point=[0.0, 1000.0],  # Worst case for each objective
        )
        ```

    Note:
        This is typically configured via TVL spec's exploration.convergence section:
        ```yaml
        exploration:
          convergence:
            metric: hypervolume_improvement
            window: 20
            threshold: 0.001
        ```
    """

    reason = "convergence"

    def __init__(
        self,
        *,
        window: int,
        threshold: float,
        objective_names: list[str],
        directions: list[str],
        reference_point: list[float] | None = None,
    ) -> None:
        """Initialize hypervolume convergence stop condition.

        Args:
            window: Number of trials to consider for convergence detection.
            threshold: Minimum hypervolume improvement required to continue.
                If improvement falls below this for `window` consecutive trials,
                optimization stops.
            objective_names: Names of the objectives being optimized.
            directions: Optimization direction for each objective ("maximize" or "minimize").
            reference_point: Reference point for hypervolume calculation. If None,
                uses worst observed values with a margin.

        Raises:
            ValueError: If arguments are invalid.
        """
        if window <= 0:
            raise ValueError("window must be a positive integer")
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        if len(objective_names) != len(directions):
            raise ValueError("objective_names and directions must have same length")
        if not objective_names:
            raise ValueError("At least one objective is required")

        self._window = window
        self._threshold = threshold
        self._objective_names = objective_names
        self._directions = directions
        self._reference_point = reference_point

        # Internal state
        self._hypervolume_history: deque[float] = deque(maxlen=window)
        self._pareto_front: list[list[float]] = []
        self._last_index = 0
        self._computed_reference: list[float] | None = None

    def reset(self) -> None:
        """Reset convergence tracking state."""
        self._hypervolume_history.clear()
        self._pareto_front = []
        self._last_index = 0
        self._computed_reference = None

    def should_stop(self, trials: Iterable[TrialResult]) -> bool:
        """Check if hypervolume has converged.

        Args:
            trials: All trial results so far.

        Returns:
            True if hypervolume improvement has fallen below threshold
            for the entire sliding window.
        """
        if isinstance(trials, Sequence):
            trial_seq = trials
        else:
            trial_seq = list(trials)

        # Process new trials
        new_trials = trial_seq[self._last_index :]
        if not new_trials:
            return self._check_convergence()

        for trial in new_trials:
            if trial.status != TrialStatus.COMPLETED:
                continue

            # Extract objective values
            point = self._extract_point(trial)
            if point is None:
                continue

            # Update reference point if needed
            self._update_reference_point(point)

            # Calculate hypervolume improvement
            improvement = self._calculate_improvement(point)
            self._hypervolume_history.append(improvement)

            # Update Pareto front
            self._update_pareto_front(point)

        self._last_index = len(trial_seq)
        return self._check_convergence()

    def _extract_point(self, trial: TrialResult) -> list[float] | None:
        """Extract objective values from trial metrics."""
        point = []
        for name in self._objective_names:
            value = trial.metrics.get(name)
            if value is None:
                return None
            point.append(float(value))
        return point

    def _update_reference_point(self, point: list[float]) -> None:
        """Update reference point based on observed values."""
        if self._reference_point is not None:
            self._computed_reference = list(self._reference_point)
            return

        if self._computed_reference is None:
            # Initialize with first point + margin
            self._computed_reference = []
            for i, (val, direction) in enumerate(zip(point, self._directions)):
                if direction == "maximize":
                    # For maximize, reference should be below all observed values
                    self._computed_reference.append(val - abs(val) * 0.1 - 0.1)
                else:
                    # For minimize, reference should be above all observed values
                    self._computed_reference.append(val + abs(val) * 0.1 + 0.1)
        else:
            # Update to ensure reference is dominated by all observed points
            for i, (val, direction) in enumerate(zip(point, self._directions)):
                if direction == "maximize":
                    self._computed_reference[i] = min(
                        self._computed_reference[i], val - abs(val) * 0.1 - 0.1
                    )
                else:
                    self._computed_reference[i] = max(
                        self._computed_reference[i], val + abs(val) * 0.1 + 0.1
                    )

    def _calculate_improvement(self, point: list[float]) -> float:
        """Calculate hypervolume improvement from adding a point."""
        if self._computed_reference is None:
            return 0.0

        # Normalize points for hypervolume calculation
        # (flip minimize objectives so all are maximize)
        def normalize(p: list[float]) -> list[float]:
            result = []
            for val, direction in zip(p, self._directions):
                if direction == "minimize":
                    result.append(-val)
                else:
                    result.append(val)
            return result

        normalized_point = normalize(point)
        normalized_front = [normalize(p) for p in self._pareto_front]
        normalized_ref = normalize(self._computed_reference)

        # Calculate current hypervolume
        current_hv = self._simple_hypervolume(normalized_front, normalized_ref)

        # Check if point is dominated
        for front_point in normalized_front:
            if self._dominates(front_point, normalized_point):
                return 0.0

        # Add point and calculate new hypervolume
        new_front = [
            p for p in normalized_front if not self._dominates(normalized_point, p)
        ]
        new_front.append(normalized_point)
        new_hv = self._simple_hypervolume(new_front, normalized_ref)

        return max(0.0, new_hv - current_hv)

    def _dominates(self, a: list[float], b: list[float]) -> bool:
        """Check if point a dominates point b (all maximize)."""
        at_least_one_better = False
        for av, bv in zip(a, b):
            if av < bv:
                return False
            if av > bv:
                at_least_one_better = True
        return at_least_one_better

    def _simple_hypervolume(
        self, front: list[list[float]], reference: list[float]
    ) -> float:
        """Calculate hypervolume using simple 2D algorithm or approximation."""
        if not front:
            return 0.0

        n_obj = len(reference)

        if n_obj == 1:
            # 1D: just the range
            max_val = max(p[0] for p in front)
            return max(0.0, max_val - reference[0])

        if n_obj == 2:
            # 2D: exact algorithm
            return self._hypervolume_2d(front, reference)

        # Higher dimensions: use maximum individual contribution as lower bound
        # This is monotonic - adding non-dominated points cannot decrease the value
        # We use max contribution rather than sum to avoid double-counting overlap
        max_contribution = 0.0
        for point in front:
            volume = 1.0
            for pv, rv in zip(point, reference):
                volume *= max(0.0, pv - rv)
            max_contribution = max(max_contribution, volume)
        return max_contribution

    def _hypervolume_2d(
        self, front: list[list[float]], reference: list[float]
    ) -> float:
        """Calculate exact 2D hypervolume."""
        if not front:
            return 0.0

        # Filter points dominated by reference
        valid = [p for p in front if p[0] > reference[0] and p[1] > reference[1]]
        if not valid:
            return 0.0

        # Sort by first objective descending
        sorted_front = sorted(valid, key=lambda p: -p[0])

        hv = 0.0
        prev_y = reference[1]

        for point in sorted_front:
            if point[1] > prev_y:
                hv += (point[0] - reference[0]) * (point[1] - prev_y)
                prev_y = point[1]

        return hv

    def _update_pareto_front(self, point: list[float]) -> None:
        """Update the Pareto front with a new point."""

        # Normalize for comparison (flip minimize objectives)
        def normalize(p: list[float]) -> list[float]:
            result = []
            for val, direction in zip(p, self._directions):
                if direction == "minimize":
                    result.append(-val)
                else:
                    result.append(val)
            return result

        normalized_point = normalize(point)

        # Check if dominated by existing front
        for front_point in self._pareto_front:
            normalized_front = normalize(front_point)
            if self._dominates(normalized_front, normalized_point):
                return  # Point is dominated, don't add

        # Remove points dominated by new point
        new_front = []
        for front_point in self._pareto_front:
            normalized_front = normalize(front_point)
            if not self._dominates(normalized_point, normalized_front):
                new_front.append(front_point)

        new_front.append(point)
        self._pareto_front = new_front

    def _check_convergence(self) -> bool:
        """Check if convergence criterion is met."""
        if len(self._hypervolume_history) < self._window:
            return False

        # Check if all improvements in window are below threshold
        return all(imp <= self._threshold for imp in self._hypervolume_history)
