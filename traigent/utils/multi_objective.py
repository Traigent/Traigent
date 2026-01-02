"""Multi-objective optimization utilities for Traigent."""

# Traceability: CONC-Layer-Core CONC-Quality-Performance CONC-Quality-Maintainability FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from ..api.types import TrialResult, TrialStatus
from ..core.objectives import ObjectiveSchema

_COMPLETED_STATUS_TOKENS = {
    TrialStatus.COMPLETED.value.lower(),
}
_COMPLETED_STATUS_TOKENS.add(getattr(TrialStatus.COMPLETED, "name", "").lower())


def _is_completed_status(status: Any) -> bool:
    """Return True if a trial status represents completion."""
    if isinstance(status, TrialStatus):
        normalized = status.value
    else:
        normalized = status

    if isinstance(normalized, str):
        return normalized.lower() in _COMPLETED_STATUS_TOKENS

    return False


@dataclass
class ParetoPoint:
    """Represents a point on the Pareto front."""

    config: dict[str, Any]
    objectives: dict[str, float]
    trial: TrialResult

    def dominates(
        self, other: ParetoPoint, maximize: dict[str, bool], epsilon: float = 1e-10
    ) -> bool:
        """Check if this point dominates another point with epsilon tolerance.

        Args:
            other: Other Pareto point
            maximize: Dictionary indicating whether to maximize each objective
            epsilon: Tolerance for numerical comparison

        Returns:
            True if this point dominates the other
        """
        better_in_at_least_one = False

        for obj_name, obj_value in self.objectives.items():
            if obj_name not in other.objectives:
                continue

            other_value = other.objectives[obj_name]
            should_maximize = maximize.get(obj_name, True)

            # Use epsilon for near-equality check
            diff = obj_value - other_value

            if should_maximize:
                if diff < -epsilon:  # obj_value significantly less than other_value
                    return False  # Other is better in this objective
                elif diff > epsilon:  # obj_value significantly greater than other_value
                    better_in_at_least_one = True
            else:  # minimize
                if diff > epsilon:  # obj_value significantly greater than other_value
                    return False  # Other is better in this objective
                elif diff < -epsilon:  # obj_value significantly less than other_value
                    better_in_at_least_one = True

        return better_in_at_least_one


class ParetoFrontCalculator:
    """Calculates Pareto fronts for multi-objective optimization."""

    def __init__(
        self,
        maximize: dict[str, bool] | None = None,
        objective_schema: ObjectiveSchema | None = None,
        *,
        random_seed: int | None = 0,
        monte_carlo_samples: int = 10000,
    ) -> None:
        """Initialize Pareto front calculator.

        Args:
            maximize: Dictionary indicating whether to maximize each objective.
                     Defaults to maximizing all objectives.
            objective_schema: ObjectiveSchema with orientations (overrides maximize).
            random_seed: Seed for Monte Carlo calculations. None for non-deterministic.
            monte_carlo_samples: Number of samples for Monte Carlo hypervolume.
        """
        if objective_schema is not None:
            # Extract orientations from schema
            self.maximize: dict[str, Any] = {}
            for obj_def in objective_schema.objectives:
                self.maximize[obj_def.name] = obj_def.orientation == "maximize"
        else:
            self.maximize = maximize or {}

        self.num_samples = max(1, monte_carlo_samples)
        self._rng = (
            random.Random(random_seed) if random_seed is not None else random.Random()
        )

    def calculate_pareto_front(
        self, trials: list[TrialResult], objectives: list[str]
    ) -> list[ParetoPoint]:
        """Calculate the Pareto front from a set of trials.

        Args:
            trials: List of completed trials
            objectives: List of objective names to consider

        Returns:
            List of Pareto-optimal points
        """
        # Convert trials to Pareto points
        points = []
        for trial in trials:
            if _is_completed_status(getattr(trial, "status", None)) and trial.metrics:
                # Extract objective values
                point_objectives = {}
                for obj in objectives:
                    if obj in trial.metrics:
                        point_objectives[obj] = trial.metrics[obj]

                if point_objectives:  # Only include points with at least one objective
                    points.append(
                        ParetoPoint(
                            config=trial.config,
                            objectives=point_objectives,
                            trial=trial,
                        )
                    )

        if not points:
            return []

        # Find non-dominated points
        pareto_front = []

        for i, point in enumerate(points):
            is_dominated = False

            for j, other_point in enumerate(points):
                if i != j and other_point.dominates(point, self.maximize):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_front.append(point)

        return pareto_front

    def calculate_hypervolume(
        self,
        pareto_front: list[ParetoPoint],
        reference_point: dict[str, float] | None = None,
    ) -> float:
        """Calculate hypervolume indicator for Pareto front.

        Args:
            pareto_front: List of Pareto-optimal points
            reference_point: Reference point for hypervolume calculation

        Returns:
            Hypervolume value
        """
        if not pareto_front:
            return 0.0

        if len(pareto_front[0].objectives) > 2:
            # For > 2D, use approximation
            return self._approximate_hypervolume(pareto_front, reference_point)

        return self._exact_hypervolume_2d(pareto_front, reference_point)

    def _exact_hypervolume_2d(
        self, pareto_front: list[ParetoPoint], reference_point: dict[str, float] | None
    ) -> float:
        """Calculate exact hypervolume for 2D case."""
        if len(pareto_front) == 0:
            return 0.0

        objectives = list(pareto_front[0].objectives.keys())
        if len(objectives) != 2:
            return 0.0

        obj1, obj2 = objectives

        # Set reference point
        if reference_point is None:
            ref1 = min(point.objectives[obj1] for point in pareto_front) - 1
            ref2 = min(point.objectives[obj2] for point in pareto_front) - 1
        else:
            ref1 = reference_point.get(obj1, 0)
            ref2 = reference_point.get(obj2, 0)

        # Sort points by first objective
        sorted_points = sorted(
            pareto_front,
            key=lambda p: p.objectives[obj1],
            reverse=self.maximize.get(obj1, True),
        )

        hypervolume = 0.0
        prev_obj1 = ref1

        for point in sorted_points:
            curr_obj1 = point.objectives[obj1]
            curr_obj2 = point.objectives[obj2]

            width = abs(curr_obj1 - prev_obj1)
            height = abs(curr_obj2 - ref2)

            hypervolume += width * height
            prev_obj1 = curr_obj1

        return hypervolume

    def _approximate_hypervolume(
        self, pareto_front: list[ParetoPoint], reference_point: dict[str, float] | None
    ) -> float:
        """Calculate approximate hypervolume for high-dimensional case using Monte Carlo."""
        if not pareto_front:
            return 0.0

        objectives = list(pareto_front[0].objectives.keys())

        # Set reference point and bounds
        if reference_point is None:
            ref_point = {
                obj: min(point.objectives[obj] for point in pareto_front) - 1
                for obj in objectives
            }
        else:
            ref_point = reference_point

        max_point = {
            obj: max(point.objectives[obj] for point in pareto_front) + 1
            for obj in objectives
        }

        # Monte Carlo sampling
        dominated_count = 0
        for _ in range(self.num_samples):
            # Generate random point in search space
            sample_point = {
                obj: self._rng.uniform(ref_point[obj], max_point[obj])
                for obj in objectives
            }

            # Check if any Pareto point dominates this sample
            for pareto_point in pareto_front:
                if self._point_dominates_sample(pareto_point, sample_point):
                    dominated_count += 1
                    break

        # Calculate volume of search space
        total_volume = 1.0
        for obj in objectives:
            total_volume *= abs(max_point[obj] - ref_point[obj])

        # Approximate hypervolume
        hypervolume = (dominated_count / self.num_samples) * total_volume
        return hypervolume

    def _point_dominates_sample(
        self, pareto_point: ParetoPoint, sample_point: dict[str, float]
    ) -> bool:
        """Check if a Pareto point dominates a sample point."""
        for obj, value in pareto_point.objectives.items():
            if obj in sample_point:
                sample_value = sample_point[obj]
                should_maximize = self.maximize.get(obj, True)

                if should_maximize:
                    if value < sample_value:
                        return False
                else:
                    if value > sample_value:
                        return False

        return True


class MultiObjectiveMetrics:
    """Calculates various multi-objective optimization metrics."""

    @staticmethod
    def calculate_diversity_metric(pareto_front: list[ParetoPoint]) -> float:
        """Calculate diversity metric for Pareto front.

        Args:
            pareto_front: List of Pareto-optimal points

        Returns:
            Diversity metric (higher is better)
        """
        if len(pareto_front) < 2:
            return 0.0

        objectives = list(pareto_front[0].objectives.keys())

        # Calculate distances between all pairs of points
        distances = []
        for i in range(len(pareto_front)):
            for j in range(i + 1, len(pareto_front)):
                point1 = pareto_front[i]
                point2 = pareto_front[j]

                # Euclidean distance in objective space
                distance = 0.0
                for obj in objectives:
                    if obj in point1.objectives and obj in point2.objectives:
                        diff = point1.objectives[obj] - point2.objectives[obj]
                        distance += diff * diff

                distances.append(math.sqrt(distance))

        if not distances:
            return 0.0

        # Return average distance as diversity metric
        return sum(distances) / len(distances)

    @staticmethod
    def calculate_convergence_metric(
        pareto_front: list[ParetoPoint],
        true_pareto_front: list[ParetoPoint] | None = None,
    ) -> float:
        """Calculate convergence metric for Pareto front.

        Args:
            pareto_front: Calculated Pareto front
            true_pareto_front: True Pareto front (if known)

        Returns:
            Convergence metric (lower is better)
        """
        if true_pareto_front is None:
            # Without true front, use approximation based on front spread
            if len(pareto_front) < 2:
                return float("inf")

            objectives = list(pareto_front[0].objectives.keys())

            # Calculate range in each objective
            total_range = 0.0
            for obj in objectives:
                values = [
                    point.objectives[obj]
                    for point in pareto_front
                    if obj in point.objectives
                ]
                if len(values) > 1:
                    total_range += max(values) - min(values)

            # Smaller ranges indicate better convergence
            return 1.0 / (total_range + 1e-10)

        # Calculate average distance from calculated front to true front
        total_distance = 0.0

        for calc_point in pareto_front:
            min_distance = float("inf")

            for true_point in true_pareto_front:
                distance = 0.0
                common_objectives = set(calc_point.objectives.keys()) & set(
                    true_point.objectives.keys()
                )

                for obj in common_objectives:
                    diff = calc_point.objectives[obj] - true_point.objectives[obj]
                    distance += diff * diff

                distance = math.sqrt(distance)
                min_distance = min(min_distance, distance)

            total_distance += min_distance

        return total_distance / len(pareto_front) if pareto_front else float("inf")


def scalarize_objectives(
    objectives: dict[str, float],
    weights: dict[str, float],
    minimize_objectives: list[str] | None = None,
    objective_schema: ObjectiveSchema | None = None,
) -> float:
    """Scalarize multiple objectives using weighted sum.

    Args:
        objectives: Dictionary of objective values
        weights: Dictionary of objective weights
        minimize_objectives: List of objective names that should be minimized (inverted).
                           If None, no auto-detection is performed (backward compatibility).
        objective_schema: ObjectiveSchema with orientations and weights (overrides other params).

    Returns:
        Scalarized objective value
    """
    if not objectives:
        return 0.0

    # Use ObjectiveSchema if provided (overrides other params)
    if objective_schema is not None:
        weights = {}
        minimize_objectives = []

        for obj_def in objective_schema.objectives:
            weights[obj_def.name] = obj_def.weight
            if obj_def.orientation == "minimize":
                minimize_objectives.append(obj_def.name)

    # Only auto-detect minimization objectives if explicitly requested
    elif minimize_objectives is None:
        minimize_objectives = []

    total_score = 0.0
    total_weight = 0.0

    for obj_name, obj_value in objectives.items():
        # Skip None values
        if obj_value is None:
            continue

        weight = weights.get(obj_name, 1.0)

        # For minimization objectives, we invert the value by using (1 - obj_value)
        # assuming objectives are normalized between 0 and 1, or use negative weight
        if obj_name in minimize_objectives:
            # Use negative objective value for minimization
            total_score += weight * (-obj_value)
        else:
            # Normal maximization
            total_score += weight * obj_value

        total_weight += weight

    # Handle case where total weight is zero (use equal weights)
    if total_weight == 0:
        # Use equal weights for all non-None objectives
        valid_objectives = {k: v for k, v in objectives.items() if v is not None}
        if not valid_objectives:
            return 0.0

        # Apply minimization logic for equal weights too
        total_value = 0.0
        for obj_name, obj_value in valid_objectives.items():
            if obj_name in minimize_objectives:
                total_value += -obj_value
            else:
                total_value += obj_value

        return total_value / len(valid_objectives)

    return total_score / total_weight


def normalize_objectives(
    trials: list[TrialResult], objectives: list[str]
) -> dict[str, tuple[float, float]]:
    """Calculate normalization ranges for objectives.

    Args:
        trials: List of trials
        objectives: List of objective names

    Returns:
        Dictionary mapping objective names to (min, max) tuples
    """
    ranges = {}

    for obj in objectives:
        values = []
        for trial in trials:
            if (
                _is_completed_status(getattr(trial, "status", None))
                and trial.metrics
                and obj in trial.metrics
            ):
                values.append(trial.metrics[obj])

        if values:
            min_value = min(values)
            max_value = max(values)

            if math.isclose(min_value, max_value, rel_tol=0.0, abs_tol=1e-12):
                padding = max(1e-9, abs(min_value) * 1e-6)
                ranges[obj] = (min_value - padding, max_value + padding)
            else:
                ranges[obj] = (min_value, max_value)
        else:
            ranges[obj] = (0.0, 1.0)

    return ranges
