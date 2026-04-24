"""Utilities for constructing and maintaining optimization stop conditions."""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from traigent.api.types import TrialResult

if TYPE_CHECKING:
    from traigent.core.cost_enforcement import CostEnforcer
from traigent.core.objectives import ObjectiveSchema
from traigent.core.stop_conditions import (
    CostLimitStopCondition,
    HypervolumeConvergenceStopCondition,
    MaxSamplesStopCondition,
    MaxTrialsStopCondition,
    MetricLimitStopCondition,
    PlateauAfterNStopCondition,
    StopCondition,
)

__all__ = ["StopConditionManager"]


class StopConditionManager:
    """Owns construction and evaluation of reusable stop conditions."""

    def __init__(
        self,
        *,
        max_trials: int | None,
        max_samples: int | None,
        samples_include_pruned: bool,
        plateau_window: int | None,
        plateau_epsilon: float | None,
        objective_schema: ObjectiveSchema | None,
        metric_limit: float | None,
        metric_name: str | None,
        metric_include_pruned: bool,
    ) -> None:
        self._conditions: list[StopCondition] = []

        self._samples_include_pruned = samples_include_pruned

        if max_trials and max_trials > 0:
            self._conditions.append(MaxTrialsStopCondition(max_trials))

        if max_samples and max_samples > 0:
            self._conditions.append(
                MaxSamplesStopCondition(
                    max_samples=max_samples,
                    include_pruned=self._samples_include_pruned,
                )
            )

        if plateau_window and plateau_window > 0:
            if objective_schema is None:
                raise ValueError(
                    "plateau_window configured but no objective schema available"
                )
            epsilon = plateau_epsilon if plateau_epsilon is not None else 1e-6
            self._conditions.append(
                PlateauAfterNStopCondition(
                    window_size=plateau_window,
                    epsilon=epsilon,
                    objective_schema=objective_schema,
                )
            )

        if metric_limit is not None:
            if not metric_name:
                raise ValueError("metric_name is required when metric_limit is set")
            self._conditions.append(
                MetricLimitStopCondition(
                    limit=metric_limit,
                    metric_name=metric_name,
                    include_pruned=metric_include_pruned,
                )
            )

    @property
    def conditions(self) -> tuple[StopCondition, ...]:
        return tuple(self._conditions)

    def reset(self) -> None:
        for condition in self._conditions:
            condition.reset()

    def update_max_trials(self, value: int | None) -> None:
        max_idx = None
        for idx, condition in enumerate(self._conditions):
            if isinstance(condition, MaxTrialsStopCondition):
                max_idx = idx
                break

        if value and value > 0:
            if max_idx is not None:
                self._conditions[max_idx] = MaxTrialsStopCondition(value)
            else:
                self._conditions.append(MaxTrialsStopCondition(value))
        elif max_idx is not None:
            self._conditions.pop(max_idx)

    def update_max_samples(self, value: int | None) -> None:
        sample_idx = None
        for idx, condition in enumerate(self._conditions):
            if isinstance(condition, MaxSamplesStopCondition):
                sample_idx = idx
                break

        if value and value > 0:
            if sample_idx is not None:
                existing = self._conditions[sample_idx]
                if isinstance(existing, MaxSamplesStopCondition):
                    existing.update_limit(value)
                else:  # defensive
                    self._conditions[sample_idx] = MaxSamplesStopCondition(
                        max_samples=value,
                        include_pruned=self._samples_include_pruned,
                    )
            else:
                self._conditions.append(
                    MaxSamplesStopCondition(
                        max_samples=value,
                        include_pruned=self._samples_include_pruned,
                    )
                )
        elif sample_idx is not None:
            self._conditions.pop(sample_idx)

    def update_samples_include_pruned(self, include_pruned: bool) -> None:
        self._samples_include_pruned = include_pruned
        for condition in self._conditions:
            if isinstance(condition, MaxSamplesStopCondition):
                condition.set_include_pruned(include_pruned)

    def should_stop(self, trials: Iterable[TrialResult]) -> tuple[bool, str | None]:
        for condition in self._conditions:
            if condition.should_stop(trials):
                reason = getattr(condition, "reason", condition.__class__.__name__)
                return True, reason
        return False, None

    def register_cost_limit_condition(
        self, cost_enforcer: CostEnforcer
    ) -> CostLimitStopCondition:
        """Register a cost limit stop condition using the shared CostEnforcer.

        Args:
            cost_enforcer: Shared CostEnforcer instance for cost tracking.

        Returns:
            The registered CostLimitStopCondition for reference.
        """
        condition = CostLimitStopCondition(cost_enforcer)
        self._conditions.append(condition)
        return condition

    def add_condition(self, condition: StopCondition) -> None:
        """Add a custom stop condition.

        Args:
            condition: The stop condition to add.
        """
        self._conditions.append(condition)

    def add_convergence_condition(
        self,
        window: int,
        threshold: float,
        objective_names: list[str],
        directions: list[str],
        reference_point: list[float] | None = None,
    ) -> HypervolumeConvergenceStopCondition:
        """Add a hypervolume convergence stop condition.

        This condition stops optimization when hypervolume improvement
        falls below the threshold for a sliding window of trials.

        Args:
            window: Number of trials to consider for convergence detection.
            threshold: Minimum hypervolume improvement required to continue.
            objective_names: Names of the objectives being optimized.
            directions: Optimization direction for each objective.
            reference_point: Optional reference point for hypervolume calculation.

        Returns:
            The registered condition for reference.
        """
        condition = HypervolumeConvergenceStopCondition(
            window=window,
            threshold=threshold,
            objective_names=objective_names,
            directions=directions,
            reference_point=reference_point,
        )
        self._conditions.append(condition)
        return condition
