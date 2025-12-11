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
    """Stop once a maximum number of trials has been reached."""

    reason = "max_trials"

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

        if isinstance(trials, Sequence):
            count = len(trials)
        else:
            count = sum(1 for _ in trials)
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
