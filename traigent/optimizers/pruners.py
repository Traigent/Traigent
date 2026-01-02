"""Custom Optuna pruners tailored to Traigent workloads."""

# Traceability: CONC-Layer-Core CONC-Quality-Performance CONC-Quality-Reliability FUNC-OPT-ALGORITHMS REQ-OPT-ALG-004 SYNC-OptimizationFlow

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import optuna


@dataclass(slots=True)
class CeilingPrunerConfig:
    """Configuration for :class:`CeilingPruner`.

    Defaults are tuned to balance exploration vs exploitation:
    - min_completed_trials=3: Gather enough baseline data before pruning
    - warmup_steps=5: Let trials warm up before making pruning decisions
    - epsilon=0.05: Allow 5% exploration margin to avoid overly aggressive pruning
    """

    min_completed_trials: int = 3
    warmup_steps: int = 5
    epsilon: float = 0.05  # 5% exploration margin
    cost_threshold: float | None = None


class CeilingPruner(optuna.pruners.BasePruner):
    """Prune trials whose optimistic ceiling falls behind the current best result.

    This pruner expects callers to report an optimistic estimate of the final objective value
    (for example the best-case accuracy achievable given remaining examples). As soon as the
    estimate sinks below the best completed trial (within ``epsilon``), the trial is pruned.

    Optionally supports absolute cost-based pruning when ``cost_threshold`` is set.

    Note:
        The ``epsilon`` parameter is an **absolute** value, not a percentage.
        For metrics on a 0-1 scale (e.g., accuracy), epsilon=0.05 means 5 percentage points.
        For metrics on a 0-100 scale, epsilon=0.05 would be negligible (0.05 points).
        Ensure your epsilon matches your metric scale.
    """

    def __init__(
        self,
        *,
        min_completed_trials: int = 3,
        warmup_steps: int = 5,
        epsilon: float = 0.05,
        cost_threshold: float | None = None,
    ) -> None:
        self._min_completed_trials = min_completed_trials
        self._warmup_steps = warmup_steps
        self._epsilon = epsilon
        self._cost_threshold = cost_threshold

    def prune(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> bool:
        last_step = trial.last_step
        if last_step is None or last_step < self._warmup_steps:
            return False

        latest = trial.intermediate_values.get(last_step)
        if latest is None:
            return False

        # Absolute cost threshold pruning.
        # Historically we only supported multi-objective payloads where cost was the
        # second entry. However, single-objective runs that optimise cost report a
        # scalar value, so we handle both representations here.
        if self._cost_threshold is not None:
            if isinstance(latest, (int, float)):
                if float(latest) > self._cost_threshold:
                    return True
            elif isinstance(latest, (list, tuple)):
                # Prefer Optuna's per-objective directions when available so that we
                # only apply the threshold to minimisation objectives (typically cost).
                directions = getattr(study, "directions", None)
                if directions:
                    for direction, value in zip(directions, latest, strict=False):
                        if (
                            direction == optuna.study.StudyDirection.MINIMIZE
                            and value is not None
                            and float(value) > self._cost_threshold
                        ):
                            return True
                elif len(latest) >= 2:
                    projected_cost = latest[1]
                    if (
                        projected_cost is not None
                        and float(projected_cost) > self._cost_threshold
                    ):
                        return True

        completed = _completed_trials_with_values(study.trials)
        if len(completed) < self._min_completed_trials:
            return False

        # Optuna exposes ``direction`` for single-objective studies and ``directions``
        # for multi-objective ones. Ceiling pruning is only well-defined for the
        # former, so we fall back to "no pruning" when multiple objectives are in
        # play (multi-objective studies currently skip intermediate reports).
        direction = getattr(study, "direction", None)
        if direction == optuna.study.StudyDirection.MAXIMIZE:
            best = max(t.value for t in completed)
            return float(latest) <= float(best) - self._epsilon

        if direction == optuna.study.StudyDirection.MINIMIZE:
            best = min(t.value for t in completed)
            return float(latest) >= float(best) + self._epsilon

        return False


def _completed_trials_with_values(
    trials: Iterable[optuna.trial.FrozenTrial],
) -> list[optuna.trial.FrozenTrial]:
    return [
        t
        for t in trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]


__all__ = ["CeilingPruner", "CeilingPrunerConfig"]
