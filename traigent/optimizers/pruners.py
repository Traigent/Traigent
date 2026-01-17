"""Custom Optuna pruners tailored to Traigent workloads."""

# Traceability: CONC-Layer-Core CONC-Quality-Performance CONC-Quality-Reliability FUNC-OPT-ALGORITHMS REQ-OPT-ALG-004 SYNC-OptimizationFlow

from __future__ import annotations

import json
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from statistics import NormalDist

import optuna

from traigent.utils.logging import get_logger

logger = get_logger(__name__)


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


@dataclass(slots=True)
class StatisticalInferiorityPrunerConfig:
    """Configuration for :class:`StatisticalInferiorityPruner`.

    Defaults are tuned for statistical robustness:
    - confidence=0.95: 95% confidence level for bounds
    - min_samples_per_config=3: Minimum trials per config before comparison
    - warmup_trials=5: Minimum total trials before pruning starts
    """

    confidence: float = 0.95
    min_samples_per_config: int = 3
    warmup_trials: int = 5


class StatisticalInferiorityPruner(optuna.pruners.BasePruner):
    """Prune trials for configs whose UCB falls below the best config's LCB.

    This pruner uses confidence intervals to determine if a configuration is
    statistically inferior. It computes confidence bounds for each config
    based on historical trial values, and prunes trials when their config's
    upper confidence bound (UCB) is below the best config's lower confidence
    bound (LCB).

    This is useful for avoiding wasted computation on configurations that are
    unlikely to be competitive, while continuing to explore promising configs.

    Uses stdlib-only normal approximation for confidence bounds (no scipy).

    Note:
        This pruner only prunes at the START of a trial (step 0) based on
        historical data. It does not prune mid-trial based on intermediate
        values.
    """

    def __init__(
        self,
        *,
        confidence: float = 0.95,
        min_samples_per_config: int = 3,
        warmup_trials: int = 5,
    ) -> None:
        """Initialize the statistical inferiority pruner.

        Args:
            confidence: Confidence level for bounds (0-1). Default 0.95.
            min_samples_per_config: Min completed trials per config before
                computing bounds. Default 3.
            warmup_trials: Min total completed trials before any pruning.
                Default 5.
        """
        self._validate_confidence(confidence)
        self._validate_min_samples(min_samples_per_config)

        self._confidence = confidence
        self._min_samples = min_samples_per_config
        self._warmup_trials = warmup_trials

    @staticmethod
    def _validate_confidence(confidence: float) -> None:
        """Validate confidence parameter."""
        if confidence <= 0 or confidence >= 1:
            raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    @staticmethod
    def _validate_min_samples(min_samples: int) -> None:
        """Validate min_samples_per_config parameter."""
        if min_samples < 2:
            raise ValueError(f"min_samples_per_config must be >= 2, got {min_samples}")

    def _config_hash(self, params: dict) -> str:
        """Create stable hash from trial params dict."""
        return json.dumps(params, sort_keys=True, default=str)

    def _compute_bounds(self, values: list[float]) -> tuple[float, float]:
        """Return (LCB, UCB) using normal approximation (stdlib-only).

        Args:
            values: List of objective values for a config.

        Returns:
            Tuple of (lower_confidence_bound, upper_confidence_bound).
        """
        n = len(values)
        mean = statistics.mean(values)
        if n < 2:
            return (mean, mean)  # No variance estimate possible
        std = statistics.stdev(values)
        stderr = std / (n**0.5)
        # Normal approximation: z-score for confidence level
        z = NormalDist().inv_cdf((1 + self._confidence) / 2)
        margin = z * stderr
        return (mean - margin, mean + margin)

    def _group_values_by_config(
        self, completed: list[optuna.trial.FrozenTrial]
    ) -> dict[str, list[float]]:
        """Group completed trial values by their config hash."""
        config_values: dict[str, list[float]] = {}
        for t in completed:
            config_key = self._config_hash(t.params)
            if config_key not in config_values:
                config_values[config_key] = []
            config_values[config_key].append(float(t.value))
        return config_values

    def _find_best_bound(
        self, config_values: dict[str, list[float]], maximize: bool
    ) -> float | None:
        """Find the best LCB (maximize) or best UCB (minimize) among configs."""
        best_bound: float | None = None
        for values in config_values.values():
            if len(values) < self._min_samples:
                continue
            lcb, ucb = self._compute_bounds(values)
            bound = lcb if maximize else ucb
            if best_bound is None:
                best_bound = bound
            elif maximize and lcb > best_bound:
                best_bound = lcb
            elif not maximize and ucb < best_bound:
                best_bound = ucb
        return best_bound

    def _is_inferior(
        self,
        current_values: list[float],
        best_bound: float,
        maximize: bool,
        config_key: str,
    ) -> bool:
        """Check if current config is statistically inferior to best."""
        lcb, ucb = self._compute_bounds(current_values)
        if maximize and ucb < best_bound:
            logger.info(
                "Config %s... pruned: UCB=%.4f < best_LCB=%.4f",
                config_key[:40],
                ucb,
                best_bound,
            )
            return True
        if not maximize and lcb > best_bound:
            logger.info(
                "Config %s... pruned: LCB=%.4f > best_UCB=%.4f",
                config_key[:40],
                lcb,
                best_bound,
            )
            return True
        return False

    def prune(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> bool:
        """Determine if trial should be pruned based on statistical inferiority.

        Pruning only happens at trial start (step 0) based on historical data.

        Args:
            study: The Optuna study.
            trial: The current trial to potentially prune.

        Returns:
            True if trial should be pruned (config is statistically inferior).
        """
        # Only prune at the start of a trial (before any steps)
        last_step = trial.last_step
        if last_step is not None and last_step > 0:
            return False

        # Determine direction (skip multi-objective) - must check BEFORE
        # accessing .value on trials (which raises RuntimeError in multi-objective)
        # Optuna raises RuntimeError when accessing .direction on multi-objective studies
        try:
            direction = study.direction
        except RuntimeError:
            # Multi-objective study - not supported
            return False
        maximize = direction == optuna.study.StudyDirection.MAXIMIZE

        # Get all completed trials (safe now that we know it's single-objective)
        completed = _completed_trials_with_values(study.trials)
        if len(completed) < self._warmup_trials:
            return False

        # Group values and find best bound
        config_values = self._group_values_by_config(completed)
        best_bound = self._find_best_bound(config_values, maximize)
        if best_bound is None:
            return False

        # Check current config
        config_key = self._config_hash(trial.params)
        current_values = config_values.get(config_key, [])
        if len(current_values) < self._min_samples:
            return False

        return self._is_inferior(current_values, best_bound, maximize, config_key)


__all__ = [
    "CeilingPruner",
    "CeilingPrunerConfig",
    "StatisticalInferiorityPruner",
    "StatisticalInferiorityPrunerConfig",
]
