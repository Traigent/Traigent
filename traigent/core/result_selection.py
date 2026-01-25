"""Utilities for selecting the best configuration from completed trials."""

# Traceability: CONC-Layer-Core CONC-Quality-Performance FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from traigent.api.types import TrialResult

__all__ = [
    "SelectionResult",
    "select_best_configuration",
    "apply_tie_breaker",
]

# Type alias matching TVL models.py
TieBreaker = Literal["min_abs_deviation", "custom"]


@dataclass(slots=True)
class SelectionResult:
    """Best-configuration selection output for orchestrator result building."""

    best_config: dict[str, Any]
    best_score: float
    session_summary: dict[str, Any] | None


def _is_minimization_objective(objective: str) -> bool:
    minimize_patterns = ("cost", "latency", "error", "loss", "time", "duration")
    objective_lower = objective.lower()
    return any(pattern in objective_lower for pattern in minimize_patterns)


def apply_tie_breaker(
    tied_trials: list[TrialResult],
    tie_breakers: dict[str, TieBreaker],
    primary_objective: str,
    *,
    band_target: float | None = None,
) -> TrialResult:
    """Apply tie-breaker rules to select among equally-scored trials.

    Tie-breaker strategies:
    - min_abs_deviation: Select trial with smallest deviation from band_target
      (for banded objectives) or best secondary metrics (maximization metrics
      are added, minimization metrics like cost/latency are subtracted).
    - custom: Reserved for user-defined tie-breaker logic (returns first)

    Args:
        tied_trials: List of trials with equal primary objective scores.
        tie_breakers: Dict mapping objective names to tie-breaker strategies.
        primary_objective: Name of the primary objective.
        band_target: Optional target value for banded objectives.

    Returns:
        The selected trial after applying tie-breaker rules.
    """
    if not tied_trials:
        raise ValueError("Cannot apply tie-breaker to empty trial list")

    if len(tied_trials) == 1:
        return tied_trials[0]

    # Get the tie-breaker for the primary objective
    tie_breaker = tie_breakers.get(primary_objective, "min_abs_deviation")

    if tie_breaker == "min_abs_deviation":
        return _apply_min_abs_deviation(tied_trials, primary_objective, band_target)

    # For "custom" or unknown, return the first trial
    return tied_trials[0]


def _apply_min_abs_deviation(
    trials: list[TrialResult],
    objective: str,
    band_target: float | None,
) -> TrialResult:
    """Select trial with minimum absolute deviation from target or best stability."""
    if band_target is not None:
        # For banded objectives: pick closest to target
        def deviation(t: TrialResult) -> float:
            value = t.get_metric(objective, 0.0) or 0.0
            return abs(value - band_target)

        return min(trials, key=deviation)

    # For non-banded: prefer trials with better secondary metrics
    # Use trial_id as final tie-breaker for determinism
    def secondary_score(t: TrialResult) -> tuple[float, str]:
        # Sum of all positive metrics (normalized approach)
        total = 0.0
        for name, value in (t.metrics or {}).items():
            if isinstance(value, (int, float)) and name != objective:
                # For cost/latency metrics, invert (lower is better)
                if _is_minimization_objective(name):
                    total -= float(value)
                else:
                    total += float(value)
        return (total, t.trial_id or "")

    return max(trials, key=secondary_score)


def _select_best_single_trial(
    successful_trials: list[TrialResult],
    primary_objective: str,
    tie_breakers: dict[str, TieBreaker],
    band_target: float | None,
) -> SelectionResult:
    """Select best trial without aggregation, applying tie-breakers."""
    minimization = _is_minimization_objective(primary_objective)
    chooser = min if minimization else max

    # Find the best score
    best_score = chooser(
        t.get_metric(primary_objective, 0.0) or 0.0 for t in successful_trials
    )

    # Find all trials with the best score (potential ties)
    tied_trials = [
        t
        for t in successful_trials
        if (t.get_metric(primary_objective, 0.0) or 0.0) == best_score
    ]

    # Apply tie-breaker if there are ties
    if len(tied_trials) > 1 and tie_breakers:
        best_trial = apply_tie_breaker(
            tied_trials, tie_breakers, primary_objective, band_target=band_target
        )
    else:
        best_trial = tied_trials[0] if tied_trials else successful_trials[0]

    return SelectionResult(
        best_config=best_trial.config or {},
        best_score=best_trial.get_metric(primary_objective, 0.0) or 0.0,
        session_summary=None,
    )


def _aggregate_trials(
    successful_trials: list[TrialResult],
    config_space_keys: set[str],
) -> dict[str, dict[str, Any]]:
    """Aggregate trials by configuration hash."""
    from traigent.utils.hashing import generate_config_hash

    replicate_counters: dict[str, int] = {}
    aggregated: dict[str, dict[str, Any]] = {}

    for trial in successful_trials:
        cfg = trial.config or {}
        filtered_cfg = (
            {k: cfg.get(k) for k in config_space_keys} if config_space_keys else cfg
        )
        config_hash = generate_config_hash(filtered_cfg)

        replicate_counters[config_hash] = replicate_counters.get(config_hash, 0) + 1
        trial.metadata = getattr(trial, "metadata", {})
        trial.metadata["replicate_index"] = replicate_counters[config_hash]

        entry = aggregated.setdefault(
            config_hash,
            {"config": trial.config, "metrics_sum": {}, "count": 0},
        )

        for metric_name, metric_value in (trial.metrics or {}).items():
            if metric_name == "examples_attempted":
                continue
            if isinstance(metric_value, (int, float)):
                entry["metrics_sum"][metric_name] = entry["metrics_sum"].get(
                    metric_name, 0.0
                ) + float(metric_value)

        entry["count"] += 1

    return aggregated


def _compute_mean_metrics(entry: dict[str, Any]) -> dict[str, float]:
    """Compute mean metrics from an aggregated entry."""
    count = entry["count"] or 1
    return {
        metric: value / count for metric, value in entry.get("metrics_sum", {}).items()
    }


def _apply_aggregated_tie_breaker(
    tied_entries: list[dict[str, Any]],
    tie_breakers: dict[str, TieBreaker],
    primary_objective: str,
    band_target: float | None,
) -> dict[str, Any]:
    """Apply tie-breaker rules to aggregated entries with equal primary scores.

    Args:
        tied_entries: List of aggregated entries with equal primary objective scores.
        tie_breakers: Dict mapping objective names to tie-breaker strategies.
        primary_objective: Name of the primary objective.
        band_target: Optional target value for banded objectives.

    Returns:
        The selected entry after applying tie-breaker rules.
    """
    if not tied_entries:
        raise ValueError("Cannot apply tie-breaker to empty entry list")

    if len(tied_entries) == 1:
        return tied_entries[0]

    tie_breaker = tie_breakers.get(primary_objective, "min_abs_deviation")

    if tie_breaker == "min_abs_deviation":
        if band_target is not None:
            # Pick entry closest to band target
            def deviation(entry: dict[str, Any]) -> float:
                value = _compute_mean_metrics(entry).get(primary_objective, 0.0)
                return abs(value - band_target)

            return min(tied_entries, key=deviation)

        # For non-banded: use secondary metrics
        def secondary_score(entry: dict[str, Any]) -> float:
            total = 0.0
            for name, value in _compute_mean_metrics(entry).items():
                if name != primary_objective:
                    if _is_minimization_objective(name):
                        total -= value
                    else:
                        total += value
            return total

        return max(tied_entries, key=secondary_score)

    # For "custom" or unknown, return the first entry
    return tied_entries[0]


def _select_best_aggregated(
    aggregated: dict[str, dict[str, Any]],
    primary_objective: str,
    tie_breakers: dict[str, TieBreaker],
    band_target: float | None,
) -> SelectionResult:
    """Select best configuration from aggregated results with tie-breaking."""
    if not aggregated:
        return SelectionResult(best_config={}, best_score=0.0, session_summary=None)

    minimization = _is_minimization_objective(primary_objective)

    def score(entry: dict[str, Any]) -> float:
        return _compute_mean_metrics(entry).get(primary_objective, 0.0)

    # Find the best score
    all_scores = [score(entry) for entry in aggregated.values()]
    best_score_val = min(all_scores) if minimization else max(all_scores)

    # Find all entries with the best score (ties)
    tied_entries = [
        entry for entry in aggregated.values() if score(entry) == best_score_val
    ]

    # Apply tie-breaking for aggregated entries
    if len(tied_entries) > 1 and tie_breakers:
        best_entry = _apply_aggregated_tie_breaker(
            tied_entries, tie_breakers, primary_objective, band_target
        )
    else:
        best_entry = tied_entries[0]

    best_metrics = _compute_mean_metrics(best_entry)

    allowed_unprefixed = {"accuracy", "latency", "cost", "custom_metric"}
    sanitized_metrics: dict[str, float] = {}
    for metric_name, metric_value in best_metrics.items():
        target_key = (
            metric_name if metric_name in allowed_unprefixed else f"run_{metric_name}"
        )
        sanitized_metrics[target_key] = metric_value

    session_summary = {
        "selection_mode": "aggregated_mean",
        "primary_objective": primary_objective,
        "samples_per_config": {
            key: entry["count"] for key, entry in aggregated.items()
        },
        "metrics": sanitized_metrics,
        "sanitized": True,
    }

    return SelectionResult(
        best_config=best_entry["config"] or {},
        best_score=best_metrics.get(primary_objective, 0.0),
        session_summary=session_summary,
    )


def select_best_configuration(
    trials: Iterable[TrialResult],
    primary_objective: str,
    *,
    config_space_keys: Iterable[str],
    aggregate_configs: bool,
    tie_breakers: dict[str, TieBreaker] | None = None,
    band_target: float | None = None,
) -> SelectionResult:
    """Return the configuration that best satisfies the primary objective.

    Args:
        trials: Iterable of completed trial results.
        primary_objective: Name of the primary objective to optimize.
        config_space_keys: Keys that define the configuration space.
        aggregate_configs: Whether to aggregate results by config.
        tie_breakers: Optional dict mapping objectives to tie-breaker strategies.
            From TVL 0.9 promotion_policy.tie_breakers.
        band_target: Optional target value for banded objectives.

    Returns:
        SelectionResult with best configuration and score.
    """
    successful_trials = [t for t in trials if t.is_successful]
    if not successful_trials:
        return SelectionResult(best_config={}, best_score=0.0, session_summary=None)

    tie_breakers = tie_breakers or {}

    if not aggregate_configs:
        return _select_best_single_trial(
            successful_trials, primary_objective, tie_breakers, band_target
        )

    aggregated = _aggregate_trials(successful_trials, set(config_space_keys))
    return _select_best_aggregated(
        aggregated, primary_objective, tie_breakers, band_target
    )
