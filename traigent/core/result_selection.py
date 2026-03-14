"""Utilities for selecting the best configuration from completed trials."""

# Traceability: CONC-Layer-Core CONC-Quality-Performance FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from traigent.api.types import TrialResult
from traigent.utils.objectives import classify_objective, is_minimization_objective

__all__ = [
    "SelectionResult",
    "select_best_configuration",
    "apply_tie_breaker",
]

# Type alias matching TVL models.py
TieBreaker = Literal["min_abs_deviation", "custom"]
ComparabilityMode = Literal["legacy", "warn", "strict"]
NO_RANKING_ELIGIBLE_TRIALS = "NO_RANKING_ELIGIBLE_TRIALS"


@dataclass(slots=True)
class SelectionResult:
    """Best-configuration selection output for orchestrator result building."""

    best_config: dict[str, Any]
    best_score: float | None
    session_summary: dict[str, Any] | None
    reason_code: str | None = None


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_trial_comparability(trial: TrialResult) -> dict[str, Any] | None:
    """Extract comparability payload from trial metadata."""
    metadata = getattr(trial, "metadata", {}) or {}
    comparability = metadata.get("comparability")
    if isinstance(comparability, dict):
        return comparability
    return None


def _infer_total_examples_from_trial(trial: TrialResult) -> int:
    """Infer total example count from trial metadata when possible."""
    metadata = getattr(trial, "metadata", {}) or {}
    attempted = _coerce_int(metadata.get("examples_attempted"))
    if attempted is not None and attempted >= 0:
        return attempted

    example_results = metadata.get("example_results")
    if isinstance(example_results, list):
        return len(example_results)

    evaluation_result = metadata.get("evaluation_result")
    if evaluation_result is not None:
        total_examples = _coerce_int(getattr(evaluation_result, "total_examples", None))
        if total_examples is not None and total_examples >= 0:
            return total_examples
    return 0


def _resolve_from_primary_match(
    comparability: dict[str, Any],
    total_examples: int,
) -> tuple[int, int, float]:
    """Resolve coverage when the comparability primary matches the objective."""
    primary_present = (
        _coerce_int(comparability.get("examples_with_primary_metric")) or 0
    )
    coverage_ratio = _coerce_float(comparability.get("coverage_ratio"))
    if coverage_ratio is None and total_examples > 0:
        coverage_ratio = primary_present / total_examples
    return primary_present, total_examples, float(coverage_ratio or 0.0)


def _lookup_per_metric_coverage(
    per_metric: dict[str, Any],
    objective_key: str,
    objective_key_lower: str,
    total_examples: int,
) -> tuple[int, int, float] | None:
    """Look up coverage for a metric in per_metric_coverage, returning None if absent."""
    metric_coverage = per_metric.get(objective_key)
    if metric_coverage is None:
        metric_coverage = next(
            (
                value
                for key, value in per_metric.items()
                if isinstance(key, str) and key.lower() == objective_key_lower
            ),
            None,
        )
    if not isinstance(metric_coverage, dict):
        return None
    present = _coerce_int(metric_coverage.get("present")) or 0
    metric_total = _coerce_int(metric_coverage.get("total"))
    if metric_total is not None and metric_total >= 0:
        total_examples = metric_total
    ratio = _coerce_float(metric_coverage.get("ratio"))
    if ratio is None and total_examples > 0:
        ratio = present / total_examples
    return present, total_examples, float(ratio or 0.0)


def _resolve_primary_coverage(
    trial: TrialResult,
    comparability: dict[str, Any],
    primary_objective: str,
) -> tuple[int, int, float]:
    """Resolve primary-objective coverage from comparability payload."""
    total_examples = _coerce_int(comparability.get("total_examples"))
    if total_examples is None:
        total_examples = _infer_total_examples_from_trial(trial)
    total_examples = max(total_examples, 0)

    objective_key = primary_objective.strip()
    objective_key_lower = objective_key.lower()
    comparability_primary = (
        str(comparability.get("primary_objective", "")).strip().lower()
    )

    if comparability_primary and comparability_primary == objective_key_lower:
        return _resolve_from_primary_match(comparability, total_examples)

    per_metric = comparability.get("per_metric_coverage")
    if isinstance(per_metric, dict):
        result = _lookup_per_metric_coverage(
            per_metric, objective_key, objective_key_lower, total_examples
        )
        if result is not None:
            return result

    primary_value = trial.get_metric(objective_key)
    if primary_value is not None and total_examples > 0:
        return total_examples, total_examples, 1.0
    return 0, total_examples, 0.0


def _find_blocking_warning_codes(
    comparability: dict[str, Any],
    primary_objective: str,
) -> list[str] | None:
    """Return sorted blocking codes if the comparability payload contains any, else None."""
    comparability_primary = (
        str(comparability.get("primary_objective", "")).strip().lower()
    )
    if comparability_primary != primary_objective.strip().lower():
        return None
    codes = comparability.get("warning_codes")
    if not isinstance(codes, list):
        return None
    blocking = {"MCI-001", "MCI-002", "MCI-004", "MCI-007"}
    found = {str(c) for c in codes} & blocking
    return sorted(found) if found else None


def _is_ranking_eligible(
    trial: TrialResult,
    primary_objective: str,
    comparability_mode: ComparabilityMode,
) -> tuple[bool, bool, list[str]]:
    """Evaluate objective-aware ranking eligibility for a trial."""
    if not trial.is_successful:
        return False, False, ["MCI-007"]

    if comparability_mode == "legacy":
        metric = trial.get_metric(primary_objective)
        return metric is not None, False, (["MCI-004"] if metric is None else [])

    comparability = _extract_trial_comparability(trial)
    if comparability is None:
        return False, True, ["UNKNOWN_COMPARABILITY"]

    objective_value = trial.get_metric(primary_objective)
    if objective_value is None:
        return False, False, ["MCI-004"]

    present, total, coverage = _resolve_primary_coverage(
        trial, comparability, primary_objective
    )
    if total <= 0:
        return False, False, ["MCI-001"]
    if present <= 0:
        return False, False, ["MCI-004"]
    if coverage < 1.0:
        return False, False, ["MCI-002"]

    objective_class = classify_objective(primary_objective)
    evaluation_mode = str(comparability.get("evaluation_mode", "unknown")).strip()
    if objective_class == "quality" and evaluation_mode != "evaluated":
        return False, False, ["MCI-004", "QUALITY_OBJECTIVE_REQUIRES_EVALUATED_MODE"]

    blocking = _find_blocking_warning_codes(comparability, primary_objective)
    if blocking is not None:
        return False, False, blocking

    return True, False, []


def _build_ranking_summary(
    *,
    total_input_trials: int,
    total_successful: int,
    non_successful_count: int,
    eligible_count: int,
    excluded_count: int,
    unknown_count: int,
    comparability_mode: ComparabilityMode,
) -> dict[str, Any]:
    return {
        "comparability_mode": comparability_mode,
        "total_input_trials": total_input_trials,
        "total_successful_trials": total_successful,
        "non_successful_count": non_successful_count,
        "eligible_count": eligible_count,
        "excluded_count": excluded_count,
        "unknown_count": unknown_count,
    }


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


def _secondary_metric_total(
    metrics: dict[str, Any],
    exclude_objective: str,
) -> float:
    """Sum secondary metrics, subtracting minimization objectives."""
    total = 0.0
    for name, value in metrics.items():
        if not isinstance(value, (int, float)) or name == exclude_objective:
            continue
        if is_minimization_objective(name):
            total -= float(value)
        else:
            total += float(value)
    return total


def _apply_min_abs_deviation(
    trials: list[TrialResult],
    objective: str,
    band_target: float | None,
) -> TrialResult:
    """Select trial with minimum absolute deviation from target or best stability."""
    if band_target is not None:

        def deviation(t: TrialResult) -> float:
            value = t.get_metric(objective)
            return float("inf") if value is None else abs(value - band_target)

        return min(trials, key=deviation)

    def secondary_score(t: TrialResult) -> tuple[float, str]:
        return (_secondary_metric_total(t.metrics or {}, objective), t.trial_id or "")

    return max(trials, key=secondary_score)


def _select_best_single_trial(
    successful_trials: list[TrialResult],
    primary_objective: str,
    tie_breakers: dict[str, TieBreaker],
    band_target: float | None,
    ranking_summary: dict[str, Any],
) -> SelectionResult:
    """Select best trial without aggregation, applying tie-breakers."""
    minimization = is_minimization_objective(primary_objective)
    chooser = min if minimization else max

    scored_trials = [
        (t, t.get_metric(primary_objective))
        for t in successful_trials
        if t.get_metric(primary_objective) is not None
    ]
    if not scored_trials:
        return SelectionResult(
            best_config={},
            best_score=None,
            session_summary={
                "reason_code": NO_RANKING_ELIGIBLE_TRIALS,
                "ranking": ranking_summary,
            },
            reason_code=NO_RANKING_ELIGIBLE_TRIALS,
        )

    best_score = chooser(score for _, score in scored_trials if score is not None)

    # Find all trials with the best score (potential ties)
    tied_trials = [trial for trial, score in scored_trials if score == best_score]

    # Apply tie-breaker if there are ties
    if len(tied_trials) > 1 and tie_breakers:
        best_trial = apply_tie_breaker(
            tied_trials, tie_breakers, primary_objective, band_target=band_target
        )
    else:
        best_trial = tied_trials[0] if tied_trials else successful_trials[0]

    return SelectionResult(
        best_config=best_trial.config or {},
        best_score=best_trial.get_metric(primary_objective),
        session_summary={"ranking": ranking_summary},
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

    if tie_breaker != "min_abs_deviation":
        return tied_entries[0]

    if band_target is not None:

        def deviation(entry: dict[str, Any]) -> float:
            value = _compute_mean_metrics(entry).get(primary_objective, 0.0)
            return abs(value - band_target)

        return min(tied_entries, key=deviation)

    def agg_secondary_score(entry: dict[str, Any]) -> float:
        return _secondary_metric_total(_compute_mean_metrics(entry), primary_objective)

    return max(tied_entries, key=agg_secondary_score)


def _select_best_aggregated(
    aggregated: dict[str, dict[str, Any]],
    primary_objective: str,
    tie_breakers: dict[str, TieBreaker],
    band_target: float | None,
    ranking_summary: dict[str, Any],
) -> SelectionResult:
    """Select best configuration from aggregated results with tie-breaking."""
    if not aggregated:
        return SelectionResult(
            best_config={},
            best_score=None,
            session_summary={
                "reason_code": NO_RANKING_ELIGIBLE_TRIALS,
                "ranking": ranking_summary,
            },
            reason_code=NO_RANKING_ELIGIBLE_TRIALS,
        )

    minimization = is_minimization_objective(primary_objective)

    def score(entry: dict[str, Any]) -> float:
        value = _compute_mean_metrics(entry).get(primary_objective)
        if value is None:
            return float("inf") if minimization else float("-inf")
        return value

    # Find the best score
    all_scores = [score(entry) for entry in aggregated.values()]
    all_scores = [s for s in all_scores if s not in (float("inf"), float("-inf"))]
    if not all_scores:
        return SelectionResult(
            best_config={},
            best_score=None,
            session_summary={
                "reason_code": NO_RANKING_ELIGIBLE_TRIALS,
                "ranking": ranking_summary,
            },
            reason_code=NO_RANKING_ELIGIBLE_TRIALS,
        )
    best_score_val = min(all_scores) if minimization else max(all_scores)

    # Find all entries with the best score (ties)
    tied_entries = [
        entry
        for entry in aggregated.values()
        if _compute_mean_metrics(entry).get(primary_objective) == best_score_val
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
        "ranking": ranking_summary,
    }

    return SelectionResult(
        best_config=best_entry["config"] or {},
        best_score=best_metrics.get(primary_objective),
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
    comparability_mode: ComparabilityMode = "warn",
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
    trial_list = list(trials)
    successful_trials = [t for t in trial_list if t.is_successful]
    non_successful_count = len(trial_list) - len(successful_trials)
    if not successful_trials:
        return SelectionResult(
            best_config={},
            best_score=None,
            session_summary={
                "reason_code": NO_RANKING_ELIGIBLE_TRIALS,
                "ranking": _build_ranking_summary(
                    total_input_trials=len(trial_list),
                    total_successful=0,
                    non_successful_count=non_successful_count,
                    eligible_count=0,
                    excluded_count=0,
                    unknown_count=0,
                    comparability_mode=comparability_mode,
                ),
            },
            reason_code=NO_RANKING_ELIGIBLE_TRIALS,
        )

    eligible_trials: list[TrialResult] = []
    excluded_count = 0
    unknown_count = 0
    for trial in successful_trials:
        eligible, unknown, _codes = _is_ranking_eligible(
            trial, primary_objective, comparability_mode
        )
        if eligible:
            eligible_trials.append(trial)
            continue
        excluded_count += 1
        if unknown:
            unknown_count += 1

    ranking_summary = _build_ranking_summary(
        total_input_trials=len(trial_list),
        total_successful=len(successful_trials),
        non_successful_count=non_successful_count,
        eligible_count=len(eligible_trials),
        excluded_count=excluded_count,
        unknown_count=unknown_count,
        comparability_mode=comparability_mode,
    )

    if not eligible_trials:
        return SelectionResult(
            best_config={},
            best_score=None,
            session_summary={
                "reason_code": NO_RANKING_ELIGIBLE_TRIALS,
                "ranking": ranking_summary,
            },
            reason_code=NO_RANKING_ELIGIBLE_TRIALS,
        )

    tie_breakers = tie_breakers or {}

    if not aggregate_configs:
        return _select_best_single_trial(
            eligible_trials,
            primary_objective,
            tie_breakers,
            band_target,
            ranking_summary,
        )

    aggregated = _aggregate_trials(eligible_trials, set(config_space_keys))
    return _select_best_aggregated(
        aggregated,
        primary_objective,
        tie_breakers,
        band_target,
        ranking_summary,
    )
