"""Utilities for selecting the best configuration from completed trials."""

# Traceability: CONC-Layer-Core CONC-Quality-Performance FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from traigent.api.types import TrialResult
from traigent.utils.objectives import (
    classify_objective,
    coerce_finite_objective_score,
    is_minimization_objective,
)

if TYPE_CHECKING:
    from traigent.core.objectives import ObjectiveSchema

__all__ = [
    "SelectionResult",
    "select_best_configuration",
    "apply_tie_breaker",
    "resolve_weighted_selection_schema",
    "resolve_result_selection_schema",
]

# Type alias matching TVL models.py
TieBreaker = Literal["min_abs_deviation", "custom"]
ComparabilityMode = Literal["legacy", "warn", "strict"]
NO_RANKING_ELIGIBLE_TRIALS = "NO_RANKING_ELIGIBLE_TRIALS"
NO_CERTIFIED_SELECTION = "NO_CERTIFIED_SELECTION"
PRIMARY_SCORE_TIE_REL_TOL = 1e-9
PRIMARY_SCORE_TIE_ABS_TOL = 1e-12


@dataclass(slots=True)
class SelectionResult:
    """Best-configuration selection output for orchestrator result building."""

    best_config: dict[str, Any] | None
    best_score: float | None
    session_summary: dict[str, Any] | None
    reason_code: str | None = None
    best_trial_id: str | None = None
    # Trial ids of the exact ranking-eligible set the winner was chosen over
    # (issue #1832). Lets the orchestrator run the inert-constant-objective
    # check over precisely the trials selection ranked, not a re-derived set.
    ranking_eligible_trial_ids: list[str] | None = None
    # Winner-vs-runner-up paired margin significance (issue #1866). Additive
    # qualification of best_config — it never changes which config wins. None
    # when there is no runner-up (< 2 distinct configs / no primary objective);
    # otherwise a dict {runner_up, delta, ci95, p_value, verdict,
    # effective_alpha, n_configs, ...} where verdict is "clear",
    # "statistical_tie", or "na" (no per-example data). The verdict uses a
    # Bonferroni-corrected effective_alpha for the best-of-n_configs selection.
    best_config_margin: dict[str, Any] | None = None


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


def _primary_scores_tied(score: float, best_score: float) -> bool:
    return math.isclose(
        score,
        best_score,
        rel_tol=PRIMARY_SCORE_TIE_REL_TOL,
        abs_tol=PRIMARY_SCORE_TIE_ABS_TOL,
    )


def _is_multi_objective_non_banded(
    objective_schema: ObjectiveSchema | None,
) -> bool:
    """True when the schema declares >1 objective and none is banded.

    The structural precondition shared by both weighted-selection gates below.
    Single-objective and banded schemas are excluded: there is nothing to
    trade off (single) or the winner is band-proximity, not a weighted
    aggregate (banded).
    """
    if objective_schema is None:
        return False
    objectives = getattr(objective_schema, "objectives", None) or []
    if len(objectives) <= 1:
        return False
    if any(
        getattr(obj, "orientation", "") == "band"
        or getattr(obj, "band", None) is not None
        for obj in objectives
    ):
        return False
    return True


def resolve_weighted_selection_schema(
    objective_schema: ObjectiveSchema | None,
) -> ObjectiveSchema | None:
    """Return the schema when weighted *live incumbent* tracking should apply.

    This is the narrow gate (issue #1682): it activates only when the declared
    ObjectiveSchema carries meaningful (non-uniform) weights over more than
    one non-banded objective. Single-objective, uniform-weight, and banded
    schemas return ``None`` so mid-run incumbent tracking (``_simple_is_better``)
    and the inert-constant-objective warning (#1832) keep the legacy
    primary-objective behavior bit-for-bit.

    NOTE: the *terminal* ``best_config`` uses the broader
    :func:`resolve_result_selection_schema` (issue #1846) so uniform/default
    weights also select by the weighted aggregate — matching the run's own
    post-hoc ``weighted_results_v2.json``. This function stays non-uniform-only
    on purpose to avoid perturbing live tracking and stop conditions.
    """
    if not _is_multi_objective_non_banded(objective_schema):
        return None
    has_meaningful = getattr(objective_schema, "has_meaningful_weights", None)
    if has_meaningful is None or not has_meaningful():
        return None
    return objective_schema


def resolve_result_selection_schema(
    objective_schema: ObjectiveSchema | None,
) -> ObjectiveSchema | None:
    """Return the schema when the *terminal* ``best_config`` must be chosen by
    the weighted aggregate (issue #1846).

    Broader than :func:`resolve_weighted_selection_schema`: it activates for
    ANY multi-objective (>1), non-banded schema — INCLUDING uniform/default
    equal weights. Declaring ``objectives=["accuracy", "cost"]`` (equal 0.5/0.5
    weights) is a genuine multi-objective request, and the run's own post-hoc
    analysis (``OptimizationResult.calculate_weighted_scores`` /
    ``weighted_results_v2.json``) already crowns its winner by the equal-weight
    aggregate. Selecting ``results.best_config`` by the same aggregate makes the
    two artifacts of one run agree instead of contradicting each other (the
    accuracy-argmax winner was rank 6/12 on the run's own declared basis).

    Single-objective and banded schemas still return ``None`` so those paths —
    and the #1184 exactly-equal-accuracy cost tie-break on the unweighted path —
    are unchanged.
    """
    if not _is_multi_objective_non_banded(objective_schema):
        return None
    return objective_schema


def observed_metric_ranges(
    trials: Iterable[TrialResult],
    objective_names: Iterable[str],
) -> dict[str, tuple[float, float]]:
    """Observed (min, max) per objective across trials' finite metric values.

    These ranges feed ``ObjectiveSchema.compute_weighted_score(..., ranges=...)``
    so weighted selection normalizes each objective over the values actually
    observed in the run — scale-independent, matching the post-hoc
    ``calculate_weighted_scores`` normalization. Without this, small-magnitude
    minimize objectives (e.g. per-trial cost in fractions of a cent) could
    never influence the winner (issue #1682 review finding).
    """
    names = list(objective_names)
    ranges: dict[str, tuple[float, float]] = {}
    for trial in trials:
        metrics = getattr(trial, "metrics", None) or {}
        for name in names:
            value = metrics.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            numeric = float(value)
            if not math.isfinite(numeric):
                continue
            low, high = ranges.get(name, (numeric, numeric))
            ranges[name] = (min(low, numeric), max(high, numeric))
    return ranges


def _weighted_trial_score(
    schema: ObjectiveSchema,
    metrics: dict[str, Any] | None,
    ranges: dict[str, tuple[float, float]],
) -> float | None:
    """Weighted aggregate for a metrics dict via the shared objectives.py scorer."""
    weighted = schema.compute_weighted_score(metrics or {}, ranges=ranges)
    if weighted is None or not math.isfinite(weighted):
        return None
    return float(weighted)


def _populate_weighted_scores(
    trials: Iterable[TrialResult],
    schema: ObjectiveSchema,
    ranges: dict[str, tuple[float, float]],
) -> None:
    """Expose the basis of selection per trial as ``metrics["score"]`` (#1682).

    NOTE (mutation blast radius): this writes into the SAME TrialResult
    objects that land in ``OptimizationResult.trials`` — that is the delivery
    mechanism the issue asks for. It runs only at terminal selection time
    (never mid-run), so live surfaces that alias ``"score"`` (stop
    conditions, trial-lifecycle fallbacks, client progress reporting) are
    unaffected; in weighted runs it overwrites any evaluator-provided
    ``"score"`` value on ranking-eligible trials.
    """
    for trial in trials:
        weighted = _weighted_trial_score(schema, trial.metrics, ranges)
        if weighted is None:
            continue
        if trial.metrics is None:
            trial.metrics = {}
        trial.metrics["score"] = weighted
        # Keep the TrialResult.score accessor consistent with metrics["score"]
        # for weighted runs: both carry the weighted selection basis (#1845
        # semantics, #1682 basis). Additive — does not alter the metrics["score"]
        # overwrite above, which remains the weighted-run source of truth.
        trial.score = weighted


def _weighted_session_extras(
    schema: ObjectiveSchema,
    best_weighted_score: float,
    ranges: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    return {
        "weighted_selection": {
            "enabled": True,
            "aggregation": "weighted_sum",
            "normalization": "observed_range_min_max",
            "normalization_ranges": {
                name: [low, high] for name, (low, high) in ranges.items()
            },
            "weights_normalized": dict(schema.weights_normalized),
            "best_weighted_score": best_weighted_score,
        }
    }


def _declared_secondary_objectives(
    primary_objective: str,
    objective_order: Iterable[str] | None,
) -> list[str]:
    if objective_order is None:
        return []

    ordered = [str(objective) for objective in objective_order]
    if primary_objective in ordered:
        ordered = ordered[ordered.index(primary_objective) + 1 :]
    return [objective for objective in ordered if objective != primary_objective]


def _schema_orientations(
    objective_schema: ObjectiveSchema | None,
) -> dict[str, str]:
    """Extract the declared per-objective orientation from a schema.

    Reads ``ObjectiveSchema.objectives[*].orientation`` defensively so a
    schema-carrying caller (weighted terminal selection, post-hoc analysis)
    can honor declared orientation without a pre-built map.
    """
    orientations: dict[str, str] = {}
    for objective in getattr(objective_schema, "objectives", None) or []:
        name = getattr(objective, "name", None)
        orientation = getattr(objective, "orientation", None)
        if isinstance(name, str) and isinstance(orientation, str):
            orientations[name] = orientation
    return orientations


def _resolve_orientations(
    objective_orientations: dict[str, str] | None,
    objective_schema: ObjectiveSchema | None,
) -> dict[str, str]:
    """Return the single declared-orientation map every secondary-comparison
    path must honor.

    This is the one source of truth for orientation-aware tie-breaking (issue
    #1955): both the explicit ``objective_orientations`` map (built by the
    orchestrator) and the declared :class:`ObjectiveSchema` feed it, so the
    unweighted AND weighted selection paths compare minimize-oriented
    secondaries in the same direction. An explicit entry wins over the
    schema-derived one; the schema fills any gaps.
    """
    merged = _schema_orientations(objective_schema)
    if objective_orientations:
        for name, orientation in objective_orientations.items():
            if isinstance(name, str) and isinstance(orientation, str):
                merged[name] = orientation
    return merged


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
        metric = coerce_finite_objective_score(trial.get_metric(primary_objective))
        return metric is not None, False, (["MCI-004"] if metric is None else [])

    comparability = _extract_trial_comparability(trial)
    if comparability is None:
        return False, True, ["UNKNOWN_COMPARABILITY"]

    objective_value = coerce_finite_objective_score(trial.get_metric(primary_objective))
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
    objective_order: Iterable[str] | None = None,
    objective_orientations: dict[str, str] | None = None,
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
        objective_order: Declared objectives in preference order.
        objective_orientations: Declared per-objective orientation map. When
            present, a minimize-oriented secondary is subtracted even when its
            NAME misses the heuristic patterns (issue #1955), so every
            selection path breaks ties in the declared direction.

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
        return _apply_min_abs_deviation(
            tied_trials,
            primary_objective,
            band_target,
            objective_order,
            objective_orientations,
        )

    # For "custom" or unknown, return the first trial
    return tied_trials[0]


def _secondary_metric_total(
    metrics: dict[str, Any],
    exclude_objective: str,
    objective_orientations: dict[str, str] | None = None,
) -> float:
    """Sum secondary metrics, subtracting minimization objectives.

    Orientation is taken from the DECLARED schema (``objective_orientations``)
    when available — the same source of truth the primary selector honors
    (issue #1955). Name-pattern guessing is only the schema-less fallback, so a
    custom-named declared-minimize secondary (e.g. ``token_budget``) is no
    longer silently inverted in tie-breaks.
    """
    orientations = objective_orientations or {}
    total = 0.0
    for name, value in metrics.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or name == exclude_objective
        ):
            continue
        if is_minimization_objective(name, orientation=orientations.get(name)):
            total -= float(value)
        else:
            total += float(value)
    return total


def _secondary_metric_key(
    metrics: dict[str, Any],
    exclude_objective: str,
    objective_order: Iterable[str] | None = None,
    objective_orientations: dict[str, str] | None = None,
) -> tuple[float, ...]:
    """Return a secondary-objective ordering key.

    Declared objectives break ties lexicographically in the run's objective order
    after the primary. If the caller has no declared secondary objectives, retain
    the legacy aggregate secondary score. The declared orientation map is
    threaded through so a minimize-oriented secondary sorts downward (#1955).
    """
    declared_secondaries = _declared_secondary_objectives(
        exclude_objective, objective_order
    )
    if not declared_secondaries:
        return (
            _secondary_metric_total(metrics, exclude_objective, objective_orientations),
        )

    ordered_scores: list[float] = []
    for objective in declared_secondaries:
        value = metrics.get(objective)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            ordered_scores.append(float("-inf"))
            continue
        ordered_scores.append(
            _secondary_metric_total(
                {objective: value}, exclude_objective, objective_orientations
            )
        )
    return tuple(ordered_scores)


def _apply_min_abs_deviation(
    trials: list[TrialResult],
    objective: str,
    band_target: float | None,
    objective_order: Iterable[str] | None = None,
    objective_orientations: dict[str, str] | None = None,
) -> TrialResult:
    """Select trial with minimum absolute deviation from target or best stability."""
    if band_target is not None:

        def deviation(t: TrialResult) -> float:
            value = coerce_finite_objective_score(t.get_metric(objective))
            return float("inf") if value is None else abs(value - band_target)

        return min(trials, key=deviation)

    def secondary_score(t: TrialResult) -> tuple[Any, ...]:
        return (
            *_secondary_metric_key(
                t.metrics or {},
                objective,
                objective_order,
                objective_orientations,
            ),
            t.trial_id or "",
        )

    return max(trials, key=secondary_score)


def _select_best_single_trial(
    successful_trials: list[TrialResult],
    primary_objective: str,
    tie_breakers: dict[str, TieBreaker],
    band_target: float | None,
    ranking_summary: dict[str, Any],
    objective_order: Iterable[str] | None,
    objective_orientations: dict[str, str] | None = None,
) -> SelectionResult:
    """Select best trial without aggregation, applying tie-breakers."""
    minimization = is_minimization_objective(
        primary_objective,
        orientation=(objective_orientations or {}).get(primary_objective),
    )
    chooser = min if minimization else max

    scored_trials = []
    for trial in successful_trials:
        score = coerce_finite_objective_score(trial.get_metric(primary_objective))
        if score is not None:
            scored_trials.append((trial, score))
    if not scored_trials:
        return SelectionResult(
            best_config=None,
            best_score=None,
            session_summary={
                "reason_code": NO_RANKING_ELIGIBLE_TRIALS,
                "ranking": ranking_summary,
            },
            reason_code=NO_RANKING_ELIGIBLE_TRIALS,
        )

    if band_target is not None:
        best_deviation = min(abs(score - band_target) for _, score in scored_trials)
        tied_trials = [
            trial
            for trial, score in scored_trials
            if _primary_scores_tied(abs(score - band_target), best_deviation)
        ]
    else:
        best_score = chooser(score for _, score in scored_trials)

        # Find all trials within the conservative primary-score tie tolerance.
        tied_trials = [
            trial
            for trial, score in scored_trials
            if _primary_scores_tied(score, best_score)
        ]

    # Apply tie-breaker if there are ties.
    if len(tied_trials) > 1:
        best_trial = apply_tie_breaker(
            tied_trials,
            tie_breakers,
            primary_objective,
            band_target=band_target,
            objective_order=objective_order,
            objective_orientations=objective_orientations,
        )
        if not tie_breakers and not _declared_secondary_objectives(
            primary_objective, objective_order
        ):
            best_trial = tied_trials[0]
    else:
        best_trial = tied_trials[0] if tied_trials else successful_trials[0]

    return SelectionResult(
        best_config=best_trial.config or {},
        best_score=coerce_finite_objective_score(
            best_trial.get_metric(primary_objective)
        ),
        session_summary={"ranking": ranking_summary},
        best_trial_id=best_trial.trial_id,
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
            {"config": trial.config, "metrics_sum": {}, "count": 0, "trial_ids": []},
        )
        entry["trial_ids"].append(trial.trial_id)

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
    objective_order: Iterable[str] | None = None,
    objective_orientations: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Apply tie-breaker rules to aggregated entries with equal primary scores.

    Args:
        tied_entries: List of aggregated entries with equal primary objective scores.
        tie_breakers: Dict mapping objective names to tie-breaker strategies.
        primary_objective: Name of the primary objective.
        band_target: Optional target value for banded objectives.
        objective_order: Declared objectives in preference order.
        objective_orientations: Declared per-objective orientation map, threaded
            into the secondary comparison so minimize-oriented secondaries sort
            downward on aggregated ties (issue #1955).

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

    def agg_secondary_score(entry: dict[str, Any]) -> tuple[float, ...]:
        return _secondary_metric_key(
            _compute_mean_metrics(entry),
            primary_objective,
            objective_order,
            objective_orientations,
        )

    return max(tied_entries, key=agg_secondary_score)


def _sanitize_mean_metrics(best_metrics: dict[str, float]) -> dict[str, float]:
    """Prefix non-allowlisted metric names for the session summary surface."""
    allowed_unprefixed = {"accuracy", "latency", "cost", "custom_metric"}
    return {
        (name if name in allowed_unprefixed else f"run_{name}"): value
        for name, value in best_metrics.items()
    }


def _select_best_weighted(
    eligible_trials: list[TrialResult],
    schema: ObjectiveSchema,
    primary_objective: str,
    tie_breakers: dict[str, TieBreaker],
    ranking_summary: dict[str, Any],
    objective_order: Iterable[str] | None,
    *,
    aggregate_configs: bool,
    config_space_keys: set[str],
    objective_orientations: dict[str, str] | None = None,
) -> SelectionResult | None:
    """Rank eligible trials by the schema's weighted aggregate (issue #1682).

    Uses ``ObjectiveSchema.compute_weighted_score`` with observed min-max
    ``ranges`` — the shared objectives.py scoring function, normalized over
    the values actually observed across eligible trials, exactly like the
    post-hoc ``calculate_weighted_scores`` path. This makes selection
    scale-independent (a cost objective in fractions of a cent weighs the
    same as one in dollars) and keeps ``best_config`` consistent with the
    post-hoc weighted ranking. ``best_score`` stays the winner's
    primary-objective value for result-shape compatibility; the weighted
    basis of selection is surfaced per-trial in ``metrics["score"]`` and in
    ``session_summary["weighted_selection"]``.

    Returns None when no eligible trial has a computable weighted score, in
    which case the caller falls back to legacy primary-objective ranking.
    """
    ranges = observed_metric_ranges(
        eligible_trials, (obj.name for obj in schema.objectives)
    )

    if aggregate_configs:
        return _select_best_weighted_aggregated(
            eligible_trials,
            schema,
            primary_objective,
            tie_breakers,
            ranking_summary,
            objective_order,
            config_space_keys,
            ranges,
            objective_orientations,
        )

    scored = [
        (trial, weighted)
        for trial in eligible_trials
        if (weighted := _weighted_trial_score(schema, trial.metrics, ranges))
        is not None
    ]
    if not scored:
        return None

    best_weighted = max(weighted for _, weighted in scored)
    tied_trials = [
        trial
        for trial, weighted in scored
        if _primary_scores_tied(weighted, best_weighted)
    ]
    if len(tied_trials) > 1:
        best_trial = apply_tie_breaker(
            tied_trials,
            tie_breakers,
            primary_objective,
            band_target=None,
            objective_order=objective_order,
            objective_orientations=objective_orientations,
        )
    else:
        best_trial = tied_trials[0]

    _populate_weighted_scores(eligible_trials, schema, ranges)

    return SelectionResult(
        best_config=best_trial.config or {},
        best_score=coerce_finite_objective_score(
            best_trial.get_metric(primary_objective)
        ),
        session_summary={
            "ranking": ranking_summary,
            **_weighted_session_extras(schema, best_weighted, ranges),
        },
        best_trial_id=best_trial.trial_id,
    )


def _select_best_weighted_aggregated(
    eligible_trials: list[TrialResult],
    schema: ObjectiveSchema,
    primary_objective: str,
    tie_breakers: dict[str, TieBreaker],
    ranking_summary: dict[str, Any],
    objective_order: Iterable[str] | None,
    config_space_keys: set[str],
    ranges: dict[str, tuple[float, float]],
    objective_orientations: dict[str, str] | None = None,
) -> SelectionResult | None:
    """Weighted ranking over config-aggregated mean metrics (issue #1682).

    Aggregation runs BEFORE per-trial score population so config means stay
    metric-pure; each config is then ranked by the weighted aggregate of its
    mean metrics via the shared objectives.py scorer. ``ranges`` are the
    per-trial observed min-max ranges (one definition shared with the
    non-aggregated path and post-hoc analysis); config means always lie
    within them.
    """
    aggregated = _aggregate_trials(eligible_trials, config_space_keys)
    entry_scores = [
        (entry, weighted)
        for entry in aggregated.values()
        if (
            weighted := _weighted_trial_score(
                schema, _compute_mean_metrics(entry), ranges
            )
        )
        is not None
    ]
    if not entry_scores:
        return None

    best_weighted = max(weighted for _, weighted in entry_scores)
    tied_entries = [
        entry
        for entry, weighted in entry_scores
        if _primary_scores_tied(weighted, best_weighted)
    ]
    if len(tied_entries) > 1:
        best_entry = _apply_aggregated_tie_breaker(
            tied_entries,
            tie_breakers,
            primary_objective,
            None,
            objective_order,
            objective_orientations,
        )
    else:
        best_entry = tied_entries[0]

    _populate_weighted_scores(eligible_trials, schema, ranges)

    best_metrics = _compute_mean_metrics(best_entry)
    session_summary = {
        "selection_mode": "aggregated_mean",
        "primary_objective": primary_objective,
        "samples_per_config": {
            key: entry["count"] for key, entry in aggregated.items()
        },
        # Authoritative replicate identity for the winner (#1854): groups are
        # keyed by the config PROJECTED onto config_space_keys, while
        # best_config keeps the first trial's FULL config — so consumers must
        # not reconstruct the group by config equality (aux keys diverge).
        "winning_trial_ids": list(best_entry.get("trial_ids") or []),
        "metrics": _sanitize_mean_metrics(best_metrics),
        "sanitized": True,
        "ranking": ranking_summary,
        **_weighted_session_extras(schema, best_weighted, ranges),
    }

    return SelectionResult(
        best_config=best_entry["config"] or {},
        best_score=coerce_finite_objective_score(best_metrics.get(primary_objective)),
        session_summary=session_summary,
        best_trial_id=(best_entry.get("trial_ids") or [None])[0],
    )


def _select_best_aggregated(
    aggregated: dict[str, dict[str, Any]],
    primary_objective: str,
    tie_breakers: dict[str, TieBreaker],
    band_target: float | None,
    ranking_summary: dict[str, Any],
    objective_order: Iterable[str] | None,
    objective_orientations: dict[str, str] | None = None,
) -> SelectionResult:
    """Select best configuration from aggregated results with tie-breaking."""
    if not aggregated:
        return SelectionResult(
            best_config=None,
            best_score=None,
            session_summary={
                "reason_code": NO_RANKING_ELIGIBLE_TRIALS,
                "ranking": ranking_summary,
            },
            reason_code=NO_RANKING_ELIGIBLE_TRIALS,
        )

    minimization = is_minimization_objective(
        primary_objective,
        orientation=(objective_orientations or {}).get(primary_objective),
    )

    def score(entry: dict[str, Any]) -> float:
        value = coerce_finite_objective_score(
            _compute_mean_metrics(entry).get(primary_objective)
        )
        if value is None:
            return float("inf") if minimization else float("-inf")
        return value

    # Find the best score
    all_scores = [score(entry) for entry in aggregated.values()]
    all_scores = [s for s in all_scores if math.isfinite(s)]
    if not all_scores:
        return SelectionResult(
            best_config=None,
            best_score=None,
            session_summary={
                "reason_code": NO_RANKING_ELIGIBLE_TRIALS,
                "ranking": ranking_summary,
            },
            reason_code=NO_RANKING_ELIGIBLE_TRIALS,
        )
    if band_target is not None:
        best_deviation = min(
            abs(score_value - band_target) for score_value in all_scores
        )
        tied_entries = [
            entry
            for entry in aggregated.values()
            if math.isfinite(score_value := score(entry))
            and _primary_scores_tied(abs(score_value - band_target), best_deviation)
        ]
    else:
        best_score_val = min(all_scores) if minimization else max(all_scores)

        # Find all entries within the conservative primary-score tie tolerance.
        tied_entries = [
            entry
            for entry in aggregated.values()
            if _primary_scores_tied(
                score_value := score(entry),
                best_score_val,
            )
            and math.isfinite(score_value)
        ]

    # Apply tie-breaking for aggregated entries
    if len(tied_entries) > 1:
        best_entry = _apply_aggregated_tie_breaker(
            tied_entries,
            tie_breakers,
            primary_objective,
            band_target,
            objective_order,
            objective_orientations,
        )
        if not tie_breakers and not _declared_secondary_objectives(
            primary_objective, objective_order
        ):
            best_entry = tied_entries[0]
    else:
        best_entry = tied_entries[0]

    best_metrics = _compute_mean_metrics(best_entry)
    sanitized_metrics = _sanitize_mean_metrics(best_metrics)

    session_summary = {
        "selection_mode": "aggregated_mean",
        "primary_objective": primary_objective,
        "samples_per_config": {
            key: entry["count"] for key, entry in aggregated.items()
        },
        # Same rationale as the weighted path (#1854): the winner's replicate
        # set by trial id, immune to aux config keys outside the space.
        "winning_trial_ids": list(best_entry.get("trial_ids") or []),
        "metrics": sanitized_metrics,
        "sanitized": True,
        "ranking": ranking_summary,
    }

    return SelectionResult(
        best_config=best_entry["config"] or {},
        best_score=coerce_finite_objective_score(best_metrics.get(primary_objective)),
        session_summary=session_summary,
        best_trial_id=(best_entry.get("trial_ids") or [None])[0],
    )


def _mark_weighted_selection_unavailable(
    result: SelectionResult,
    unavailable: bool,
) -> None:
    """Flag a legacy-fallback result that should have been weighted (#1846).

    Additive, warning-only: when a multi-objective schema was declared but no
    eligible trial produced a finite weighted score, terminal selection falls
    back to primary-objective ranking. Recording the flag in the session
    summary keeps the divergence from the post-hoc weighted artifact visible
    downstream; it never changes the chosen ``best_config``.
    """
    if not unavailable:
        return
    summary = dict(result.session_summary or {})
    summary["weighted_selection_unavailable"] = True
    summary.setdefault(
        "weighted_selection_unavailable_reason",
        "declared multi-objective weights could not be applied "
        "(no eligible trial had a finite weighted score); "
        "best_config fell back to primary-objective ranking",
    )
    result.session_summary = summary


def _trial_produced_outputs(trial: TrialResult) -> bool:
    """True iff the trial both completed AND produced at least one successful example.

    A trial can complete with 0 successful examples — for instance, every provider
    call inside it raised (e.g. an Anthropic 404 on a deprecated model ID). Such a
    trial is "completed" in the trial-status sense but has no rankable output, so
    naming it the "best" is misleading. When the evaluator reports
    metadata["successful_examples"] explicitly, we trust it; if the metadata is
    absent we fall back to trial.is_successful so older or external evaluators
    that don't surface per-example success counts still work.
    """
    if not trial.is_successful:
        return False
    metadata = trial.metadata or {}
    successful_examples = metadata.get("successful_examples")
    if isinstance(successful_examples, (int, float)) and successful_examples == 0:
        return False
    return True


def _attach_best_config_margin(
    result: SelectionResult,
    eligible_trials: list[TrialResult],
    primary_objective: str,
    objective_orientations: dict[str, str] | None,
) -> None:
    """Qualify a winning ``best_config`` with margin significance (issue #1866).

    Additive only: this sets ``result.best_config_margin`` and never touches
    ``best_config`` / ``best_score``. The import is local because
    ``stat_significance`` references ``TrialResult`` from ``api.types`` (which
    lazy-imports this module) — a module-level import would risk a load cycle.
    """
    if result.best_config is None and result.best_trial_id is None:
        return
    from traigent.core.stat_significance import compute_best_config_margin

    orientation = (objective_orientations or {}).get(primary_objective)
    result.best_config_margin = compute_best_config_margin(
        eligible_trials,
        winner_trial_id=result.best_trial_id,
        winner_config=result.best_config,
        primary_objective=primary_objective,
        orientation=orientation,
    )


def select_best_configuration(
    trials: Iterable[TrialResult],
    primary_objective: str | None,
    *,
    config_space_keys: Iterable[str],
    aggregate_configs: bool,
    tie_breakers: dict[str, TieBreaker] | None = None,
    band_target: float | None = None,
    objective_order: Iterable[str] | None = None,
    comparability_mode: ComparabilityMode = "warn",
    require_certified: bool = False,
    certified_config: dict[str, Any] | None = None,
    certified_score: float | None = None,
    objective_orientations: dict[str, str] | None = None,
    objective_schema: ObjectiveSchema | None = None,
) -> SelectionResult:
    """Return the configuration that best satisfies the primary objective.

    Strict evidence modes (RFC 0001 / FR-SDK-FAIL-CLOSED-PROMOTION-V1):
    with ``require_certified=True`` the selector NEVER re-derives a winner by
    raw score. It returns the gate-certified incumbent verbatim, or — when no
    certified winner exists — an explicit empty result with
    ``reason_code=NO_CERTIFIED_SELECTION``. Silent fallback to
    highest-score-wins is exactly the fail-open behavior this closes.

    Args:
        trials: Iterable of completed trial results.
        primary_objective: Name of the primary objective to optimize, or
            ``None`` for objectives-free runs (a supported BaseOptimizer
            mode) — ranking is disabled and the selector returns its honest
            no-eligible shape instead of a winner-by-score.
        config_space_keys: Keys that define the configuration space.
        aggregate_configs: Whether to aggregate results by config.
        tie_breakers: Optional dict mapping objectives to tie-breaker strategies.
            From TVL 0.9 promotion_policy.tie_breakers.
        band_target: Optional target value for banded objectives.
        objective_order: Declared objectives in preference order.
        objective_orientations: Optional mapping of objective name → declared
            orientation (``"minimize"`` / ``"maximize"`` / ``"band"``).  When
            present, the declared orientation for the primary objective is used
            directly and name-pattern heuristics are bypassed.  Populated from
            ``ObjectiveSchema.objectives[*].orientation`` by the orchestrator.
        objective_schema: Optional declared ObjectiveSchema. When it declares
            more than one non-banded objective, the terminal ``best_config``
            is ranked by the schema's weighted aggregate
            (``ObjectiveSchema.compute_weighted_score``) instead of the primary
            objective alone — INCLUDING uniform/default equal weights (issue
            #1846), so ``best_config`` equals the run's own post-hoc weighted
            winner (``best_weighted_config`` / ``weighted_results_v2.json``).
            Non-uniform weights already did this (issue #1682); #1846 extends it
            to the uniform case declared by a bare ``objectives=[...]`` list.
            Single-objective and banded schemas keep legacy primary-objective
            behavior, preserving the #1184 exactly-equal-accuracy cost
            tie-break on the unweighted path.

    Returns:
        SelectionResult with best configuration and score.
    """
    if require_certified:
        if certified_config is None:
            return SelectionResult(
                best_config={},
                best_score=None,
                session_summary={
                    "reason": (
                        "strict evidence mode: no certified winner exists; "
                        "refusing winner-by-objective fallback"
                    ),
                    "reason_code": NO_CERTIFIED_SELECTION,
                },
                reason_code=NO_CERTIFIED_SELECTION,
            )
        return SelectionResult(
            best_config=dict(certified_config),
            best_score=certified_score,
            session_summary={
                "reason": "strict evidence mode: gate-certified incumbent",
            },
        )

    trial_list = list(trials)
    successful_trials = [t for t in trial_list if _trial_produced_outputs(t)]
    non_successful_count = len(trial_list) - len(successful_trials)

    if primary_objective is None:
        # Objectives-free run: nothing to rank by. The honest result is the
        # same no-eligible shape as below — never a winner-by-score.
        return SelectionResult(
            best_config={},
            best_score=None,
            session_summary={
                "reason": "no objectives declared; ranking disabled",
                "reason_code": NO_RANKING_ELIGIBLE_TRIALS,
                "ranking": _build_ranking_summary(
                    total_input_trials=len(trial_list),
                    total_successful=len(successful_trials),
                    non_successful_count=non_successful_count,
                    eligible_count=0,
                    excluded_count=len(successful_trials),
                    unknown_count=0,
                    comparability_mode=comparability_mode,
                ),
            },
            reason_code=NO_RANKING_ELIGIBLE_TRIALS,
        )
    if not successful_trials:
        return SelectionResult(
            best_config=None,
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
            best_config=None,
            best_score=None,
            session_summary={
                "reason_code": NO_RANKING_ELIGIBLE_TRIALS,
                "ranking": ranking_summary,
            },
            reason_code=NO_RANKING_ELIGIBLE_TRIALS,
        )

    tie_breakers = tie_breakers or {}

    # One declared-orientation map for EVERY selection sub-path (issue #1955):
    # the explicit orchestrator-built map and the declared schema are merged
    # once here so the weighted and unweighted tie-breaks compare
    # minimize-oriented secondaries in the same declared direction.
    resolved_orientations = _resolve_orientations(
        objective_orientations, objective_schema
    )

    # Trial ids of the exact ranking-eligible set (issue #1832). Attached to
    # whichever selection path wins so the orchestrator can run its
    # inert-constant-objective check over precisely these trials.
    eligible_trial_ids = [trial.trial_id for trial in eligible_trials]

    # Terminal best_config uses the broader gate (issue #1846): uniform/default
    # weights also select by the weighted aggregate so best_config equals the
    # run's own post-hoc weighted winner. Live incumbent tracking keeps the
    # narrower resolve_weighted_selection_schema (non-uniform only).
    weighted_schema = resolve_result_selection_schema(objective_schema)
    if weighted_schema is not None:
        weighted_result = _select_best_weighted(
            eligible_trials,
            weighted_schema,
            primary_objective,
            tie_breakers,
            ranking_summary,
            objective_order,
            aggregate_configs=aggregate_configs,
            config_space_keys=set(config_space_keys),
            objective_orientations=resolved_orientations,
        )
        if weighted_result is not None:
            weighted_result.ranking_eligible_trial_ids = eligible_trial_ids
            _attach_best_config_margin(
                weighted_result,
                eligible_trials,
                primary_objective,
                objective_orientations,
            )
            return weighted_result
        # Defensive (issue #1846 fix-direction #4): a multi-objective schema was
        # declared but NO eligible trial had a computable weighted score
        # (e.g. non-finite/absent secondary objective). We fall back to legacy
        # primary-objective ranking, but mark the result so the silent
        # divergence from the post-hoc weighted artifact is visible rather than
        # invisible.
        weighted_selection_unavailable = True
    else:
        weighted_selection_unavailable = False

    if not aggregate_configs:
        single_result = _select_best_single_trial(
            eligible_trials,
            primary_objective,
            tie_breakers,
            band_target,
            ranking_summary,
            objective_order,
            objective_orientations=resolved_orientations,
        )
        single_result.ranking_eligible_trial_ids = eligible_trial_ids
        _mark_weighted_selection_unavailable(
            single_result, weighted_selection_unavailable
        )
        _attach_best_config_margin(
            single_result,
            eligible_trials,
            primary_objective,
            objective_orientations,
        )
        return single_result

    aggregated = _aggregate_trials(eligible_trials, set(config_space_keys))
    aggregated_result = _select_best_aggregated(
        aggregated,
        primary_objective,
        tie_breakers,
        band_target,
        ranking_summary,
        objective_order,
        objective_orientations=resolved_orientations,
    )
    aggregated_result.ranking_eligible_trial_ids = eligible_trial_ids
    _mark_weighted_selection_unavailable(
        aggregated_result, weighted_selection_unavailable
    )
    _attach_best_config_margin(
        aggregated_result,
        eligible_trials,
        primary_objective,
        objective_orientations,
    )
    return aggregated_result
