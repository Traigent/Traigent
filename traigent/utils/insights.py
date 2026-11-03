"""Optimization insights and analysis utilities.

This module provides functions to analyze optimization results and generate
business intelligence about parameter performance, configuration trade-offs,
and optimization effectiveness.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Performance CONC-Quality-Observability FUNC-ANALYTICS REQ-ANLY-011 SYNC-Observability

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from typing import Any, cast

from traigent.api.types import OptimizationResult, TrialResult
from traigent.utils.logging import get_logger
from traigent.utils.relative_change import relative_change
from traigent.utils.trial_costs import is_cost_unmeasured

logger = get_logger(__name__)

_Orientations = Mapping[str, str | None]


def _resolve_orientations(
    results: OptimizationResult, objectives: list[str]
) -> dict[str, str | None]:
    """Resolve each objective's direction from the result (declared, else SDK-owned).

    The primary objective must be resolvable: ranking it without a direction
    would silently treat a cost or latency objective as higher-is-better
    (#2819), so the standard "declare an orientation" error is raised. A
    secondary objective with no known direction is reported with
    ``orientation=None`` and no best/improvement instead of a guessed one.
    """
    orientations: dict[str, str | None] = {}
    for index, name in enumerate(objectives):
        try:
            orientations[name] = results._objective_orientation(name)
        except ValueError:
            if index == 0:
                raise
            orientations[name] = None
    return orientations


def _is_minimize(orientations: _Orientations | None, objective: str) -> bool:
    return orientations is not None and orientations.get(objective) == "minimize"


def _metric_value(trial: TrialResult, objective: str) -> float | None:
    """A trial's finite numeric value for ``objective``; None when not reported."""
    value = trial.metrics.get(objective)
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _best_of(values: list[float], minimize: bool) -> float:
    return min(values) if minimize else max(values)


def _worst_of(values: list[float], minimize: bool) -> float:
    return max(values) if minimize else min(values)


def get_optimization_insights(results: OptimizationResult) -> dict[str, Any]:
    """Generate comprehensive insights from optimization results.

    Analyzes optimization results to provide business intelligence about:
    - Top performing configurations with trade-off analysis
    - Parameter importance and impact on performance
    - Optimization effectiveness and convergence
    - Cost-benefit analysis and recommendations

    Args:
        results: OptimizationResult from completed optimization

    Returns:
        Dictionary containing structured insights:
        - top_configurations: Ranked list of best configurations
        - performance_summary: Overall optimization statistics
        - parameter_insights: Analysis of parameter importance
        - recommendations: Actionable recommendations

        Each ``multi_objective_analysis`` entry carries the objective's
        ``orientation``; ``best`` and ``improvement`` are ``None`` for a
        secondary objective whose direction is unknown. Trials that did not
        report the primary metric are excluded, and ``improvement`` is the
        best versus the worst trial in the objective's direction.

    Raises:
        ValueError: The primary objective has no declared orientation and no
            SDK-owned default (a custom metric such as ``f1``). Declare it in
            the objective schema or ``results.metadata["objective_orientations"]``.

    Example::

        results = my_function.optimize()
        insights = traigent.get_optimization_insights(results)
        print(f"Best model: {insights['top_configurations'][0]['config']['model']}")
        print(f"Performance improvement: {insights['performance_summary']['improvement']:.1%}")
    """
    if not results or not results.trials:
        return {
            "error": "No optimization results available",
            "top_configurations": [],
            "performance_summary": {},
            "parameter_insights": {},
            "recommendations": [],
        }

    # Get successful trials only
    successful_trials = [t for t in results.trials if t.is_successful and t.metrics]

    if not successful_trials:
        return {
            "error": "No successful trials found",
            "top_configurations": [],
            "performance_summary": {},
            "parameter_insights": {},
            "recommendations": [],
        }

    orientations = _resolve_orientations(results, list(results.objectives or []))
    primary_objective = results.objectives[0] if results.objectives else "score"
    if results.objectives and not any(
        _metric_value(t, primary_objective) is not None for t in successful_trials
    ):
        return {
            "error": f"No successful trial reported the primary objective {primary_objective!r}",
            "top_configurations": [],
            "performance_summary": {},
            "parameter_insights": {},
            "recommendations": [],
        }

    # Generate insights
    top_configs = _analyze_top_configurations(
        successful_trials, results.objectives, orientations=orientations
    )
    performance_summary = _analyze_performance_summary(
        successful_trials, results.objectives, orientations=orientations
    )
    parameter_insights = _analyze_parameter_importance(
        successful_trials, results.objectives, orientations=orientations
    )
    recommendations = _generate_recommendations(
        top_configs, parameter_insights, performance_summary, orientations
    )

    return {
        "top_configurations": top_configs,
        "performance_summary": performance_summary,
        "parameter_insights": parameter_insights,
        "recommendations": recommendations,
    }


def _analyze_top_configurations(
    trials: list[TrialResult],
    objectives: list[str],
    top_k: int = 3,
    orientations: _Orientations | None = None,
) -> list[dict[str, Any]]:
    """Analyze top performing configurations.

    Trials are ranked best-first for the primary objective's direction, and a
    trial that did not report the primary metric is excluded (it is not scored
    as 0.0).
    """
    # Sort trials by primary objective (first in list)
    primary_objective = objectives[0] if objectives else "score"
    minimize = _is_minimize(orientations, primary_objective)

    scored = [
        (value, t)
        for t in trials
        if (value := _metric_value(t, primary_objective)) is not None
    ]
    sorted_trials = [
        t for _, t in sorted(scored, key=lambda item: item[0], reverse=not minimize)
    ]

    top_configurations = []
    for i, trial in enumerate(sorted_trials[:top_k]):
        config_analysis = {
            "rank": i + 1,
            "config": trial.config.copy(),
            "metrics": trial.metrics.copy(),
            "score": _metric_value(trial, primary_objective),
            "trial_id": trial.trial_id,
            "relative_performance": _calculate_relative_performance(
                trial, sorted_trials[0], primary_objective, minimize
            ),
        }

        # Add cost analysis if any cost metric is available. Prefer cost_per_1k,
        # then the per-config ``cost`` metric, then ``total_cost`` so that a run
        # whose per-config ``cost`` was decoupled from the real spend still
        # reports a non-zero cost-per-query (#1423).
        # A trial flagged ``cost_unmeasured`` has no cost: no cost analysis is
        # reported for it, never a ``cost_per_query`` of 0.0 (#2517).
        if not is_cost_unmeasured(trial.metrics) and any(
            k in trial.metrics for k in ("cost", "cost_per_1k", "total_cost")
        ):
            cost_per_query = next(
                (
                    value
                    for value in (
                        trial.metrics.get("cost_per_1k"),
                        trial.metrics.get("cost"),
                        trial.metrics.get("total_cost"),
                    )
                    if value
                ),
                None,
            )
            if cost_per_query is None and any(
                trial.metrics.get(k) == 0 for k in ("cost_per_1k", "cost", "total_cost")
            ):
                cost_per_query = 0.0  # a measured zero, not an absent cost
            if cost_per_query is not None:
                config_analysis["cost_analysis"] = {
                    "cost_per_query": cost_per_query,
                    "cost_efficiency": (
                        None
                        if minimize
                        else _calculate_cost_efficiency(trial, primary_objective)
                    ),
                }

        top_configurations.append(config_analysis)

    return top_configurations


def _analyze_performance_summary(
    trials: list[TrialResult],
    objectives: list[str],
    orientations: _Orientations | None = None,
) -> dict[str, Any]:
    """Analyze overall optimization performance.

    ``best_score``/``worst_score`` follow the primary objective's direction.
    ``improvement`` is the relative gain of the best trial over the worst trial
    in that direction (``relative_change(worst, best, orientation)``): 0.99
    means the best trial is 99% better than the worst, whether the objective is
    maximized or minimized. Trials that did not report the primary metric are
    excluded rather than scored as 0.0.
    """
    primary_objective = objectives[0] if objectives else "score"
    minimize = _is_minimize(orientations, primary_objective)

    # Extract scores for primary objective
    scores = [
        v for t in trials if (v := _metric_value(t, primary_objective)) is not None
    ]

    if not scores:
        return {}

    best_score = _best_of(scores, minimize)
    worst_score = _worst_of(scores, minimize)
    avg_score = statistics.mean(scores)

    improvement = relative_change(
        worst_score, best_score, "minimize" if minimize else "maximize"
    )

    summary = {
        "total_trials": len(trials),
        "successful_trials": len(trials),
        "primary_objective": primary_objective,
        "best_score": best_score,
        "worst_score": worst_score,
        "average_score": avg_score,
        "score_std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "improvement": (
            improvement if improvement is not None and worst_score > 0 else 0.0
        ),
        "consistency": (
            1.0 - (statistics.stdev(scores) / avg_score)
            if avg_score > 0 and len(scores) > 1
            else 1.0
        ),
    }

    # Add multi-objective analysis if applicable
    if len(objectives) > 1:
        summary["multi_objective_analysis"] = _summarize_multi_objective_scores(
            trials, objectives, orientations
        )

    return summary


def _summarize_multi_objective_scores(
    trials: list[TrialResult],
    objectives: list[str],
    orientations: _Orientations | None = None,
) -> dict[str, dict[str, Any]]:
    """Summarize each objective with its best value in that objective's direction.

    An objective whose direction is unknown reports ``orientation=None`` and no
    ``best``/``improvement`` (``None``) rather than a guessed direction.
    """
    multi_obj_analysis: dict[str, dict[str, Any]] = {}
    for obj in objectives:
        obj_scores = [v for t in trials if (v := _metric_value(t, obj)) is not None]
        if not obj_scores:
            continue
        if orientations is None:
            orientation: str | None = "maximize"  # direct (legacy) call
        else:
            orientation = orientations.get(obj)
        if orientation in ("maximize", "minimize"):
            minimize = orientation == "minimize"
            best_value: float | None = _best_of(obj_scores, minimize)
            worst_value = _worst_of(obj_scores, minimize)
            change = relative_change(worst_value, best_value, orientation)
            improvement: float | None = (
                change if change is not None and worst_value > 0 else 0.0
            )
        else:
            best_value, improvement = None, None
        multi_obj_analysis[obj] = {
            "orientation": orientation,
            "best": best_value,
            "average": statistics.mean(obj_scores),
            "improvement": improvement,
        }
    return multi_obj_analysis


def _analyze_parameter_importance(
    trials: list[TrialResult],
    objectives: list[str],
    orientations: _Orientations | None = None,
) -> dict[str, Any]:
    """Analyze parameter importance and impact."""
    if not trials or not objectives:
        return {}

    primary_objective = objectives[0]
    minimize = _is_minimize(orientations, primary_objective)
    parameter_analysis = {}

    # Get all unique parameters
    all_params: set[str] = set()
    for trial in trials:
        all_params.update(trial.config.keys())

    # Analyze each parameter
    for param in all_params:
        analysis = _analyze_single_parameter_importance(
            param, trials, primary_objective, minimize
        )
        if analysis is not None:
            parameter_analysis[param] = analysis

    return parameter_analysis


def _analyze_single_parameter_importance(
    param: str,
    trials: list[TrialResult],
    primary_objective: str,
    minimize: bool = False,
) -> dict[str, Any] | None:
    param_values = []
    param_scores = []

    for trial in trials:
        if param not in trial.config:
            continue
        value = trial.config[param]
        score = _metric_value(trial, primary_objective)
        if score is None:
            continue  # did not report the metric: excluded, not scored 0.0
        param_values.append(value)
        param_scores.append(score)

    if param_values and param_scores:
        # Find best value for this parameter
        best_idx = param_scores.index(_best_of(param_scores, minimize))
        best_value = param_values[best_idx]

        # Calculate performance impact (association-style measure)
        performance_impact = _calculate_parameter_impact(param_values, param_scores)

        if performance_impact > 0.1:
            optimization_priority = "high"
        elif performance_impact > 0.05:
            optimization_priority = "medium"
        else:
            optimization_priority = "low"

        return {
            "best_value": best_value,
            "performance_impact": performance_impact,
            "value_distribution": _analyze_value_distribution(
                param_values, param_scores, minimize
            ),
            "optimization_priority": optimization_priority,
        }

    return None


def _generate_recommendations(
    top_configs: list[dict[str, Any]],
    param_insights: dict[str, Any],
    performance_summary: dict[str, Any],
    orientations: _Orientations | None = None,
) -> list[str]:
    """Generate actionable recommendations based on analysis."""
    recommendations = []

    if not top_configs:
        return ["Run optimization with more trials to generate recommendations"]

    best_config = top_configs[0]

    # Performance recommendations
    if performance_summary.get("improvement", 0) > 0.1:
        recommendations.append(
            f"🎯 Optimization achieved {performance_summary['improvement']:.1%} improvement. "
            f"Consider using the best configuration: {_format_config_summary(best_config['config'])}"
        )

    # Parameter recommendations
    high_impact_params = [
        param
        for param, info in param_insights.items()
        if info.get("optimization_priority") == "high"
    ]

    if high_impact_params:
        recommendations.append(
            f"🔧 High-impact parameters: {', '.join(high_impact_params)}. "
            "Focus future optimization efforts on these parameters."
        )

    # Cost efficiency recommendations
    if len(top_configs) > 1:
        best_config = top_configs[0]
        second_config = top_configs[1]

        if "cost_analysis" in best_config and "cost_analysis" in second_config:
            cost_diff = (
                best_config["cost_analysis"]["cost_per_query"]
                - second_config["cost_analysis"]["cost_per_query"]
            )
            perf_diff = best_config["score"] - second_config["score"]
            if _is_minimize(
                orientations, performance_summary.get("primary_objective", "")
            ):
                perf_diff = -perf_diff  # positive = best is better

            if (
                cost_diff > 0 and perf_diff < 0.05
            ):  # Higher cost, minimal performance gain
                recommendations.append(
                    f"💰 Consider configuration #{second_config['rank']} for better cost efficiency. "
                    f"Only {perf_diff:.1%} performance difference but {abs(cost_diff):.3f} lower cost."
                )

    # Consistency recommendations
    if performance_summary.get("consistency", 0) < 0.8:
        recommendations.append(
            "⚠️ High score variance detected. Consider running more trials or adjusting parameter ranges."
        )

    return recommendations


def _calculate_relative_performance(
    trial: TrialResult,
    best_trial: TrialResult,
    primary_objective: str | None = None,
    minimize: bool = False,
) -> float:
    """Calculate relative performance compared to the best trial (1.0 = as good).

    For a minimized objective the ratio is inverted (best / trial) so a worse
    trial scores below 1.0 in both directions.
    """
    if not best_trial.metrics:
        return 1.0

    # Primary objective when known; otherwise the first metric (legacy callers).
    primary_metric = primary_objective or next(iter(best_trial.metrics.keys()))
    best_score = _metric_value(best_trial, primary_metric)
    trial_score = _metric_value(trial, primary_metric)
    if best_score is None or trial_score is None:
        return 1.0

    if minimize:
        return 1.0 if trial_score == 0 else best_score / trial_score
    if best_score == 0:
        return 1.0

    return trial_score / best_score


def _calculate_cost_efficiency(trial: TrialResult, primary_objective: str) -> float:
    """Calculate cost efficiency score (quality per unit cost).

    Falls back to ``total_cost`` when the per-config ``cost`` metric is missing,
    and guards against a zero/missing cost. Returning ``inf`` for cost==0 (the
    old behavior) corrupted every downstream ranking/aggregation that consumed
    this value, because a real run whose per-config ``cost`` was wrongly $0
    produced ``inf`` efficiency for EVERY config (#1423). A non-finite efficiency
    is not actionable, so we return ``0.0`` when cost is unavailable.
    """
    score = trial.metrics.get(primary_objective, 0.0)
    cost = trial.metrics.get("cost_per_1k")
    if cost is None:
        cost = trial.metrics.get("cost")
    if cost is None:
        cost = trial.metrics.get("total_cost")

    if not cost:  # None or 0.0
        return 0.0

    return score / cost


def _calculate_parameter_impact(values: list[Any], scores: list[float]) -> float:
    """Calculate parameter impact on performance."""
    if len(set(values)) == 1:  # All same value
        return 0.0

    # For categorical parameters, use variance between groups
    if isinstance(values[0], str):
        value_groups: dict[str, list[float]] = {}
        for val, score in zip(values, scores, strict=False):
            if val not in value_groups:
                value_groups[val] = []
            value_groups[val].append(score)

        if len(value_groups) < 2:
            return 0.0

        group_means = [statistics.mean(scores) for scores in value_groups.values()]
        overall_mean = statistics.mean(scores)

        if overall_mean == 0:
            return 0.0

        # Calculate coefficient of variation between groups
        return statistics.stdev(group_means) / overall_mean if overall_mean > 0 else 0.0

    # For numeric parameters, use a statistical association approximation
    try:
        numeric_values = [float(v) for v in values]
        return abs(_simple_correlation(numeric_values, scores))
    except (ValueError, TypeError):
        return 0.0


def _simple_correlation(x: list[float], y: list[float]) -> float:
    """Simple association calculation."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0

    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)

    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y, strict=False))

    sum_sq_x = sum((xi - mean_x) ** 2 for xi in x)
    sum_sq_y = sum((yi - mean_y) ** 2 for yi in y)

    denominator = (sum_sq_x * sum_sq_y) ** 0.5

    if denominator == 0:
        return 0.0

    return cast(float, numerator / denominator)


def _analyze_value_distribution(
    values: list[Any], scores: list[float], minimize: bool = False
) -> dict[str, Any]:
    """Analyze distribution of parameter values and their performance."""
    unique_values = list(set(values))

    if len(unique_values) == 1:
        return {"type": "constant", "best_value": unique_values[0]}

    # Group scores by value
    value_performance: dict[Any, list[float]] = {}
    for val, score in zip(values, scores, strict=False):
        if val not in value_performance:
            value_performance[val] = []
        value_performance[val].append(score)

    # Calculate average performance for each value
    avg_performance = {
        val: statistics.mean(scores) for val, scores in value_performance.items()
    }

    pick = min if minimize else max
    best_value = pick(avg_performance.keys(), key=lambda k: avg_performance[k])

    return {
        "type": "categorical" if isinstance(values[0], str) else "numeric",
        "unique_values": len(unique_values),
        "best_value": best_value,
        "value_performance": avg_performance,
    }


def _format_config_summary(config: dict[str, Any]) -> str:
    """Format configuration for readable summary."""
    items = []
    for key, value in config.items():
        if isinstance(value, float):
            items.append(f"{key}={value:.3f}")
        else:
            items.append(f"{key}={value}")

    return ", ".join(items)
