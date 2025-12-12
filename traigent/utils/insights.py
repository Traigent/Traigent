"""Optimization insights and analysis utilities.

This module provides functions to analyze optimization results and generate
business intelligence about parameter performance, configuration trade-offs,
and optimization effectiveness.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Performance CONC-Quality-Observability FUNC-ANALYTICS REQ-ANLY-011 SYNC-Observability

from __future__ import annotations

import statistics
from typing import Any, cast

from traigent.api.types import OptimizationResult, TrialResult
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


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

    Example:
        >>> results = my_function.optimize()
        >>> insights = traigent.get_optimization_insights(results)
        >>> print(f"Best model: {insights['top_configurations'][0]['config']['model']}")
        >>> print(f"Performance improvement: {insights['performance_summary']['improvement']:.1%}")
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

    # Generate insights
    top_configs = _analyze_top_configurations(successful_trials, results.objectives)
    performance_summary = _analyze_performance_summary(
        successful_trials, results.objectives
    )
    parameter_insights = _analyze_parameter_importance(
        successful_trials, results.objectives
    )
    recommendations = _generate_recommendations(
        top_configs, parameter_insights, performance_summary
    )

    return {
        "top_configurations": top_configs,
        "performance_summary": performance_summary,
        "parameter_insights": parameter_insights,
        "recommendations": recommendations,
    }


def _analyze_top_configurations(
    trials: list[TrialResult], objectives: list[str], top_k: int = 3
) -> list[dict[str, Any]]:
    """Analyze top performing configurations."""
    # Sort trials by primary objective (first in list)
    primary_objective = objectives[0] if objectives else "score"

    # Sort trials by primary metric score (descending)
    sorted_trials = sorted(
        trials, key=lambda t: t.metrics.get(primary_objective, 0.0), reverse=True
    )

    top_configurations = []
    for i, trial in enumerate(sorted_trials[:top_k]):
        config_analysis = {
            "rank": i + 1,
            "config": trial.config.copy(),
            "metrics": trial.metrics.copy(),
            "score": trial.metrics.get(primary_objective, 0.0),
            "trial_id": trial.trial_id,
            "relative_performance": _calculate_relative_performance(
                trial, sorted_trials[0]
            ),
        }

        # Add cost analysis if cost metric is available
        if "cost" in trial.metrics or "cost_per_1k" in trial.metrics:
            cost_key = "cost_per_1k" if "cost_per_1k" in trial.metrics else "cost"
            config_analysis["cost_analysis"] = {
                "cost_per_query": trial.metrics.get(cost_key, 0.0),
                "cost_efficiency": _calculate_cost_efficiency(trial, primary_objective),
            }

        top_configurations.append(config_analysis)

    return top_configurations


def _analyze_performance_summary(
    trials: list[TrialResult], objectives: list[str]
) -> dict[str, Any]:
    """Analyze overall optimization performance."""
    primary_objective = objectives[0] if objectives else "score"

    # Extract scores for primary objective
    scores = [t.metrics.get(primary_objective, 0.0) for t in trials]

    if not scores:
        return {}

    best_score = max(scores)
    worst_score = min(scores)
    avg_score = statistics.mean(scores)

    summary = {
        "total_trials": len(trials),
        "successful_trials": len(trials),
        "primary_objective": primary_objective,
        "best_score": best_score,
        "worst_score": worst_score,
        "average_score": avg_score,
        "score_std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "improvement": (
            (best_score - worst_score) / worst_score if worst_score > 0 else 0.0
        ),
        "consistency": (
            1.0 - (statistics.stdev(scores) / avg_score)
            if avg_score > 0 and len(scores) > 1
            else 1.0
        ),
    }

    # Add multi-objective analysis if applicable
    if len(objectives) > 1:
        multi_obj_analysis = {}
        for obj in objectives:
            obj_scores = [t.metrics.get(obj, 0.0) for t in trials]
            if obj_scores:
                multi_obj_analysis[obj] = {
                    "best": max(obj_scores),
                    "average": statistics.mean(obj_scores),
                    "improvement": (
                        (max(obj_scores) - min(obj_scores)) / min(obj_scores)
                        if min(obj_scores) > 0
                        else 0.0
                    ),
                }
        summary["multi_objective_analysis"] = multi_obj_analysis

    return summary


def _analyze_parameter_importance(
    trials: list[TrialResult], objectives: list[str]
) -> dict[str, Any]:
    """Analyze parameter importance and impact."""
    if not trials or not objectives:
        return {}

    primary_objective = objectives[0]
    parameter_analysis = {}

    # Get all unique parameters
    all_params: set[str] = set()
    for trial in trials:
        all_params.update(trial.config.keys())

    # Analyze each parameter
    for param in all_params:
        param_values = []
        param_scores = []

        for trial in trials:
            if param in trial.config:
                param_values.append(trial.config[param])
                param_scores.append(trial.metrics.get(primary_objective, 0.0))

        if param_values and param_scores:
            # Find best value for this parameter
            best_idx = param_scores.index(max(param_scores))
            best_value = param_values[best_idx]

            # Calculate performance impact (correlation-like measure)
            performance_impact = _calculate_parameter_impact(param_values, param_scores)

            parameter_analysis[param] = {
                "best_value": best_value,
                "performance_impact": performance_impact,
                "value_distribution": _analyze_value_distribution(
                    param_values, param_scores
                ),
                "optimization_priority": (
                    "high"
                    if performance_impact > 0.1
                    else "medium" if performance_impact > 0.05 else "low"
                ),
            }

    return parameter_analysis


def _generate_recommendations(
    top_configs: list[dict[str, Any]],
    param_insights: dict[str, Any],
    performance_summary: dict[str, Any],
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
    trial: TrialResult, best_trial: TrialResult
) -> float:
    """Calculate relative performance compared to best trial."""
    if not best_trial.metrics:
        return 1.0

    # Use first metric as primary
    primary_metric = next(iter(best_trial.metrics.keys()))
    best_score = best_trial.metrics.get(primary_metric, 0.0)
    trial_score = trial.metrics.get(primary_metric, 0.0)

    if best_score == 0:
        return 1.0

    return trial_score / best_score


def _calculate_cost_efficiency(trial: TrialResult, primary_objective: str) -> float:
    """Calculate cost efficiency score."""
    score = trial.metrics.get(primary_objective, 0.0)
    cost = trial.metrics.get("cost_per_1k", trial.metrics.get("cost", 1.0))

    if cost == 0:
        return float("inf")

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

    # For numeric parameters, use correlation approximation
    try:
        numeric_values = [float(v) for v in values]
        return abs(_simple_correlation(numeric_values, scores))
    except (ValueError, TypeError):
        return 0.0


def _simple_correlation(x: list[float], y: list[float]) -> float:
    """Simple correlation calculation."""
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
    values: list[Any], scores: list[float]
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

    best_value = max(avg_performance.keys(), key=lambda k: avg_performance[k])

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
