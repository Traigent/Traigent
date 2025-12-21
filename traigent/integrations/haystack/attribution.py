"""Attribution and insights for Haystack pipeline optimization.

This module provides per-node metrics collection, quality contribution analysis,
cost/latency contribution analysis, parameter sensitivity analysis, and
optimization recommendations.

Example usage:
    from traigent.integrations.haystack import (
        compute_attribution,
        NodeMetrics,
        ComponentAttribution,
    )

    # After optimization completes
    attribution = compute_attribution(result.history)

    # Access component-level insights
    for name, attr in attribution.items():
        print(f"{name}: quality={attr.quality_contribution:.2f}")
        print(f"  Cost fraction: {attr.cost_contribution:.2%}")
        print(f"  Most sensitive: {attr.most_sensitive_param}")
        print(f"  Recommendation: {attr.recommendation}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from traigent.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class NodeMetrics:
    """Per-run metrics for a single pipeline component.

    Attributes:
        component_name: Name of the component.
        invocation_count: Number of times the component was invoked.
        total_latency_ms: Total latency across all invocations.
        total_cost: Total cost from this component.
        total_input_tokens: Total input tokens processed.
        total_output_tokens: Total output tokens generated.
        error_count: Number of errors from this component.
    """

    component_name: str
    invocation_count: int = 0
    total_latency_ms: float = 0.0
    total_cost: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    error_count: int = 0

    @property
    def avg_latency_ms(self) -> float:
        """Average latency per invocation."""
        if self.invocation_count == 0:
            return 0.0
        return self.total_latency_ms / self.invocation_count

    @property
    def avg_cost(self) -> float:
        """Average cost per invocation."""
        if self.invocation_count == 0:
            return 0.0
        return self.total_cost / self.invocation_count

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component_name": self.component_name,
            "invocation_count": self.invocation_count,
            "total_latency_ms": self.total_latency_ms,
            "total_cost": self.total_cost,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "error_count": self.error_count,
            "avg_latency_ms": self.avg_latency_ms,
            "avg_cost": self.avg_cost,
        }


@dataclass
class ComponentAttribution:
    """Attribution insights for a pipeline component.

    Aggregated analysis across all optimization runs.

    Attributes:
        component_name: Name of the component.
        quality_contribution: Impact on quality (-1 to 1).
        cost_contribution: Fraction of total cost (0 to 1).
        latency_contribution: Fraction of total latency (0 to 1).
        sensitivity_scores: Impact score per parameter.
        most_sensitive_param: Parameter with highest sensitivity.
        optimization_opportunity: 'high', 'medium', or 'low'.
        recommendation: Human-readable suggestion.
    """

    component_name: str
    quality_contribution: float = 0.0
    cost_contribution: float = 0.0
    latency_contribution: float = 0.0
    sensitivity_scores: dict[str, float] = field(default_factory=dict)
    most_sensitive_param: str | None = None
    optimization_opportunity: str = "low"
    recommendation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component_name": self.component_name,
            "quality_contribution": self.quality_contribution,
            "cost_contribution": self.cost_contribution,
            "latency_contribution": self.latency_contribution,
            "sensitivity_scores": self.sensitivity_scores,
            "most_sensitive_param": self.most_sensitive_param,
            "optimization_opportunity": self.optimization_opportunity,
            "recommendation": self.recommendation,
        }


def extract_node_metrics(
    run_result: Any,
    component_names: list[str] | None = None,
) -> dict[str, NodeMetrics]:
    """Extract per-node metrics from a run result.

    Args:
        run_result: Result object with per-example metrics.
        component_names: Optional list of component names to include.

    Returns:
        Dict mapping component names to NodeMetrics.
    """
    node_metrics: dict[str, NodeMetrics] = {}

    # Try to get node-level metrics from the result
    if hasattr(run_result, "node_metrics"):
        raw_metrics = run_result.node_metrics
        if isinstance(raw_metrics, dict):
            for name, data in raw_metrics.items():
                if component_names and name not in component_names:
                    continue
                if isinstance(data, NodeMetrics):
                    node_metrics[name] = data
                elif isinstance(data, dict):
                    node_metrics[name] = NodeMetrics(
                        component_name=name,
                        invocation_count=data.get("invocation_count", 0),
                        total_latency_ms=data.get("total_latency_ms", 0.0),
                        total_cost=data.get("total_cost", 0.0),
                        total_input_tokens=data.get("total_input_tokens", 0),
                        total_output_tokens=data.get("total_output_tokens", 0),
                        error_count=data.get("error_count", 0),
                    )

    # Try to extract from aggregated metrics
    if not node_metrics and hasattr(run_result, "aggregated_metrics"):
        metrics = run_result.aggregated_metrics
        if isinstance(metrics, dict):
            # Look for per-component cost/latency metrics
            for key, value in metrics.items():
                if "_cost" in key or "_latency" in key:
                    # Extract component name from key like "generator_cost"
                    parts = key.rsplit("_", 1)
                    if len(parts) == 2:
                        name = parts[0]
                        metric_type = parts[1]
                        if name not in node_metrics:
                            node_metrics[name] = NodeMetrics(component_name=name)
                        if metric_type == "cost":
                            node_metrics[name].total_cost = float(value)
                        elif metric_type == "latency":
                            node_metrics[name].total_latency_ms = float(value)

    return node_metrics


def compute_quality_contribution(
    trials: list[Any],
    target_metric: str = "accuracy",
) -> dict[str, float]:
    """Compute quality contribution per component using variance analysis.

    Args:
        trials: List of trial results.
        target_metric: Metric to analyze quality contribution for.

    Returns:
        Dict mapping component names to quality contribution scores.
    """
    if not trials:
        return {}

    # Collect successful trials with the target metric
    successful = []
    for t in trials:
        if hasattr(t, "metrics") and target_metric in t.metrics:
            if hasattr(t, "is_successful") and t.is_successful:
                successful.append(t)
            elif not hasattr(t, "is_successful"):
                successful.append(t)

    if len(successful) < 3:
        logger.warning(
            f"Not enough trials ({len(successful)}) for quality contribution"
        )
        return {}

    # Get all parameter names and identify components
    all_params: set[str] = set()
    for trial in successful:
        if hasattr(trial, "config"):
            all_params.update(trial.config.keys())

    # Group parameters by component
    component_params: dict[str, list[str]] = {}
    for param in all_params:
        if "." in param:
            component = param.split(".")[0]
        else:
            component = param
        if component not in component_params:
            component_params[component] = []
        component_params[component].append(param)

    # Compute quality contribution via parameter impact analysis
    metric_values = [t.metrics[target_metric] for t in successful]
    metric_mean = sum(metric_values) / len(metric_values)
    metric_var = sum((v - metric_mean) ** 2 for v in metric_values)

    if metric_var == 0:
        return dict.fromkeys(component_params, 0.0)

    contributions: dict[str, float] = {}

    for component, params in component_params.items():
        # Compute correlation between component params and quality
        component_impact = 0.0
        param_count = 0

        for param in params:
            param_values = []
            paired_metrics = []

            for trial in successful:
                if hasattr(trial, "config") and param in trial.config:
                    val = trial.config[param]
                    if isinstance(val, (int, float)):
                        param_values.append(float(val))
                        paired_metrics.append(trial.metrics[target_metric])

            if len(param_values) >= 3:
                # Compute correlation
                param_mean = sum(param_values) / len(param_values)
                param_var = sum((v - param_mean) ** 2 for v in param_values)

                if param_var > 0:
                    metric_mean_local = sum(paired_metrics) / len(paired_metrics)
                    covar = sum(
                        (pv - param_mean) * (mv - metric_mean_local)
                        for pv, mv in zip(param_values, paired_metrics)
                    )
                    corr = covar / (
                        (param_var**0.5)
                        * (
                            sum((m - metric_mean_local) ** 2 for m in paired_metrics)
                            ** 0.5
                        )
                        + 1e-10
                    )
                    component_impact += abs(corr)
                    param_count += 1

        if param_count > 0:
            contributions[component] = component_impact / param_count
        else:
            contributions[component] = 0.0

    # Normalize to -1 to 1 range
    max_contrib = max(abs(c) for c in contributions.values()) if contributions else 1.0
    if max_contrib > 0:
        contributions = {k: v / max_contrib for k, v in contributions.items()}

    return contributions


def compute_cost_contribution(
    trials: list[Any],
) -> dict[str, float]:
    """Compute cost contribution per component.

    Args:
        trials: List of trial results.

    Returns:
        Dict mapping component names to cost fraction (0-1).
    """
    if not trials:
        return {}

    # Aggregate costs per component across all trials
    component_costs: dict[str, float] = {}
    total_cost = 0.0

    for trial in trials:
        # Try to get node-level metrics
        node_metrics = extract_node_metrics(trial)
        for name, metrics in node_metrics.items():
            if name not in component_costs:
                component_costs[name] = 0.0
            component_costs[name] += metrics.total_cost
            total_cost += metrics.total_cost

        # Also check aggregated metrics for total cost
        if hasattr(trial, "metrics") and not node_metrics:
            cost = trial.metrics.get("total_cost", 0.0)
            total_cost += cost

    if total_cost == 0:
        return dict.fromkeys(component_costs, 0.0)

    return {k: v / total_cost for k, v in component_costs.items()}


def compute_latency_contribution(
    trials: list[Any],
) -> dict[str, float]:
    """Compute latency contribution per component.

    Args:
        trials: List of trial results.

    Returns:
        Dict mapping component names to latency fraction (0-1).
    """
    if not trials:
        return {}

    # Aggregate latencies per component across all trials
    component_latencies: dict[str, float] = {}
    total_latency = 0.0

    for trial in trials:
        # Try to get node-level metrics
        node_metrics = extract_node_metrics(trial)
        for name, metrics in node_metrics.items():
            if name not in component_latencies:
                component_latencies[name] = 0.0
            component_latencies[name] += metrics.total_latency_ms
            total_latency += metrics.total_latency_ms

    if total_latency == 0:
        return dict.fromkeys(component_latencies, 0.0)

    return {k: v / total_latency for k, v in component_latencies.items()}


def compute_sensitivity_scores(
    trials: list[Any],
    target_metric: str = "accuracy",
) -> dict[str, dict[str, float]]:
    """Compute parameter sensitivity per component.

    Args:
        trials: List of trial results.
        target_metric: Metric to compute sensitivity for.

    Returns:
        Dict mapping component names to dict of parameter sensitivity scores.
    """
    if not trials:
        return {}

    # Collect successful trials
    successful = []
    for t in trials:
        if hasattr(t, "metrics") and target_metric in t.metrics:
            if hasattr(t, "is_successful") and t.is_successful:
                successful.append(t)
            elif not hasattr(t, "is_successful"):
                successful.append(t)

    if len(successful) < 3:
        return {}

    # Get all parameters and group by component
    all_params: set[str] = set()
    for trial in successful:
        if hasattr(trial, "config"):
            all_params.update(trial.config.keys())

    component_sensitivity: dict[str, dict[str, float]] = {}

    for param in all_params:
        # Determine component name
        if "." in param:
            component = param.split(".")[0]
        else:
            component = param

        if component not in component_sensitivity:
            component_sensitivity[component] = {}

        # Compute sensitivity via variance contribution
        param_values = []
        metric_values = []

        for trial in successful:
            if hasattr(trial, "config") and param in trial.config:
                val = trial.config[param]
                if isinstance(val, (int, float)):
                    param_values.append(float(val))
                    metric_values.append(trial.metrics[target_metric])

        if len(param_values) >= 3 and len(set(param_values)) > 1:
            # Compute correlation-based sensitivity
            param_mean = sum(param_values) / len(param_values)
            metric_mean = sum(metric_values) / len(metric_values)

            param_var = sum((v - param_mean) ** 2 for v in param_values)
            metric_var = sum((m - metric_mean) ** 2 for m in metric_values)

            if param_var > 0 and metric_var > 0:
                covar = sum(
                    (pv - param_mean) * (mv - metric_mean)
                    for pv, mv in zip(param_values, metric_values)
                )
                corr = abs(covar / ((param_var**0.5) * (metric_var**0.5)))
                component_sensitivity[component][param] = min(corr, 1.0)
            else:
                component_sensitivity[component][param] = 0.0
        else:
            component_sensitivity[component][param] = 0.0

    return component_sensitivity


def generate_recommendation(
    component_name: str,
    sensitivity_scores: dict[str, float],
    optimization_opportunity: str,
) -> str | None:
    """Generate optimization recommendation for a component.

    Args:
        component_name: Name of the component.
        sensitivity_scores: Parameter sensitivity scores.
        optimization_opportunity: 'high', 'medium', or 'low'.

    Returns:
        Human-readable recommendation string or None.
    """
    if optimization_opportunity == "low":
        return None

    if not sensitivity_scores:
        return f"Consider adding more tunable parameters to {component_name}"

    # Find most sensitive parameter
    sorted_params = sorted(sensitivity_scores.items(), key=lambda x: x[1], reverse=True)

    if not sorted_params:
        return None

    top_param, top_score = sorted_params[0]
    param_name = top_param.split(".")[-1] if "." in top_param else top_param

    if optimization_opportunity == "high":
        if "temperature" in param_name.lower():
            return f"Consider expanding {param_name} range for {component_name}"
        elif "model" in param_name.lower():
            return (
                f"Model selection ({param_name}) has high impact - "
                f"try additional models for {component_name}"
            )
        elif "top_k" in param_name.lower() or "k" in param_name.lower():
            return (
                f"Retrieval parameter {param_name} is sensitive - "
                f"explore wider ranges for {component_name}"
            )
        else:
            return (
                f"Parameter {param_name} has high impact on {component_name} - "
                f"consider expanding its search range"
            )
    else:  # medium
        return (
            f"{component_name} has moderate optimization potential - "
            f"focus on {param_name} parameter"
        )


def compute_attribution(
    trials: list[Any],
    target_metric: str = "accuracy",
) -> dict[str, ComponentAttribution]:
    """Compute full attribution analysis for all components.

    Args:
        trials: List of trial results from optimization.
        target_metric: Primary quality metric.

    Returns:
        Dict mapping component names to ComponentAttribution objects.
    """
    if not trials:
        return {}

    # Compute all contributions
    quality = compute_quality_contribution(trials, target_metric)
    cost = compute_cost_contribution(trials)
    latency = compute_latency_contribution(trials)
    sensitivity = compute_sensitivity_scores(trials, target_metric)

    # Get all component names
    all_components = set(quality.keys()) | set(cost.keys()) | set(latency.keys())
    all_components.update(sensitivity.keys())

    # Build attribution objects
    attribution: dict[str, ComponentAttribution] = {}

    for component in all_components:
        sens_scores = sensitivity.get(component, {})

        # Find most sensitive parameter
        most_sensitive = None
        if sens_scores:
            sorted_params = sorted(
                sens_scores.items(), key=lambda x: x[1], reverse=True
            )
            if sorted_params:
                most_sensitive = sorted_params[0][0]

        # Determine optimization opportunity
        quality_score = abs(quality.get(component, 0.0))
        max_sensitivity = max(sens_scores.values()) if sens_scores else 0.0
        combined_score = (quality_score + max_sensitivity) / 2

        if combined_score >= 0.5:
            opportunity = "high"
        elif combined_score >= 0.2:
            opportunity = "medium"
        else:
            opportunity = "low"

        # Generate recommendation
        rec = generate_recommendation(component, sens_scores, opportunity)

        attribution[component] = ComponentAttribution(
            component_name=component,
            quality_contribution=quality.get(component, 0.0),
            cost_contribution=cost.get(component, 0.0),
            latency_contribution=latency.get(component, 0.0),
            sensitivity_scores=sens_scores,
            most_sensitive_param=most_sensitive,
            optimization_opportunity=opportunity,
            recommendation=rec,
        )

    return attribution


def get_attribution_ranked(
    attribution: dict[str, ComponentAttribution],
    by: str = "quality",
) -> list[ComponentAttribution]:
    """Get components ranked by specified metric.

    Args:
        attribution: Dict of component attributions.
        by: Ranking criteria - 'quality', 'cost', 'latency', or 'opportunity'.

    Returns:
        List of ComponentAttribution sorted by the specified metric.
    """
    if not attribution:
        return []

    if by == "quality":
        return sorted(
            attribution.values(),
            key=lambda x: abs(x.quality_contribution),
            reverse=True,
        )
    elif by == "cost":
        return sorted(
            attribution.values(),
            key=lambda x: x.cost_contribution,
            reverse=True,
        )
    elif by == "latency":
        return sorted(
            attribution.values(),
            key=lambda x: x.latency_contribution,
            reverse=True,
        )
    elif by == "opportunity":
        # Order by opportunity level (high, medium, low)
        order = {"high": 0, "medium": 1, "low": 2}
        return sorted(
            attribution.values(),
            key=lambda x: order.get(x.optimization_opportunity, 3),
        )
    else:
        return list(attribution.values())


def export_attribution(
    attribution: dict[str, ComponentAttribution],
    format: str = "dict",
    output_path: str | None = None,
) -> dict[str, Any] | str:
    """Export attribution analysis to various formats.

    Args:
        attribution: Dict of component attributions.
        format: 'dict', 'json', or 'csv'.
        output_path: Optional path to write output.

    Returns:
        Exported data as dict, JSON string, or CSV string.
    """
    import csv
    import io
    import json

    data = {
        "components": {name: attr.to_dict() for name, attr in attribution.items()},
        "summary": {
            "total_components": len(attribution),
            "high_opportunity": sum(
                1 for a in attribution.values() if a.optimization_opportunity == "high"
            ),
            "medium_opportunity": sum(
                1
                for a in attribution.values()
                if a.optimization_opportunity == "medium"
            ),
            "low_opportunity": sum(
                1 for a in attribution.values() if a.optimization_opportunity == "low"
            ),
        },
    }

    if format == "dict":
        return data

    elif format == "json":
        json_str = json.dumps(data, indent=2, default=str)
        if output_path:
            with open(output_path, "w") as f:
                f.write(json_str)
        return json_str

    elif format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(
            [
                "component",
                "quality_contribution",
                "cost_contribution",
                "latency_contribution",
                "most_sensitive_param",
                "optimization_opportunity",
                "recommendation",
            ]
        )

        # Rows
        for name, attr in sorted(attribution.items()):
            writer.writerow(
                [
                    name,
                    f"{attr.quality_contribution:.4f}",
                    f"{attr.cost_contribution:.4f}",
                    f"{attr.latency_contribution:.4f}",
                    attr.most_sensitive_param or "",
                    attr.optimization_opportunity,
                    attr.recommendation or "",
                ]
            )

        csv_str = output.getvalue()
        if output_path:
            with open(output_path, "w") as f:
                f.write(csv_str)
        return csv_str

    else:
        raise ValueError(f"Unsupported format: {format}. Use 'dict', 'json', or 'csv'")
