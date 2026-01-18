"""Agent-specific metric utilities for multi-agent workflow evaluation.

This module provides utilities for computing, extracting, and aggregating
metrics on a per-agent basis in multi-agent workflows. It integrates with
the namespace parsing utilities from traigent.core.namespace and the RAGAS
evaluation framework.

Example:
    >>> from traigent.metrics.agent_metrics import (
    ...     compute_per_agent_metrics,
    ...     aggregate_agent_metrics,
    ...     get_reference_free_metrics,
    ... )
    >>>
    >>> # Compute per-agent metrics from evaluation results
    >>> agent_metrics = compute_per_agent_metrics(
    ...     measures={"grader_cost": 0.002, "generator_cost": 0.004},
    ...     agents=["grader", "generator"],
    ... )
    >>> agent_metrics["grader"]["cost"]
    0.002
"""

# Traceability: CONC-Layer-Core FUNC-MULTI-AGENT FUNC-EVAL-METRICS REQ-EVAL-005

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from traigent.core.namespace import (
    NAMESPACE_DELIMITER,
    build_per_agent_objectives,
    extract_agent_metrics,
    sanitize_metric_name,
)
from traigent.utils.logging import get_logger

logger = get_logger(__name__)

# Reference-free RAGAS metrics that don't require ground truth
# These can be used when expected_output is not available
REFERENCE_FREE_RAGAS_METRICS: tuple[str, ...] = (
    "answer_relevancy",  # Uses LLM to judge if response addresses query
    "faithfulness",  # Uses LLM to check if response is supported by context
)

# Metrics that require reference/ground truth
REFERENCE_REQUIRED_RAGAS_METRICS: tuple[str, ...] = (
    "context_precision",  # Needs reference_contexts
    "context_recall",  # Needs reference_contexts
    "answer_similarity",  # Compares response to reference
)

# All supported RAGAS metrics
ALL_RAGAS_METRICS: tuple[str, ...] = (
    *REFERENCE_FREE_RAGAS_METRICS,
    *REFERENCE_REQUIRED_RAGAS_METRICS,
)

# Agent-specific quality metrics for multi-agent workflows
AGENT_QUALITY_METRICS: tuple[str, ...] = (
    "tool_call_accuracy",  # % of correct tool calls
    "task_completion",  # Did agent complete its task?
    "flow_adherence",  # Did agent follow expected flow?
    "response_quality",  # Quality score of agent's outputs
)

# Agent-specific cost/performance metrics
AGENT_PERFORMANCE_METRICS: tuple[str, ...] = (
    "cost",
    "latency_ms",
    "input_tokens",
    "output_tokens",
    "total_tokens",
)


@dataclass
class AgentMetricsSummary:
    """Summary of metrics for a single agent.

    Attributes:
        agent_id: Unique identifier for the agent
        metrics: Dictionary of metric name to value
        sample_count: Number of samples used to compute metrics
        metadata: Optional additional metadata
    """

    agent_id: str
    metrics: dict[str, float] = field(default_factory=dict)
    sample_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_measures_dict(self, prefix: str = "") -> dict[str, float]:
        """Convert to MeasuresDict-compatible format.

        Args:
            prefix: Optional prefix for metric keys

        Returns:
            Dict with underscore-separated keys and numeric values
        """
        result: dict[str, float] = {}
        for name, value in self.metrics.items():
            # Use underscore (not double underscore) for MeasuresDict
            # e.g., "grader_cost" not "grader__cost"
            key = (
                f"{prefix}{self.agent_id}_{name}"
                if self.agent_id
                else f"{prefix}{name}"
            )
            result[sanitize_metric_name(key)] = float(value)
        return result


@dataclass
class MultiAgentMetricsSummary:
    """Aggregated metrics across multiple agents.

    Attributes:
        agents: Dictionary mapping agent_id to AgentMetricsSummary
        global_metrics: Metrics not tied to any specific agent (e.g., total_cost)
        workflow_id: Optional workflow identifier
    """

    agents: dict[str, AgentMetricsSummary] = field(default_factory=dict)
    global_metrics: dict[str, float] = field(default_factory=dict)
    workflow_id: str | None = None

    def to_measures_dict(self, include_totals: bool = True) -> dict[str, float]:
        """Convert all metrics to MeasuresDict-compatible format.

        Args:
            include_totals: Whether to include global/total metrics

        Returns:
            Flat dict with all metrics using underscore naming
        """
        result: dict[str, float] = {}

        # Add global metrics
        if include_totals:
            for name, value in self.global_metrics.items():
                result[sanitize_metric_name(name)] = float(value)

        # Add per-agent metrics
        for agent_summary in self.agents.values():
            result.update(agent_summary.to_measures_dict())

        return result

    def get_agent(self, agent_id: str) -> AgentMetricsSummary | None:
        """Get metrics summary for a specific agent."""
        return self.agents.get(agent_id)


def compute_per_agent_metrics(
    measures: Mapping[str, float | int],
    agents: Sequence[str],
    *,
    metric_names: Sequence[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Extract per-agent metrics from a flat measures dictionary.

    Uses single underscore convention (MeasuresDict format) for metric names:
    - "grader_cost" → agent="grader", metric="cost"
    - "generator_latency_ms" → agent="generator", metric="latency_ms"

    Args:
        measures: Flat dictionary of metrics (e.g., from MeasuresDict)
        agents: List of agent names to extract metrics for
        metric_names: Optional list of specific metrics to extract
            (e.g., ["cost", "latency_ms"]). If None, extracts all.

    Returns:
        Dictionary mapping agent_id to dict of metric_name to value

    Example:
        >>> measures = {
        ...     "grader_cost": 0.002,
        ...     "grader_latency_ms": 150,
        ...     "generator_cost": 0.004,
        ...     "generator_latency_ms": 300,
        ...     "total_cost": 0.006,
        ... }
        >>> per_agent = compute_per_agent_metrics(measures, ["grader", "generator"])
        >>> per_agent["grader"]
        {'cost': 0.002, 'latency_ms': 150}
    """
    result: dict[str, dict[str, float]] = {}

    for agent in agents:
        agent_metrics = extract_agent_metrics(measures, agent)

        # Strip agent prefix to get raw metric names
        prefix = f"{agent}_"
        stripped: dict[str, float] = {}

        for key, value in agent_metrics.items():
            if key.startswith(prefix):
                metric_name = key[len(prefix) :]
                if metric_names is None or metric_name in metric_names:
                    stripped[metric_name] = float(value)

        if stripped:
            result[agent] = stripped

    return result


def aggregate_agent_metrics(
    per_agent_metrics: Mapping[str, Mapping[str, float]],
    *,
    aggregation: str = "sum",
) -> dict[str, float]:
    """Aggregate metrics across all agents.

    Args:
        per_agent_metrics: Dict mapping agent_id to metrics dict
        aggregation: Aggregation method ("sum", "mean", "max", "min")

    Returns:
        Dictionary of aggregated metrics

    Example:
        >>> per_agent = {
        ...     "grader": {"cost": 0.002, "latency_ms": 150},
        ...     "generator": {"cost": 0.004, "latency_ms": 300},
        ... }
        >>> totals = aggregate_agent_metrics(per_agent, aggregation="sum")
        >>> totals["total_cost"]
        0.006
    """
    if not per_agent_metrics:
        return {}

    # Collect all values per metric
    metric_values: dict[str, list[float]] = {}
    for agent_metrics in per_agent_metrics.values():
        for name, value in agent_metrics.items():
            if name not in metric_values:
                metric_values[name] = []
            metric_values[name].append(float(value))

    # Aggregate
    result: dict[str, float] = {}
    for name, values in metric_values.items():
        if aggregation == "sum":
            result[f"total_{name}"] = sum(values)
        elif aggregation == "mean":
            result[f"mean_{name}"] = sum(values) / len(values) if values else 0.0
        elif aggregation == "max":
            result[f"max_{name}"] = max(values) if values else 0.0
        elif aggregation == "min":
            result[f"min_{name}"] = min(values) if values else 0.0
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

    return result


def get_reference_free_metrics(
    requested_metrics: Sequence[str] | None = None,
) -> list[str]:
    """Get RAGAS metrics that don't require ground truth.

    These metrics can be used when expected_output is not available,
    making them suitable for production monitoring or when ground truth
    is expensive to obtain.

    Args:
        requested_metrics: Optional list of metrics to filter.
            Returns intersection with reference-free metrics.

    Returns:
        List of reference-free metric names

    Example:
        >>> get_reference_free_metrics()
        ['answer_relevancy', 'faithfulness']
        >>> get_reference_free_metrics(["faithfulness", "context_precision"])
        ['faithfulness']
    """
    if requested_metrics is None:
        return list(REFERENCE_FREE_RAGAS_METRICS)

    return [m for m in requested_metrics if m in REFERENCE_FREE_RAGAS_METRICS]


def get_metrics_for_available_data(
    has_reference: bool = False,
    has_contexts: bool = False,
    has_llm: bool = False,
    requested_metrics: Sequence[str] | None = None,
) -> list[str]:
    """Determine which RAGAS metrics can be computed based on available data.

    This is useful for automatically selecting appropriate metrics when
    not all data (reference, contexts, LLM) is available.

    Args:
        has_reference: Whether ground truth/expected output is available
        has_contexts: Whether retrieved/reference contexts are available
        has_llm: Whether an LLM is configured for evaluation
        requested_metrics: Optional list of metrics to filter

    Returns:
        List of computable metric names

    Example:
        >>> # Only have contexts, no reference, no LLM
        >>> get_metrics_for_available_data(
        ...     has_reference=False,
        ...     has_contexts=True,
        ...     has_llm=False,
        ... )
        []  # No metrics can be computed without LLM or reference

        >>> # Have LLM and contexts
        >>> get_metrics_for_available_data(
        ...     has_reference=False,
        ...     has_contexts=True,
        ...     has_llm=True,
        ... )
        ['answer_relevancy', 'faithfulness']
    """
    available: list[str] = []

    # LLM-based reference-free metrics
    if has_llm:
        available.extend(["answer_relevancy", "faithfulness"])

    # Reference-based metrics
    if has_reference:
        # answer_similarity can work without LLM (falls back to string similarity)
        available.append("answer_similarity")

        # Context metrics need both reference and contexts
        if has_contexts:
            available.extend(["context_precision", "context_recall"])

    # Filter by requested metrics if provided
    if requested_metrics is not None:
        available = [m for m in available if m in requested_metrics]

    return available


def build_agent_objectives(
    agents: Sequence[str],
    *,
    include_cost: bool = True,
    include_latency: bool = True,
    include_quality: bool = False,
    include_totals: bool = True,
) -> list[str]:
    """Build a list of objectives for multi-agent optimization.

    Creates objective names for each agent using underscore convention
    (MeasuresDict-compatible).

    Args:
        agents: List of agent names
        include_cost: Include cost metrics (default: True)
        include_latency: Include latency metrics (default: True)
        include_quality: Include quality metrics like accuracy (default: False)
        include_totals: Include total/aggregate metrics (default: True)

    Returns:
        List of objective names

    Example:
        >>> build_agent_objectives(["grader", "generator"], include_quality=False)
        ['total_cost', 'total_latency_ms', 'grader_cost', 'grader_latency_ms',
         'generator_cost', 'generator_latency_ms']
    """
    metrics: list[str] = []

    if include_cost:
        metrics.append("cost")
    if include_latency:
        metrics.append("latency_ms")
    if include_quality:
        metrics.extend(["accuracy", "quality_score"])

    return build_per_agent_objectives(
        agents=list(agents),
        metrics=metrics,
        include_totals=include_totals,
    )


def extract_namespaced_config_for_agent(
    config: Mapping[str, Any],
    agent: str,
) -> dict[str, Any]:
    """Extract configuration values for a specific agent.

    Uses double underscore (__) convention for namespaced parameters:
    - "grader__temperature" → agent="grader", param="temperature"

    Args:
        config: Configuration dictionary with namespaced keys
        agent: Agent name to extract config for

    Returns:
        Dictionary with parameter names (without agent prefix) and values

    Example:
        >>> config = {
        ...     "grader__temperature": 0.3,
        ...     "grader__model": "gpt-4o-mini",
        ...     "generator__temperature": 0.7,
        ...     "max_retries": 3,
        ... }
        >>> extract_namespaced_config_for_agent(config, "grader")
        {'temperature': 0.3, 'model': 'gpt-4o-mini'}
    """
    result: dict[str, Any] = {}
    prefix = f"{agent}{NAMESPACE_DELIMITER}"

    for key, value in config.items():
        if key.startswith(prefix):
            param_name = key[len(prefix) :]
            result[param_name] = value

    return result


def validate_agent_metrics(
    measures: Mapping[str, float | int],
    agents: Sequence[str],
    required_metrics: Sequence[str],
) -> tuple[bool, list[str]]:
    """Validate that all required metrics are present for each agent.

    Args:
        measures: Flat metrics dictionary
        agents: List of agents to validate
        required_metrics: Metrics that must be present for each agent

    Returns:
        Tuple of (is_valid, list of missing metric names)

    Example:
        >>> measures = {"grader_cost": 0.002, "generator_cost": 0.004}
        >>> valid, missing = validate_agent_metrics(
        ...     measures, ["grader", "generator"], ["cost", "latency_ms"]
        ... )
        >>> valid
        False
        >>> missing
        ['grader_latency_ms', 'generator_latency_ms']
    """
    missing: list[str] = []

    for agent in agents:
        for metric in required_metrics:
            key = f"{agent}_{metric}"
            if key not in measures:
                missing.append(key)

    return len(missing) == 0, missing


__all__ = [
    # Constants
    "REFERENCE_FREE_RAGAS_METRICS",
    "REFERENCE_REQUIRED_RAGAS_METRICS",
    "ALL_RAGAS_METRICS",
    "AGENT_QUALITY_METRICS",
    "AGENT_PERFORMANCE_METRICS",
    # Data classes
    "AgentMetricsSummary",
    "MultiAgentMetricsSummary",
    # Functions
    "compute_per_agent_metrics",
    "aggregate_agent_metrics",
    "get_reference_free_metrics",
    "get_metrics_for_available_data",
    "build_agent_objectives",
    "extract_namespaced_config_for_agent",
    "validate_agent_metrics",
]
