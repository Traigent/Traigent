"""Namespace parsing utilities for multi-agent parameter optimization.

This module provides utilities for parsing namespaced configuration parameters
and objectives, enabling per-agent optimization in multi-agent workflows.

Naming Convention:
    Double underscore (__) is used as the delimiter between agent name and parameter:
    - "grader__temperature" → agent="grader", param="temperature"
    - "generator__model" → agent="generator", param="model"

    This avoids conflicts with MeasuresDict which uses single underscore for
    compound names (e.g., "total_cost", "latency_ms").

Example:
    >>> from traigent.core.namespace import parse_namespaced_param, is_namespaced
    >>>
    >>> # Parse namespaced parameter
    >>> parse_namespaced_param("grader__temperature")
    ("grader", "temperature")
    >>>
    >>> # Check if parameter is namespaced
    >>> is_namespaced("grader__temperature")
    True
    >>> is_namespaced("global_temperature")  # Single underscore = not namespaced
    False
"""

# Traceability: CONC-Layer-Core FUNC-MULTI-AGENT REQ-INT-008

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from traigent.utils.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# Namespace delimiter - double underscore to avoid MeasuresDict conflicts
NAMESPACE_DELIMITER = "__"


@dataclass(frozen=True)
class ParsedNamespace:
    """Result of parsing a namespaced parameter or metric name.

    Attributes:
        agent: The agent name (e.g., "grader", "generator")
        name: The parameter/metric name without namespace (e.g., "temperature", "cost")
        original: The original full string (e.g., "grader__temperature")
    """

    agent: str
    name: str
    original: str

    @property
    def is_namespaced(self) -> bool:
        """Return True if this was originally a namespaced parameter."""
        return self.agent != ""


def is_namespaced(param: str) -> bool:
    """Check if a parameter/metric name uses namespace convention.

    Uses double underscore (__) as the namespace delimiter.
    Single underscores are NOT treated as namespace delimiters.

    Args:
        param: Parameter or metric name to check

    Returns:
        True if the name contains __ delimiter, False otherwise

    Examples:
        >>> is_namespaced("grader__temperature")
        True
        >>> is_namespaced("total_cost")  # Single underscore
        False
        >>> is_namespaced("temperature")  # No underscore
        False
    """
    return NAMESPACE_DELIMITER in param


def parse_namespaced_param(param: str) -> tuple[str, str]:
    """Parse a namespaced parameter into (agent, param_name).

    Uses double underscore (__) as the namespace delimiter.

    Args:
        param: Parameter name, optionally namespaced (e.g., "grader__temperature")

    Returns:
        Tuple of (agent_name, param_name). If not namespaced, returns ("", param).

    Examples:
        >>> parse_namespaced_param("grader__temperature")
        ("grader", "temperature")
        >>> parse_namespaced_param("generator__model")
        ("generator", "model")
        >>> parse_namespaced_param("temperature")
        ("", "temperature")
        >>> parse_namespaced_param("grader__nested__param")  # Multiple delimiters
        ("grader", "nested__param")
    """
    if NAMESPACE_DELIMITER in param:
        # Split on first __ only (in case param name contains __)
        parts = param.split(NAMESPACE_DELIMITER, 1)
        return (parts[0], parts[1])
    return ("", param)


def parse_namespace(name: str) -> ParsedNamespace:
    """Parse a namespaced name into a ParsedNamespace object.

    Args:
        name: Name to parse, optionally namespaced

    Returns:
        ParsedNamespace with agent, name, and original fields

    Examples:
        >>> ns = parse_namespace("grader__temperature")
        >>> ns.agent
        'grader'
        >>> ns.name
        'temperature'
        >>> ns.is_namespaced
        True
    """
    agent, param_name = parse_namespaced_param(name)
    return ParsedNamespace(agent=agent, name=param_name, original=name)


def create_namespaced_param(agent: str, param: str) -> str:
    """Create a namespaced parameter name.

    Args:
        agent: Agent name (e.g., "grader")
        param: Parameter name (e.g., "temperature")

    Returns:
        Namespaced parameter name (e.g., "grader__temperature")

    Example:
        >>> create_namespaced_param("grader", "temperature")
        'grader__temperature'
    """
    return f"{agent}{NAMESPACE_DELIMITER}{param}"


def extract_agents_from_config(
    configuration_space: dict[str, Any],
) -> set[str]:
    """Extract unique agent names from namespaced configuration space.

    Args:
        configuration_space: Dictionary with parameter names as keys

    Returns:
        Set of unique agent names found in namespaced parameters

    Example:
        >>> config = {
        ...     "grader__temperature": [0.0, 0.3],
        ...     "grader__model": ["gpt-4"],
        ...     "generator__temperature": [0.5, 0.7],
        ...     "max_retries": [3],  # Global, not namespaced
        ... }
        >>> extract_agents_from_config(config)
        {'grader', 'generator'}
    """
    agents: set[str] = set()
    for param in configuration_space:
        agent, _ = parse_namespaced_param(param)
        if agent:
            agents.add(agent)
    return agents


def group_params_by_agent(
    configuration_space: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Group configuration space parameters by agent.

    Args:
        configuration_space: Dictionary with parameter names as keys

    Returns:
        Dictionary mapping agent names to their parameter dicts.
        Global parameters (non-namespaced) are grouped under empty string key "".

    Example:
        >>> config = {
        ...     "grader__temperature": [0.0, 0.3],
        ...     "generator__model": ["gpt-4"],
        ...     "max_retries": [3],
        ... }
        >>> grouped = group_params_by_agent(config)
        >>> grouped["grader"]
        {'temperature': [0.0, 0.3]}
        >>> grouped["generator"]
        {'model': ['gpt-4']}
        >>> grouped[""]  # Global params
        {'max_retries': [3]}
    """
    grouped: dict[str, dict[str, Any]] = {}

    for param, value in configuration_space.items():
        agent, param_name = parse_namespaced_param(param)
        if agent not in grouped:
            grouped[agent] = {}
        grouped[agent][param_name] = value

    return grouped


def flatten_agent_config(
    agent_configs: dict[str, dict[str, Any]],
    exclude_global: bool = False,
) -> dict[str, Any]:
    """Flatten agent-grouped config back to namespaced format.

    Args:
        agent_configs: Dictionary mapping agent names to their parameter dicts
        exclude_global: If True, exclude global parameters (empty string agent)

    Returns:
        Flattened dictionary with namespaced parameter names

    Example:
        >>> agent_configs = {
        ...     "grader": {"temperature": 0.3},
        ...     "": {"max_retries": 3},
        ... }
        >>> flatten_agent_config(agent_configs)
        {'grader__temperature': 0.3, 'max_retries': 3}
    """
    flattened: dict[str, Any] = {}

    for agent, params in agent_configs.items():
        for param_name, value in params.items():
            if agent:
                # Namespaced parameter
                full_name = create_namespaced_param(agent, param_name)
            elif exclude_global:
                continue
            else:
                # Global parameter (no namespace)
                full_name = param_name
            flattened[full_name] = value

    return flattened


def sanitize_metric_name(name: str) -> str:
    """Sanitize a metric name for MeasuresDict compatibility.

    Replaces non-alphanumeric characters (except underscore) with underscore.
    Ensures the name starts with a letter or underscore.

    Args:
        name: Metric name to sanitize

    Returns:
        Sanitized name safe for MeasuresDict keys

    Example:
        >>> sanitize_metric_name("grader-v2.cost")
        'grader_v2_cost'
        >>> sanitize_metric_name("123invalid")
        '_123invalid'
    """
    # Replace invalid characters with underscore
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)

    # Ensure starts with letter or underscore
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized

    return sanitized


def extract_agent_metrics(
    measures: dict[str, float | int],
    agent: str,
    metric_suffix: str | None = None,
) -> dict[str, float | int]:
    """Extract metrics for a specific agent from a flat measures dict.

    Args:
        measures: Flat dictionary of all metrics (e.g., from MeasuresDict)
        agent: Agent name to extract metrics for
        metric_suffix: Optional suffix to filter by (e.g., "cost", "latency_ms")

    Returns:
        Dictionary of metrics for the specified agent

    Example:
        >>> measures = {
        ...     "total_cost": 0.01,
        ...     "grader_cost": 0.003,
        ...     "grader_latency_ms": 150,
        ...     "generator_cost": 0.007,
        ... }
        >>> extract_agent_metrics(measures, "grader")
        {'grader_cost': 0.003, 'grader_latency_ms': 150}
        >>> extract_agent_metrics(measures, "grader", metric_suffix="cost")
        {'grader_cost': 0.003}
    """
    prefix = f"{agent}_"
    result: dict[str, float | int] = {}

    for key, value in measures.items():
        if key.startswith(prefix):
            if metric_suffix is None or key.endswith(f"_{metric_suffix}"):
                result[key] = value

    return result


def build_per_agent_objectives(
    agents: list[str],
    metrics: list[str],
    include_totals: bool = True,
) -> list[str]:
    """Build a list of per-agent objective names.

    Creates objective names using single underscore (MeasuresDict format),
    not double underscore (configuration space format).

    Args:
        agents: List of agent names
        metrics: List of base metric names (e.g., ["cost", "latency_ms"])
        include_totals: If True, include total_* metrics as well

    Returns:
        List of objective names

    Example:
        >>> build_per_agent_objectives(["grader", "generator"], ["cost", "latency_ms"])
        ['total_cost', 'total_latency_ms', 'grader_cost', 'grader_latency_ms',
         'generator_cost', 'generator_latency_ms']
    """
    objectives: list[str] = []

    if include_totals:
        for metric in metrics:
            objectives.append(f"total_{metric}")

    for agent in agents:
        for metric in metrics:
            objectives.append(f"{agent}_{metric}")

    return objectives
