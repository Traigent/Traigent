"""Agent configuration inference and validation for multi-agent experiments.

This module provides utilities to build AgentConfiguration from various input
sources, including explicit agent definitions, per-parameter agent assignments,
and prefix-based inference.

Example:
    >>> from traigent.api.agent_inference import build_agent_configuration
    >>> from traigent.api.parameter_ranges import Choices, Range
    >>>
    >>> # Using explicit parameter agents
    >>> config = build_agent_configuration(
    ...     configuration_space={
    ...         "model": Choices(["gpt-4"]),
    ...         "temperature": Range(0.0, 1.0),
    ...     },
    ...     parameter_agents={"model": "financial", "temperature": "financial"},
    ...     agent_measures={"financial": ["accuracy"]},
    ... )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from traigent.api.types import (
    AgentConfiguration,
    AgentDefinition,
    GlobalConfiguration,
)
from traigent.utils.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


def build_agent_configuration(
    configuration_space: dict[str, Any],
    explicit_agents: dict[str, AgentDefinition] | None = None,
    agent_prefixes: list[str] | None = None,
    agent_measures: dict[str, list[str]] | None = None,
    global_measures: list[str] | None = None,
    parameter_agents: dict[str, str] | None = None,
) -> AgentConfiguration | None:
    """Build AgentConfiguration from various input sources.

    This function determines agent groupings for parameters and measures using
    the following priority:
    1. explicit_agents - Full agent definitions provided by user (highest priority)
    2. parameter_agents - Agent assignments from Range(..., agent="x")
    3. agent_prefixes - Prefix-based inference from parameter naming patterns

    Args:
        configuration_space: Dictionary of parameter names to ranges/values.
        explicit_agents: Explicit agent definitions (overrides all inference).
        agent_prefixes: List of prefixes for prefix-based parameter grouping.
            Each prefix should match parameters named "<prefix>_<param>".
        agent_measures: Map of agent_id to list of measure_ids.
        global_measures: List of measure IDs not tied to any specific agent.
        parameter_agents: Direct mapping of parameter names to agent IDs,
            typically extracted from Range(..., agent="x") parameters.

    Returns:
        AgentConfiguration if multiple agents detected, None for single-agent
        experiments (no grouping needed).

    Raises:
        ValueError: If agent_prefixes contains prefixes with no matching parameters.

    Example:
        >>> # Method 1: Explicit agents
        >>> config = build_agent_configuration(
        ...     configuration_space={"model": ["gpt-4"], "temp": (0, 1)},
        ...     explicit_agents={
        ...         "financial": AgentDefinition(
        ...             display_name="Financial",
        ...             parameter_keys=["model", "temp"],
        ...         )
        ...     }
        ... )

        >>> # Method 2: Per-parameter agents
        >>> config = build_agent_configuration(
        ...     configuration_space={"model": ["gpt-4"], "temp": (0, 1)},
        ...     parameter_agents={"model": "financial", "temp": "financial"},
        ... )

        >>> # Method 3: Prefix-based inference
        >>> config = build_agent_configuration(
        ...     configuration_space={
        ...         "financial_model": ["gpt-4"],
        ...         "legal_model": ["claude"],
        ...     },
        ...     agent_prefixes=["financial", "legal"],
        ... )
    """
    # Priority 1: Explicit agent definitions
    if explicit_agents:
        return _build_from_explicit_agents(
            explicit_agents=explicit_agents,
            global_measures=global_measures,
        )

    # Collect parameter-to-agent assignments
    param_to_agent: dict[str, str] = {}

    # Priority 2: From Range(..., agent="x") parameters
    if parameter_agents:
        param_to_agent.update(parameter_agents)

    # Priority 3: From agent_prefixes
    if agent_prefixes:
        _validate_prefixes(agent_prefixes, list(configuration_space.keys()))
        _assign_by_prefix(
            param_keys=list(configuration_space.keys()),
            prefixes=agent_prefixes,
            param_to_agent=param_to_agent,
        )

    # Check if we have multiple agents
    unique_agents = set(param_to_agent.values())
    if len(unique_agents) < 2:
        # Single agent or no agents detected - no grouping needed
        logger.debug(
            "Single agent or no multi-agent configuration detected. "
            f"Unique agents: {unique_agents}"
        )
        return None

    return _build_from_assignments(
        configuration_space=configuration_space,
        param_to_agent=param_to_agent,
        agent_measures=agent_measures,
        global_measures=global_measures,
    )


def extract_parameter_agents(
    configuration_space: dict[str, Any],
) -> dict[str, str]:
    """Extract agent assignments from ParameterRange objects.

    Scans configuration_space for ParameterRange instances that have an
    `agent` attribute set, and returns a mapping of parameter names to
    their assigned agent IDs.

    Args:
        configuration_space: Dictionary of parameter names to ranges/values.

    Returns:
        Dictionary mapping parameter names to agent IDs for parameters
        that have explicit agent assignments.

    Example:
        >>> from traigent.api.parameter_ranges import Choices, Range
        >>> config_space = {
        ...     "model": Choices(["gpt-4"], agent="financial"),
        ...     "temperature": Range(0.0, 1.0, agent="financial"),
        ...     "max_tokens": Range(100, 4096),  # No agent
        ... }
        >>> agents = extract_parameter_agents(config_space)
        >>> agents
        {'model': 'financial', 'temperature': 'financial'}
    """
    # Import here to avoid circular dependency
    from traigent.api.parameter_ranges import ParameterRange

    result: dict[str, str] = {}
    for key, value in configuration_space.items():
        if isinstance(value, ParameterRange):
            agent = getattr(value, "agent", None)
            if agent:
                result[key] = agent
    return result


def _build_from_explicit_agents(
    explicit_agents: dict[str, AgentDefinition],
    global_measures: list[str] | None,
) -> AgentConfiguration:
    """Build AgentConfiguration from explicit agent definitions."""
    global_config = None
    if global_measures:
        global_config = GlobalConfiguration(
            parameter_keys=[],
            measure_ids=global_measures,
            order=99,
        )

    return AgentConfiguration(
        version="1.0",
        agents=explicit_agents,
        global_config=global_config,
        auto_inferred=False,
    )


def _validate_prefixes(prefixes: list[str], param_keys: list[str]) -> None:
    """Validate that all prefixes match at least one parameter.

    Raises:
        ValueError: If any prefix has no matching parameters.
    """
    for prefix in prefixes:
        matches = [k for k in param_keys if k.startswith(prefix + "_") or k == prefix]
        if not matches:
            raise ValueError(
                f"agent_prefixes contains '{prefix}' but no parameters start with "
                f"'{prefix}_'. Available parameters: {param_keys}"
            )


def _assign_by_prefix(
    param_keys: list[str],
    prefixes: list[str],
    param_to_agent: dict[str, str],
) -> None:
    """Assign parameters to agents based on prefix matching.

    Modifies param_to_agent in place, only for parameters not already assigned.
    Matches both 'prefix_*' patterns and exact 'prefix' matches.
    """
    for param_key in param_keys:
        if param_key in param_to_agent:
            continue  # Already assigned by higher priority source

        for prefix in prefixes:
            # Match 'prefix_*' patterns OR exact 'prefix' match
            if param_key.startswith(prefix + "_") or param_key == prefix:
                param_to_agent[param_key] = prefix
                break


def _build_from_assignments(
    configuration_space: dict[str, Any],
    param_to_agent: dict[str, str],
    agent_measures: dict[str, list[str]] | None,
    global_measures: list[str] | None,
) -> AgentConfiguration:
    """Build AgentConfiguration from parameter-to-agent assignments.

    Raises:
        ValueError: If agent_measures references agents not found in parameters.
    """
    agent_measures = agent_measures or {}

    # Get unique agents sorted for consistent ordering
    unique_agents = sorted(set(param_to_agent.values()))

    # Validate agent_measures references only defined agents (fail fast)
    if agent_measures:
        unknown_agents = set(agent_measures.keys()) - set(unique_agents)
        if unknown_agents:
            raise ValueError(
                f"agent_measures contains unknown agents: {sorted(unknown_agents)}. "
                f"Available agents: {unique_agents}"
            )

    # Build agent definitions
    agents: dict[str, AgentDefinition] = {}
    for order, agent_id in enumerate(unique_agents):
        agent_params = [k for k, a in param_to_agent.items() if a == agent_id]
        agent_measure_ids = agent_measures.get(agent_id, [])

        agents[agent_id] = AgentDefinition(
            display_name=_humanize(agent_id),
            parameter_keys=agent_params,
            measure_ids=agent_measure_ids,
            primary_model=_find_model_key(agent_params),
            order=order,
        )

    # Collect global (unassigned) parameters
    global_params = [k for k in configuration_space if k not in param_to_agent]

    # Build global config if we have global params or measures
    global_config = None
    if global_params or global_measures:
        global_config = GlobalConfiguration(
            parameter_keys=global_params,
            measure_ids=global_measures or [],
            order=99,
        )

    return AgentConfiguration(
        version="1.0",
        agents=agents,
        global_config=global_config,
        auto_inferred=True,
    )


def _humanize(agent_id: str) -> str:
    """Convert agent_id to human-readable display name.

    Examples:
        >>> _humanize("financial")
        'Financial'
        >>> _humanize("financial_agent")
        'Financial Agent'
        >>> _humanize("my_llm_router")
        'My Llm Router'
    """
    return agent_id.replace("_", " ").title()


def _find_model_key(param_keys: list[str]) -> str | None:
    """Find the primary model key from a list of parameter names.

    Looks for parameter names containing "model" (case-insensitive).

    Args:
        param_keys: List of parameter names.

    Returns:
        The first parameter name containing "model", or None if not found.
    """
    for key in param_keys:
        if "model" in key.lower():
            return key
    return None
