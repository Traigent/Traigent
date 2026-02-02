"""Configuration space discovery for Hybrid API mode.

Provides auto-discovery of TVAR definitions from external services
and conversion to Traigent configuration space format.
"""

# Traceability: HYBRID-MODE-OPTIMIZATION TVAR-DISCOVERY TVL-0.9

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from traigent.hybrid.protocol import ConfigSpaceResponse, TVARDefinition
from traigent.utils.logging import get_logger

if TYPE_CHECKING:
    from traigent.hybrid.transport import HybridTransport

logger = get_logger(__name__)


class ConfigSpaceDiscovery:
    """Auto-discover configuration space from external API.

    Fetches TVAR definitions from external services and normalizes
    them to Traigent configuration space format for use with the
    optimizer.

    TVL Type Mapping:
        - enum: list of values -> ["a", "b", "c"]
        - bool: -> [True, False]
        - int: range -> {"low": min, "high": max, "type": "int"}
        - float: range + resolution -> {"low": min, "high": max, "step": step}
        - str: allowed values -> ["value1", "value2"]

    Example:
        discovery = ConfigSpaceDiscovery(transport)
        config_space = await discovery.fetch_and_normalize()
        # config_space = {"temperature": {"low": 0.0, "high": 2.0, "step": 0.1}, ...}
    """

    def __init__(self, transport: HybridTransport) -> None:
        """Initialize discovery with transport.

        Args:
            transport: HybridTransport for fetching config space.
        """
        self._transport = transport
        self._cached_response: ConfigSpaceResponse | None = None

    async def fetch(self) -> ConfigSpaceResponse:
        """Fetch raw config space response from external service.

        Returns:
            ConfigSpaceResponse with TVARs and constraints.

        Raises:
            TransportError: If fetch fails.
        """
        if self._cached_response is not None:
            return self._cached_response

        self._cached_response = await self._transport.discover_config_space()
        logger.info(
            f"Discovered {len(self._cached_response.tvars)} TVARs "
            f"for capability '{self._cached_response.capability_id}'"
        )
        return self._cached_response

    async def fetch_and_normalize(self) -> dict[str, Any]:
        """Fetch TVARs and convert to Traigent config space format.

        Returns:
            Configuration space dictionary suitable for Traigent optimizer.

        Raises:
            TransportError: If fetch fails.
        """
        response = await self.fetch()
        return response.to_traigent_config_space()

    def get_capability_id(self) -> str | None:
        """Get the capability ID from cached response.

        Returns:
            Capability ID if available, None if not fetched yet.
        """
        return self._cached_response.capability_id if self._cached_response else None

    def get_constraints(self) -> dict[str, list[str]] | None:
        """Get constraints from cached response.

        Returns:
            Constraints dict if available, None if not fetched yet.
        """
        return self._cached_response.constraints if self._cached_response else None

    def get_tvars(self) -> list[TVARDefinition]:
        """Get raw TVAR definitions from cached response.

        Returns:
            List of TVARDefinition objects (empty if not fetched yet).
        """
        return self._cached_response.tvars if self._cached_response else []

    def get_tool_tvars(self) -> list[TVARDefinition]:
        """Get TVARs that represent tool configurations.

        Returns:
            List of TVARDefinition objects where is_tool=True.
        """
        return [tvar for tvar in self.get_tvars() if tvar.is_tool]

    def get_agents(self) -> list[str]:
        """Get list of agent names from TVARs.

        For multi-agent configurations, TVARs can be grouped by agent.

        Returns:
            List of unique agent names (empty if no multi-agent grouping).
        """
        agents = set()
        for tvar in self.get_tvars():
            if tvar.agent:
                agents.add(tvar.agent)
        return sorted(agents)

    def get_tvars_for_agent(self, agent_name: str) -> list[TVARDefinition]:
        """Get TVARs for a specific agent.

        Args:
            agent_name: Agent name to filter by.

        Returns:
            List of TVARDefinition objects for the specified agent.
        """
        return [tvar for tvar in self.get_tvars() if tvar.agent == agent_name]

    def clear_cache(self) -> None:
        """Clear cached config space response.

        Call this to force re-fetching on next access.
        """
        self._cached_response = None


def normalize_tvar_to_config_space(tvar: TVARDefinition) -> Any:
    """Convert a single TVAR to Traigent config space format.

    Args:
        tvar: TVAR definition to convert.

    Returns:
        Configuration space entry in Traigent format.

    Examples:
        >>> tvar = TVARDefinition(name="temp", type="float", domain={"range": [0.0, 1.0]})
        >>> normalize_tvar_to_config_space(tvar)
        {"low": 0.0, "high": 1.0}

        >>> tvar = TVARDefinition(name="model", type="enum", domain={"values": ["a", "b"]})
        >>> normalize_tvar_to_config_space(tvar)
        ["a", "b"]
    """
    return tvar.to_traigent_config_space()


def merge_config_spaces(
    base: dict[str, Any],
    override: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge configuration spaces with override taking precedence.

    Useful for combining auto-discovered TVARs with user overrides.

    Args:
        base: Base configuration space (e.g., from discovery).
        override: Override configuration space (e.g., from user).

    Returns:
        Merged configuration space.

    Examples:
        >>> base = {"temperature": {"low": 0.0, "high": 2.0}}
        >>> override = {"temperature": {"low": 0.0, "high": 1.0}}
        >>> merge_config_spaces(base, override)
        {"temperature": {"low": 0.0, "high": 1.0}}
    """
    if override is None:
        return dict(base)

    result = dict(base)
    for key, value in override.items():
        result[key] = value

    return result


def validate_config_against_tvars(
    config: dict[str, Any],
    tvars: list[TVARDefinition],
) -> list[str]:
    """Validate a configuration against TVAR definitions.

    Checks that:
    - All required TVARs are present in config
    - Values are within defined domains
    - Constraints are satisfied (basic check)

    Args:
        config: Configuration to validate.
        tvars: TVAR definitions to validate against.

    Returns:
        List of validation error messages (empty if valid).
    """
    errors: list[str] = []

    tvar_map = {tvar.name: tvar for tvar in tvars}

    # Check for unknown config keys
    for key in config:
        if key not in tvar_map:
            errors.append(f"Unknown configuration key: {key}")

    # Validate each TVAR
    for tvar in tvars:
        if tvar.name not in config:
            if tvar.default is None:
                errors.append(f"Missing required configuration: {tvar.name}")
            continue

        value = config[tvar.name]

        # Type-specific validation
        if tvar.type == "bool":
            if not isinstance(value, bool):
                errors.append(f"{tvar.name}: expected bool, got {type(value).__name__}")

        elif tvar.type == "int":
            if not isinstance(value, int) or isinstance(value, bool):
                errors.append(f"{tvar.name}: expected int, got {type(value).__name__}")
            else:
                range_spec = tvar.domain.get("range", [])
                if len(range_spec) >= 2:
                    if value < range_spec[0] or value > range_spec[1]:
                        errors.append(
                            f"{tvar.name}: {value} not in range "
                            f"[{range_spec[0]}, {range_spec[1]}]"
                        )

        elif tvar.type == "float":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                errors.append(
                    f"{tvar.name}: expected float, got {type(value).__name__}"
                )
            else:
                range_spec = tvar.domain.get("range", [])
                if len(range_spec) >= 2:
                    if value < range_spec[0] or value > range_spec[1]:
                        errors.append(
                            f"{tvar.name}: {value} not in range "
                            f"[{range_spec[0]}, {range_spec[1]}]"
                        )

        elif tvar.type == "enum" or tvar.type == "str":
            allowed = tvar.domain.get("values", [])
            if allowed and value not in allowed:
                errors.append(f"{tvar.name}: '{value}' not in allowed values {allowed}")

    return errors
