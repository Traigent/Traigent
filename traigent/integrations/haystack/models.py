"""Data models for Haystack pipeline integration.

This module defines the PipelineSpec and TVARScope dataclasses used to
represent discovered pipeline structure and tuned variables (TVARs).

Terminology (aligned with TVL - Tuning Variable Language):
- TVAR (Tuned Variable): A parameter that can be optimized
- DiscoveredTVAR: A TVAR discovered through pipeline introspection
- TVARScope: A namespace/container for TVARs (maps to Haystack Component)
- PipelineSpec: The complete discovered pipeline structure with all scopes
- Connection: A data flow edge between TVARScopes

These types represent *discovered* configuration from introspection.
For optimization, these will be converted to core TVL types (TVAR, ExplorationSpace)
in Epic 2+.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Parameter names that may contain sensitive values (used for masking in __repr__)
# Use exact matches and specific patterns to avoid false positives like "tokenizer"
_SENSITIVE_PARAM_EXACT = frozenset(
    {"api_key", "apikey", "secret", "password", "token", "credential", "auth_token"}
)
_SENSITIVE_PARAM_SUFFIXES = ("_key", "_secret", "_password", "_token", "_credential")


@dataclass
class DiscoveredTVAR:
    """A tuned variable (TVAR) discovered through pipeline introspection.

    In TVL terminology, a TVAR is a parameter that can be optimized.
    DiscoveredTVAR represents a TVAR found by inspecting a pipeline component's
    __init__ signature and current values.

    Attributes:
        name: The TVAR name (parameter name in the component)
        value: The current value of the TVAR
        python_type: The Python type as a string
            (e.g., "int", "float", "str", "Literal")
        type_hint: The original type hint as a string representation
        is_tunable: Whether this TVAR can be tuned during optimization
        literal_choices: For Literal types, the allowed values
        is_optional: Whether the TVAR accepts None (Optional[T])
        non_tunable_reason: Explanation of why the TVAR is not tunable
        default_range: Suggested (min, max) range for tunable numeric TVARs
        range_type: The scale type: "continuous", "discrete", or "log"
    """

    name: str
    value: Any
    python_type: str
    type_hint: str | None = None
    is_tunable: bool = True
    literal_choices: list[Any] | None = None
    is_optional: bool = False
    non_tunable_reason: str | None = None
    default_range: tuple[Any, Any] | None = None
    range_type: str | None = None  # "continuous", "discrete", or "log"

    def _is_sensitive(self) -> bool:
        """Check if this TVAR might contain sensitive data.

        Uses exact matches and specific suffixes to avoid false positives
        like 'tokenizer', 'keyboard_layout', or 'authenticate'.
        """
        name_lower = self.name.lower()
        # Check exact matches first
        if name_lower in _SENSITIVE_PARAM_EXACT:
            return True
        # Check suffixes (e.g., "openai_api_key", "client_secret")
        return name_lower.endswith(_SENSITIVE_PARAM_SUFFIXES)

    def _get_display_value(self) -> str:
        """Get a safe display value, masking secrets."""
        if self._is_sensitive() and self.value is not None:
            return "'***'"
        return repr(self.value)

    def __repr__(self) -> str:
        """Return a human-readable representation with secrets masked."""
        tunable_str = "tunable" if self.is_tunable else "fixed"
        if self.literal_choices:
            return (
                f"DiscoveredTVAR(name='{self.name}', type='{self.python_type}', "
                f"choices={self.literal_choices}, {tunable_str})"
            )
        range_str = ""
        if self.default_range:
            range_str = f", range={self.default_range}"
        display_value = self._get_display_value()
        return (
            f"DiscoveredTVAR(name='{self.name}', type='{self.python_type}', "
            f"value={display_value}{range_str}, {tunable_str})"
        )


@dataclass
class Connection:
    """Represents a data flow connection between two TVARScopes in the pipeline.

    In Haystack terms, this is an edge between components. In TVL terms,
    connections define the pipeline topology which may affect TVAR dependencies.

    Attributes:
        source: Name of the source TVARScope
        target: Name of the target TVARScope
        sender_socket: Output socket name on source (e.g., "documents", "replies")
        receiver_socket: Input socket name on target (e.g., "documents", "prompt")
    """

    source: str
    target: str
    sender_socket: str | None = None
    receiver_socket: str | None = None

    def __repr__(self) -> str:
        """Return a human-readable representation."""
        if self.sender_socket and self.receiver_socket:
            return (
                f"Connection({self.source}.{self.sender_socket} -> "
                f"{self.target}.{self.receiver_socket})"
            )
        return f"Connection({self.source} -> {self.target})"


@dataclass
class TVARScope:
    """A namespace/container for TVARs discovered in a Haystack component.

    In TVL terminology, a scope groups related TVARs under a common namespace.
    Each Haystack component becomes a TVARScope, with the component name as
    the scope prefix (e.g., "generator.temperature", "retriever.top_k").

    Attributes:
        name: The scope name (component name in the pipeline)
        class_name: The short class name (e.g., "OpenAIGenerator")
        class_type: The fully qualified class name
            (e.g., "haystack.components.generators.OpenAIGenerator")
        category: The component category
            (e.g., "Generator", "Retriever", "Router")
        tvars: Dictionary mapping TVAR names to DiscoveredTVAR objects
        max_runs: Maximum number of runs for this scope in a loop (None if not in loop)
    """

    name: str
    class_name: str
    class_type: str
    category: str = "Component"
    tvars: dict[str, DiscoveredTVAR] = field(default_factory=dict)
    max_runs: int | None = None

    # Backwards-compatible alias for tvars
    @property
    def parameters(self) -> dict[str, DiscoveredTVAR]:
        """Alias for tvars (backwards compatibility)."""
        return self.tvars

    def __repr__(self) -> str:
        """Return a human-readable representation."""
        return (
            f"TVARScope(name='{self.name}', "
            f"class_name='{self.class_name}', category='{self.category}')"
        )


@dataclass
class PipelineSpec:
    """Complete specification of a discovered Haystack Pipeline.

    PipelineSpec is the output of pipeline introspection. It contains all
    discovered TVARScopes (components), their TVARs (parameters), and the
    pipeline topology (connections/edges).

    In Epic 2+, PipelineSpec will be convertible to an ExplorationSpace for
    optimization, extracting only the tunable TVARs with their ranges.

    Attributes:
        scopes: List of TVARScope objects discovered in the pipeline
        connections: List of Connection objects representing data flow
        loops: List of detected cycles in the pipeline graph
    """

    scopes: list[TVARScope] = field(default_factory=list)
    connections: list[Connection] = field(default_factory=list)
    loops: list[list[str]] = field(default_factory=list)

    # Backwards-compatible aliases
    @property
    def components(self) -> list[TVARScope]:
        """Alias for scopes (backwards compatibility)."""
        return self.scopes

    @property
    def edges(self) -> list[Connection]:
        """Alias for connections (backwards compatibility)."""
        return self.connections

    @property
    def has_loops(self) -> bool:
        """Check if pipeline contains cycles.

        Returns:
            True if the pipeline has at least one cycle, False otherwise.
        """
        return len(self.loops) > 0

    @property
    def unbounded_loops(self) -> list[list[str]]:
        """Return loops where no scope has max_runs set.

        Returns:
            List of cycles where all scopes have max_runs=None.
        """
        unbounded: list[list[str]] = []
        for loop in self.loops:
            has_bound = False
            for scope_name in loop:
                scope = self.get_scope(scope_name)
                if scope is not None and scope.max_runs is not None:
                    has_bound = True
                    break
            if not has_bound:
                unbounded.append(loop)
        return unbounded

    @property
    def scope_names(self) -> list[str]:
        """Return list of scope names."""
        return [s.name for s in self.scopes]

    # Backwards-compatible alias
    @property
    def component_names(self) -> list[str]:
        """Alias for scope_names (backwards compatibility)."""
        return self.scope_names

    @property
    def scope_count(self) -> int:
        """Return the number of scopes."""
        return len(self.scopes)

    # Backwards-compatible alias
    @property
    def component_count(self) -> int:
        """Alias for scope_count (backwards compatibility)."""
        return self.scope_count

    def get_scope(self, name: str) -> TVARScope | None:
        """Get a scope by name.

        Args:
            name: The scope name to look up

        Returns:
            The TVARScope if found, None otherwise
        """
        for scope in self.scopes:
            if scope.name == name:
                return scope
        return None

    # Backwards-compatible alias
    def get_component(self, name: str) -> TVARScope | None:
        """Alias for get_scope (backwards compatibility)."""
        return self.get_scope(name)

    def get_tvar(self, scope_name: str, tvar_name: str) -> DiscoveredTVAR | None:
        """Get a TVAR by scope name and TVAR name.

        Provides namespaced access to TVARs
        (e.g., get_tvar("generator", "temperature")).

        Args:
            scope_name: The scope name (e.g., "generator")
            tvar_name: The TVAR name (e.g., "temperature")

        Returns:
            The DiscoveredTVAR if found, None otherwise
        """
        scope = self.get_scope(scope_name)
        if scope is None:
            return None
        return scope.tvars.get(tvar_name)

    # Backwards-compatible alias
    def get_parameter(
        self, component_name: str, param_name: str
    ) -> DiscoveredTVAR | None:
        """Alias for get_tvar (backwards compatibility)."""
        return self.get_tvar(component_name, param_name)

    def get_scopes_by_category(self, category: str) -> list[TVARScope]:
        """Get all scopes of a specific category.

        Args:
            category: The category to filter by
                (e.g., "Generator", "Retriever")

        Returns:
            List of TVARScope objects matching the category
        """
        return [s for s in self.scopes if s.category == category]

    # Backwards-compatible alias
    def get_components_by_category(self, category: str) -> list[TVARScope]:
        """Alias for get_scopes_by_category (backwards compatibility)."""
        return self.get_scopes_by_category(category)

    def to_networkx(self) -> Any:
        """Convert to NetworkX DiGraph for graph algorithms.

        Returns:
            A NetworkX DiGraph with scopes as nodes and connections as edges.

        Raises:
            ImportError: If networkx is not installed.

        Example:
            >>> pipeline_spec = from_pipeline(pipeline)
            >>> G = pipeline_spec.to_networkx()
            >>> print(list(G.nodes()))
            ['retriever', 'generator']
        """
        try:
            import networkx as nx
        except ImportError as e:
            raise ImportError(
                "networkx is required for graph operations. "
                "Install it with: pip install networkx"
            ) from e

        G: Any = nx.DiGraph()

        # Add nodes (scopes)
        for scope in self.scopes:
            G.add_node(
                scope.name,
                class_name=scope.class_name,
                category=scope.category,
            )

        # Add edges with metadata
        for conn in self.connections:
            G.add_edge(
                conn.source,
                conn.target,
                sender_socket=conn.sender_socket,
                receiver_socket=conn.receiver_socket,
            )

        return G

    def __repr__(self) -> str:
        """Return a human-readable representation."""
        loops_str = f", loops={len(self.loops)}" if self.loops else ""
        return (
            f"PipelineSpec(scopes={len(self.scopes)}, "
            f"connections={len(self.connections)}{loops_str})"
        )

    def __len__(self) -> int:
        """Return the number of scopes."""
        return len(self.scopes)

    def __iter__(self):
        """Iterate over scopes."""
        return iter(self.scopes)


# =============================================================================
# Backwards-compatible type aliases and factory functions
# =============================================================================
# These aliases ensure existing code continues to work during migration.
# They will be deprecated in a future version.

#: Alias for DiscoveredTVAR (deprecated, use DiscoveredTVAR)
Parameter = DiscoveredTVAR

#: Alias for Connection (deprecated, use Connection)
Edge = Connection


def _create_tvar_scope(
    name: str,
    class_name: str,
    class_type: str,
    category: str = "Component",
    tvars: dict[str, DiscoveredTVAR] | None = None,
    parameters: dict[str, DiscoveredTVAR] | None = None,  # Backwards compat
    max_runs: int | None = None,
) -> TVARScope:
    """Factory function for TVARScope with backwards-compatible 'parameters' kwarg."""
    # Support old 'parameters' keyword
    if parameters is not None and tvars is None:
        tvars = parameters
    return TVARScope(
        name=name,
        class_name=class_name,
        class_type=class_type,
        category=category,
        tvars=tvars if tvars is not None else {},
        max_runs=max_runs,
    )


def _create_pipeline_spec(
    scopes: list[TVARScope] | None = None,
    connections: list[Connection] | None = None,
    loops: list[list[str]] | None = None,
    # Backwards-compatible keywords
    components: list[TVARScope] | None = None,
    edges: list[Connection] | None = None,
) -> PipelineSpec:
    """Factory function for PipelineSpec with backwards-compatible kwargs."""
    # Support old keyword names
    if components is not None and scopes is None:
        scopes = components
    if edges is not None and connections is None:
        connections = edges
    return PipelineSpec(
        scopes=scopes if scopes is not None else [],
        connections=connections if connections is not None else [],
        loops=loops if loops is not None else [],
    )


class _ComponentFactory:
    """Factory class that creates TVARScope with backwards-compatible kwargs.

    Supports both new (tvars=) and old (parameters=) keyword arguments.
    """

    def __call__(
        self,
        name: str,
        class_name: str,
        class_type: str,
        category: str = "Component",
        tvars: dict[str, DiscoveredTVAR] | None = None,
        parameters: dict[str, DiscoveredTVAR] | None = None,
        max_runs: int | None = None,
    ) -> TVARScope:
        return _create_tvar_scope(
            name=name,
            class_name=class_name,
            class_type=class_type,
            category=category,
            tvars=tvars,
            parameters=parameters,
            max_runs=max_runs,
        )


class _ConfigSpaceFactory:
    """Factory class that creates PipelineSpec with backwards-compatible kwargs.

    Supports both new (scopes=, connections=) and old (components=, edges=) keywords.
    """

    def __call__(
        self,
        scopes: list[TVARScope] | None = None,
        connections: list[Connection] | None = None,
        loops: list[list[str]] | None = None,
        components: list[TVARScope] | None = None,
        edges: list[Connection] | None = None,
    ) -> PipelineSpec:
        return _create_pipeline_spec(
            scopes=scopes,
            connections=connections,
            loops=loops,
            components=components,
            edges=edges,
        )


#: Backwards-compatible alias for TVARScope (deprecated, use TVARScope)
#: Accepts both tvars= and parameters= keyword arguments
Component = _ComponentFactory()

#: Backwards-compatible alias for PipelineSpec (deprecated, use PipelineSpec)
#: Accepts both scopes=/connections= and components=/edges= keyword arguments
ConfigSpace = _ConfigSpaceFactory()
