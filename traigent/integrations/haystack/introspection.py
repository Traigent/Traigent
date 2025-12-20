"""Pipeline introspection for Haystack integration.

This module provides the from_pipeline() function that extracts TVARScopes
(components) and their TVARs (tunable parameters) from a Haystack Pipeline.

Terminology (aligned with TVL - Tuning Variable Language):
- TVAR: Tuned Variable - a parameter that can be optimized
- TVARScope: A namespace for TVARs (corresponds to a Haystack component)
- PipelineSpec: Complete discovered pipeline structure
"""

from __future__ import annotations

import inspect
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from .models import (  # Backwards-compatible aliases
    Connection,
    DiscoveredTVAR,
    PipelineSpec,
    TVARScope,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from haystack import Pipeline

    from .configspace import ExplorationSpace


# Types that are considered tunable primitives
TUNABLE_TYPES = {"int", "float", "str", "bool", "Literal"}

# Default ranges for common LLM and retriever TVARs
# Maps TVAR name patterns to (range, scale_type)
TVAR_SEMANTICS: dict[str, dict[str, Any]] = {
    "temperature": {"range": (0.0, 2.0), "scale": "continuous"},
    "top_p": {"range": (0.0, 1.0), "scale": "continuous"},
    "top_k": {"range": (1, 100), "scale": "discrete"},
    "max_tokens": {"range": (1, 4096), "scale": "discrete"},
    "presence_penalty": {"range": (-2.0, 2.0), "scale": "continuous"},
    "frequency_penalty": {"range": (-2.0, 2.0), "scale": "continuous"},
    "score_threshold": {"range": (0.0, 1.0), "scale": "continuous"},
    "similarity_threshold": {"range": (0.0, 1.0), "scale": "continuous"},
}

# Backwards-compatible alias
PARAMETER_SEMANTICS = TVAR_SEMANTICS

# TVAR names that indicate non-tunable complex objects
NON_TUNABLE_TVAR_NAMES = {
    "document_store",
    "embedding_backend",
    "tokenizer",
    "client",
    "api_key",
    "secret",
    "callback",
    "callbacks",
    "streaming_callback",
}

# Backwards-compatible alias
NON_TUNABLE_PARAM_NAMES = NON_TUNABLE_TVAR_NAMES

# Model catalogs by provider for auto-discovery
# Updated: December 2024
MODEL_CATALOGS: dict[str, list[str]] = {
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
        "o1",
        "o1-mini",
    ],
    "anthropic": [
        "claude-3-5-sonnet-latest",
        "claude-3-5-haiku-latest",
        "claude-3-opus-latest",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ],
    "azure_openai": [
        # Azure uses deployment names, but common model patterns
        "gpt-4o",
        "gpt-4",
        "gpt-35-turbo",
    ],
    "cohere": [
        "command-r-plus",
        "command-r",
        "command",
        "command-light",
    ],
    "google": [
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-pro",
    ],
}

# Map component class names to provider keys
PROVIDER_DETECTION: dict[str, str] = {
    # OpenAI
    "OpenAIGenerator": "openai",
    "OpenAIChatGenerator": "openai",
    "GPTGenerator": "openai",
    # Anthropic
    "AnthropicGenerator": "anthropic",
    "AnthropicChatGenerator": "anthropic",
    "ClaudeGenerator": "anthropic",
    # Azure OpenAI
    "AzureOpenAIGenerator": "azure_openai",
    "AzureOpenAIChatGenerator": "azure_openai",
    # Cohere
    "CohereGenerator": "cohere",
    "CohereChatGenerator": "cohere",
    # Google
    "GoogleAIGeminiGenerator": "google",
    "GoogleAIGeminiChatGenerator": "google",
    "VertexAIGeminiGenerator": "google",
}


@overload
def from_pipeline(
    pipeline: Pipeline,
    *,
    as_exploration_space: Literal[False] = ...,
) -> PipelineSpec: ...


@overload
def from_pipeline(
    pipeline: Pipeline,
    *,
    as_exploration_space: Literal[True],
) -> ExplorationSpace: ...


def from_pipeline(
    pipeline: Pipeline,
    *,
    as_exploration_space: bool = False,
) -> PipelineSpec | ExplorationSpace:
    """Extract TVARScopes and TVARs from a Haystack Pipeline.

    Analyzes the given pipeline and extracts all components (TVARScopes)
    with their tunable parameters (TVARs) for optimization.

    Args:
        pipeline: A Haystack Pipeline instance
        as_exploration_space: If True, return an ExplorationSpace for optimization
            instead of a PipelineSpec. Default is False for backwards compatibility.

    Returns:
        PipelineSpec containing discovered scopes, connections, and TVARs,
        or ExplorationSpace if as_exploration_space=True.

    Raises:
        TypeError: If pipeline is not a valid Haystack Pipeline

    Example:
        >>> from haystack import Pipeline
        >>> from traigent.integrations.haystack import from_pipeline
        >>> pipeline = Pipeline()
        >>> # ... add components ...
        >>> spec = from_pipeline(pipeline)
        >>> print(spec.scopes)  # List of discovered TVARScopes
        []

        # For optimization, get an ExplorationSpace:
        >>> space = from_pipeline(pipeline, as_exploration_space=True)
        >>> print(space.tunable_tvars)  # Dict of tunable TVARs
    """
    # Validate input type
    if not _is_haystack_pipeline(pipeline):
        raise TypeError(
            f"Expected a Haystack Pipeline instance, " f"got {type(pipeline).__name__}"
        )

    # Extract scopes (components) from pipeline
    scopes = _extract_scopes(pipeline)

    # Extract connections (edges between scopes)
    connections = _extract_connections(pipeline)

    # Extract max_runs_per_component from pipeline (Haystack 2.x)
    max_runs = _extract_max_runs(pipeline)

    # Apply max_runs to all scopes if pipeline has it set
    if max_runs is not None:
        for scope in scopes:
            scope.max_runs = max_runs

    # Create initial PipelineSpec
    spec = PipelineSpec(scopes=scopes, connections=connections)

    # Detect loops using the graph
    loops = _detect_loops(spec)
    spec.loops = loops

    # Log warning for unbounded loops
    _warn_unbounded_loops(spec)

    # Return ExplorationSpace if requested
    if as_exploration_space:
        from .configspace import ExplorationSpace

        return ExplorationSpace.from_pipeline_spec(spec)

    return spec


def _is_haystack_pipeline(obj: object) -> bool:
    """Check if object is a Haystack Pipeline.

    Uses duck typing to avoid hard dependency on haystack import.
    """
    walk = getattr(obj, "walk", None)
    if callable(walk):
        return True

    components = getattr(obj, "_components", None)
    return isinstance(components, dict)


def _extract_scopes(pipeline: Pipeline) -> list[TVARScope]:
    """Extract all TVARScopes from a Haystack Pipeline.

    Args:
        pipeline: A valid Haystack Pipeline instance

    Returns:
        List of TVARScope objects with name and class information
    """
    scopes: list[TVARScope] = []

    # Try walk() method first (public API in Haystack 2.x)
    walk = getattr(pipeline, "walk", None)
    if callable(walk):
        for name, component in walk():
            scopes.append(_create_scope(name, component))
    # Fallback to internal _components dict
    elif hasattr(pipeline, "_components"):
        for name, component in pipeline._components.items():
            scopes.append(_create_scope(name, component))

    return scopes


# Backwards-compatible alias
def _extract_components(pipeline: Pipeline) -> list[TVARScope]:
    """Alias for _extract_scopes (backwards compatibility)."""
    return _extract_scopes(pipeline)


def _create_scope(name: str, component: object) -> TVARScope:
    """Create a TVARScope dataclass from a pipeline component.

    Args:
        name: The scope name (component key in pipeline)
        component: The actual component instance

    Returns:
        TVARScope dataclass with extracted information
    """
    component_type = type(component)
    class_name = component_type.__name__
    module_path = component_type.__module__
    fully_qualified_name = f"{module_path}.{class_name}"

    # Determine scope category from class hierarchy
    category = _detect_scope_category(component_type)

    # Extract TVARs from the component
    tvars = _extract_tvars(component, class_name)

    return TVARScope(
        name=name,
        class_name=class_name,
        class_type=fully_qualified_name,
        category=category,
        tvars=tvars,
    )


# Backwards-compatible alias
def _create_component(name: str, component: object) -> TVARScope:
    """Alias for _create_scope (backwards compatibility)."""
    return _create_scope(name, component)


def _detect_scope_category(component_type: type) -> str:
    """Detect the category of a scope from its class hierarchy.

    Args:
        component_type: The type of the component

    Returns:
        Category string (e.g., "Generator", "Retriever", "Router")
    """
    type_name = component_type.__name__.lower()
    module_name = component_type.__module__.lower()

    # Check common Haystack component categories
    if "generator" in type_name or "generators" in module_name:
        return "Generator"
    elif "retriever" in type_name or "retrievers" in module_name:
        return "Retriever"
    elif "embedder" in type_name or "embedders" in module_name:
        return "Embedder"
    elif "router" in type_name or "routers" in module_name:
        return "Router"
    elif "converter" in type_name or "converters" in module_name:
        return "Converter"
    elif "builder" in type_name or "builders" in module_name:
        return "Builder"
    elif "writer" in type_name or "writers" in module_name:
        return "Writer"
    elif "reader" in type_name or "readers" in module_name:
        return "Reader"
    elif "ranker" in type_name or "rankers" in module_name:
        return "Ranker"
    else:
        return "Component"


# Backwards-compatible alias
def _detect_component_category(component_type: type) -> str:
    """Alias for _detect_scope_category (backwards compatibility)."""
    return _detect_scope_category(component_type)


def _get_type_hints_safe(component: object) -> dict[str, Any]:
    """Get evaluated type hints for a component's __init__ method.

    Handles 'from __future__ import annotations' by using get_type_hints().

    Args:
        component: The component instance

    Returns:
        Dictionary of parameter names to type hints, empty dict on failure
    """
    try:
        return get_type_hints(type(component).__init__)
    except Exception:
        return {}


def _resolve_type_hint(
    param_name: str,
    param: inspect.Parameter,
    type_hints: dict[str, Any],
) -> Any | None:
    """Resolve the type hint for a parameter.

    Args:
        param_name: The parameter name
        param: The inspect.Parameter object
        type_hints: Evaluated type hints dict

    Returns:
        The resolved type hint or None
    """
    type_hint = type_hints.get(param_name)
    if type_hint is None:
        type_hint = param.annotation
        if type_hint is inspect.Parameter.empty:
            return None
    return type_hint


def _get_tvar_value(
    component: object,
    tvar_name: str,
    param: inspect.Parameter,
) -> Any:
    """Get the current value of a TVAR from the component.

    Args:
        component: The component instance
        tvar_name: The TVAR name
        param: The inspect.Parameter object

    Returns:
        The current TVAR value or None
    """
    try:
        value = getattr(component, tvar_name, param.default)
        return None if value is inspect.Parameter.empty else value
    except AttributeError:
        if param.default is not inspect.Parameter.empty:
            return param.default
        return None


# Backwards-compatible alias
def _get_param_value(
    component: object,
    param_name: str,
    param: inspect.Parameter,
) -> Any:
    """Alias for _get_tvar_value (backwards compatibility)."""
    return _get_tvar_value(component, param_name, param)


def _extract_tvars(
    component: object, component_class_name: str | None = None
) -> dict[str, DiscoveredTVAR]:
    """Extract all __init__ parameters as TVARs from a component.

    Args:
        component: The component instance to extract TVARs from
        component_class_name: Optional class name for provider detection

    Returns:
        Dictionary mapping TVAR names to DiscoveredTVAR objects
    """
    tvars: dict[str, DiscoveredTVAR] = {}

    # Get class name if not provided
    if component_class_name is None:
        component_class_name = type(component).__name__

    try:
        sig = inspect.signature(component.__init__)  # type: ignore[misc]
    except (ValueError, TypeError):
        return tvars

    type_hints = _get_type_hints_safe(component)

    for tvar_name, param in sig.parameters.items():
        if tvar_name == "self":
            continue

        type_hint = _resolve_type_hint(tvar_name, param, type_hints)
        current_value = _get_tvar_value(component, tvar_name, param)
        python_type, literal_choices, is_optional = _parse_type_hint(type_hint)
        is_tunable, non_tunable_reason = _is_tunable_tvar(
            tvar_name, python_type, current_value
        )

        # Infer default range for tunable numeric TVARs
        default_range, range_type = (None, None)
        if is_tunable:
            default_range, range_type = _infer_tvar_semantics(tvar_name, python_type)

        # Apply model catalog for model parameters (AC3, AC4 of Story 2.8)
        if tvar_name == "model" and literal_choices is None:
            literal_choices, non_tunable_reason = _get_model_choices(
                component_class_name, current_value, non_tunable_reason
            )

        tvars[tvar_name] = DiscoveredTVAR(
            name=tvar_name,
            value=current_value,
            python_type=python_type,
            type_hint=_type_hint_to_string(type_hint),
            is_tunable=is_tunable,
            literal_choices=literal_choices,
            is_optional=is_optional,
            non_tunable_reason=non_tunable_reason,
            default_range=default_range,
            range_type=range_type,
        )

    return tvars


def _get_model_choices(
    component_class_name: str,
    current_value: Any,
    existing_reason: str | None,
) -> tuple[list[str] | None, str | None]:
    """Get model choices from provider catalog or fallback to current value.

    Args:
        component_class_name: The component class name for provider detection
        current_value: The current model value
        existing_reason: Existing non-tunable reason if any

    Returns:
        Tuple of (literal_choices, non_tunable_reason)
    """
    provider = PROVIDER_DETECTION.get(component_class_name)

    if provider and provider in MODEL_CATALOGS:
        # Provider catalog available - use it
        choices = MODEL_CATALOGS[provider].copy()
        # Ensure current value is in choices if not None
        if current_value and current_value not in choices:
            choices.insert(0, current_value)
        return (choices, existing_reason)

    # Fallback: use current value as single choice (AC4)
    if current_value:
        logger.warning(
            f"Could not determine model catalog for {component_class_name}. "
            f"Using current value '{current_value}' as only choice. "
            f"Use set_choices() to specify available models."
        )
        return (
            [current_value],
            existing_reason or "Unknown provider - specify model choices manually",
        )

    return (None, existing_reason)


# Backwards-compatible alias
def _extract_parameters(
    component: object, component_class_name: str | None = None
) -> dict[str, DiscoveredTVAR]:
    """Alias for _extract_tvars (backwards compatibility)."""
    return _extract_tvars(component, component_class_name)


def _parse_type_hint(hint: Any) -> tuple[str, list[Any] | None, bool]:
    """Parse a type hint into its components.

    Args:
        hint: The type hint to parse

    Returns:
        Tuple of (python_type, literal_choices, is_optional)
    """
    if hint is None:
        return ("unknown", None, False)

    # Handle Literal types
    origin = get_origin(hint)
    if origin is Literal:
        choices = list(get_args(hint))
        return ("Literal", choices, False)

    # Handle Optional[T] (Union[T, None])
    if origin is Union:
        return _parse_union_type(hint)

    # Handle basic and complex types
    return _parse_basic_type(hint)


def _parse_union_type(hint: Any) -> tuple[str, list[Any] | None, bool]:
    """Parse a Union type hint (including Optional[T]).

    Args:
        hint: The Union type hint to parse

    Returns:
        Tuple of (python_type, literal_choices, is_optional)
    """
    args = get_args(hint)
    non_none_args = [a for a in args if a is not type(None)]
    is_optional = type(None) in args

    if len(non_none_args) == 1:
        # Unwrap Optional[T] and recurse
        inner_type, inner_choices, _ = _parse_type_hint(non_none_args[0])
        return (inner_type, inner_choices, is_optional)
    elif len(non_none_args) > 1:
        # Union of multiple types - not easily tunable
        return ("Union", None, is_optional)

    return ("unknown", None, is_optional)


def _parse_basic_type(hint: Any) -> tuple[str, list[Any] | None, bool]:
    """Parse a basic or complex type hint.

    Args:
        hint: The type hint to parse

    Returns:
        Tuple of (python_type, literal_choices, is_optional)
    """
    # Map of basic types to their string names
    basic_types = {int: "int", float: "float", str: "str", bool: "bool"}

    if hint in basic_types:
        return (basic_types[hint], None, False)

    # Handle type objects (e.g., from typing module)
    if isinstance(hint, type):
        type_name = hint.__name__
        if type_name in ("int", "float", "str", "bool"):
            return (type_name, None, False)
        return ("object", None, False)

    # Try to get the name from the hint
    return _extract_type_name(hint)


def _extract_type_name(hint: Any) -> tuple[str, list[Any] | None, bool]:
    """Extract type name from a complex type hint.

    Args:
        hint: The type hint to extract name from

    Returns:
        Tuple of (python_type, literal_choices, is_optional)
    """
    try:
        if hasattr(hint, "__name__"):
            return (hint.__name__, None, False)
        if hasattr(hint, "_name") and hint._name:
            return (hint._name, None, False)
    except Exception:
        pass

    return ("unknown", None, False)


def _is_tunable_tvar(
    tvar_name: str, python_type: str, value: Any
) -> tuple[bool, str | None]:
    """Determine if a TVAR is tunable.

    Args:
        tvar_name: The TVAR name
        python_type: The detected Python type
        value: The current TVAR value

    Returns:
        Tuple of (is_tunable, non_tunable_reason)
    """
    # Check if TVAR name indicates non-tunable
    tvar_name_lower = tvar_name.lower()
    for non_tunable_name in NON_TUNABLE_TVAR_NAMES:
        if non_tunable_name in tvar_name_lower:
            return (False, f"TVAR '{tvar_name}' is a complex object")

    # Check if it's a callable
    if callable(value) and not isinstance(value, type):
        return (False, "TVAR is a callable")

    # Check if the type is tunable
    if python_type in TUNABLE_TYPES:
        return (True, None)

    # Unknown or object types are not tunable by default
    if python_type in ("unknown", "object", "Union"):
        return (False, f"Type '{python_type}' is not automatically tunable")

    # Default to not tunable for complex types
    return (False, f"Type '{python_type}' requires manual specification")


# Backwards-compatible alias
def _is_tunable_parameter(
    param_name: str, python_type: str, value: Any
) -> tuple[bool, str | None]:
    """Alias for _is_tunable_tvar (backwards compatibility)."""
    return _is_tunable_tvar(param_name, python_type, value)


def _infer_tvar_semantics(
    tvar_name: str, python_type: str
) -> tuple[tuple[Any, Any] | None, str | None]:
    """Infer default range and scale for a TVAR based on its name.

    Args:
        tvar_name: The TVAR name to check
        python_type: The detected Python type

    Returns:
        Tuple of (default_range, range_type) or (None, None) if unknown
    """
    # Only infer ranges for numeric types
    if python_type not in ("int", "float"):
        return (None, None)

    tvar_lower = tvar_name.lower()
    for pattern, semantics in TVAR_SEMANTICS.items():
        if pattern in tvar_lower:
            return (semantics["range"], semantics["scale"])

    return (None, None)


# Backwards-compatible alias
def _infer_parameter_semantics(
    param_name: str, python_type: str
) -> tuple[tuple[Any, Any] | None, str | None]:
    """Alias for _infer_tvar_semantics (backwards compatibility)."""
    return _infer_tvar_semantics(param_name, python_type)


def _type_hint_to_string(hint: Any) -> str | None:
    """Convert a type hint to a string representation.

    Args:
        hint: The type hint to convert

    Returns:
        String representation of the type hint, or None if not available
    """
    if hint is None:
        return None

    try:
        # For basic types
        if isinstance(hint, type):
            return hint.__name__

        # For typing constructs, use repr and clean it up
        repr_str = repr(hint)

        # Clean up common patterns
        repr_str = repr_str.replace("typing.", "")

        return repr_str
    except Exception:
        return None


def _extract_connections(pipeline: Pipeline) -> list[Connection]:
    """Extract connection structure from a Haystack Pipeline.

    Haystack Pipelines use NetworkX internally to manage component connections.
    This function extracts the connection information for use in analysis.

    Args:
        pipeline: A valid Haystack Pipeline instance

    Returns:
        List of Connection objects representing scope connections
    """
    connections: list[Connection] = []

    # Check if pipeline has graph attribute (Haystack 2.x uses NetworkX internally)
    graph = getattr(pipeline, "graph", None)
    if graph is None:
        return connections

    # Check if the graph has edges method (NetworkX DiGraph)
    edges_method = getattr(graph, "edges", None)
    if not callable(edges_method):
        return connections

    try:
        # Extract edges from NetworkX graph
        # Haystack stores connection info in edge data
        for sender, receiver, data in graph.edges(data=True):
            conn = Connection(
                source=sender,
                target=receiver,
                sender_socket=data.get("sender_socket"),
                receiver_socket=data.get("receiver_socket"),
            )
            connections.append(conn)
    except Exception:
        # If graph traversal fails, return empty connections
        pass

    return connections


# Backwards-compatible alias
def _extract_graph(pipeline: Pipeline) -> list[Connection]:
    """Alias for _extract_connections (backwards compatibility)."""
    return _extract_connections(pipeline)


def _extract_max_runs(pipeline: Pipeline) -> int | None:
    """Extract max_runs_per_component from a Haystack Pipeline.

    Haystack 2.x pipelines can have a max_runs_per_component setting
    that limits the number of times a component can run in a loop.

    Args:
        pipeline: A valid Haystack Pipeline instance

    Returns:
        The max_runs value if set, None otherwise
    """
    return getattr(pipeline, "max_runs_per_component", None)


def _detect_loops(spec: PipelineSpec) -> list[list[str]]:
    """Detect all simple cycles in the pipeline graph.

    Uses NetworkX simple_cycles() algorithm to find all cycles
    in the directed graph representation of the pipeline.

    Args:
        spec: The PipelineSpec with scopes and connections

    Returns:
        List of cycles, where each cycle is a list of scope names
    """
    if not spec.connections:
        return []

    try:
        import networkx as nx  # noqa: F401 - used for type checking
    except ImportError:
        return []

    try:
        G = spec.to_networkx()
        import networkx as nx

        return list(nx.simple_cycles(G))
    except Exception:
        return []


def _warn_unbounded_loops(spec: PipelineSpec) -> None:
    """Log warning for loops without max_runs set.

    When a pipeline has cycles but no max_runs_per_component configured,
    this could lead to infinite loops during execution.

    Args:
        spec: The PipelineSpec to check for unbounded loops
    """
    import logging

    logger = logging.getLogger(__name__)

    unbounded = spec.unbounded_loops
    if unbounded:
        for loop in unbounded:
            loop_str = " -> ".join(loop + [loop[0]])  # Show cycle
            logger.warning(
                f"Unbounded loop detected: {loop_str}. "
                f"Consider setting max_runs_per_component on the pipeline."
            )
