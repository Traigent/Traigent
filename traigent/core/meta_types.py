"""Type definitions for __traigent_meta__ structure.

This module provides TypedDict definitions and type guards for the reserved
`__traigent_meta__` key used by multi-agent workflows to report costs and
token usage to Traigent.

Example usage:
    from traigent.core.meta_types import is_traigent_metadata

    meta = output.get("__traigent_meta__")
    if is_traigent_metadata(meta):
        # Type checker knows meta has correct structure
        cost = meta["total_cost"]
        if "usage" in meta:
            tokens = meta["usage"]["input_tokens"]
"""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability FUNC-EVAL-METRICS REQ-EVAL-005

from typing import Any, TypedDict, TypeGuard

try:
    from typing import NotRequired  # Python 3.11+
except ImportError:
    from typing_extensions import NotRequired  # Python 3.8-3.10


class UsageMetadata(TypedDict):
    """Token usage metadata within __traigent_meta__.

    Attributes:
        input_tokens: Number of input tokens consumed
        output_tokens: Number of output tokens generated
    """

    input_tokens: int
    output_tokens: int


class TraigentMetadata(TypedDict):
    """Structure of __traigent_meta__ dict injected by with_usage().

    This is the canonical structure for reporting multi-agent workflow
    costs and token usage to Traigent's optimization engine.

    Attributes:
        total_cost: Total cost in USD for the operation
        usage: Optional token usage breakdown
    """

    total_cost: float
    usage: NotRequired[UsageMetadata]


def is_traigent_metadata(obj: Any) -> TypeGuard[TraigentMetadata]:
    """Type guard for runtime validation of __traigent_meta__ structure.

    Validates that an object conforms to the TraigentMetadata structure:
    - Must be a dict
    - Must have "total_cost" key with numeric value
    - If "usage" key present, must be dict with integer token counts

    Args:
        obj: Object to validate

    Returns:
        True if obj is a valid TraigentMetadata dict, False otherwise

    Example:
        >>> meta = {"total_cost": 0.001, "usage": {"input_tokens": 100, "output_tokens": 50}}
        >>> if is_traigent_metadata(meta):
        ...     cost = meta["total_cost"]  # Type checker knows this is safe
    """
    if not isinstance(obj, dict):
        return False

    # Required field: total_cost must be numeric
    if "total_cost" not in obj:
        return False
    if not isinstance(obj["total_cost"], (int, float)):
        return False

    # Optional field: usage must have correct structure if present
    if "usage" in obj:
        usage = obj["usage"]
        if not isinstance(usage, dict):
            return False

        # Validate token fields if present
        if "input_tokens" in usage:
            if not isinstance(usage["input_tokens"], int):
                return False
        if "output_tokens" in usage:
            if not isinstance(usage["output_tokens"], int):
                return False

    return True
