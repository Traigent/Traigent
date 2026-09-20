"""Canonical orientation policy for SDK-owned objective metrics.

Only exact metric names whose meaning is owned by this SDK receive a default.
Custom metrics must declare whether greater or smaller values are better.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Literal, cast

ObjectiveOrientation = Literal["maximize", "minimize"]

_CANONICAL_OBJECTIVE_ORIENTATIONS: dict[str, ObjectiveOrientation] = {
    "accuracy": "maximize",
    "success": "maximize",
    "success_rate": "maximize",
    "exact_match_default": "maximize",
    "cost": "minimize",
    "total_cost": "minimize",
    "cost_per_example_mean": "minimize",
    "input_cost": "minimize",
    "output_cost": "minimize",
    "latency": "minimize",
    "duration": "minimize",
    "response_time_ms": "minimize",
    "avg_response_time": "minimize",
    "avg_response_time_ms": "minimize",
    "execution_time_ms": "minimize",
    "error_rate": "minimize",
    "empty_output_rate": "minimize",
    "truncated_output_rate": "minimize",
}

CANONICAL_OBJECTIVE_ORIENTATIONS = MappingProxyType(
    _CANONICAL_OBJECTIVE_ORIENTATIONS
)


def _missing_orientation_error(name: str) -> ValueError:
    return ValueError(
        f"Objective {name!r} has no declared orientation, and Traigent does not "
        "infer directions for custom metric names. Declare it explicitly, for "
        "example:\n\n"
        "from traigent.core.objectives import create_default_objectives\n"
        f"objectives = create_default_objectives([{name!r}], "
        f"orientations={{{name!r}: 'minimize'}})"
    )


def validate_objective_orientation(
    name: str,
    orientation: str,
) -> ObjectiveOrientation:
    """Validate and narrow an explicitly declared scalar orientation."""
    if orientation not in {"maximize", "minimize"}:
        raise ValueError(
            f"Orientation for objective {name!r} must be 'maximize' or "
            f"'minimize'; got {orientation!r}."
        )
    return cast(ObjectiveOrientation, orientation)


def resolve_objective_orientation(
    name: str,
    explicit: str | None = None,
) -> ObjectiveOrientation:
    """Return an explicit direction or the exact SDK-owned default.

    Explicit valid declarations always win, including declarations that reverse
    the SDK-owned default for a known metric. Unknown bare names fail closed.
    """
    if explicit is not None:
        return validate_objective_orientation(name, explicit)
    try:
        return CANONICAL_OBJECTIVE_ORIENTATIONS[name]
    except KeyError:
        raise _missing_orientation_error(name) from None


__all__ = [
    "CANONICAL_OBJECTIVE_ORIENTATIONS",
    "ObjectiveOrientation",
    "resolve_objective_orientation",
    "validate_objective_orientation",
]
