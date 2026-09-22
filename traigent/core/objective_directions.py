"""Canonical orientation policy for SDK-owned objective metrics.

Only exact metric names whose meaning is owned by this SDK receive a default.
Custom metrics must declare whether greater or smaller values are better.
"""

from __future__ import annotations

import warnings
from types import MappingProxyType
from typing import Literal, cast

from traigent.utils.exceptions import ObjectiveDirectionOverrideWarning

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

CANONICAL_OBJECTIVE_ORIENTATIONS = MappingProxyType(_CANONICAL_OBJECTIVE_ORIENTATIONS)


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


def _direction_conflict_warning(
    name: str,
    declared: ObjectiveOrientation,
    default: ObjectiveOrientation,
) -> ObjectiveDirectionOverrideWarning:
    return ObjectiveDirectionOverrideWarning(
        f"Objective {name!r} is a reserved keyword: its built-in direction is "
        f"{default!r}, but you declared {declared!r}. Traigent will use your "
        f"declared direction for this run, but a Traigent certificate for this "
        f"run will refuse to assert a comparison claim for {name!r}, because the "
        "declared direction contradicts the reserved keyword's preset."
    )


def resolve_objective_orientation(
    name: str,
    explicit: str | None = None,
) -> ObjectiveOrientation:
    """Return an explicit direction or the exact SDK-owned default.

    Explicit valid declarations always win, including declarations that reverse
    the SDK-owned default for a known metric -- this function never raises for
    that case. When ``name`` is a reserved keyword (has a canonical default)
    and ``explicit`` disagrees with it, an :class:`ObjectiveDirectionOverrideWarning`
    is emitted so the caller learns, at declaration time, that a Traigent
    certificate for the run will refuse to assert a comparison claim for this
    objective. Custom (non-reserved) names never trigger this warning, since
    they have no built-in direction to conflict with. Unknown bare names fail
    closed.
    """
    if explicit is not None:
        validated = validate_objective_orientation(name, explicit)
        default = CANONICAL_OBJECTIVE_ORIENTATIONS.get(name)
        if default is not None and validated != default:
            warnings.warn(
                _direction_conflict_warning(name, validated, default),
                stacklevel=3,
            )
        return validated
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
