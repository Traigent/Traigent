"""Objective utility helpers."""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability FUNC-ORCH-LIFECYCLE REQ-ORCH-003

from __future__ import annotations

import math
from typing import Any, Literal

from traigent.core.objective_directions import resolve_objective_orientation

_QUALITY_OBJECTIVE_NAMES = {
    "accuracy",
    "score",
}

_OPERATIONAL_OBJECTIVE_NAMES = {
    "cost",
    "total_cost",
    "latency",
    "response_time_ms",
}


def is_minimization_objective(
    objective_name: str,
    orientation: str | None = None,
) -> bool:
    """Return True when the objective should be minimized.

    When *orientation* is supplied (the value from an
    ``ObjectiveDefinition.orientation`` field), it is used directly and
    name-pattern heuristics are bypassed:

    * ``"minimize"`` → ``True``
    * ``"maximize"`` → ``False``
    * ``"band"``     → ``False`` (banded objectives use deviation, not direction)

    When *orientation* is ``None``, only exact SDK-owned metric defaults are
    accepted. Unknown custom metrics must declare an orientation.
    """
    if orientation == "band":
        return False
    return resolve_objective_orientation(objective_name, orientation) == "minimize"


def coerce_finite_objective_score(value: Any) -> float | None:
    """Return a finite numeric objective score, or None when unrankable."""
    if value is None or isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def classify_objective(
    objective_name: str,
) -> Literal["quality", "operational", "other"]:
    """Classify an objective for ranking-eligibility policy decisions."""
    lowered = objective_name.strip().lower()
    if lowered in _QUALITY_OBJECTIVE_NAMES or lowered.endswith("_accuracy"):
        return "quality"
    if lowered in _OPERATIONAL_OBJECTIVE_NAMES:
        return "operational"
    return "other"


def is_quality_objective(objective_name: str) -> bool:
    """Whether objective is a quality objective."""
    return classify_objective(objective_name) == "quality"


def is_operational_objective(objective_name: str) -> bool:
    """Whether objective is an operational objective."""
    return classify_objective(objective_name) == "operational"


__all__ = [
    "classify_objective",
    "coerce_finite_objective_score",
    "is_minimization_objective",
    "is_operational_objective",
    "is_quality_objective",
]
