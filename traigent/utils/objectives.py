"""Objective utility helpers."""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability FUNC-ORCH-LIFECYCLE REQ-ORCH-003

from __future__ import annotations

from typing import Literal

_MINIMIZE_OBJECTIVE_PATTERNS = (
    "cost",
    "latency",
    "error",
    "loss",
    "time",
    "duration",
)

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

    When *orientation* is ``None`` this falls back to substring matching of
    *objective_name* against ``_MINIMIZE_OBJECTIVE_PATTERNS``.  This heuristic
    is retained for backward compatibility with legacy string-only objective
    flows and can misclassify compound names like ``"accuracy_cost_ratio"``.
    """
    if orientation is not None:
        return orientation == "minimize"
    lowered = objective_name.lower()
    return any(pattern in lowered for pattern in _MINIMIZE_OBJECTIVE_PATTERNS)


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
    "is_minimization_objective",
    "is_operational_objective",
    "is_quality_objective",
]
