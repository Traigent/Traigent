"""Objective utility helpers."""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability FUNC-ORCH-LIFECYCLE REQ-ORCH-003

from __future__ import annotations

_MINIMIZE_OBJECTIVE_PATTERNS = (
    "cost",
    "latency",
    "error",
    "loss",
    "time",
    "duration",
)


def is_minimization_objective(objective_name: str) -> bool:
    """Infer objective direction from objective name patterns.

    This is a heuristic fallback for legacy objective-name-only flows.
    It performs substring matching and can misclassify compound names like
    ``"accuracy_cost_ratio"`` as minimization objectives.

    Prefer explicit objective orientation metadata when available.
    """
    lowered = objective_name.lower()
    return any(pattern in lowered for pattern in _MINIMIZE_OBJECTIVE_PATTERNS)


__all__ = ["is_minimization_objective"]
