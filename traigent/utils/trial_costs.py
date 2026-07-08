"""Shared helpers for extracting trial cost telemetry."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from traigent.storage.local_storage import TRIAL_COST_FIELDS, extract_trial_cost_fields

__all__ = [
    "TRIAL_COST_FIELDS",
    "coerce_finite_cost",
    "extract_trial_cost_metric",
    "is_finite_numeric_cost",
]


def coerce_finite_cost(value: Any) -> float | None:
    """Coerce a cost-like value to a finite float without accepting booleans."""

    if value is None or isinstance(value, bool):
        return None
    try:
        cost = float(value)
    except (TypeError, ValueError):
        return None
    return cost if math.isfinite(cost) else None


def is_finite_numeric_cost(value: Any) -> bool:
    """Return True only for finite JSON numeric cost values."""

    if value is None or isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _iter_cost_mappings(source: Any) -> Any:
    if isinstance(source, Mapping):
        yield source
        for key in ("metadata", "measures", "summary_stats"):
            nested = source.get(key)
            if nested is not None:
                yield from _iter_cost_mappings(nested)
        return
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
        for item in source:
            yield from _iter_cost_mappings(item)


def extract_trial_cost_metric(*sources: Any) -> float | None:
    """Return the first finite cost found in trial telemetry sources."""

    for source in sources:
        for cost_mapping in _iter_cost_mappings(source):
            costs = extract_trial_cost_fields(cost_mapping)
            cost = coerce_finite_cost(costs.get("cost"))
            if cost is not None:
                return cost
    return None
