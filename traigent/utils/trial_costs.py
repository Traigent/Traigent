"""Shared helpers for extracting trial cost telemetry."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from traigent.storage.local_storage import TRIAL_COST_FIELDS, extract_trial_cost_fields

__all__ = [
    "COST_ROW_KEYS",
    "TRIAL_COST_FIELDS",
    "coerce_finite_cost",
    "extract_trial_cost_metric",
    "is_cost_unmeasured",
    "is_finite_numeric_cost",
    "known_spend_lower_bound",
    "lift_flagged_cost_to_floor",
    "row_has_measured_cost",
    "trial_cost_unknown",
]

# Keys under which a per-example result row may carry a measured cost.
COST_ROW_KEYS = ("input_cost", "output_cost", "total_cost", "cost")


def coerce_finite_cost(value: Any) -> float | None:
    """Coerce a cost-like value to a finite float without accepting booleans."""

    if value is None or isinstance(value, bool):
        return None
    try:
        cost = float(value)
    except (TypeError, ValueError):
        return None
    return cost if math.isfinite(cost) else None


def is_cost_unmeasured(metrics: Any) -> bool:
    """True when a metrics mapping flags its cost as unmeasured (Traigent#2517).

    Upload and budget paths must omit the cost for such a trial; a default of
    0.0 (or the score) would record an unknown cost as a measured one.
    """
    if not isinstance(metrics, Mapping):
        return False
    flag = metrics.get("cost_unmeasured")
    if flag is True:
        # Row dataclasses (``HybridExampleResult``) serialize the flag as a
        # bool. An explicit ``True`` is the same claim as ``1.0``: a flag that
        # disowns a cost must never be ignored for being the wrong type.
        return True
    return (
        isinstance(flag, (int, float))
        and not isinstance(flag, bool)
        and math.isfinite(flag)
        and flag > 0
    )


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


def row_has_measured_cost(row: Any) -> bool:
    """True when a result row carries a finite numeric cost and no unknown flag."""
    if not isinstance(row, Mapping) or is_cost_unmeasured(row):
        return False
    return any(is_finite_numeric_cost(row.get(key)) for key in COST_ROW_KEYS)


def trial_cost_unknown(
    rows: Sequence[Any], successes: Sequence[bool] | None = None
) -> bool:
    """The ONE classifier: is a trial's cost unknown, given its result rows?

    A row is *measured* when it carries a finite numeric cost and no
    ``cost_unmeasured`` flag. The trial's cost is unknown (Traigent#2517) when

    * any row is explicitly flagged unmeasured, or
    * any *successful* row has no measured cost (response not captured, key
      absent): summing the rest would give a floor that can outrank a fully
      measured trial, or
    * rows exist but none is measured (e.g. every example errored): the
      trial's spend is unknown, never a fabricated $0.

    A failed row without a cost is skipped (an error before any usage is not
    evidence that the rest of the spend is unknown). ``successes`` is aligned
    with ``rows``; when omitted or misaligned every row counts as successful.
    No rows at all means nothing ran: not unknown.
    """
    rows = list(rows)
    if not rows:
        return False
    if successes is None or len(successes) != len(rows):
        flags = [True] * len(rows)
    else:
        flags = [bool(flag) for flag in successes]
    measured = 0
    for row, ok in zip(rows, flags, strict=True):
        if isinstance(row, Mapping) and is_cost_unmeasured(row):
            return True
        if row_has_measured_cost(row):
            measured += 1
        elif ok:
            return True
    return measured == 0


def extract_trial_cost_metric(*sources: Any) -> float | None:
    """Return the first finite cost found in trial telemetry sources.

    A source whose mappings flag ``cost_unmeasured`` yields nothing: telemetry
    or metadata zeros next to the flag are not a measurement.
    """

    for source in sources:
        if any(is_cost_unmeasured(m) for m in _iter_cost_mappings(source)):
            continue
        for cost_mapping in _iter_cost_mappings(source):
            costs = extract_trial_cost_fields(cost_mapping)
            cost = coerce_finite_cost(costs.get("cost"))
            if cost is not None:
                return cost
    return None


def known_spend_lower_bound(*sources: Any) -> float:
    """Spend actually reported in the sources, ignoring any unmeasured flag.

    For a trial flagged ``cost_unmeasured`` the number is only a lower bound
    (the reported part of the spend); ``0.0`` when nothing was reported. It is
    never a measurement: callers that send it must send the flag with it.

    Every mapping in every source is scanned (not just the first usable one),
    so a ``0`` in one place can never mask spend reported in another. An
    explicit ``cost_lower_bound`` never lowers a larger reported
    ``cost``/``total_cost``: the largest of all bounds and reported
    non-negative costs is returned. The maximum (not the sum) is used because
    the same spend is routinely echoed at several nesting levels (``metrics``,
    ``metadata``, ``summary_stats``); summing would double count it, while a
    maximum is still a valid floor. Producers must therefore sum independent
    components (e.g. per-example costs) themselves before emitting one value.
    """

    bounds: list[float] = []
    costs_seen: list[float] = []
    for source in sources:
        for cost_mapping in _iter_cost_mappings(source):
            bound = coerce_finite_cost(cost_mapping.get("cost_lower_bound"))
            if bound is not None and bound >= 0:
                bounds.append(bound)
            costs = extract_trial_cost_fields(cost_mapping)
            cost = coerce_finite_cost(costs.get("cost"))
            if cost is not None and cost >= 0:
                costs_seen.append(cost)
    candidates = bounds + costs_seen
    return max(candidates) if candidates else 0.0


def lift_flagged_cost_to_floor(metrics: dict[str, Any], *sources: Any) -> bool:
    """Make a flagged-unmeasured ``metrics["cost"]`` carry the reported floor.

    A budget-armed submission needs a numeric ``cost``; for a trial flagged
    ``cost_unmeasured`` that number is only the reported lower bound (``0.0``
    when nothing was reported), sent together with the flag. An existing
    ``cost`` (a flagged ``0`` included) never lowers the reported spend: the
    value becomes ``known_spend_lower_bound(metrics, *sources)``, which already
    counts ``metrics["cost"]``. A non-numeric original (e.g. the string
    ``"0.4"``) is replaced by the floor. Returns True when ``metrics["cost"]``
    changed.
    """

    floor = known_spend_lower_bound(metrics, *sources)
    original = metrics.get("cost")
    # Only a finite *numeric* original may be returned unchanged: a numeric
    # string ("0.4") would otherwise survive as a string cost.
    current = (
        coerce_finite_cost(original)
        if isinstance(original, (int, float)) and not isinstance(original, bool)
        else None
    )
    if current is not None and current >= floor:
        return False
    metrics["cost"] = floor
    return True
