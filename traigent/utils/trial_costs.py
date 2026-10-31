"""Shared helpers for extracting trial cost telemetry."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, TypeGuard

from traigent.storage.local_storage import TRIAL_COST_FIELDS, extract_trial_cost_fields

__all__ = [
    "COST_FLOOR_WIRE_KEY",
    "COST_ROW_KEYS",
    "TRIAL_COST_FIELDS",
    "coerce_finite_cost",
    "extract_trial_cost_metric",
    "is_cost_unmeasured",
    "is_finite_numeric_cost",
    "known_spend_lower_bound",
    "lift_cost_floor_to_body",
    "move_flagged_cost_to_floor_field",
    "row_has_measured_cost",
    "row_reports_zero_provider_attempts",
    "trial_cost_unknown",
    "unmeasured_example_coverage",
    "unmeasured_example_ids",
]

#: Key carrying the exact known spend floor of an unmeasured-cost trial or run
#: (top-level of the ``POST /sessions/{id}/results`` body; TraigentSchema field
#: ``cost_floor_usd``). It is NOT a cost.
COST_FLOOR_WIRE_KEY = "cost_floor_usd"

# Keys under which a per-example result row may carry a measured cost.
COST_ROW_KEYS = ("input_cost", "output_cost", "total_cost", "cost")

#: A failed row that sets this to ``0`` explicitly reports that no provider call
#: was attempted, so it has no spend to measure (see :func:`trial_cost_unknown`).
ZERO_ATTEMPTS_KEY = "provider_attempts"


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


def is_cost_unranked(metrics: Any) -> bool:
    """True when a trial's cost must not take part in cost ranking (Traigent#2539).

    Covers ``cost_unmeasured`` (unknown spend) and ``cost_unpriced`` (some call
    could not be priced, so a non-strict run recorded a placeholder ``$0``).
    Either way the number is a floor or a stray zero, never a ranking value.
    Ranking surfaces use this; upload/budget paths keep ``is_cost_unmeasured``.
    """
    if is_cost_unmeasured(metrics):
        return True
    if not isinstance(metrics, Mapping):
        return False
    flag = metrics.get("cost_unpriced")
    return (
        isinstance(flag, (int, float))
        and not isinstance(flag, bool)
        and math.isfinite(flag)
        and flag > 0
    )


def is_finite_numeric_cost(value: Any) -> TypeGuard[int | float]:
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


def row_reports_zero_provider_attempts(row: Any) -> bool:
    """True only when a row explicitly reports ``provider_attempts == 0``."""
    if not isinstance(row, Mapping):
        return False
    attempts = row.get(ZERO_ATTEMPTS_KEY)
    return (
        isinstance(attempts, (int, float))
        and not isinstance(attempts, bool)
        and math.isfinite(attempts)
        and attempts == 0
    )


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

    * a *failed* row has no measured cost, unless it reports a cost explicitly
      (``0`` included) or reports ``provider_attempts == 0``: an error does not
      prove nothing was spent (a provider call can succeed and the evaluator
      raise afterwards), so a failed row with no cost telemetry is unknown.

    ``successes`` is aligned with ``rows``; when omitted or misaligned every
    row counts as successful. No rows at all means nothing ran: not unknown.
    """
    rows = list(rows)
    if not rows:
        return False
    if successes is None or len(successes) != len(rows):
        flags = [True] * len(rows)
    else:
        flags = [bool(flag) for flag in successes]
    measured = 0
    zero_attempts = 0
    for row, ok in zip(rows, flags, strict=True):
        if isinstance(row, Mapping) and is_cost_unmeasured(row):
            return True
        if row_has_measured_cost(row):
            measured += 1
        elif not ok and row_reports_zero_provider_attempts(row):
            zero_attempts += 1
        else:
            return True
    return measured == 0 and zero_attempts == 0


def _example_row_fields(row: Any) -> tuple[Any, Any, Any]:
    """Read ``(metrics, example_id, success)`` from a mapping or object row."""
    if isinstance(row, Mapping):
        return row.get("metrics"), row.get("example_id"), row.get("success", True)
    return (
        getattr(row, "metrics", None),
        getattr(row, "example_id", None),
        getattr(row, "success", True),
    )


def _example_cost_unmeasured(metrics: Mapping[str, Any], ok: Any) -> bool:
    """Per-row rule shared with :func:`trial_cost_unknown`."""
    flagged = is_cost_unmeasured(metrics)
    no_cost = not row_has_measured_cost(metrics) and not (
        not bool(ok) and row_reports_zero_provider_attempts(metrics)
    )
    return flagged or no_cost


def unmeasured_example_ids(example_results: Any, limit: int = 5) -> list[str]:
    """Ids (or ``#index``) of the examples whose cost could not be measured.

    Uses the same per-row rule as :func:`trial_cost_unknown`: a row flagged
    ``cost_unmeasured``, or a row with no measured cost (a failed row that
    reports zero provider attempts excepted). Only the
    example id or position is returned, never example content. At most
    ``limit`` entries; ``[]`` when the rows are not readable.
    """
    if not isinstance(example_results, Sequence) or isinstance(
        example_results, (str, bytes, bytearray)
    ):
        return []
    found: list[str] = []
    for index, row in enumerate(example_results):
        metrics, example_id, ok = _example_row_fields(row)
        if not isinstance(metrics, Mapping):
            continue
        if _example_cost_unmeasured(metrics, ok):
            found.append(str(example_id) if example_id else f"#{index}")
            if len(found) >= limit:
                break
    return found


def unmeasured_example_coverage(
    example_results: Any, attempted: Any
) -> tuple[int, int] | None:
    """Complete ``(missing, attempted)`` example counts for one trial, or None.

    Counts example *attempts* (not distinct ids) with the same per-row rule as
    :func:`unmeasured_example_ids`, over every row, never a truncated id list.
    ``attempted`` is the trial's own attempted-example count; the counts are
    returned only when the readable rows account for exactly that many
    attempts. Otherwise (rows absent, redacted, partial, or unreadable) the
    coverage is unknown and ``None`` is returned: a count is never guessed.
    """
    if (
        isinstance(attempted, bool)
        or not isinstance(attempted, int)
        or attempted <= 0
        or not isinstance(example_results, Sequence)
        or isinstance(example_results, (str, bytes, bytearray))
        or len(example_results) != attempted
    ):
        return None
    missing = 0
    for row in example_results:
        metrics, _example_id, ok = _example_row_fields(row)
        if not isinstance(metrics, Mapping):
            return None
        if _example_cost_unmeasured(metrics, ok):
            missing += 1
    return missing, attempted


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


def _collect_known_spend(
    cost_mapping: Mapping[str, Any], bounds: list[float], costs_seen: list[float]
) -> None:
    bound = coerce_finite_cost(cost_mapping.get("cost_lower_bound"))
    if bound is not None and bound >= 0:
        bounds.append(bound)
    # Every alias counts, even when another alias reports zero.
    costs = extract_trial_cost_fields(cost_mapping)
    aliases = [
        *(costs.get(field) for field in TRIAL_COST_FIELDS),
        *(cost_mapping.get(field) for field in TRIAL_COST_FIELDS),
        cost_mapping.get("total_example_cost"),
    ]
    for alias in aliases:
        cost = coerce_finite_cost(alias)
        if cost is not None and cost >= 0:
            costs_seen.append(cost)
    parts = [
        coerce_finite_cost(cost_mapping.get("input_cost")),
        coerce_finite_cost(cost_mapping.get("output_cost")),
    ]
    if all(part is not None and part >= 0 for part in parts):
        costs_seen.append(sum(p for p in parts if p is not None))


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
            _collect_known_spend(cost_mapping, bounds, costs_seen)
    candidates = bounds + costs_seen
    return max(candidates) if candidates else 0.0


def move_flagged_cost_to_floor_field(
    metrics: dict[str, Any], *sources: Any, bridge_cost: bool = False
) -> bool:
    """Report a flagged-unmeasured trial's floor in its own field, not as cost.

    ``metrics["cost"]`` is removed (a flagged placeholder ``0`` or a reported
    floor must not read as a measured number) and the exact known floor, the
    largest of ``metrics["cost"]`` and every reported bound
    (:func:`known_spend_lower_bound`), is written under
    :data:`COST_FLOOR_WIRE_KEY`. The floor never lowers reported spend.

    ``bridge_cost=True`` (budget-armed sessions only) also writes the floor
    back as ``metrics["cost"]``: the backend still requires a numeric
    ``metrics.cost`` on every completed trial of a budget-armed session.
    Returns True when ``metrics`` changed.
    """

    floor = known_spend_lower_bound(metrics, *sources)
    before = dict(metrics)
    metrics.pop("cost", None)
    existing = coerce_finite_cost(metrics.get(COST_FLOOR_WIRE_KEY))
    if existing is None or existing < floor:
        metrics[COST_FLOOR_WIRE_KEY] = floor
    if bridge_cost:
        # TEMPORARY bridge until backend reads cost_floor_usd (Traigent#2688)
        metrics["cost"] = metrics[COST_FLOOR_WIRE_KEY]
    return metrics != before


def lift_cost_floor_to_body(body: dict[str, Any]) -> None:
    """Move the floor from ``body["metrics"]`` to the top level of the body."""

    metrics = body.get("metrics")
    if isinstance(metrics, dict) and COST_FLOOR_WIRE_KEY in metrics:
        body[COST_FLOOR_WIRE_KEY] = metrics.pop(COST_FLOOR_WIRE_KEY)
