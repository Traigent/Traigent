"""Session budget tracking and result cost guarantees."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from traigent.utils.trial_costs import (
    coerce_finite_cost,
    extract_trial_cost_metric,
    is_cost_unmeasured,
    is_finite_numeric_cost,
    move_flagged_cost_to_floor_field,
)


def cost_budget_is_armed(budget: Any) -> bool:
    """Return True when a session-create budget carries a positive cost cap."""

    if not isinstance(budget, Mapping):
        return False
    max_cost = coerce_finite_cost(budget.get("max_cost_usd"))
    return max_cost is not None and max_cost > 0


def remember_cost_budget_armed_session(
    client: Any, session_id: str, budget: Any
) -> None:
    """Record a successfully created budget-armed session on a client instance."""

    sessions = getattr(client, "__dict__", {}).get("_cost_budget_armed_sessions")
    if not isinstance(sessions, set):
        sessions = set()
        client._cost_budget_armed_sessions = sessions
    if cost_budget_is_armed(budget):
        sessions.add(session_id)
    else:
        sessions.discard(session_id)


def is_cost_budget_armed_session(client: Any, session_id: str) -> bool:
    """Return whether the client remembers a positive cost budget for a session."""

    sessions = getattr(client, "__dict__", {}).get("_cost_budget_armed_sessions")
    return isinstance(sessions, set) and session_id in sessions


def is_completed_status(status: Any) -> bool:
    """Return True for SDK or backend completed status tokens."""

    raw_status = getattr(status, "value", status)
    return str(raw_status or "").strip().upper() == "COMPLETED"


def ensure_cost_metric_for_budgeted_completed_submission(
    *,
    client: Any,
    session_id: str,
    metrics: dict[str, Any],
    status: Any,
    telemetry_sources: tuple[Any, ...] = (),
    logger: Any | None = None,
) -> bool:
    """Resolve metrics.cost for completed submissions on budgeted sessions.

    A cost found in telemetry is copied into ``metrics["cost"]``. When no cost
    is available and none is flagged, the trial's spend is unknown, so the
    mapping gets ``cost_unmeasured = 1.0``. A flagged trial's spend actually
    reported (a lower bound; ``0.0`` when nothing was reported) travels in
    ``COST_FLOOR_WIRE_KEY`` together with the flag (#2688, owner decision A).
    On budget-armed sessions only, ``metrics["cost"]`` also carries the floor
    (TEMPORARY bridge: the backend requires a numeric ``metrics.cost``).

    Returns True when the metrics mapping was mutated.
    """

    if not is_completed_status(status):
        return False
    armed = is_cost_budget_armed_session(client, session_id)
    if is_cost_unmeasured(metrics):
        # Unknown spend: the floor travels in its own field WITH the flag. Only
        # a budget-armed session also gets the TEMPORARY metrics.cost bridge.
        return move_flagged_cost_to_floor_field(
            metrics, *telemetry_sources, bridge_cost=armed
        )
    if not armed:
        return False

    if is_finite_numeric_cost(metrics.get("cost")):
        return False

    cost = extract_trial_cost_metric(metrics, *telemetry_sources)
    if cost is not None:
        metrics["cost"] = cost
        return True

    # No cost and no flag: nothing measured this trial's spend, so it is
    # unknown. Say so with the flag; the reported floor (0.0 here) goes in
    # its own field, never in ``cost``.
    metrics["cost_unmeasured"] = 1.0
    move_flagged_cost_to_floor_field(metrics, *telemetry_sources, bridge_cost=True)
    if logger is not None:
        logger.debug(
            "no cost telemetry; submitting budgeted trial as cost_unmeasured (not $0)"
        )
    return True
