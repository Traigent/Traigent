"""Session budget tracking and result cost guarantees."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from traigent.utils.trial_costs import (
    coerce_finite_cost,
    extract_trial_cost_metric,
    is_finite_numeric_cost,
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
    """Guarantee metrics.cost for completed submissions on budgeted sessions.

    Returns True when the metrics mapping was mutated.
    """

    if not is_completed_status(status):
        return False
    if not is_cost_budget_armed_session(client, session_id):
        return False
    if is_finite_numeric_cost(metrics.get("cost")):
        return False

    cost = extract_trial_cost_metric(metrics, *telemetry_sources)
    if cost is not None:
        metrics["cost"] = cost
        return True

    metrics["cost"] = 0.0
    if logger is not None:
        logger.debug("no cost telemetry; backfilling 0.0 for budget accounting")
    return True
