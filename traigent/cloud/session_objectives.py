"""Typed session objective normalization helpers."""

from __future__ import annotations

from typing import Any

from traigent.cloud.models import SessionObjectiveDefinition

_DIRECTION_OBJECTIVES = frozenset({"maximize", "minimize"})


def session_objective_to_wire(
    objective: str | SessionObjectiveDefinition | dict[str, Any],
) -> str | dict[str, Any]:
    """Serialize one typed objective to the session-create wire shape."""

    if isinstance(objective, str):
        return objective
    if isinstance(objective, SessionObjectiveDefinition):
        payload: dict[str, Any] = {"metric": objective.metric}
        if objective.band is not None:
            payload["band"] = dict(objective.band)
            if objective.test is not None:
                payload["test"] = objective.test
            if objective.alpha is not None:
                payload["alpha"] = objective.alpha
        elif objective.direction is not None:
            payload["direction"] = objective.direction
        if objective.weight is not None:
            payload["weight"] = objective.weight
        return payload
    if isinstance(objective, dict):
        return dict(objective)
    raise TypeError(
        "Session objectives must be strings, dicts, or SessionObjectiveDefinition objects"
    )


def _score_objective_direction(
    objective: Any,
) -> tuple[dict[str, Any], str] | None:
    """Return a score objective and direction when the wire value is valid."""
    if not isinstance(objective, dict):
        return None
    metric = objective.get("metric")
    direction: Any = objective.get("direction")
    if (
        isinstance(metric, str)
        and metric == "score"
        and isinstance(direction, str)
        and direction.lower() in _DIRECTION_OBJECTIVES
    ):
        return objective, direction.lower()
    return None


def normalize_typed_objectives(objectives: Any) -> list[Any]:
    """Normalize typed objective shorthands without changing legacy semantics.

    Bare direction words are legacy optimization-goal placeholders, not metric
    names. The typed session contract uses "score" for that fallback because
    BackendSessionManager backfills the score metric before result submission.
    """

    normalized: list[Any] = []
    seen_score_directions: set[str] = set()
    raw_objectives = list(objectives or ["maximize"])

    for raw_objective in raw_objectives:
        objective = session_objective_to_wire(raw_objective)
        if isinstance(objective, str):
            direction = objective.strip().lower()
            if direction in _DIRECTION_OBJECTIVES:
                objective = {"metric": "score", "direction": direction}

        score_objective = _score_objective_direction(objective)

        if score_objective is not None:
            objective, score_direction = score_objective
            if score_direction in seen_score_directions:
                continue
            seen_score_directions.add(score_direction)
            objective["direction"] = score_direction

        normalized.append(objective)

    return normalized
