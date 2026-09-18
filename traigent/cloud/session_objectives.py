"""Typed session objective normalization helpers.

The wire shape is CANONICAL: ``{name, orientation, weight, band}``. The legacy
``{metric, direction, weight, band}`` form is still accepted as INPUT -- a caller
may hand us a ``SessionObjectiveDefinition`` (whose fields are named ``metric``
and ``direction``) or a raw dict in either spelling -- but it is translated here
rather than forwarded, so only one shape ever reaches the API (#304).

That split is deliberate: a convenient SDK input is a feature, a second wire
format is a contract we would have to keep honouring. The backend accepts both
spellings today, which is why this can migrate without a coordinated release;
once it stops, the SDK is already sending the form it will keep accepting.
"""

from __future__ import annotations

from typing import Any

from traigent.cloud.models import SessionObjectiveDefinition

_DIRECTION_OBJECTIVES = frozenset({"maximize", "minimize"})

# Canonical spelling <- the legacy key it replaces.
_LEGACY_KEY_ALIASES = {"name": "metric", "orientation": "direction"}


def _canonical_band(
    band: Any, test: Any = None, alpha: Any = None
) -> dict[str, Any] | None:
    """Translate a band in any accepted input spelling to the canonical BandTarget.

    Accepts ``{low, high}``, ``{center, tol}`` or an already-canonical
    ``{target: [lower, upper]}``. Emits ``{target, test?, alpha?}`` and nothing
    else: objective_definition_schema.json declares ``additionalProperties: false``
    on BandTarget, so forwarding ``center``/``tol`` produces a payload the contract
    rejects -- the defect fixed in #2385, not to be reintroduced here.
    """
    if not isinstance(band, dict):
        return None

    target = band.get("target")
    if target is None:
        low, high = band.get("low"), band.get("high")
        if low is None and high is None:
            center, tol = band.get("center"), band.get("tol")
            if center is not None and tol is not None:
                low, high = center - tol, center + tol
        if low is None or high is None:
            return None
        target = [low, high]

    canonical: dict[str, Any] = {"target": list(target)}
    # test/alpha may arrive beside the band or nested inside it; nested wins only
    # when the outer value is absent, matching the backend's own precedence.
    effective_test = test if test is not None else band.get("test")
    effective_alpha = alpha if alpha is not None else band.get("alpha")
    if effective_test is not None:
        canonical["test"] = effective_test
    if effective_alpha is not None:
        canonical["alpha"] = effective_alpha
    return canonical


def _canonicalize_objective_dict(objective: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a caller-supplied objective dict into the canonical wire shape.

    Legacy keys are renamed, never duplicated: emitting both spellings would make
    the payload ambiguous, and the backend rejects a ``direction`` that disagrees
    with an ``orientation``.
    """
    canonical = dict(objective)

    for canonical_key, legacy_key in _LEGACY_KEY_ALIASES.items():
        if legacy_key in canonical:
            legacy_value = canonical.pop(legacy_key)
            canonical.setdefault(canonical_key, legacy_value)

    band = _canonical_band(
        canonical.get("band"), canonical.pop("test", None), canonical.pop("alpha", None)
    )
    if band is not None:
        canonical["band"] = band
        # A banded objective's orientation is "band"; the legacy form signalled it
        # structurally, by the band key being present, and never as a direction value.
        canonical["orientation"] = "band"
    elif "band" in canonical:
        canonical.pop("band")

    return canonical


def session_objective_to_wire(
    objective: str | SessionObjectiveDefinition | dict[str, Any],
) -> str | dict[str, Any]:
    """Serialize one typed objective to the canonical session-create wire shape."""

    if isinstance(objective, str):
        return objective
    if isinstance(objective, SessionObjectiveDefinition):
        payload: dict[str, Any] = {"name": objective.metric}
        if objective.band is not None:
            band = _canonical_band(objective.band, objective.test, objective.alpha)
            if band is not None:
                payload["band"] = band
                payload["orientation"] = "band"
        elif objective.direction is not None:
            payload["orientation"] = objective.direction
        if objective.weight is not None:
            payload["weight"] = objective.weight
        return payload
    if isinstance(objective, dict):
        return _canonicalize_objective_dict(objective)
    raise TypeError(
        "Session objectives must be strings, dicts, or SessionObjectiveDefinition objects"
    )


def _score_objective_orientation(
    objective: Any,
) -> tuple[dict[str, Any], str] | None:
    """Return a score objective and orientation when the wire value is valid."""
    if not isinstance(objective, dict):
        return None
    name = objective.get("name")
    orientation: Any = objective.get("orientation")
    if (
        isinstance(name, str)
        and name == "score"
        and isinstance(orientation, str)
        and orientation.lower() in _DIRECTION_OBJECTIVES
    ):
        return objective, orientation.lower()
    return None


def normalize_typed_objectives(objectives: Any) -> list[Any]:
    """Normalize typed objective shorthands without changing legacy semantics.

    Bare direction words are legacy optimization-goal placeholders, not metric
    names. The typed session contract uses "score" for that fallback because
    BackendSessionManager backfills the score metric before result submission.
    """

    normalized: list[Any] = []
    seen_score_orientations: set[str] = set()
    raw_objectives = list(objectives or ["maximize"])

    for raw_objective in raw_objectives:
        objective = session_objective_to_wire(raw_objective)
        if isinstance(objective, str):
            orientation = objective.strip().lower()
            if orientation in _DIRECTION_OBJECTIVES:
                objective = {"name": "score", "orientation": orientation}

        score_objective = _score_objective_orientation(objective)

        if score_objective is not None:
            objective, score_orientation = score_objective
            if score_orientation in seen_score_orientations:
                continue
            seen_score_orientations.add(score_orientation)
            objective["orientation"] = score_orientation

        normalized.append(objective)

    return normalized
