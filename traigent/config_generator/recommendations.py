"""Public TVAR recommendation catalog query helpers.

These helpers expose only public-safe catalog metadata: tuned-variable names,
range suggestions, impact estimates, measured evidence notes, effectuation
status, and application guidance. Impacts are task-dependent and were measured
on specific benchmark slices, models, and metrics; validate recommendations on
your own eval set before relying on them.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from traigent.config_generator.catalog import catalog_entries, entry_to_recommendation
from traigent.config_generator.types import EvidenceRef, TVarRecommendation

__all__ = [
    "RECOMMENDATION_CAVEAT",
    "list_recommendation_agent_types",
    "recommend_config_space",
]

RECOMMENDATION_CAVEAT = (
    "Recommendation impacts are task-dependent. The supporting evidence was "
    "measured on specific benchmark slices, models, and metrics; treat these "
    "as search-space starting points and validate them on your own eval set."
)

_IMPACT_ORDER = {"low": 0, "medium": 1, "high": 2}
_CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}
_OBSERVATIONAL_LIMITATIONS = frozenset({"observational_not_causal"})


def list_recommendation_agent_types() -> tuple[str, ...]:
    """Return valid agent/task types for public catalog recommendations."""
    return tuple(
        sorted(
            {
                str(agent_type)
                for entry in _active_catalog_entries()
                for agent_type in entry.get("agent_types", ())
                if str(agent_type).strip()
            }
        )
    )


def recommend_config_space(
    agent_type: str,
    *,
    min_impact: str | None = None,
    min_confidence: str | None = None,
) -> dict[str, Any]:
    """Return evidence-backed TVAR recommendations for an agent/task type.

    Args:
        agent_type: Catalog agent/task type. Use
            ``list_recommendation_agent_types()`` to list valid values.
        min_impact: Optional minimum impact estimate: ``"low"``, ``"medium"``,
            or ``"high"``.
        min_confidence: Optional minimum public evidence-strength label:
            ``"low"``, ``"medium"``, or ``"high"``. This is derived only from
            catalog evidence metadata and is not a statistical confidence
            interval.

    Returns:
        A JSON-serializable dict with the normalized agent type, filters,
        caveat, and recommendation rows. Each row contains the knob name,
        suggested range, impact, evidence note, effectuation status, and apply
        guidance.

    Raises:
        ValueError: If ``agent_type`` or a filter value is unknown.

    Recommendation impacts are task-dependent and were measured on specific
    benchmark slices, models, and metrics. Treat the returned knobs as starting
    points for your own evaluation, not universal performance claims.
    """
    normalized_agent_type = _normalize_agent_type(agent_type)
    impact_threshold = _normalize_level(min_impact, _IMPACT_ORDER, "min_impact")
    confidence_threshold = _normalize_level(
        min_confidence,
        _CONFIDENCE_ORDER,
        "min_confidence",
    )

    rows = [
        _recommendation_to_row(entry_to_recommendation(entry))
        for entry in _active_catalog_entries(normalized_agent_type)
    ]
    filtered_rows = [
        row
        for row in rows
        if _passes_threshold(row["impact"], impact_threshold, _IMPACT_ORDER)
        and _passes_threshold(
            row["confidence"], confidence_threshold, _CONFIDENCE_ORDER
        )
    ]

    return {
        "agent_type": normalized_agent_type,
        "valid_agent_types": list(list_recommendation_agent_types()),
        "filters": {
            "min_impact": impact_threshold,
            "min_confidence": confidence_threshold,
        },
        "caveat": RECOMMENDATION_CAVEAT,
        "recommendations": filtered_rows,
    }


def _active_catalog_entries(agent_type: str | None = None) -> list[dict[str, Any]]:
    return [
        entry
        for entry in catalog_entries(agent_type)
        if str(entry.get("status", "active")) == "active"
    ]


def _normalize_agent_type(agent_type: str) -> str:
    normalized = str(agent_type).strip().lower().replace("-", "_")
    valid_types = list_recommendation_agent_types()
    if normalized in valid_types:
        return normalized

    valid = ", ".join(valid_types) or "<none>"
    raise ValueError(
        f"Unknown agent_type {agent_type!r}. "
        f"Valid recommendation agent types: {valid}."
    )


def _normalize_level(
    value: str | None,
    order: Mapping[str, int],
    label: str,
) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in order:
        return normalized

    valid = ", ".join(order)
    raise ValueError(f"Unknown {label} {value!r}. Valid values: {valid}.")


def _passes_threshold(
    value: Any,
    threshold: str | None,
    order: Mapping[str, int],
) -> bool:
    if threshold is None:
        return True
    return order.get(str(value), -1) >= order[threshold]


def _recommendation_to_row(rec: TVarRecommendation) -> dict[str, Any]:
    confidence = _evidence_confidence(rec.evidence_refs)
    return {
        "name": rec.name,
        "range_type": rec.range_type,
        "range_kwargs": copy.deepcopy(rec.range_kwargs),
        "range_code": rec.to_range_code(),
        "suggested_values": _suggested_values(rec),
        "category": rec.category,
        "kind": rec.kind,
        "impact": rec.impact_estimate,
        "confidence": confidence,
        "evidence_note": _evidence_note(rec.evidence_refs),
        "effectuation_status": rec.effectuation_status,
        "effectuation_strategy": rec.effectuation_strategy,
        "apply_guidance": rec.apply_guidance,
        "catalog_entry_id": rec.catalog_entry_id or rec.entry_id,
    }


def _suggested_values(rec: TVarRecommendation) -> list[Any]:
    if rec.recommended_values:
        return list(rec.recommended_values)

    values = rec.range_kwargs.get("values")
    if (
        rec.range_type == "Choices"
        and isinstance(values, Sequence)
        and not isinstance(values, (str, bytes))
    ):
        return list(values)
    return []


def _evidence_confidence(evidence_refs: Sequence[EvidenceRef]) -> str:
    if not evidence_refs:
        return "low"

    has_causal_positive = any(_is_causal_positive_ref(ref) for ref in evidence_refs)
    if not has_causal_positive:
        return "low"

    has_large_clean_slice = any(
        _is_causal_positive_ref(ref) and ref.n >= 100 and not ref.limitations
        for ref in evidence_refs
    )
    if has_large_clean_slice:
        return "high"

    return "medium"


def _is_causal_positive_ref(ref: EvidenceRef) -> bool:
    if ref.delta is None:
        return False
    if set(ref.limitations) & _OBSERVATIONAL_LIMITATIONS:
        return False
    return abs(float(ref.delta)) > 0.0


def _evidence_note(evidence_refs: Sequence[EvidenceRef]) -> str:
    if not evidence_refs:
        return "No public evidence note is attached; validate before using."
    return "; ".join(_format_evidence_ref(ref) for ref in evidence_refs)


def _format_evidence_ref(ref: EvidenceRef) -> str:
    limitations = _format_limitations(ref.limitations)
    comparison = f"{_format_value(ref.baseline)} -> {_format_value(ref.candidate)}"
    if ref.delta is None:
        summary = "observational support; no measured delta"
    else:
        summary = f"{ref.metric} delta {_format_delta(ref.delta)}"
    return (
        f"{summary} on {ref.scope} benchmark "
        f"(n={ref.n}, model={ref.model}, {comparison}; limitations: {limitations})"
    )


def _format_delta(delta: float) -> str:
    if delta > 0:
        return f"+{delta:g}"
    return f"{delta:g}"


def _format_limitations(limitations: Sequence[str]) -> str:
    if not limitations:
        return "none listed"
    return ", ".join(str(item) for item in limitations)


def _format_value(value: Any) -> str:
    return repr(value) if isinstance(value, str) else str(value)
