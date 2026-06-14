"""Public metric/evaluator recommendation catalog query helpers.

These helpers expose public-safe metric and evaluator selection metadata:
task type, measure type, metric shape, evaluator binding, provenance, impact
estimates, evidence-strength labels, limitations, and cost notes. Fit is
task-dependent and benchmark-specific; validate choices on your own evaluation
dataset before relying on them for optimization or release decisions.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from traigent.evaluators.catalog_loader import catalog_entries, catalog_version

__all__ = [
    "EVAL_RECOMMENDATION_CAVEAT",
    "list_eval_recommendation_task_types",
    "recommend_evaluator",
    "recommend_metrics",
]

EVAL_RECOMMENDATION_SCHEMA_VERSION = "1"
EVAL_RECOMMENDATION_CAVEAT = (
    "Metric and evaluator fit is task-dependent. The supporting evidence comes "
    "from specific benchmarks, papers, or operational practice; treat these as "
    "selection starting points and validate them on your own evaluation dataset."
)

_CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}
_IMPACT_ORDER = {"low": 0, "medium": 1, "high": 2}
_COST_TIER_ORDER = {"low": 0, "medium": 1, "high": 2}
_MEASURE_TYPES = (
    "sanity_check",
    "accuracy",
    "quality",
    "latency",
    "safety",
    "efficiency",
    "reliability",
)
_METHOD_RANK_DETERMINISTIC_FIRST = {
    "deterministic": 0,
    "statistical": 1,
    "hybrid": 2,
    "llm_based": 3,
}


def list_eval_recommendation_task_types() -> tuple[str, ...]:
    """Return valid task types for metric/evaluator recommendations."""
    return tuple(
        sorted(
            {
                str(task_type)
                for entry in _active_catalog_entries()
                for task_type in entry.get("task_types", ())
                if str(task_type).strip()
            }
        )
    )


def recommend_metrics(
    task_type: str,
    *,
    measure_types: Sequence[str] | str | None = None,
    min_confidence: str | None = None,
) -> dict[str, Any]:
    """Return task-specific metric recommendations.

    Args:
        task_type: Catalog task type. Use
            ``list_eval_recommendation_task_types()`` to list valid values.
        measure_types: Optional measure-type filter. Values must match the
            canonical MeasureType enum used by TraigentSchema.
        min_confidence: Optional minimum evidence-strength label:
            ``"low"``, ``"medium"``, or ``"high"``. This is not a statistical
            confidence interval.

    Returns:
        JSON-serializable dict with catalog metadata, caveat, recommendation
        rows, and a ``metric_functions_stub`` mapping from metric name to the
        SDK built-in function name when one exists.
    """
    normalized_task_type = _normalize_task_type(task_type)
    normalized_measure_types = _normalize_measure_types(measure_types)
    confidence_threshold = _normalize_level(
        min_confidence,
        _CONFIDENCE_ORDER,
        "min_confidence",
    )

    rows = [
        _metric_recommendation_row(entry)
        for entry in _active_catalog_entries(normalized_task_type)
    ]
    filtered_rows = [
        row
        for row in rows
        if _passes_measure_filter(row, normalized_measure_types)
        and _passes_min_threshold(
            row["confidence"],
            confidence_threshold,
            _CONFIDENCE_ORDER,
        )
    ]

    return {
        "schema_version": EVAL_RECOMMENDATION_SCHEMA_VERSION,
        "catalog_version": catalog_version(),
        "task_type": normalized_task_type,
        "valid_task_types": list(list_eval_recommendation_task_types()),
        "filters": {
            "measure_types": (
                list(normalized_measure_types) if normalized_measure_types else None
            ),
            "min_confidence": confidence_threshold,
        },
        "caveat": EVAL_RECOMMENDATION_CAVEAT,
        "recommendations": filtered_rows,
        "metric_functions_stub": _metric_functions_stub(filtered_rows),
    }


def recommend_evaluator(
    task_type: str,
    *,
    prefer_deterministic: bool = True,
    max_cost_tier: str | None = None,
) -> dict[str, Any]:
    """Return ranked evaluator approaches for a task type.

    Args:
        task_type: Catalog task type. Use
            ``list_eval_recommendation_task_types()`` to list valid values.
        prefer_deterministic: Put deterministic evaluators first when true.
        max_cost_tier: Optional maximum derived cost tier:
            ``"low"``, ``"medium"``, or ``"high"``.

    Returns:
        JSON-serializable dict with evaluator bindings, cost notes, provenance,
        limitations, and ranking metadata.
    """
    normalized_task_type = _normalize_task_type(task_type)
    cost_threshold = _normalize_level(max_cost_tier, _COST_TIER_ORDER, "max_cost_tier")

    rows = [
        _evaluator_recommendation_row(entry)
        for entry in _active_catalog_entries(normalized_task_type)
    ]
    filtered_rows = [
        row
        for row in rows
        if _passes_max_threshold(row["cost_tier"], cost_threshold, _COST_TIER_ORDER)
    ]
    ranked_rows = sorted(
        filtered_rows,
        key=lambda row: _evaluator_rank_key(
            row,
            prefer_deterministic=prefer_deterministic,
        ),
    )

    return {
        "schema_version": EVAL_RECOMMENDATION_SCHEMA_VERSION,
        "catalog_version": catalog_version(),
        "task_type": normalized_task_type,
        "valid_task_types": list(list_eval_recommendation_task_types()),
        "filters": {
            "prefer_deterministic": prefer_deterministic,
            "max_cost_tier": cost_threshold,
        },
        "caveat": EVAL_RECOMMENDATION_CAVEAT,
        "recommendations": ranked_rows,
    }


def _active_catalog_entries(task_type: str | None = None) -> list[dict[str, Any]]:
    return [
        entry
        for entry in catalog_entries(task_type)
        if str(entry.get("status", "active")) == "active"
    ]


def _normalize_task_type(task_type: str) -> str:
    normalized = str(task_type).strip().lower().replace("-", "_")
    valid_types = list_eval_recommendation_task_types()
    if normalized in valid_types:
        return normalized

    valid = ", ".join(valid_types) or "<none>"
    raise ValueError(
        f"Unknown task_type {task_type!r}. "
        f"Valid evaluator recommendation task types: {valid}."
    )


def _normalize_measure_types(
    values: Sequence[str] | str | None,
) -> tuple[str, ...] | None:
    if values is None:
        return None
    raw_values: tuple[str, ...]
    if isinstance(values, str):
        raw_values = (values,)
    else:
        raw_values = tuple(values)

    normalized_values = tuple(
        str(value).strip().lower().replace("-", "_")
        for value in raw_values
        if str(value).strip()
    )
    unknown = sorted(set(normalized_values) - set(_MEASURE_TYPES))
    if unknown:
        valid = ", ".join(_MEASURE_TYPES)
        raise ValueError(f"Unknown measure_types {unknown!r}. Valid values: {valid}.")
    return normalized_values or None


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


def _passes_measure_filter(
    row: Mapping[str, Any],
    measure_types: tuple[str, ...] | None,
) -> bool:
    if measure_types is None:
        return True
    return str(row["measure_type"]) in set(measure_types)


def _passes_min_threshold(
    value: Any,
    threshold: str | None,
    order: Mapping[str, int],
) -> bool:
    if threshold is None:
        return True
    return order.get(str(value), -1) >= order[threshold]


def _passes_max_threshold(
    value: Any,
    threshold: str | None,
    order: Mapping[str, int],
) -> bool:
    if threshold is None:
        return True
    return order.get(str(value), 99) <= order[threshold]


def _metric_recommendation_row(entry: Mapping[str, Any]) -> dict[str, Any]:
    metric = _copy_mapping(entry["metric"])
    return {
        "catalog_entry_id": str(entry["entry_id"]),
        "measure_type": str(entry["measure_type"]),
        "metric": metric,
        "evaluation_method": str(entry["evaluation_method"]),
        "evaluator_binding": _copy_mapping(entry["evaluator_binding"]),
        "impact_estimate": str(entry["impact_estimate"]),
        "confidence": str(entry["confidence"]),
        "provenance": copy.deepcopy(list(entry["provenance"])),
        "limitations": copy.deepcopy(list(entry["limitations"])),
        "cost_note": entry.get("cost_note"),
    }


def _evaluator_recommendation_row(entry: Mapping[str, Any]) -> dict[str, Any]:
    row = _metric_recommendation_row(entry)
    row["metric_name"] = row["metric"]["name"]
    row["cost_tier"] = _entry_cost_tier(row)
    row["cost_note"] = row["cost_note"] or _default_cost_note(row["evaluation_method"])
    return row


def _copy_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Catalog entry contains a non-object field")
    return copy.deepcopy(dict(value))


def _metric_functions_stub(rows: Sequence[Mapping[str, Any]]) -> dict[str, str | None]:
    return {
        str(row["metric"]["name"]): row["metric"].get("builtin_function")
        for row in rows
    }


def _entry_cost_tier(row: Mapping[str, Any]) -> str:
    method = str(row["evaluation_method"])
    if method == "llm_based":
        return "high"
    if method in {"hybrid", "statistical"}:
        return "medium"
    return "low"


def _default_cost_note(evaluation_method: str) -> str:
    if evaluation_method == "statistical":
        return "No judge-model calls required; cost depends on sample count and instrumentation."
    if evaluation_method == "hybrid":
        return "Cost depends on the deterministic and judge-model portions configured."
    return "No judge-model calls required beyond the system under evaluation."


def _evaluator_rank_key(
    row: Mapping[str, Any],
    *,
    prefer_deterministic: bool,
) -> tuple[int, int, int, str]:
    method = str(row["evaluation_method"])
    method_rank = (
        _METHOD_RANK_DETERMINISTIC_FIRST.get(method, 99) if prefer_deterministic else 0
    )
    impact_rank = -_IMPACT_ORDER.get(str(row["impact_estimate"]), 0)
    confidence_rank = -_CONFIDENCE_ORDER.get(str(row["confidence"]), 0)
    return method_rank, impact_rank, confidence_rank, str(row["catalog_entry_id"])
