"""Closed vocabularies for text that crosses to the backend (#2241).

Customer content must never ride on free-text fields. Two channels were open:

* the trial ``error_message`` (``str(exception)``, which can quote a prompt or a
  model reply), and
* three free-form metadata dictionaries (``agent_configuration``,
  ``surrogate_evaluator`` and ``summary_stats.metadata``).

This module closes both. Errors collapse to one value of a fixed list
(:data:`WIRE_ERROR_CODES`). Metadata is rebuilt from an allowlist of keys;
anything not listed is dropped, and values are bounded scalars.

The wire field stays a string, so the contract shape does not change; the
TraigentSchema/backend side of the closed list is tracked separately.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from traigent.core.exception_handler import (
    VendorErrorCategory,
    classify_vendor_error,
)

ERROR_TRIAL_FAILED = "trial_failed"
ERROR_TRIAL_CANCELLED = "trial_cancelled"
ERROR_TRIAL_PRUNED = "trial_pruned"
ERROR_NO_EXAMPLES_SUCCEEDED = "no_examples_succeeded"

#: The complete set of values ``error_message`` may carry on the wire.
WIRE_ERROR_CODES: frozenset[str] = frozenset(
    {
        ERROR_TRIAL_FAILED,
        ERROR_TRIAL_CANCELLED,
        ERROR_TRIAL_PRUNED,
        ERROR_NO_EXAMPLES_SUCCEEDED,
        *(f"vendor_{category.value}" for category in VendorErrorCategory),
    }
)


def error_code_for_wire(error_message: str | None) -> str | None:
    """Collapse an error text to one closed code; the text itself never leaves.

    Classification reads the text locally but the return value is always a
    member of :data:`WIRE_ERROR_CODES`. Anything not recognised is
    ``trial_failed``.
    """
    if not error_message:
        return None
    text = str(error_message)
    lowered = text.strip().lower()
    if lowered.startswith("no examples succeeded"):
        return ERROR_NO_EXAMPLES_SUCCEEDED
    if lowered.startswith("trial cancelled"):
        return ERROR_TRIAL_CANCELLED
    if lowered == "trial_pruned" or lowered.startswith("trial pruned"):
        return ERROR_TRIAL_PRUNED
    category = classify_vendor_error(RuntimeError(text))
    if category is not None:
        return f"vendor_{category.value}"
    return ERROR_TRIAL_FAILED


_MAX_STR = 128
_MAX_LIST = 256

# Spec grammar: ``None`` = bounded scalar, ``"list"`` = list of bounded scalars,
# ``{...}`` = nested mapping with its own allowlist, ``("map", spec)`` = a
# mapping with caller-chosen keys whose values follow ``spec``.
_Spec = Any

AGENT_CONFIGURATION_SPEC: dict[str, _Spec] = {
    "version": None,
    "auto_inferred": None,
    "agents": (
        "map",
        {
            "display_name": None,
            "parameter_keys": "list",
            "measure_ids": "list",
            "primary_model": None,
            "order": None,
            "agent_type": None,
            "meta": {"color": None, "icon": None},
        },
    ),
    "global": {"parameter_keys": "list", "measure_ids": "list", "order": None},
}

SURROGATE_EVALUATOR_SPEC: dict[str, _Spec] = {
    "evaluator_id": None,
    "metric_name": None,
    "judge_model": None,
    "config": {"fingerprint_source": None},
}

SUMMARY_STATS_METADATA_SPEC: dict[str, _Spec] = {
    "sdk_version": None,
    "aggregation_method": None,
    "aggregation_level": None,
    "timestamp": None,
    "source": None,
    "sample_budget_exhausted": None,
    "examples_consumed": None,
    "mode": None,
    "total_examples": None,
}


def _scalar(value: Any) -> tuple[bool, Any]:
    if value is None or isinstance(value, (bool, int, float)):
        return True, value
    if isinstance(value, str):
        return True, value[:_MAX_STR]
    return False, None


def _filter(value: Any, spec: _Spec) -> tuple[bool, Any]:
    if spec is None:
        return _scalar(value)
    if spec == "list":
        if not isinstance(value, (list, tuple)):
            return False, None
        items = []
        for item in list(value)[:_MAX_LIST]:
            ok, kept = _scalar(item)
            if ok:
                items.append(kept)
        return True, items
    if isinstance(spec, tuple) and spec[0] == "map":
        if not isinstance(value, Mapping):
            return False, None
        out: dict[str, Any] = {}
        for key, item in list(value.items())[:_MAX_LIST]:
            ok, kept = _filter(item, spec[1])
            if ok and isinstance(key, str):
                out[key[:_MAX_STR]] = kept
        return True, out
    if isinstance(spec, dict):
        if not isinstance(value, Mapping):
            return False, None
        nested: dict[str, Any] = {}
        for key, sub in spec.items():
            if key in value:
                ok, kept = _filter(value[key], sub)
                if ok:
                    nested[key] = kept
        return True, nested
    return False, None


def _apply(value: Any, spec: dict[str, _Spec]) -> dict[str, Any]:
    ok, kept = _filter(value, spec)
    return kept if ok else {}


def filter_agent_configuration(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only allowlisted keys of an ``agent_configuration`` payload."""
    return _apply(value, AGENT_CONFIGURATION_SPEC)


def filter_surrogate_evaluator(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only allowlisted keys of a ``surrogate_evaluator`` descriptor."""
    return _apply(value, SURROGATE_EVALUATOR_SPEC)


def filter_summary_stats_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only allowlisted keys of ``summary_stats.metadata``."""
    return _apply(value, SUMMARY_STATS_METADATA_SPEC)
