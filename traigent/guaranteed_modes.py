"""Client-side helpers for the guaranteed optimizer modes.

Builds a ``guaranteed_selection`` request and applies the FAIL-CLOSED guard on the result: the
SDK must never deploy a config the backend did not certify. :func:`is_deployable` /
:func:`deployable_config` yield a deployable config ONLY for a ``CERTIFIED_SELECTION``; every
other status (``NO_DECISION_*``, ``BEST_EFFORT_UNCERTIFIED``) yields no deployable config, so a
worse-but-cheaper candidate can never be auto-applied through the SDK (legacy best-config apply
paths included). Mirrors TraigentSchema ``guaranteed_selection_{request,result}_schema``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

CERTIFIED_STATUS = "CERTIFIED_SELECTION"
SELECTION_MODES = frozenset({"keep_accuracy_reduce_cost", "accuracy_then_cost"})


def build_guaranteed_selection_request(
    selection_mode: str,
    *,
    delta: float,
    epsilon: float = 0.10,
    baseline_ref: str | None = None,
    eta: float = 0.0,
    epsilon_allocation: Mapping[str, float] | None = None,
    min_pairs: int = 2,
    cost_value_range: float | None = None,
    allow_best_effort_uncertified: bool = False,
) -> dict[str, Any]:
    """Build a guaranteed_selection request payload (validated client-side)."""
    if selection_mode not in SELECTION_MODES:
        raise ValueError(f"unknown selection_mode: {selection_mode!r}")
    if selection_mode == "keep_accuracy_reduce_cost" and not baseline_ref:
        raise ValueError("keep_accuracy_reduce_cost requires baseline_ref")
    request: dict[str, Any] = {
        "schema_version": "traigent.guaranteed_selection_request.v1",
        "selection_mode": selection_mode,
        "delta": delta,
        "epsilon": epsilon,
        "eta": eta,
        "min_pairs": min_pairs,
        "allow_best_effort_uncertified": allow_best_effort_uncertified,
    }
    if baseline_ref is not None:
        request["baseline_ref"] = baseline_ref
    if epsilon_allocation is not None:
        request["epsilon_allocation"] = dict(epsilon_allocation)
    if cost_value_range is not None:
        request["cost_value_range"] = cost_value_range
    return request


def is_deployable(result: object) -> bool:
    """Fail-closed: True only for a CERTIFIED_SELECTION that carries a config to deploy."""
    if not isinstance(result, Mapping):
        return False
    if result.get("status") != CERTIFIED_STATUS:
        return False
    if result.get("deployable") is not True:
        return False
    selected = result.get("selected_config")
    return isinstance(selected, str) and bool(selected)


def deployable_config(result: object) -> str | None:
    """The config to deploy, or ``None`` when the result is not a certified selection.

    The SDK must NEVER auto-apply a config from a non-certified result.
    """
    if not isinstance(result, Mapping):
        return None
    if not is_deployable(result):
        return None
    selected = result.get("selected_config")
    return selected if isinstance(selected, str) else None
