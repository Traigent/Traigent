"""Sugar kwargs -> :class:`RuntimeRequirement`."""

from __future__ import annotations

from collections.abc import Callable

from traigent.runtime.model import (
    DEFAULT_ENVIRONMENT,
    E_ACTION_UNSUPPORTED,
    E_BAD_APPROVAL,
    E_BAD_WINDOW,
    E_FIX_CEILING,
    E_INCOMPLETE,
    E_INVALID,
    PROFILE_DEFAULTS,
    RuntimeEvent,
    RuntimeRequirement,
    RuntimeRequirementError,
    WINDOWS,
    build_rules,
    normalise_on_breach,
    positive_finite,
)


def from_kwargs(
    *,
    latency_ms: int | None,
    max_violation_rate: float | None,
    window: str = "6h",
    environment: str = DEFAULT_ENVIRONMENT,
    min_samples: int | None = None,
    confidence: float | None = None,
    on_breach: str = "alert",
    approval: str = "human",
    max_fix_cost_usd: float | None = None,
    name: str = "latency",
    agent_id: str | None = None,
    on_event: Callable[[RuntimeEvent], None] | None = None,
) -> RuntimeRequirement:
    """Build and validate a requirement from the simplified kwargs."""
    if latency_ms is None or max_violation_rate is None:
        raise RuntimeRequirementError(E_INCOMPLETE, "latency_ms/max_violation_rate")
    if approval == "standing":
        # Standing grants are not implemented (BE-4); never a silent downgrade.
        raise RuntimeRequirementError(E_ACTION_UNSUPPORTED, "approval")
    if approval != "human":
        raise RuntimeRequirementError(E_BAD_APPROVAL, "approval")
    mode = normalise_on_breach(on_breach)
    if not isinstance(window, str) or window not in WINDOWS:
        raise RuntimeRequirementError(E_BAD_WINDOW, "window")
    if mode == "fix" and max_fix_cost_usd is None:
        raise RuntimeRequirementError(E_FIX_CEILING, "max_fix_cost_usd")
    if max_fix_cost_usd is not None and not positive_finite(max_fix_cost_usd):
        raise RuntimeRequirementError(E_FIX_CEILING, "max_fix_cost_usd")
    if not isinstance(environment, str):
        raise RuntimeRequirementError(E_INVALID, "environment")
    requirement = RuntimeRequirement(
        name=name,
        latency_threshold_ms=latency_ms,
        environment=environment,
        max_violation_rate=max_violation_rate,
        confidence=confidence
        if confidence is not None
        else PROFILE_DEFAULTS["confidence"],  # type: ignore[arg-type]
        window_hours=WINDOWS[window],
        min_samples=min_samples
        if min_samples is not None
        else PROFILE_DEFAULTS["min_samples"],  # type: ignore[arg-type]
        rules=build_rules(mode, approval),
        on_breach=mode,
        max_fix_cost_usd=max_fix_cost_usd,
        agent_id=agent_id,
        source="kwargs",
        on_event=on_event,
    )
    errors = [issue for issue in requirement.validate() if issue.severity == "error"]
    if errors:
        raise RuntimeRequirementError(errors[0].code, errors[0].path)
    return requirement


__all__ = ["from_kwargs"]
