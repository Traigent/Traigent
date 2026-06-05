"""Framework-parameter effectuation strategy.

Value knobs are already applied by Traigent's integration override path. This
strategy formalizes that executable path without re-applying or clobbering
framework kwargs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from traigent.effectuation.contracts import (
    KnobKind,
    knob_kind,
    project_knob_for_optimizer,
)


@dataclass(frozen=True)
class FrameworkParamEffect:
    """Compiled value-knob effect that leaves runtime execution unchanged."""

    knob: Any

    def project_for_optimizer(self) -> dict[str, Any]:
        return project_knob_for_optimizer(self.knob)

    def wrap_callable(self, fn: Callable[..., Any], plan: Any) -> Callable[..., Any]:
        return fn

    def emit_events(self) -> list[dict[str, Any]]:
        return []


class FrameworkParamStrategy:
    """Strategy for already-supported framework value parameters."""

    strategy_id = "framework_param"
    supported_kinds = {KnobKind.VALUE.value}

    def validate(self, knob: Any, runtime_ctx: Any) -> None:
        kind = knob_kind(knob, default=KnobKind.VALUE.value)
        if kind != KnobKind.VALUE.value:
            raise ValueError(
                f"{self.strategy_id} only supports value knobs, got {kind!r}"
            )
        project_knob_for_optimizer(knob)

    def compile(self, knob: Any) -> FrameworkParamEffect:
        self.validate(knob, runtime_ctx=None)
        return FrameworkParamEffect(knob=knob)


__all__ = ["FrameworkParamEffect", "FrameworkParamStrategy"]
