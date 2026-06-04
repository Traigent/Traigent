"""Adapters between ParameterRange (the existing surface) and Knob[Tuned].

``ParameterRange`` stays unchanged (RFC 0001 decision): ``Range.default``
remains a default CANDIDATE value, and the adapters preserve that — a Tuned
default is never coerced into a Fixed binding.
"""

from __future__ import annotations

from traigent.api.parameter_ranges import ParameterRange

from .bindings import Knob, Tuned
from .kinds import KnobKind

__all__ = ["knob_to_parameter_range", "parameter_range_to_knob"]


def parameter_range_to_knob(
    name: str,
    parameter_range: ParameterRange,
    *,
    kind: KnobKind = KnobKind.VALUE,
    description: str | None = None,
) -> Knob:
    """Wrap an existing ParameterRange as a Tuned knob (identity-preserving)."""
    return Knob(
        name=name,
        binding=Tuned(range=parameter_range, default=parameter_range.get_default()),
        kind=kind,
        description=description,
    )


def knob_to_parameter_range(knob: Knob) -> ParameterRange | None:
    """Project a knob onto the optimizer-visible surface.

    Returns the underlying ParameterRange for Tuned knobs and ``None`` for
    Fixed/Calibrated knobs — they are optimizer-invisible (P2), which is
    exactly what makes ``ConfigSpace.tvars`` the Tuned-only projection.
    """
    binding = knob.binding
    if isinstance(binding, Tuned):
        return binding.range
    return None
