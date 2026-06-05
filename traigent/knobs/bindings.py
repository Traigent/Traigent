"""The one-Knob model: typed bindings Tuned | Fixed | Calibrated (RFC 0001).

Every configuration variable is a knob with exactly one binding:

- ``Tuned`` — optimizer-visible, searched over a :class:`ParameterRange`.
  Its ``default`` mirrors ``ParameterRange.default``: a default CANDIDATE
  value inside the searched domain, never a Fixed binding.
- ``Fixed`` — a runtime-supplied constant; never searched.
- ``Calibrated`` — a CVAR: produced by a calibrator from calibration
  evidence, optimizer-invisible, certificate-backed.

The binding union is discriminated by ``isinstance`` over the three frozen
classes — deliberately NO string discriminator, so no second kind taxonomy
competes with :class:`~traigent.knobs.kinds.KnobKind` (which classifies what
a knob effectuates, not how it binds).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from traigent.api.parameter_ranges import ParameterRange

from .certificates import Certificate, TargetProperty
from .kinds import KnobKind
from .signals import SignalSpec

__all__ = ["Binding", "Calibrated", "Fixed", "Knob", "Ref", "Tuned", "is_governed"]

TValue = TypeVar("TValue")


@dataclass(frozen=True, slots=True)
class Tuned(Generic[TValue]):
    """Optimizer-visible binding over a searched domain."""

    range: ParameterRange
    default: TValue | None = None  # default CANDIDATE, never Fixed


@dataclass(frozen=True, slots=True)
class Fixed(Generic[TValue]):
    """Runtime-supplied constant binding; never searched."""

    value: TValue


_IDENT_RE = __import__("re").compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$"
)


@dataclass(frozen=True, slots=True)
class Ref:
    """A namespace reference to another knob (a CVAR's tuned parent).

    Per RFC 0001 §3.3, v1 refs MUST name Tuned knobs (CVAR→CVAR parents are
    deferred). Dependencies are a freshness relation, not a constraint.
    """

    knob: str
    field: str | None = None

    def __post_init__(self) -> None:
        if not _IDENT_RE.match(self.knob):
            raise ValueError(f"Ref.knob is not a valid identifier: {self.knob!r}")


@dataclass(frozen=True, slots=True)
class Calibrated(Generic[TValue]):
    """A CVAR binding: value produced by a calibrator, certificate-backed.

    ``fallback`` may only be consumed OUTSIDE strict evidence modes, and its
    use is always observable in resolution output — never silent.
    """

    signal: SignalSpec
    target: TargetProperty
    value_type: str = "float"  # declared CVAR type (R5/R6 conformance)
    depends_on: tuple[Ref, ...] = field(default=())
    fallback: Fixed[TValue] | None = None
    certificate: Certificate | None = None
    require_calibration: bool = False
    target_epsilon: float | None = None  # chance-style level for the R8 floor


def is_governed(binding: Calibrated[Any]) -> bool:
    """RFC 0001 §3.6: is this Calibrated binding under DECLARED governance?

    Governance is declared on the binding/target — per-CVAR
    ``require_calibration``, a declaration-pinned certificate, or a
    certificate-backed / guaranteed-selection target property. It is never
    inferred from runtime-presented evidence (presenting a certificate at
    resolution time is evidence, not an opt-in to strictness).

    Shared by the resolver's per-CVAR resolution rules and the orchestrator's
    strict-evidence-mode detection so the two can never drift.
    """
    return (
        binding.require_calibration
        or binding.certificate is not None
        or binding.target.mode in ("certificate_backed", "guaranteed_selection")
    )


#: The binding union — isinstance-discriminated.
Binding = Tuned[TValue] | Fixed[TValue] | Calibrated[TValue]


@dataclass(frozen=True, slots=True)
class Knob(Generic[TValue]):
    """A named configuration variable with exactly one typed binding."""

    name: str
    binding: Tuned[TValue] | Fixed[TValue] | Calibrated[TValue]
    kind: KnobKind = KnobKind.VALUE
    description: str | None = None

    def __post_init__(self) -> None:
        if not _IDENT_RE.match(self.name):
            raise ValueError(f"Knob.name is not a valid identifier: {self.name!r}")
        if not isinstance(self.binding, (Tuned, Fixed, Calibrated)):
            raise TypeError(
                f"Knob binding must be Tuned, Fixed, or Calibrated; "
                f"got {type(self.binding).__name__}"
            )
        if isinstance(self.binding, Calibrated):
            for ref in self.binding.depends_on:
                if ref.knob == self.name:
                    raise ValueError(f"Knob {self.name!r} cannot depend on itself")

    def is_tuned(self) -> bool:
        return isinstance(self.binding, Tuned)

    def is_fixed(self) -> bool:
        return isinstance(self.binding, Fixed)

    def is_calibrated(self) -> bool:
        return isinstance(self.binding, Calibrated)

    @property
    def optimizer_visible(self) -> bool:
        """P2: only Tuned knobs enter the optimizer's search space."""
        return self.is_tuned()
