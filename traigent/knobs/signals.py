"""Signal specifications and content-free observations (RFC 0001 §3.5).

``SignalObservation`` is a CLOSED shape by design (Property P8): every field
is an identifier, finite number, count, or split label — there is no
metadata map and no content-typed field to leak. Anything richer belongs
outside the persisted-signal surface.
"""

from __future__ import annotations

from dataclasses import dataclass

from .canonical import canonical_hash

__all__ = ["SignalObservation", "SignalSpec"]


@dataclass(frozen=True, slots=True)
class SignalSpec:
    """Versioned identification of a measurement procedure.

    The spec hash participates in certificate freshness (``ctx_core``): any
    change to the signal, its score function, or its comparator — including
    version bumps — makes previously issued certificates stale by
    construction.
    """

    name: str
    version: str
    score_function: str
    score_function_version: str
    comparator: str
    comparator_version: str

    def spec_hash(self) -> str:
        """Stable canonical hash over every identifying field."""
        return canonical_hash(
            {
                "name": self.name,
                "version": self.version,
                "score_function": self.score_function,
                "score_function_version": self.score_function_version,
                "comparator": self.comparator,
                "comparator_version": self.comparator_version,
            }
        )


@dataclass(frozen=True, slots=True)
class SignalObservation:
    """One content-free observation of a signal (closed shape — P8).

    Args:
        signal: Signal spec name this observation belongs to.
        value: The observed signal value (finite float).
        n: Sample count behind the observation.
        split: Which data split produced it (e.g. ``"calibration"``,
            ``"eval"``) — the resolver's evidence-leakage rejection (R7)
            depends on faithful split tagging.
    """

    signal: str
    value: float
    n: int
    split: str

    def __post_init__(self) -> None:
        if self.value != self.value or self.value in (float("inf"), float("-inf")):
            raise ValueError("SignalObservation.value must be finite")
        if self.n < 0:
            raise ValueError("SignalObservation.n must be non-negative")
