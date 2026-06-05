"""Resolution types: nodes, errors, and the Resolver contract (RFC 0001 §3.4).

This module ships the TYPES in packet 6-B; the deterministic topological
resolver LOGIC and its orchestrator injection land in packet 6-C2
(``feature/knobs-resolver``). The rejection vocabulary R1–R8 is normative
and shared with the tvl repo's model-checking suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

__all__ = ["ResolutionError", "ResolutionNode", "ResolutionRejection"]


class ResolutionRejection(StrEnum):
    """RFC 0001 §3.4 rejection conditions (R1–R8) + the no-decision verdict."""

    CYCLE = "cycle"  # R1 — unreachable in v1 (CVAR→TVAR only); kept for the deferred extension
    MISSING_REF = "missing_ref"  # R2
    DUPLICATE_PROVIDER = "duplicate_provider"  # R3
    PHASE_MISMATCH = "phase_mismatch"  # R4
    INFEASIBLE_VALUE = "infeasible_value"  # R5
    STALE_CERTIFICATE = "stale_certificate"  # R6
    EVIDENCE_LEAKAGE = "evidence_leakage"  # R7
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"  # R8
    NO_DECISION = "no_decision"  # calibrator ⊥ with no R_i holding


class ResolutionError(Exception):
    """Typed, fail-closed resolution failure.

    Carries the full set of rejections so callers can report every reason,
    not just the first — mirrors the model checker's collect-all semantics.
    """

    def __init__(
        self, rejections: tuple[ResolutionRejection, ...], detail: str = ""
    ) -> None:
        self.rejections = rejections
        codes = ", ".join(r.value for r in rejections)
        super().__init__(
            f"resolution rejected [{codes}]" + (f": {detail}" if detail else "")
        )


@dataclass(frozen=True, slots=True)
class ResolutionNode:
    """One node of the resolution DAG (built by the 6-C2 resolver)."""

    knob: str
    binding_kind: str  # "tuned" | "fixed" | "calibrated"
    depends_on: tuple[str, ...] = ()
    phase: str = "post_suggest"  # pre_suggest | post_suggest | pre_eval

    _BINDING_KINDS = ("tuned", "fixed", "calibrated")
    _PHASES = ("pre_suggest", "post_suggest", "pre_eval")

    def __post_init__(self) -> None:
        if self.binding_kind not in self._BINDING_KINDS:
            raise ValueError(f"unknown binding_kind: {self.binding_kind!r}")
        if self.phase not in self._PHASES:
            raise ValueError(f"unknown phase: {self.phase!r}")
