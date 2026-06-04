"""Target properties, freshness contexts, and certificates (RFC 0001 §3.5).

A certificate binds a SPECIFIC calibrated value of a SPECIFIC CVAR to the
exact context it was fit under. Validity is fail-closed: every field
participates, the audit copies must agree with the live context, and the
live context must itself name the queried CVAR (the forged-subject guard).

The certificate is a CLOSED shape (Property P8): identifiers, types, hashes,
counts, and an enum — no open payload field exists to leak content.

The ``decision`` vocabulary is shared with TraigentSchema's per-config
guarantee certificates (``guarantee_certificate_schema.json``): one
certificate ontology, two scopes — per-config selection there, per-CVAR
value here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .canonical import CTX_SCHEMA_VERSION, canonical_hash

__all__ = [
    "CTX_EXT_KEYS",
    "Certificate",
    "CertificateDecision",
    "EvidenceRef",
    "FreshnessContext",
    "TargetProperty",
    "conformal_evidence_floor",
    "issue_certificate",
]

#: Optional freshness-context EXTENSION keys. The mandatory core (see
#: ``FreshnessContext``) is implicit and immutable — module configuration
#: can only ADD coverage, never remove parent-specificity.
CTX_EXT_KEYS = frozenset(
    {"stage_versions", "model_versions", "budget_assumptions", "cost_assumptions"}
)


class CertificateDecision(str, Enum):
    """Shared decision vocabulary (TraigentSchema guarantee certificates)."""

    CERTIFIED = "CERTIFIED_SELECTION"
    NO_DECISION = "NO_DECISION"
    BEST_EFFORT_UNCERTIFIED = "BEST_EFFORT_UNCERTIFIED"


@dataclass(frozen=True, slots=True)
class TargetProperty:
    """The property a calibrated value is certified against."""

    name: str
    mode: str  # require_calibration | chance_constraint | guaranteed_selection | certificate_backed
    threshold: float | None = None
    confidence: float | None = None

    _MODES = (
        "require_calibration",
        "chance_constraint",
        "guaranteed_selection",
        "certificate_backed",
    )

    def __post_init__(self) -> None:
        if self.mode not in self._MODES:
            raise ValueError(f"unknown TargetProperty mode: {self.mode!r}")

    def property_hash(self) -> str:
        return canonical_hash(
            {
                "name": self.name,
                "mode": self.mode,
                "threshold": self.threshold,
                "confidence": self.confidence,
            }
        )


def conformal_evidence_floor(epsilon: float) -> int:
    """The split-conformal minimum calibration size ``ceil(1/epsilon) - 1``.

    Below this floor even a perfect calibrator cannot certify a chance-style
    target at level ``epsilon`` — the resolver's R8 rejection uses it as a
    closed-form ``insufficient_evidence`` precondition.
    """
    if not (0.0 < epsilon < 1.0):
        raise ValueError("epsilon must be in (0, 1)")
    return math.ceil(1.0 / epsilon) - 1


@dataclass(frozen=True, slots=True)
class FreshnessContext:
    """The hash-covered calibration context (RFC 0001 §3.5).

    The MANDATORY core is every constructor field below except
    ``extensions`` — it always participates in the hash, so certificates are
    parent-specific (``tuned_parent_values`` is core), evidence-specific
    (dataset hash + count), and calibrator-specific (id, version, params) by
    construction. ``extensions`` selects OPTIONAL extra keys from
    ``CTX_EXT_KEYS`` only.
    """

    cvar_name: str
    tuned_parent_values: tuple[tuple[str, Any], ...]
    calibration_source_id: str
    signal_spec_hash: str
    calibrator_id: str
    calibrator_version: str
    calibrator_params_hash: str
    dataset_hash: str
    evidence_n: int
    calibration_split: str
    eval_split: str
    target: TargetProperty
    extensions: tuple[tuple[str, Any], ...] = field(default=())

    def __post_init__(self) -> None:
        if not isinstance(self.evidence_n, int) or isinstance(self.evidence_n, bool):
            raise ValueError("evidence_n must be a natural number (int)")
        if self.evidence_n < 0:
            raise ValueError("evidence_n must be non-negative")

    def freshness_hash(self) -> str:
        """Canonical hash over the core + selected extensions.

        Extensions are a MAP: sorted by key before hashing (order-insensitive)
        with duplicate keys rejected. Parent values are likewise sorted by
        name with duplicates rejected — the definition enforces the RFC's
        "sorted by name", never trusting the caller.
        """
        parent_names = [name for name, _ in self.tuned_parent_values]
        if len(parent_names) != len(set(parent_names)):
            raise ValueError("duplicate tuned parent name")
        ext_keys = [key for key, _ in self.extensions]
        for key in ext_keys:
            if key not in CTX_EXT_KEYS:
                raise ValueError(f"invalid_calibration_context: {key!r}")
        if len(ext_keys) != len(set(ext_keys)):
            raise ValueError("duplicate_calibration_context_key")

        return canonical_hash(
            {
                "core": {
                    "ctx_schema_version": CTX_SCHEMA_VERSION,
                    "cvar_name": self.cvar_name,
                    "tuned_parent_values": sorted(
                        ([name, value] for name, value in self.tuned_parent_values),
                        key=lambda pair: pair[0],
                    ),
                    "calibration_source_id": self.calibration_source_id,
                    "signal_spec_hash": self.signal_spec_hash,
                    "calibrator_id": self.calibrator_id,
                    "calibrator_version": self.calibrator_version,
                    "calibrator_params_hash": self.calibrator_params_hash,
                    "dataset_hash": self.dataset_hash,
                    "evidence_n": self.evidence_n,
                    "calibration_split": self.calibration_split,
                    "eval_split": self.eval_split,
                    "target": {
                        "name": self.target.name,
                        "mode": self.target.mode,
                        "threshold": self.target.threshold,
                        "confidence": self.target.confidence,
                    },
                },
                "ext": sorted(
                    ([key, value] for key, value in self.extensions),
                    key=lambda pair: pair[0],
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    """Audit copy of the evidence behind a certificate (closed shape)."""

    n: int
    pool_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.n, int) or isinstance(self.n, bool):
            raise ValueError("EvidenceRef.n must be a natural number (int)")
        if self.n < 0:
            raise ValueError("EvidenceRef.n must be non-negative")


@dataclass(frozen=True, slots=True)
class Certificate:
    """Validity object for one calibrated value of one CVAR.

    ``payload``-style open fields are deliberately ABSENT (P8). The audit
    copies (``target``, ``evidence``) duplicate context fields for
    inspection; ``valid_for`` cross-checks them against the live context so
    a certificate cannot DISPLAY one context while HASHING another.
    """

    subject_cvar: str
    subject_type: str
    subject_value_hash: str
    target: TargetProperty
    issued_hash: str
    decision: CertificateDecision
    evidence: EvidenceRef

    def valid_for(
        self,
        cvar_name: str,
        cvar_type: str,
        value: Any,
        ctx_now: FreshnessContext,
    ) -> bool:
        """RFC 0001 §3.5 validity — all nine conjuncts, fail-closed."""
        return (
            ctx_now.cvar_name == cvar_name
            and self.subject_cvar == cvar_name
            and self.subject_type == cvar_type
            and self.subject_value_hash == canonical_hash(value)
            and self.target == ctx_now.target
            and self.evidence.n == ctx_now.evidence_n
            and self.evidence.pool_hash == ctx_now.dataset_hash
            and self.issued_hash == ctx_now.freshness_hash()
            and self.decision is CertificateDecision.CERTIFIED
        )


def issue_certificate(
    cvar_name: str,
    cvar_type: str,
    value: Any,
    ctx: FreshnessContext,
    *,
    decision: CertificateDecision = CertificateDecision.CERTIFIED,
) -> Certificate:
    """Issue a certificate binding ``value`` of ``cvar_name`` to ``ctx``."""
    if ctx.cvar_name != cvar_name:
        raise ValueError(
            f"context names {ctx.cvar_name!r}, cannot issue for {cvar_name!r}"
        )
    return Certificate(
        subject_cvar=cvar_name,
        subject_type=cvar_type,
        subject_value_hash=canonical_hash(value),
        target=ctx.target,
        issued_hash=ctx.freshness_hash(),
        decision=decision,
        evidence=EvidenceRef(n=ctx.evidence_n, pool_hash=ctx.dataset_hash),
    )
