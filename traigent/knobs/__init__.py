"""traigent.knobs — the one-Knob model (RFC 0001: knob bindings).

Pure types in this packet: bindings (Tuned | Fixed | Calibrated), signal
specs and content-free observations, certificates with hash-covered
freshness contexts, resolution vocabulary, and ParameterRange adapters.
No runtime behavior changes anywhere else; nothing here is wired into the
orchestrator yet (that is the knobs-resolver packet) and nothing crosses a
wire boundary (see the program's contract-decision record).

Import-light by design: importing this package must not pull in
``traigent.core`` or the cloud client.

DESIGN NOTE for the deferred CVAR optimizer: when many candidate
configurations share one calibration pool, certification must use
fixed-sequence testing in a pre-committed (e.g. cost) order — never
Bonferroni over an unordered grid, and never silent pool reuse
(multiplicity accounting belongs INSIDE the certificate; see Pareto
Testing, arXiv:2210.07913).
"""

from __future__ import annotations

from .adapters import knob_to_parameter_range, parameter_range_to_knob
from .bindings import Binding, Calibrated, Fixed, Knob, Ref, Tuned, is_governed
from .canonical import CTX_SCHEMA_VERSION, CanonicalizationError, canonical_hash
from .cascade import (
    CascadePolicy,
    CascadeStep,
    Gate,
    GateKind,
    StageSpec,
    VoteStats,
    vote_over,
)
from .certificates import (
    CTX_EXT_KEYS,
    Certificate,
    CertificateDecision,
    EvidenceRef,
    FreshnessContext,
    TargetProperty,
    conformal_evidence_floor,
    issue_certificate,
)
from .kinds import KnobKind
from .resolution import ResolutionError, ResolutionNode, ResolutionRejection
from .resolver import CalibratedInput, KnobResolver, ResolvedConfiguration
from .signals import SignalObservation, SignalSpec
from .telemetry import composite_measures

__all__ = [
    "CTX_EXT_KEYS",
    "CTX_SCHEMA_VERSION",
    "Binding",
    "Calibrated",
    "CalibratedInput",
    "CanonicalizationError",
    "CascadePolicy",
    "CascadeStep",
    "Certificate",
    "CertificateDecision",
    "EvidenceRef",
    "Fixed",
    "FreshnessContext",
    "Gate",
    "GateKind",
    "Knob",
    "is_governed",
    "KnobKind",
    "KnobResolver",
    "Ref",
    "ResolutionError",
    "ResolutionNode",
    "ResolutionRejection",
    "ResolvedConfiguration",
    "SignalObservation",
    "SignalSpec",
    "StageSpec",
    "TargetProperty",
    "Tuned",
    "VoteStats",
    "canonical_hash",
    "composite_measures",
    "conformal_evidence_floor",
    "issue_certificate",
    "knob_to_parameter_range",
    "parameter_range_to_knob",
    "vote_over",
]
