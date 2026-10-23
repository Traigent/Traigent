"""Promotion-gate evidence for ``deploy: if_promoted`` (TRAIG-101 plan section 8).

A candidate of an optimize run is reported to the Backend as a CLOSED fact:

* ``fingerprint``: the sha256 (hex) of the canonical JSON of the candidate
  configuration (the knob values).  The configuration itself never leaves the
  customer side with this fact.
* ``legacy_decision``: the existing promotion decision (``Promote`` / ``Reject`` /
  ``NoDecision``), e.g. the SDK ``PromotionGate`` result.
* ``monitor_verdicts``: one verdict per promotion-phase monitor of the requirement,
  from the TVL reference ``evaluate_promotion_monitor`` over the candidate's
  evaluation-set counts (RFC 0003 6.11).  A monitor with no counts is
  ``UNKNOWN(insufficient)``: absence never promotes.
* ``gate_result_digest``: sha256 over the gate result under the CURRENT contract digest
  (format, contract digest, fingerprint, legacy decision, verdicts keyed by the
  monitors' hosted tokens, and the joined decision).  A gate result computed under
  another contract digest is a different digest, so it is not current.

The join is the reference ``tvl.promotion_join.join_promotion_decisions`` (Reject over
NoDecision over Promote); this module never re-implements it.  Evidence that cannot be
established (a count that violates conservation, an unusable configuration, an unknown
decision) is ``{"evidence_valid": False}`` with nothing else, never a guessed verdict.
Needs the ``runtime-tvl`` extra (``tvl-spec``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from traigent.runtime.hosted import (
    NS_MONITOR,
    UNKNOWN_REASONS,
    TokenMap,
    build_candidate,
)
from traigent.runtime.model import (
    E_COMPILER,
    RuntimeRequirement,
    RuntimeRequirementError,
    digest_of,
)

GATE_FORMAT = "traigent-gate-result/1"
_LEGACY = {
    "promote": "Promote",
    "reject": "Reject",
    "no_decision": "NoDecision",
    "Promote": "Promote",
    "Reject": "Reject",
    "NoDecision": "NoDecision",
}


@dataclass(frozen=True)
class CandidateEvidence:
    """What the customer side knows about one candidate (never sent as is)."""

    config: Mapping[str, Any]
    legacy_decision: str
    #: promotion monitor name -> evaluation-set counts ``{"n", "k", ["u", "pl", "complete"]}``
    monitor_counts: Mapping[str, Mapping[str, Any]]


def candidate_fingerprint(config: Mapping[str, Any]) -> str:
    """sha256 hex of the canonical JSON of the candidate configuration."""
    if not isinstance(config, Mapping) or not config:
        raise RuntimeRequirementError("candidate_unfingerprintable")
    try:
        return digest_of(dict(config))
    except (TypeError, ValueError):
        raise RuntimeRequirementError("candidate_unfingerprintable") from None


def latency_counts(
    latencies_ms: Sequence[float | None], threshold_ms: float
) -> dict[str, int]:
    """Counts of an evaluation set against a latency threshold.

    ``n`` resolved units, ``k`` of them above ``threshold_ms``, ``u`` unresolved (no
    usable latency).  Unresolved units make the verdict UNKNOWN in the reference
    evaluator; they are never counted as met.
    """
    resolved = [
        v
        for v in latencies_ms
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    ]
    return {
        "n": len(resolved),
        "k": sum(1 for v in resolved if v > threshold_ms),
        "u": len(latencies_ms) - len(resolved),
    }


def _tvl() -> tuple[Any, Any, Any]:
    try:
        from tvl.canonical import RuntimeClosedError
        from tvl.promotion_join import join_promotion_decisions
        from tvl.promotion_monitor import evaluate_promotion_monitor
    except ImportError:
        raise RuntimeRequirementError(E_COMPILER) from None
    return RuntimeClosedError, join_promotion_decisions, evaluate_promotion_monitor


def _verdict_text(result: Mapping[str, Any]) -> str:
    verdict = str(result.get("verdict"))
    reason = result.get("reason")
    if verdict == "UNKNOWN" and reason in UNKNOWN_REASONS:
        return f"UNKNOWN({reason})"
    return verdict


def gate_result_digest(
    *,
    contract_digest: str,
    fingerprint: str,
    legacy_decision: str,
    verdicts_by_monitor_token: Mapping[str, str],
    decision: str,
) -> str:
    return digest_of(
        {
            "format": GATE_FORMAT,
            "contract_digest": contract_digest,
            "fingerprint": fingerprint,
            "legacy_decision": legacy_decision,
            "monitors": dict(verdicts_by_monitor_token),
            "decision": decision,
        }
    )


def evaluate_candidate(
    requirement: RuntimeRequirement,
    evidence: CandidateEvidence,
    *,
    contract_digest: str,
    tokens: TokenMap,
) -> dict[str, Any]:
    """The closed candidate fact for ``report_finished`` (see the module docstring)."""
    closed_error, join, evaluate = _tvl()
    invalid: dict[str, Any] = {"evidence_valid": False}
    legacy = (
        _LEGACY.get(evidence.legacy_decision)
        if isinstance(evidence.legacy_decision, str)
        else None
    )
    if legacy is None:
        return invalid
    try:
        fingerprint = candidate_fingerprint(evidence.config)
    except RuntimeRequirementError:
        return invalid
    doc = requirement.to_doc()
    req = next(
        (r for r in doc.get("requirements", []) if r.get("name") == requirement.name),
        None,
    )
    monitors = [m for m in doc.get("monitors", []) if m.get("phase") == "promotion"]
    if req is None:
        return invalid
    verdicts: list[str] = []
    by_token: dict[str, str] = {}
    for mon in monitors:
        counts = evidence.monitor_counts.get(mon["name"])
        if counts is None:
            text = "UNKNOWN(insufficient)"  # absence never promotes
        else:
            try:
                text = _verdict_text(evaluate(req, mon, counts))
            except closed_error:
                return invalid
            except (TypeError, ValueError, KeyError):
                return invalid
        verdicts.append(text)
        by_token[tokens.hosted(NS_MONITOR, mon["name"])] = text
    try:
        joined = join(legacy, verdicts)
    except closed_error:
        return invalid
    return build_candidate(
        {
            "evidence_valid": True,
            "legacy_decision": legacy,
            "monitor_verdicts": verdicts,
            "fingerprint": fingerprint,
            "gate_result_digest": gate_result_digest(
                contract_digest=contract_digest,
                fingerprint=fingerprint,
                legacy_decision=legacy,
                verdicts_by_monitor_token=by_token,
                decision=str(joined),
            ),
        }
    )


def candidate_extractor(
    requirement: RuntimeRequirement,
    tokens: TokenMap,
    evidence_of: Callable[[Any], Iterable[CandidateEvidence]],
) -> Callable[[Any, Mapping[str, Any]], list[dict[str, Any]]]:
    """A ``candidates=`` callable for ``respond_hosted`` (it supplies the grant's digest)."""

    def extract(result: Any, grant: Mapping[str, Any]) -> list[dict[str, Any]]:
        return [
            evaluate_candidate(
                requirement, ev, contract_digest=grant["contract_digest"], tokens=tokens
            )
            for ev in evidence_of(result)
        ]

    return extract
