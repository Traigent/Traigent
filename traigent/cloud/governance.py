# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Traigent-Commercial
# Copyright (c) 2024-2026 Traigent Ltd. Dual-licensed: AGPL-3.0 or commercial.
"""Content-free governance serializers for the typed session wire.

RFC 0001 P8: anything that crosses the wire is restricted to names, types,
decisions, and sha256 freshness hashes — never calibrated values, signals,
evidence, or prompts. These builders are the ONLY place the SDK shapes
governance for the backend; keep them allowlist-style so nothing new leaks
by accident. Contract source of truth: TraigentSchema
``promotion_policy_schema.json`` + ``tvl_governance_schema.json`` (v4.5.0).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: The promotion-policy keys the wire contract knows (mirrors the backend's
#: top-level-closed normalizer — anything else would be rejected there).
_POLICY_WIRE_KEYS = (
    "dominance",
    "alpha",
    "min_effect",
    "adjust",
    "chance_constraints",
    "tie_breakers",
    "require_calibration",
)


def promotion_policy_to_wire(policy: Any) -> dict[str, Any] | None:
    """Serialize a promotion policy (PromotionPolicy dataclass or raw dict)
    into the closed wire shape. Returns None when there is nothing to send.

    Only contract-known keys are emitted; ``require_calibration`` keeps the
    strict {enabled, hash_covered_context} shape. Values must already be
    JSON-safe — this is a projection, not a validator (the backend's
    normalizer is the enforcement point).
    """
    if policy is None:
        return None

    if isinstance(policy, dict):
        source: dict[str, Any] = policy
    else:
        source = {}
        for key in _POLICY_WIRE_KEYS:
            value = getattr(policy, key, None)
            if value is None:
                continue
            source[key] = value

    wire: dict[str, Any] = {}
    for key in _POLICY_WIRE_KEYS:
        if key not in source or source[key] is None:
            continue
        value = source[key]
        if key == "require_calibration":
            enabled = bool(
                getattr(value, "enabled", None)
                or (isinstance(value, dict) and value.get("enabled"))
            )
            covered = getattr(value, "hash_covered_context", None)
            if covered is None and isinstance(value, dict):
                covered = value.get("hash_covered_context")
            entry: dict[str, Any] = {"enabled": enabled}
            if covered:
                entry["hash_covered_context"] = [str(item) for item in covered]
            wire[key] = entry
        elif key == "chance_constraints":
            serialized = [_chance_constraint_to_wire(item) for item in value]
            serialized = [item for item in serialized if item]
            if serialized:
                wire[key] = serialized
        elif key == "tie_breakers":
            if isinstance(value, dict) and value:
                wire[key] = {
                    str(metric): {
                        "strategy": str(
                            getattr(rule, "strategy", None)
                            or (
                                rule.get("strategy") if isinstance(rule, dict) else rule
                            )
                        )
                    }
                    for metric, rule in value.items()
                }
        elif key == "min_effect":
            if isinstance(value, dict) and value:
                wire[key] = {str(metric): float(eps) for metric, eps in value.items()}
        else:
            wire[key] = value

    return wire or None


def _chance_constraint_to_wire(constraint: Any) -> dict[str, Any] | None:
    """Project a chance constraint to its wire fields (names + numbers only)."""
    if isinstance(constraint, dict):
        getter = constraint.get
    else:

        def getter(name: str, default: Any = None) -> Any:  # noqa: E306
            return getattr(constraint, name, default)

    entry: dict[str, Any] = {}
    for field_name in ("metric", "objective", "threshold", "confidence", "direction"):
        value = getter(field_name)
        if value is not None:
            entry[field_name] = value
    return entry or None


def build_tvl_governance(config_space: Any) -> dict[str, Any] | None:
    """Build the content-free governance summary from DECLARED bindings.

    Emits ONLY: cvar name, declared TVL value type, governed flag — straight
    from the ConfigSpace knob declarations (never runtime/calibrated state).
    Returns None when the space declares no calibrated variables, so
    ungoverned sessions stay byte-identical on the wire.
    """
    knobs = getattr(config_space, "knobs", None)
    if not knobs:
        return None

    try:
        from traigent.knobs.bindings import Calibrated, is_governed
    except ImportError:  # pragma: no cover - knobs is a hard dep of ConfigSpace
        logger.warning("knobs bindings unavailable; omitting tvl_governance")
        return None

    cvars: list[dict[str, Any]] = []
    for name, knob in dict(knobs).items():
        binding = getattr(knob, "binding", None)
        if not isinstance(binding, Calibrated):
            continue
        cvars.append(
            {
                "name": str(getattr(knob, "name", name)),
                "type": str(getattr(binding, "value_type", "float")),
                "governed": bool(is_governed(binding)),
            }
        )

    if not cvars:
        return None
    return {"cvars": cvars}


def tvl_governance_to_wire(governance: Any) -> dict[str, Any] | None:
    """Project a governance summary onto the closed wire shape (allowlist).

    Review round 3 (codex): the summary rode the create payload VERBATIM, so
    a direct SessionCreationRequest caller could put value/evidence blobs on
    the wire — and the leak happens at TRANSMISSION, regardless of what the
    backend's normalizer later rejects. Only the contract-known fields
    survive: cvars {name, type, governed}, certificates {cvar_name,
    decision, freshness_hash}, policies {name, strategy}. A projection, not
    a validator — the backend's allowlist rebuild is the enforcement point.
    """
    if not isinstance(governance, dict) or not governance:
        return None

    wire: dict[str, Any] = {}

    cvars = governance.get("cvars")
    if isinstance(cvars, list):
        projected = [
            {key: entry[key] for key in ("name", "type", "governed") if key in entry}
            for entry in cvars
            if isinstance(entry, dict)
        ]
        projected = [entry for entry in projected if entry.get("name")]
        if projected:
            wire["cvars"] = projected

    certificates = governance.get("certificates")
    if isinstance(certificates, list):
        projected = [
            {
                key: entry[key]
                for key in ("cvar_name", "decision", "freshness_hash")
                if key in entry
            }
            for entry in certificates
            if isinstance(entry, dict)
        ]
        projected = [entry for entry in projected if entry.get("cvar_name")]
        if projected:
            wire["certificates"] = projected

    policies = governance.get("policies")
    if isinstance(policies, list):
        projected = [
            {key: entry[key] for key in ("name", "strategy") if key in entry}
            for entry in policies
            if isinstance(entry, dict)
        ]
        projected = [entry for entry in projected if entry.get("name")]
        if projected:
            wire["policies"] = projected

    return wire or None


def build_certified_selection(
    trial_id: Any, certificates_by_cvar: Any
) -> dict[str, Any] | None:
    """Build the client-attested certified-selection finalize report.

    Mirrors TraigentSchema ``certified_selection_schema.json`` AND the
    server's coverage rule CLIENT-side: a report is built only when EVERY
    entry is a CERTIFIED decision with an issued (freshness) hash — any gap
    means NO report (the honest no-winner), never a partial one (the server
    would 400 it; partial reports are bugs, not flows).

    Content-free (RFC 0001 P8): cvar names, the decision enum value, and the
    certificate's issued hash ONLY — never subject_value_hash (dictionary-
    invertible on low-entropy calibrated values), evidence counts, pool
    hashes, or target details.
    """
    if not trial_id or not certificates_by_cvar:
        return None

    entries: list[dict[str, Any]] = []
    for name, certificate in dict(certificates_by_cvar).items():
        decision = getattr(certificate, "decision", None)
        decision_value = getattr(decision, "value", None) or (
            str(decision) if decision is not None else ""
        )
        issued_hash = getattr(certificate, "issued_hash", None)
        if decision_value != "CERTIFIED_SELECTION" or not issued_hash:
            logger.debug(
                "certified_selection withheld: cvar %s decision=%s issued=%s",
                name,
                decision_value or "<none>",
                bool(issued_hash),
            )
            return None
        entries.append(
            {
                "cvar_name": str(name),
                "decision": "CERTIFIED_SELECTION",
                "freshness_hash": str(issued_hash),
            }
        )

    return {
        "trial_id": str(trial_id),
        "certificates": entries,
        "attestation": "sdk_client_attested",
    }
