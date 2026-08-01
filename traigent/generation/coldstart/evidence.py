"""Concrete construction-evidence admission for cold-start datasets.

This module is intentionally not an extension point.  Generators may propose
inputs and oracles may produce ground truth, but neither can bypass this
admission check or alter the persisted provenance shape.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .writer import canonical_json_bytes, sha256_bytes


_ALLOWED_GROUND_TRUTH_SOURCES = frozenset({"spec_derived", "oracle_computed"})
_INJECTION_MARKERS = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "disregard previous instructions",
    "reveal the system prompt",
    "developer message",
    "<|system|>",
    "jailbreak",
)
MAX_INPUT_BYTES = 65_536


@dataclass(frozen=True, slots=True)
class EvidenceAdmission:
    """The non-overridable outcome for one proposed scenario."""

    admitted: bool
    input_digest: str
    quarantine_reason: str | None = None


def _value(value: Any) -> str:
    """Normalize a string-or-enum descriptor without accepting arbitrary objects."""
    raw = getattr(value, "value", value)
    return raw.strip().lower() if isinstance(raw, str) else ""


def _attribute_or_mapping(value: Any, *names: str) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return None


def candidate_inputs(candidate: Any) -> Mapping[str, Any] | None:
    """Extract a proposed input mapping from the shared candidate contract."""
    inputs = _attribute_or_mapping(candidate, "inputs", "input_data", "input")
    return inputs if isinstance(inputs, Mapping) else None


def candidate_ground_truth(candidate: Any) -> Any:
    """Extract a candidate's independently created ground truth, if any."""
    return _attribute_or_mapping(candidate, "ground_truth")


def _contains_injection_marker(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_injection_marker(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_injection_marker(item) for item in value)
    if not isinstance(value, str):
        return False
    normalized = value.casefold()
    return any(marker in normalized for marker in _INJECTION_MARKERS)


def input_digest(inputs: Mapping[str, Any]) -> str:
    """Digest an input payload without exposing it in audit or manifest records."""
    return sha256_bytes(canonical_json_bytes(dict(inputs)))


def _supported_scoring_contract(ground_truth: Any) -> str:
    contract = _attribute_or_mapping(
        ground_truth,
        "scoring_contract",
        "supported_scoring_contract",
    )
    normalized = _value(contract)
    if normalized in {"", "unknown", "unsupported", "none"}:
        return ""
    return normalized


def admit_candidate(
    candidate: Any,
    *,
    seen_input_digests: set[str],
    max_input_bytes: int,
) -> EvidenceAdmission:
    """Admit only independent gold that has passed concrete input screening.

    The result has no override.  In particular, a model-created label is never
    a substitute for absent oracle/spec evidence because its source is not one
    of the two accepted construction-evidence sources.
    """
    inputs = candidate_inputs(candidate)
    if not inputs:
        return EvidenceAdmission(False, "", "missing_or_empty_input")

    try:
        digest = input_digest(inputs)
        serialized_size = len(canonical_json_bytes(dict(inputs)))
    except ValueError:
        return EvidenceAdmission(False, "", "input_not_json_serializable")

    if serialized_size > max_input_bytes:
        return EvidenceAdmission(False, digest, "input_too_large")
    if _contains_injection_marker(inputs):
        return EvidenceAdmission(False, digest, "input_injection_marker")
    if digest in seen_input_digests:
        return EvidenceAdmission(False, digest, "duplicate_input")

    ground_truth = candidate_ground_truth(candidate)
    if ground_truth is None:
        return EvidenceAdmission(False, digest, "missing_ground_truth")
    source = _value(_attribute_or_mapping(ground_truth, "source"))
    if source not in _ALLOWED_GROUND_TRUTH_SOURCES:
        return EvidenceAdmission(False, digest, "ineligible_ground_truth_source")
    expected_output = _attribute_or_mapping(ground_truth, "expected_output", "output")
    if expected_output is None:
        return EvidenceAdmission(False, digest, "missing_expected_output")
    if not _supported_scoring_contract(ground_truth):
        return EvidenceAdmission(False, digest, "unsupported_scoring_contract")

    try:
        canonical_json_bytes(expected_output)
    except ValueError:
        return EvidenceAdmission(False, digest, "expected_output_not_json_serializable")

    seen_input_digests.add(digest)
    return EvidenceAdmission(True, digest)


def build_tuning_row(
    *,
    candidate: Any,
    schema_version: str,
    oracle_id: str,
    generator_id: str,
    seed: int,
    system_fingerprint: str,
) -> dict[str, Any]:
    """Create the sole Dataset-compatible persistence shape for an admitted row."""
    inputs = candidate_inputs(candidate)
    ground_truth = candidate_ground_truth(candidate)
    if inputs is None or ground_truth is None:
        raise ValueError("Cannot persist a scenario that lacks admissible evidence.")

    source = _value(_attribute_or_mapping(ground_truth, "source"))
    scoring_contract = _supported_scoring_contract(ground_truth)
    expected_output = _attribute_or_mapping(ground_truth, "expected_output", "output")
    if (
        source not in _ALLOWED_GROUND_TRUTH_SOURCES
        or not scoring_contract
        or expected_output is None
    ):
        raise ValueError("Cannot persist a scenario that lacks admissible evidence.")

    provenance: dict[str, Any] = {
        "schema_version": schema_version,
        "ground_truth_source": source,
        "scoring_contract": scoring_contract,
        "oracle_id": oracle_id,
        "generator_id": generator_id,
        "seed": seed,
        "system_fingerprint": system_fingerprint,
        "split": "tune",
    }
    unsigned_row: dict[str, Any] = {
        "input": dict(inputs),
        "expected_output": expected_output,
        "example_id": f"coldstart_{input_digest(inputs)[:24]}",
        "traigent_coldstart": provenance,
    }
    provenance["row_digest"] = sha256_bytes(canonical_json_bytes(unsigned_row))
    return unsigned_row


def validate_tuning_row(
    row: Mapping[str, Any],
    *,
    expected_schema_version: str,
    seen_input_digests: set[str],
    max_input_bytes: int,
) -> str | None:
    """Re-derive construction eligibility for the integrity check.

    This detects malformed or policy-ineligible rows.  It is deliberately not
    authentication: a party able to alter both a row and the manifest can still
    recompute hashes, and semantic oracle correctness remains external.
    """
    inputs = row.get("input")
    expected_output = row.get("expected_output")
    provenance = row.get("traigent_coldstart")
    if not isinstance(inputs, Mapping) or not inputs:
        return "missing_or_empty_input"
    if expected_output is None:
        return "missing_expected_output"
    if not isinstance(row.get("example_id"), str) or not row["example_id"]:
        return "missing_example_id"
    if not isinstance(provenance, Mapping):
        return "missing_coldstart_provenance"
    if provenance.get("schema_version") != expected_schema_version:
        return "schema_version_mismatch"
    if provenance.get("split") != "tune":
        return "non_tune_split"
    if (
        _value(provenance.get("ground_truth_source"))
        not in _ALLOWED_GROUND_TRUTH_SOURCES
    ):
        return "ineligible_ground_truth_source"
    if not _value(provenance.get("scoring_contract")):
        return "unsupported_scoring_contract"
    if not all(
        isinstance(provenance.get(field), str) and provenance[field]
        for field in ("oracle_id", "generator_id", "system_fingerprint", "row_digest")
    ):
        return "missing_required_provenance"
    if not isinstance(provenance.get("seed"), int):
        return "invalid_seed"

    try:
        digest = input_digest(inputs)
        if len(canonical_json_bytes(dict(inputs))) > max_input_bytes:
            return "input_too_large"
        if _contains_injection_marker(inputs):
            return "input_injection_marker"
        canonical_json_bytes(expected_output)
    except ValueError:
        return "row_not_json_serializable"
    if digest in seen_input_digests:
        return "duplicate_input"

    without_digest = dict(row)
    unsigned_provenance = dict(provenance)
    supplied_digest = unsigned_provenance.pop("row_digest")
    without_digest["traigent_coldstart"] = unsigned_provenance
    if supplied_digest != sha256_bytes(canonical_json_bytes(without_digest)):
        return "row_digest_mismatch"

    seen_input_digests.add(digest)
    return None


def build_audit_row(
    *,
    candidate: Any,
    admission: EvidenceAdmission,
    schema_version: str,
) -> dict[str, Any]:
    """Build a non-Dataset audit record without copying user input or gold."""
    digest = admission.input_digest
    if not digest:
        try:
            inputs = candidate_inputs(candidate)
            digest = input_digest(inputs) if inputs else ""
        except ValueError:
            digest = ""
    return {
        "artifact": "coldstart_audit",
        "schema_version": schema_version,
        "candidate_digest": digest,
        "state": "admitted" if admission.admitted else "quarantined",
        "quarantine_reason": admission.quarantine_reason,
    }


__all__ = [
    "EvidenceAdmission",
    "admit_candidate",
    "build_audit_row",
    "build_tuning_row",
    "candidate_ground_truth",
    "candidate_inputs",
    "input_digest",
    "validate_tuning_row",
]
