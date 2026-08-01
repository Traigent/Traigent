"""Concrete construction-evidence admission for cold-start datasets.

This module is intentionally not an extension point.  Generators may propose
inputs and oracles may produce ground truth, but neither can bypass this
admission check or alter the persisted provenance shape.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from traigent.generation.validators import looks_like_injection

from .contracts import (
    GroundTruth,
    GroundTruthSource,
    ScenarioCandidate,
    ScoringContract,
)
from .writer import _has_expected_output, canonical_json_bytes, sha256_bytes

MAX_INPUT_BYTES = 65_536


@dataclass(frozen=True, slots=True)
class EvidenceAdmission:
    """The non-overridable outcome for one proposed scenario."""

    admitted: bool
    input_digest: str
    quarantine_reason: str | None = None


@dataclass(frozen=True, slots=True)
class _TuningRowParts:
    """Typed row fields after the structural tuning-row checks pass."""

    inputs: Mapping[str, Any]
    expected_output: Any
    example_id: str
    provenance: Mapping[str, Any]


def _contains_injection_marker(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            _contains_injection_marker(item) for pair in value.items() for item in pair
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_injection_marker(item) for item in value)
    return isinstance(value, str) and looks_like_injection(value)


def _contains_non_string_mapping_key(value: Any) -> bool:
    """Reject mappings JSON would silently coerce into different input shapes."""
    if isinstance(value, Mapping):
        return any(
            not isinstance(key, str) or _contains_non_string_mapping_key(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_non_string_mapping_key(item) for item in value)
    return False


def input_digest(inputs: Mapping[str, Any]) -> str:
    """Digest an input payload without exposing it in audit or manifest records."""
    return cast(str, sha256_bytes(canonical_json_bytes(dict(inputs))))


def _screen_candidate_inputs(
    candidate: ScenarioCandidate,
    *,
    seen_input_digests: set[str],
    max_input_bytes: int,
) -> tuple[Mapping[str, Any] | None, EvidenceAdmission | None]:
    """Return screened inputs or the precise admission rejection."""
    if not isinstance(candidate, ScenarioCandidate):
        return None, EvidenceAdmission(False, "", "invalid_candidate")
    inputs = candidate.inputs
    if not isinstance(inputs, Mapping) or not inputs:
        return None, EvidenceAdmission(False, "", "missing_or_empty_input")
    if _contains_non_string_mapping_key(inputs):
        return None, EvidenceAdmission(False, "", "input_not_json_serializable")

    try:
        digest = input_digest(inputs)
        serialized_size = len(canonical_json_bytes(dict(inputs)))
    except ValueError:
        return None, EvidenceAdmission(False, "", "input_not_json_serializable")

    if serialized_size > max_input_bytes:
        return None, EvidenceAdmission(False, digest, "input_too_large")
    if _contains_injection_marker(inputs):
        return None, EvidenceAdmission(False, digest, "input_injection_marker")
    if digest in seen_input_digests:
        return None, EvidenceAdmission(False, digest, "duplicate_input")
    return inputs, EvidenceAdmission(True, digest)


def _ground_truth_rejection(candidate: ScenarioCandidate) -> str | None:
    """Return why a candidate lacks independently grounded exact-match gold."""
    ground_truth = candidate.ground_truth
    if ground_truth is None:
        return "missing_ground_truth"
    if not isinstance(ground_truth, GroundTruth):
        return "invalid_ground_truth"
    if ground_truth.source is not GroundTruthSource.ORACLE_COMPUTED:
        return "ineligible_ground_truth_source"
    if not _has_expected_output(ground_truth.expected_output):
        return "missing_expected_output"
    if isinstance(ground_truth.expected_output, Mapping):
        return "unsupported_expected_output"
    if ground_truth.scoring_contract is not ScoringContract.EXACT_MATCH:
        return "unsupported_scoring_contract"
    try:
        canonical_json_bytes(ground_truth.expected_output)
    except ValueError:
        return "expected_output_not_json_serializable"
    return None


def admit_candidate(
    candidate: ScenarioCandidate,
    *,
    seen_input_digests: set[str],
    max_input_bytes: int,
) -> EvidenceAdmission:
    """Admit only independent gold that has passed concrete input screening.

    The result has no override. Only a real ``GroundTruth`` from the injected
    oracle with the one supported scoring contract can become a tuning row.
    """
    inputs, admission = _screen_candidate_inputs(
        candidate,
        seen_input_digests=seen_input_digests,
        max_input_bytes=max_input_bytes,
    )
    if inputs is None or admission is None:
        assert admission is not None
        return admission

    rejection = _ground_truth_rejection(candidate)
    if rejection is not None:
        return EvidenceAdmission(False, admission.input_digest, rejection)
    seen_input_digests.add(admission.input_digest)
    return admission


def build_tuning_row(
    *,
    candidate: ScenarioCandidate,
    schema_version: str,
    oracle_id: str,
    generator_id: str,
    seed: int,
    system_fingerprint: str,
) -> dict[str, Any]:
    """Create the sole Dataset-compatible persistence shape for an admitted row."""
    if not isinstance(candidate, ScenarioCandidate):
        raise ValueError("Cannot persist a non-ScenarioCandidate value.")
    inputs = candidate.inputs
    ground_truth = candidate.ground_truth
    if not isinstance(inputs, Mapping) or not isinstance(ground_truth, GroundTruth):
        raise ValueError("Cannot persist a scenario that lacks admissible evidence.")
    if _contains_non_string_mapping_key(inputs):
        raise ValueError("Cannot persist an input mapping with non-string keys.")
    if (
        ground_truth.source is not GroundTruthSource.ORACLE_COMPUTED
        or ground_truth.scoring_contract is not ScoringContract.EXACT_MATCH
        or not _has_expected_output(ground_truth.expected_output)
        or isinstance(ground_truth.expected_output, Mapping)
    ):
        raise ValueError("Cannot persist a scenario that lacks admissible evidence.")
    try:
        persisted_input = cast(
            dict[str, Any], json.loads(canonical_json_bytes(dict(inputs)))
        )
    except ValueError as exc:
        raise ValueError("Cannot persist a non-JSON input mapping.") from exc
    try:
        expected_output = json.loads(canonical_json_bytes(ground_truth.expected_output))
    except ValueError as exc:
        raise ValueError("Cannot persist a non-JSON expected output.") from exc

    provenance: dict[str, Any] = {
        "schema_version": schema_version,
        "ground_truth_source": ground_truth.source.value,
        "scoring_contract": ground_truth.scoring_contract.value,
        "oracle_id": oracle_id,
        "generator_id": generator_id,
        "seed": seed,
        "system_fingerprint": system_fingerprint,
        "split": "tune",
    }
    unsigned_row: dict[str, Any] = {
        "input": persisted_input,
        "expected_output": expected_output,
        "example_id": f"coldstart_{input_digest(persisted_input)[:24]}",
        "traigent_coldstart": provenance,
    }
    provenance["row_digest"] = sha256_bytes(canonical_json_bytes(unsigned_row))
    return unsigned_row


def _row_payload_parts(
    row: Mapping[str, Any],
) -> tuple[_TuningRowParts | None, str | None]:
    """Extract required row fields before validating their provenance."""
    inputs = row.get("input")
    if not isinstance(inputs, Mapping) or not inputs:
        return None, "missing_or_empty_input"
    expected_output = row.get("expected_output")
    if not _has_expected_output(expected_output):
        return None, "missing_expected_output"
    if isinstance(expected_output, Mapping):
        return None, "unsupported_expected_output"
    example_id = row.get("example_id")
    if not isinstance(example_id, str) or not example_id:
        return None, "missing_example_id"
    provenance = row.get("traigent_coldstart")
    if not isinstance(provenance, Mapping):
        return None, "missing_coldstart_provenance"
    return _TuningRowParts(inputs, expected_output, example_id, provenance), None


def _provenance_rejection(
    provenance: Mapping[str, Any], *, expected_schema_version: str
) -> str | None:
    """Return the first fixed cold-start provenance invariant that fails."""
    if provenance.get("schema_version") != expected_schema_version:
        return "schema_version_mismatch"
    if provenance.get("split") != "tune":
        return "non_tune_split"
    if provenance.get("ground_truth_source") != GroundTruthSource.ORACLE_COMPUTED.value:
        return "ineligible_ground_truth_source"
    if provenance.get("scoring_contract") != ScoringContract.EXACT_MATCH.value:
        return "unsupported_scoring_contract"
    if not all(
        isinstance(provenance.get(field), str) and provenance[field]
        for field in ("oracle_id", "generator_id", "system_fingerprint", "row_digest")
    ):
        return "missing_required_provenance"
    if not isinstance(provenance.get("seed"), int):
        return "invalid_seed"
    return None


def _serialized_input_rejection(
    parts: _TuningRowParts, *, max_input_bytes: int
) -> tuple[str | None, str | None]:
    """Return an input digest or the serialization and policy rejection."""
    try:
        digest = input_digest(parts.inputs)
        if len(canonical_json_bytes(dict(parts.inputs))) > max_input_bytes:
            return None, "input_too_large"
        if _contains_injection_marker(parts.inputs):
            return None, "input_injection_marker"
        canonical_json_bytes(parts.expected_output)
    except ValueError:
        return None, "row_not_json_serializable"
    return digest, None


def _has_valid_row_digest(
    row: Mapping[str, Any], provenance: Mapping[str, Any]
) -> bool:
    """Recompute the row digest without its self-referential digest field."""
    without_digest = dict(row)
    unsigned_provenance = dict(provenance)
    supplied_digest = unsigned_provenance.pop("row_digest")
    without_digest["traigent_coldstart"] = unsigned_provenance
    return bool(supplied_digest == sha256_bytes(canonical_json_bytes(without_digest)))


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
    parts, reason = _row_payload_parts(row)
    if reason is not None:
        return reason
    assert parts is not None
    reason = _provenance_rejection(
        parts.provenance, expected_schema_version=expected_schema_version
    )
    if reason is not None:
        return reason
    digest, reason = _serialized_input_rejection(parts, max_input_bytes=max_input_bytes)
    if reason is not None:
        return reason
    assert digest is not None
    if digest in seen_input_digests:
        return "duplicate_input"
    if not _has_valid_row_digest(row, parts.provenance):
        return "row_digest_mismatch"
    if parts.example_id != f"coldstart_{digest[:24]}":
        return "example_id_mismatch"

    seen_input_digests.add(digest)
    return None


def build_audit_row(
    *,
    candidate: ScenarioCandidate,
    admission: EvidenceAdmission,
    schema_version: str,
) -> dict[str, Any]:
    """Build a non-Dataset audit record without copying user input or gold."""
    if not isinstance(candidate, ScenarioCandidate):
        raise ValueError("Cannot audit a non-ScenarioCandidate value.")
    digest = admission.input_digest
    if not digest:
        try:
            digest = input_digest(candidate.inputs) if candidate.inputs else ""
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
    "input_digest",
    "validate_tuning_row",
]
