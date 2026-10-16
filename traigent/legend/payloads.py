"""Closed request bodies (``legend_requests_v0.json``). Only enums, digests and opaque refs.

Idempotency (design note section 3.4): ``idempotency_key`` is a FRESH random key per
logical request and is reused unchanged on every retry of it; it is never derived from
content. ``request_digest`` is ``sha256:`` + hex SHA-256(domain || 0x00 || jcs_v1(body))
over the body excluding ``idempotency_key`` and ``request_digest``.
"""

from __future__ import annotations

import hashlib
import secrets
from typing import Any

from traigent.identity.content_identity import canonical_bytes
from traigent.legend import _contract as c
from traigent.legend.errors import LegendError
from traigent.legend.pillars import Pillar

_EXCLUDED_FROM_DIGEST = ("idempotency_key", "request_digest")


def new_idempotency_key() -> str:
    """A fresh opaque ``idk_<32 hex>`` key (128 random bits)."""
    return "idk_" + secrets.token_hex(16)


def request_digest(body: dict[str, Any]) -> str:
    """The retry-stable digest of a request body (key and digest fields excluded)."""
    clean = {k: v for k, v in body.items() if k not in _EXCLUDED_FROM_DIGEST}
    framed = c.REQUEST_DOMAIN.encode("utf-8") + b"\x00" + canonical_bytes(clean)
    return "sha256:" + hashlib.sha256(framed).hexdigest()


def seal(body: dict[str, Any], idempotency_key: str | None = None) -> dict[str, Any]:
    """Return ``body`` with ``idempotency_key`` (fresh unless supplied) and ``request_digest``."""
    key = idempotency_key if idempotency_key is not None else new_idempotency_key()
    if not c.IDEMPOTENCY_KEY_RE.fullmatch(key):
        raise LegendError("invalid_request", "idempotency_key must be idk_<32 hex>")
    sealed = {k: v for k, v in body.items() if k not in _EXCLUDED_FROM_DIGEST}
    sealed["idempotency_key"] = key
    sealed["request_digest"] = request_digest(sealed)
    return sealed


def build_declaration_payload(
    pillar: Pillar,
    *,
    entity_ref: str | None,
    previous_version_ref: str | None,
    change_reason: str | None,
    initial_parent_version_ref: str | None = None,
    idempotency_key: str | None = None,
    provenance: str = c.PROVENANCE_EXPLICIT,
) -> dict[str, Any]:
    """``DeclareRequestV0``.

    ``provenance`` defaults to ``explicit`` (a caller declaring a version on purpose); the run
    binder passes ``implicit_run`` for what a run merely observed.

    ``initial_parent_version_ref`` is honoured by the server only on the version's FIRST
    declaration (send ``None`` otherwise); ``previous_version_ref`` (where the declarer
    moved from) is independent of it and may be set on every declaration.
    """
    for value, what, pattern in (
        (entity_ref, "entity", c.ENTITY_REF_RE),
        (previous_version_ref, "previous_version", c.VERSION_REF_RE),
        (initial_parent_version_ref, "initial_parent", c.VERSION_REF_RE),
    ):
        if value is not None and not pattern.fullmatch(value):
            raise LegendError("invalid_request", f"{what} must be an opaque ref")
    if change_reason is not None and change_reason not in c.CHANGE_REASONS:
        raise LegendError("invalid_request", "change_reason is not in the closed list")
    if provenance not in c.CLIENT_PROVENANCES:
        raise LegendError("invalid_request", "provenance is not client-sendable")
    body: dict[str, Any] = {
        "schema_version": c.SCHEMA_VERSION,
        "entity_ref": entity_ref,
        **pillar.wire_identity(),
        "initial_parent_version_ref": initial_parent_version_ref,
        "identity_source": pillar.identity_source,
        "provenance": provenance,
    }
    if previous_version_ref is not None:
        body["previous_version_ref"] = previous_version_ref
    if change_reason is not None:
        body["change_reason"] = change_reason
    return seal(body, idempotency_key)


def build_query_payload(
    entity_ref: str,
    page_size: int,
    page_cursor: str | None,
    kind: str | None = None,
    as_of_ledger_position: int | None = None,
) -> dict[str, Any]:
    """``QueryRequestV0`` (no ``instruction_ref``: that exists only on the Director path)."""
    if not c.ENTITY_REF_RE.fullmatch(entity_ref or ""):
        raise LegendError("invalid_request", "entity must be an opaque entity ref")
    if type(page_size) is not int or not 1 <= page_size <= c.QUERY_MAX_PAGE_SIZE:
        raise LegendError("invalid_request", "page_size out of range")
    if page_cursor is not None and not c.CURSOR_RE.fullmatch(page_cursor):
        raise LegendError("invalid_request", "page_cursor must be an opaque cursor")
    if kind is not None and kind not in c.ENTITY_KINDS:
        raise LegendError("invalid_request", "unknown pillar kind")
    body: dict[str, Any] = {
        "schema_version": c.SCHEMA_VERSION,
        "entity_ref": entity_ref,
        "kind": kind,
        "page_size": page_size,
        "page_cursor": page_cursor,
    }
    if as_of_ledger_position is not None:
        # bool is an int subclass: refuse it explicitly. Cursors stay opaque.
        if type(as_of_ledger_position) is not int or as_of_ledger_position < 0:
            raise LegendError(
                "invalid_request", "as_of_ledger_position must be an integer >= 0"
            )
        body["as_of_ledger_position"] = as_of_ledger_position
    return body


def _closed_coverage(kind: str, coverage: dict[str, Any]) -> dict[str, Any]:
    # imported here: manifest imports pillars/payload-free helpers, keep module graph acyclic
    from traigent.legend.manifest import validate_coverage

    if set(coverage) != {"status", "covered", "missing"}:
        raise LegendError("invalid_request", "coverage has unknown or missing members")
    clean = {
        "status": coverage["status"],
        "covered": list(coverage["covered"]),
        "missing": list(coverage["missing"]),
    }
    validate_coverage(kind, clean)
    return clean


def build_capture_payload(
    *,
    version_ref: str,
    kind: str,
    manifest_fingerprint: str,
    coverage: dict[str, Any],
    recovery_state: str,
    conformance_passed: bool,
    exporter_version: str,
    backend_keyed: bool,
    identity_verification_ref: str | None,
    representation_commitment: str,
    identity_digests_reproduced: bool,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """``RecordCaptureRequestV0`` (sealed). Enums, digests and opaque refs only.

    Coverage is validated against ``RequiredComponents(kind)`` before anything is
    built; complete backend-keyed coverage must cite the ``identity_verified`` proof,
    and complete coverage needs ``identity_digests_reproduced``.
    """
    if not c.VERSION_REF_RE.fullmatch(version_ref or ""):
        raise LegendError(
            "invalid_request", "version_ref must be an opaque version ref"
        )
    if not c.MANIFEST_FINGERPRINT_RE.fullmatch(manifest_fingerprint or ""):
        raise LegendError("invalid_request", "manifest fingerprint malformed")
    if recovery_state not in (
        c.RECOVERY_STORED_UNVERIFIED,
        c.RECOVERY_ROUND_TRIP_VERIFIED,
    ):
        raise LegendError("invalid_request", "unknown recovery state")
    if recovery_state == c.RECOVERY_ROUND_TRIP_VERIFIED and not conformance_passed:
        raise LegendError("conformance_failed", "round trip needs conformance")
    if not c.SEMVER_RE.fullmatch(exporter_version or "") or len(exporter_version) > 32:
        raise LegendError("invalid_request", "exporter_version must be semver")
    if not c.SHA256_DIGEST_RE.fullmatch(representation_commitment or ""):
        raise LegendError("invalid_request", "representation_commitment malformed")
    if identity_verification_ref is not None and not c.EVENT_REF_RE.fullmatch(
        identity_verification_ref
    ):
        raise LegendError("invalid_request", "identity_verification_ref malformed")
    if kind in c.BACKEND_KEYED_KINDS and not backend_keyed:
        raise LegendError("invalid_request", "this kind is backend-keyed")
    clean = _closed_coverage(kind, coverage)
    if clean["status"] == "complete":
        if not identity_digests_reproduced:
            raise LegendError(
                "coverage_insufficient", "complete needs reproduced digests"
            )
        if backend_keyed and identity_verification_ref is None:
            raise LegendError(
                "coverage_insufficient", "complete backend-keyed needs a proof"
            )
    return seal(
        {
            "schema_version": c.SCHEMA_VERSION,
            "version_ref": version_ref,
            "recovery_manifest_fingerprint": manifest_fingerprint,
            "coverage": clean,
            "recovery_state": recovery_state,
            "conformance_passed": bool(conformance_passed),
            "exporter_version": exporter_version,
            "backend_keyed": bool(backend_keyed),
            "identity_verification_ref": identity_verification_ref,
            "representation_commitment": representation_commitment,
            "identity_digests_reproduced": bool(identity_digests_reproduced),
        },
        idempotency_key,
    )


def build_restore_payload(
    *,
    version_ref: str,
    kind: str,
    manifest_fingerprint: str,
    outcome: str,
    coverage: dict[str, Any],
    isolated_destination: bool,
    representation_digests_match: bool,
    pillar_fingerprint_match: bool,
    historical_key_available: bool,
    dependencies_verified: bool,
    verified_artifact_digest: str | None,
    backend_keyed: bool,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """``RecordRestoreRequestV0`` (sealed). A backend-keyed pillar is never ``exact`` here."""
    if not c.VERSION_REF_RE.fullmatch(version_ref or ""):
        raise LegendError(
            "invalid_request", "version_ref must be an opaque version ref"
        )
    if kind not in c.ENTITY_KINDS or not c.MANIFEST_FINGERPRINT_RE.fullmatch(
        manifest_fingerprint or ""
    ):
        raise LegendError("invalid_request", "kind or manifest fingerprint malformed")
    if outcome not in (
        c.GRADE_EXACT,
        c.GRADE_PARTIAL,
        c.GRADE_SUBSTITUTED,
        c.GRADE_FAILED,
    ):
        raise LegendError("invalid_request", "unknown restore outcome")
    if isolated_destination is not True:
        raise LegendError(
            "invalid_request", "restores are always into an isolated destination"
        )
    if verified_artifact_digest is not None and not c.SHA256_DIGEST_RE.fullmatch(
        verified_artifact_digest
    ):
        raise LegendError("invalid_request", "verified_artifact_digest malformed")
    if kind in c.BACKEND_KEYED_KINDS and not backend_keyed:
        raise LegendError("invalid_request", "this kind is backend-keyed")
    if outcome == c.GRADE_EXACT:
        if backend_keyed:
            raise LegendError(
                "coverage_insufficient", "backend-keyed is never exact client-side"
            )
        if not (
            isolated_destination
            and representation_digests_match
            and pillar_fingerprint_match
            and historical_key_available
            and dependencies_verified
            and verified_artifact_digest is not None
        ):
            raise LegendError(
                "invalid_request", "exact needs every restore measurement"
            )
    clean = _closed_coverage(kind, coverage)
    if outcome == c.GRADE_EXACT and clean["status"] != "complete":
        raise LegendError("coverage_insufficient", "exact needs complete coverage")
    return seal(
        {
            "schema_version": c.SCHEMA_VERSION,
            "version_ref": version_ref,
            "kind": kind,
            "recovery_manifest_fingerprint": manifest_fingerprint,
            "outcome": outcome,
            "coverage": clean,
            "isolated_destination": bool(isolated_destination),
            "representation_digests_match": bool(representation_digests_match),
            "pillar_fingerprint_match": bool(pillar_fingerprint_match),
            "historical_key_available": bool(historical_key_available),
            "dependencies_verified": bool(dependencies_verified),
            "verified_artifact_digest": verified_artifact_digest,
            "backend_keyed": bool(backend_keyed),
        },
        idempotency_key,
    )


def build_availability_payload(
    version_ref: str,
    manifest_fingerprint: str,
    status: str,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """``RecordAvailabilityRequestV0`` (sealed): version ref, manifest fingerprint, closed status."""
    if not c.VERSION_REF_RE.fullmatch(version_ref or ""):
        raise LegendError(
            "invalid_request", "version_ref must be an opaque version ref"
        )
    if not c.MANIFEST_FINGERPRINT_RE.fullmatch(manifest_fingerprint or ""):
        raise LegendError("invalid_request", "manifest fingerprint malformed")
    if status not in c.AVAILABILITY_STATUSES:
        raise LegendError("invalid_request", "status is not in the closed list")
    return seal(
        {
            "schema_version": c.SCHEMA_VERSION,
            "version_ref": version_ref,
            "recovery_manifest_fingerprint": manifest_fingerprint,
            "status": status,
        },
        idempotency_key,
    )
