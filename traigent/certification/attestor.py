"""Client co-attestation and issuer-signature verification primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .manifest import ClientEvidenceBuild, compute_manifest_root
from .signers import (
    CertificationError,
    VerificationError,
    _sign_material,
    canonical_manifest,
    client_signed_material,
    decode_signature,
    issuer_signed_material,
    manifest_digest,
    verify_signature,
)

__all__ = [
    "ExpectedCertificateContext",
    "create_client_co_attestation",
    "verify_certificate_signatures",
    "verify_client_co_attestation",
]

_CO_KEYS = frozenset(
    {"algorithm", "client_key_ref", "signed_manifest_digest", "nonce", "signature"}
)
_ISSUER_KEYS = frozenset(
    {"algorithm", "issuer_key_ref", "trust_ring_ref", "signed_payload", "signature"}
)
_KEY_RING_KEYS = frozenset(
    {
        "issuer_key_ref",
        "trust_ring_ref",
        "issuer_signature_algorithm",
        "client_key_ref",
        "client_signature_algorithm",
    }
)
_CLAIM_IDS = frozenset({"D2", "G1"})


@dataclass(frozen=True, slots=True)
class ExpectedCertificateContext:
    """Mandatory relying-party context that prevents cross-session replay."""

    expected_nonce: str
    expected_build_session_ref: str
    expected_session_commitment_digest: str
    expected_client_key_ref: str | None
    expected_issuer_key_ref: str
    expected_trust_ring_ref: str


def _raise(
    code: str, error_type: type[CertificationError] = CertificationError
) -> None:
    raise error_type(code)


def _require_exact_dict(
    value: object,
    code: str,
    error_type: type[CertificationError] = CertificationError,
) -> dict[str, Any]:
    if type(value) is not dict:
        _raise(code, error_type)
    return value


def _validate_key_ring(
    manifest: dict[str, Any],
    error_type: type[CertificationError] = CertificationError,
) -> dict[str, Any]:
    key_ring = manifest.get("key_ring_identifiers")
    if type(key_ring) is not dict:
        _raise("KEY_RING", error_type)
    if any(type(key) is not str for key in key_ring) or not set(key_ring).issubset(
        _KEY_RING_KEYS
    ):
        _raise("KEY_RING", error_type)
    required = {
        "issuer_key_ref",
        "trust_ring_ref",
        "issuer_signature_algorithm",
    }
    if not required.issubset(key_ring):
        _raise("KEY_RING", error_type)
    for field in required:
        if type(key_ring[field]) is not str:
            _raise("KEY_RING", error_type)
    has_client_ref = "client_key_ref" in key_ring
    has_client_algorithm = "client_signature_algorithm" in key_ring
    if has_client_ref != has_client_algorithm:
        _raise("KEY_RING", error_type)
    if has_client_ref and (
        type(key_ring["client_key_ref"]) is not str
        or type(key_ring["client_signature_algorithm"]) is not str
    ):
        _raise("KEY_RING", error_type)
    if type(key_ring["issuer_signature_algorithm"]) is not str or key_ring[
        "issuer_signature_algorithm"
    ] not in {
        "ed25519",
        "ecdsa_p256_sha256",
    }:
        _raise("KEY_RING", error_type)
    if has_client_ref and key_ring["client_signature_algorithm"] not in {
        "ed25519",
        "ecdsa_p256_sha256",
    }:
        _raise("KEY_RING", error_type)
    return key_ring


def _validate_freshness(manifest: dict[str, Any], expected_nonce: str) -> None:
    freshness = manifest.get("freshness")
    if type(freshness) is not dict or set(freshness) != {"nonce"}:
        _raise("FRESHNESS")
    if type(expected_nonce) is not str or type(freshness["nonce"]) is not str:
        _raise("NONCE")
    if freshness["nonce"] != expected_nonce:
        _raise("NONCE")


def _validate_claims(
    manifest: dict[str, Any],
    error_type: type[CertificationError] = CertificationError,
) -> tuple[bool, str | None]:
    claims = manifest.get("claims")
    if type(claims) is not list:
        _raise("CLAIMS", error_type)
    has_claims = bool(claims)
    g1_root: str | None = None
    for claim in claims:
        if type(claim) is not dict:
            _raise("CLAIMS", error_type)
        claim_id = claim.get("claim_id")
        if (
            type(claim_id) is not str
            or claim_id not in _CLAIM_IDS
            or claim.get("tier") != 1
        ):
            _raise("CLAIMS", error_type)
        if claim_id == "G1":
            payload = claim.get("payload")
            params = payload.get("params") if type(payload) is dict else None
            if (
                type(params) is not dict
                or type(params.get("manifest_root_digest")) is not str
            ):
                _raise("CLAIMS", error_type)
            if g1_root is not None:
                _raise("CLAIMS", error_type)
            g1_root = params["manifest_root_digest"]
    return has_claims, g1_root


def _canonical_for_attestation(
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], bytes]:
    try:
        copied, manifest_bytes = canonical_manifest(manifest)
    except CertificationError as exc:
        raise VerificationError(exc.code) from None
    for stream in copied["seal"]["expected_stream_projection"].values():
        if stream["chain_status"] not in {"sealed", "empty_sealed"}:
            _raise("STREAM_STATE", VerificationError)
    return copied, manifest_bytes


def _validate_context(
    manifest: dict[str, Any], context: ExpectedCertificateContext
) -> None:
    if type(context) is not ExpectedCertificateContext:
        _raise("CONTEXT", VerificationError)
    subject = manifest["subject"]
    freshness = manifest["freshness"]
    key_ring = manifest["key_ring_identifiers"]
    if (
        type(context.expected_nonce) is not str
        or freshness["nonce"] != context.expected_nonce
        or subject["build_session_ref"] != context.expected_build_session_ref
        or subject["session_commitment_digest"]
        != context.expected_session_commitment_digest
        or (
            context.expected_client_key_ref is not None
            and key_ring.get("client_key_ref") != context.expected_client_key_ref
        )
        or key_ring["issuer_key_ref"] != context.expected_issuer_key_ref
        or key_ring["trust_ring_ref"] != context.expected_trust_ring_ref
    ):
        _raise("CONTEXT", VerificationError)


def _check_g1_root(manifest: dict[str, Any], local_build: ClientEvidenceBuild) -> None:
    if type(local_build) is not ClientEvidenceBuild:
        _raise("BUILD", VerificationError)
    for claim in manifest["claims"]:
        if claim["claim_id"] == "G1":
            expected_root = claim["payload"]["params"]["manifest_root_digest"]
            if expected_root != compute_manifest_root(local_build.manifest):
                _raise("ROOT", VerificationError)


def create_client_co_attestation(
    manifest: dict[str, Any],
    expected_nonce: str,
    client_key_ref: str,
    client_private_key: object,
    algorithm: str,
    local_build: ClientEvidenceBuild,
) -> dict[str, str]:
    """Create the exact Schema co-attestation block for a final manifest."""

    if type(local_build) is not ClientEvidenceBuild:
        _raise("BUILD")
    copied, manifest_bytes = _canonical_for_attestation(manifest)
    key_ring = _validate_key_ring(copied)
    _validate_freshness(copied, expected_nonce)
    has_claims, g1_root = _validate_claims(copied)
    if (
        type(client_key_ref) is not str
        or key_ring.get("client_key_ref") != client_key_ref
    ):
        _raise("KEY_REF")
    if key_ring.get("client_signature_algorithm") != algorithm:
        _raise("ALGORITHM")
    if g1_root is not None and g1_root != compute_manifest_root(local_build.manifest):
        _raise("ROOT")
    if not has_claims:
        _raise("CO_ATTESTATION_FORBIDDEN")
    if "client_key_ref" not in key_ring:
        _raise("CO_ATTESTATION_REQUIRED")
    digest = manifest_digest(copied)
    signature = _sign_material(
        client_private_key,
        algorithm,
        client_signed_material(manifest_bytes),
    )
    return {
        "algorithm": algorithm,
        "client_key_ref": client_key_ref,
        "signed_manifest_digest": digest,
        "nonce": expected_nonce,
        "signature": signature,
    }


def _validate_co_shape(
    manifest: dict[str, Any], co_attestation: object
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    copied, manifest_bytes = _canonical_for_attestation(manifest)
    co = _require_exact_dict(co_attestation, "CO_SHAPE", VerificationError)
    if any(type(key) is not str for key in co) or set(co) != _CO_KEYS:
        _raise("CO_SHAPE", VerificationError)
    key_ring = _validate_key_ring(copied, VerificationError)
    freshness = copied.get("freshness")
    if type(freshness) is not dict or set(freshness) != {"nonce"}:
        _raise("FRESHNESS", VerificationError)
    if type(co["algorithm"]) is not str or co["algorithm"] != key_ring.get(
        "client_signature_algorithm"
    ):
        _raise("ALGORITHM", VerificationError)
    if type(co["client_key_ref"]) is not str or co["client_key_ref"] != key_ring.get(
        "client_key_ref"
    ):
        _raise("KEY_REF", VerificationError)
    if type(co["nonce"]) is not str or co["nonce"] != freshness.get("nonce"):
        _raise("NONCE", VerificationError)
    if type(co["signed_manifest_digest"]) is not str or co[
        "signed_manifest_digest"
    ] != manifest_digest(copied):
        _raise("MANIFEST_DIGEST", VerificationError)
    decode_signature(co["signature"])
    return copied, manifest_bytes, co


def verify_client_co_attestation(
    manifest: dict[str, Any],
    co_attestation: dict[str, Any],
    client_public_key: object,
    context: ExpectedCertificateContext,
    local_build: ClientEvidenceBuild,
) -> bool:
    """Verify a co-attestation locally, including the G1 opening root."""

    copied, manifest_bytes, co = _validate_co_shape(manifest, co_attestation)
    has_claims, _ = _validate_claims(copied, VerificationError)
    if not has_claims:
        _raise("CO_ATTESTATION_FORBIDDEN", VerificationError)
    _validate_context(copied, context)
    if co["client_key_ref"] != context.expected_client_key_ref:
        _raise("KEY_REF", VerificationError)
    _check_g1_root(copied, local_build)
    verify_signature(
        client_public_key,
        co["algorithm"],
        client_signed_material(manifest_bytes),
        co["signature"],
    )
    return True


def _verify_issuer_signature_only(
    manifest: dict[str, Any],
    issuer_signature: dict[str, Any],
    issuer_public_key: object,
    context: ExpectedCertificateContext,
    co_attestation: dict[str, Any] | None = None,
) -> bool:
    """Verify only the issuer signature over exact manifest/co bytes."""

    copied, manifest_bytes = _canonical_for_attestation(manifest)
    issuer = _require_exact_dict(issuer_signature, "ISSUER_SHAPE", VerificationError)
    if any(type(key) is not str for key in issuer) or set(issuer) != _ISSUER_KEYS:
        _raise("ISSUER_SHAPE", VerificationError)
    key_ring = _validate_key_ring(copied, VerificationError)
    has_claims, _ = _validate_claims(copied, VerificationError)
    _validate_context(copied, context)
    if has_claims and context.expected_client_key_ref is None:
        _raise("CONTEXT", VerificationError)
    if issuer["algorithm"] != key_ring.get("issuer_signature_algorithm"):
        _raise("ALGORITHM", VerificationError)
    if issuer["issuer_key_ref"] != key_ring.get("issuer_key_ref"):
        _raise("KEY_REF", VerificationError)
    if issuer["trust_ring_ref"] != key_ring.get("trust_ring_ref"):
        _raise("TRUST_RING", VerificationError)
    if issuer["issuer_key_ref"] != context.expected_issuer_key_ref:
        _raise("KEY_REF", VerificationError)
    if issuer["trust_ring_ref"] != context.expected_trust_ring_ref:
        _raise("TRUST_RING", VerificationError)

    co_raw = b""
    if co_attestation is not None:
        _, _, co = _validate_co_shape(copied, co_attestation)
        co_raw = decode_signature(co["signature"])
        if not has_claims:
            _raise("CO_ATTESTATION_FORBIDDEN", VerificationError)
        if issuer["signed_payload"] != ["unsigned_manifest", "co_attestation"]:
            _raise("SIGNED_PAYLOAD", VerificationError)
    elif has_claims:
        _raise("CO_ATTESTATION_REQUIRED", VerificationError)
    elif (
        "client_key_ref" in key_ring
        or "client_signature_algorithm" in key_ring
        or context.expected_client_key_ref is not None
    ):
        _raise("CO_ATTESTATION_FORBIDDEN", VerificationError)
    elif issuer["signed_payload"] != ["unsigned_manifest"]:
        _raise("SIGNED_PAYLOAD", VerificationError)
    if issuer["signed_payload"] not in (
        ["unsigned_manifest"],
        ["unsigned_manifest", "co_attestation"],
    ):
        _raise("SIGNED_PAYLOAD", VerificationError)
    verify_signature(
        issuer_public_key,
        issuer["algorithm"],
        issuer_signed_material(manifest_bytes, co_raw),
        issuer["signature"],
    )
    return True


def verify_certificate_signatures(
    manifest: dict[str, Any],
    co_attestation: dict[str, Any] | None,
    issuer_signature: dict[str, Any],
    client_public_key: object | None,
    issuer_public_key: object,
    context: ExpectedCertificateContext,
) -> bool:
    """Verify client co-attestation and issuer signature as one composition."""

    copied, _ = _canonical_for_attestation(manifest)
    has_claims, _ = _validate_claims(copied, VerificationError)
    if not has_claims:
        if co_attestation is not None or client_public_key is not None:
            _raise("CO_ATTESTATION_FORBIDDEN", VerificationError)
        if (
            "client_key_ref" in copied["key_ring_identifiers"]
            or "client_signature_algorithm" in copied["key_ring_identifiers"]
            or context.expected_client_key_ref is not None
        ):
            _raise("CO_ATTESTATION_FORBIDDEN", VerificationError)
        return _verify_issuer_signature_only(
            copied, issuer_signature, issuer_public_key, context, None
        )
    if co_attestation is None or client_public_key is None:
        _raise("CO_ATTESTATION_REQUIRED", VerificationError)
    if context.expected_client_key_ref is None:
        _raise("CONTEXT", VerificationError)
    co_copied, manifest_bytes, co = _validate_co_shape(copied, co_attestation)
    if co["client_key_ref"] != context.expected_client_key_ref:
        _raise("KEY_REF", VerificationError)
    verify_signature(
        client_public_key,
        co["algorithm"],
        client_signed_material(manifest_bytes),
        co["signature"],
    )
    return _verify_issuer_signature_only(
        co_copied,
        issuer_signature,
        issuer_public_key,
        context,
        co,
    )
