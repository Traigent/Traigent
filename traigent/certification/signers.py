"""Role-separated Certificate v0 signing material and verification primitives."""

from __future__ import annotations

import base64
import hashlib
import re
import struct
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, utils

from traigent.utils import fp2

__all__ = [
    "CertificationError",
    "SignatureError",
    "VerificationError",
    "canonical_manifest",
    "client_signed_material",
    "decode_signature",
    "encode_signature",
    "issuer_signed_material",
    "manifest_digest",
    "verify_signature",
]

CLIENT_DOMAIN = b"traigent.agent_certificate.client_co_attestation.v0"
ISSUER_DOMAIN = b"traigent.agent_certificate.issuer_signature.v0"
UNSIGNED_MANIFEST_DOMAIN = b"traigent.agent_certificate.unsigned_manifest.v1"
_MANIFEST_KEYS = frozenset(
    {
        "subject",
        "seal",
        "claims",
        "tiers",
        "evidence_digests",
        "non_claims",
        "privacy_mode",
        "sdk_identity",
        "compiler_register_versions",
        "key_ring_identifiers",
        "freshness",
    }
)
_ALGORITHMS = frozenset({"ed25519", "ecdsa_p256_sha256"})
_ECDSA_ORDER = int(
    "FFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551", 16
)
_ECDSA_HALF_ORDER = _ECDSA_ORDER // 2
_DIGEST_RE = re.compile(r"^sha256:[a-f0-9]{64}$")
_OPAQUE_REF_RE = re.compile(r"^[a-z][a-z0-9_.-]{1,63}:[A-Za-z0-9_-]{8,128}$")
_SHA_RE = re.compile(r"^[a-f0-9]{40}$")
_SEMVER_RE = re.compile(r"^[0-9]{1,4}\.[0-9]{1,4}\.[0-9]{1,4}$")
_EVIDENCE_KINDS = frozenset(
    {
        "ledger_entry_commitment",
        "seal_statement",
        "registry_record_digest",
        "sdk_witness_bundle",
        "verifier_report_digest",
        "trust_ring_artifact",
        "audit_report_digest",
    }
)
_CLAIM_IDS = frozenset({"D2", "G1"})
_NON_CLAIMS = (
    ("A3", "tmpl.noncert.a3_no_deployment_binding.v1"),
    ("A4", "tmpl.noncert.a4_no_drift_detection.v1"),
    ("B2", "tmpl.noncert.b2_no_closeout_reconciliation.v1"),
    ("E2", "tmpl.noncert.e2_statistical_validity_suspended.v1"),
    ("E3", "tmpl.noncert.e3_winner_stability_suspended.v1"),
    ("F3", "tmpl.noncert.f3_no_offline_verifiability.v1"),
    ("G2", "tmpl.noncert.g2_no_selective_disclosure.v1"),
    ("H1", "tmpl.noncert.h1_no_criteria_epochs.v1"),
    ("NC_STEP_CAPTURE", "tmpl.noncert.nc_step_capture.v1"),
    ("NC_PRESEAL_RECORDER", "tmpl.noncert.nc_preseal_recorder.v1"),
    ("NC_TIER4_GAPCHECKED", "tmpl.noncert.nc_tier4_gapchecked.v1"),
    ("NC_TARGET_MINIMIZED", "tmpl.noncert.nc_target_minimized.v1"),
    ("NC_CURRENT_ONLINE_FREETEXT", "tmpl.noncert.nc_current_online_freetext.v1"),
    ("NC_BUILD_SESSION_SCOPE", "tmpl.noncert.nc_build_session_scope.v1"),
)


class CertificationError(ValueError):
    """Base error carrying only a fixed machine-readable code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class SignatureError(CertificationError):
    """Malformed signing material or signing-key failure."""


class VerificationError(CertificationError):
    """Malformed or invalid verification material."""


def _plain_copy(value: Any) -> Any:
    if type(value) is dict:
        return {_plain_copy(key): _plain_copy(item) for key, item in value.items()}
    if type(value) is list:
        return [_plain_copy(item) for item in value]
    return value


def _reject_floats(value: Any, seen: set[int] | None = None) -> None:
    pending = [value]
    visited = set() if seen is None else seen
    while pending:
        item = pending.pop()
        if type(item) is float:
            raise CertificationError("MANIFEST_FLOAT")
        if type(item) is dict or type(item) is list:
            identity = id(item)
            if identity in visited:
                continue
            visited.add(identity)
            pending.extend(item.values() if type(item) is dict else item)


def _schema_fail() -> None:
    raise CertificationError("MANIFEST_SCHEMA")


def _keys(
    value: object, expected: set[str], *, optional: set[str] | None = None
) -> None:
    if type(value) is not dict:
        _schema_fail()
    allowed = expected | (optional or set())
    if any(type(key) is not str for key in value) or not set(value).issubset(allowed):
        _schema_fail()
    if not expected.issubset(value):
        _schema_fail()


def _string(value: object) -> str:
    if type(value) is not str:
        _schema_fail()
    return value


def _digest(value: object) -> str:
    value = _string(value)
    if _DIGEST_RE.fullmatch(value) is None:
        _schema_fail()
    return value


def _opaque_ref(value: object) -> str:
    value = _string(value)
    if _OPAQUE_REF_RE.fullmatch(value) is None:
        _schema_fail()
    return value


def _semver(value: object) -> str:
    value = _string(value)
    if _SEMVER_RE.fullmatch(value) is None:
        _schema_fail()
    return value


def _unique(items: list[Any]) -> None:
    try:
        if len(items) != len({fp2.canonicalize(item) for item in items}):
            _schema_fail()
    except (RecursionError, TypeError, ValueError, fp2.Fp2UnsupportedValue):
        _schema_fail()


def _validate_evidence_ref(value: object) -> dict[str, Any]:
    _keys(value, {"evidence_kind", "evidence_digest"}, optional={"evidence_ref"})
    evidence = value
    kind = _string(evidence["evidence_kind"])
    if type(kind) is not str or kind not in _EVIDENCE_KINDS:
        _schema_fail()
    _digest(evidence["evidence_digest"])
    if "evidence_ref" in evidence:
        _opaque_ref(evidence["evidence_ref"])
    return evidence


def _validate_claim(value: object) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    _keys(
        value,
        {"record_type", "claim_id", "tier", "payload", "verifier", "evidence_refs"},
    )
    claim = value
    if (
        claim["record_type"] != "claim"
        or type(claim["claim_id"]) is not str
        or claim["claim_id"] not in _CLAIM_IDS
    ):
        _schema_fail()
    claim_id = _string(claim["claim_id"])
    if type(claim["tier"]) is not int or claim["tier"] != 1:
        _schema_fail()
    refs = claim["evidence_refs"]
    if type(refs) is not list or not 1 <= len(refs) <= 64:
        _schema_fail()
    validated_refs = [_validate_evidence_ref(ref) for ref in refs]
    _unique(validated_refs)
    audit_refs = [
        ref for ref in validated_refs if ref["evidence_kind"] == "audit_report_digest"
    ]
    if len(audit_refs) != 1:
        _schema_fail()

    verifier = claim["verifier"]
    _keys(verifier, {"verifier_id", "verifier_version", "result"})
    verifier_id = _string(verifier["verifier_id"])
    if re.fullmatch(r"ver\.cert\.[a-z0-9_]{1,64}", verifier_id) is None:
        _schema_fail()
    _semver(verifier["verifier_version"])
    if verifier["result"] != "PASS":
        _schema_fail()

    payload = claim["payload"]
    _keys(payload, {"claim_id", "template_id", "params"})
    if payload["claim_id"] != claim_id or type(payload["template_id"]) is not str:
        _schema_fail()
    params = payload["params"]
    if claim_id == "D2":
        _keys(
            payload,
            {"claim_id", "template_id", "params"},
        )
        if payload["template_id"] != "tmpl.cert.d2.offline_backend_egress_witness.v1":
            _schema_fail()
        _keys(
            params,
            {
                "declared_mode",
                "witness_kind",
                "sdk_ref",
                "workload_class",
                "witness_bundle_digest",
            },
        )
        if (
            params["declared_mode"] != "offline"
            or params["witness_kind"] != "strace_network_trace"
            or params["workload_class"]
            != "mock_grid_no_integrations_no_analytics_no_langfuse"
        ):
            _schema_fail()
        if _SHA_RE.fullmatch(_string(params["sdk_ref"])) is None:
            _schema_fail()
        _digest(params["witness_bundle_digest"])
        sdk_refs = [
            ref
            for ref in validated_refs
            if ref["evidence_kind"] == "sdk_witness_bundle"
        ]
        if (
            len(sdk_refs) != 1
            or sdk_refs[0]["evidence_digest"] != params["witness_bundle_digest"]
        ):
            _schema_fail()
    else:
        if (
            payload["template_id"]
            != "tmpl.cert.g1.client_evidence_manifest_commitment.v1"
        ):
            _schema_fail()
        _keys(
            params,
            {"manifest_root_digest", "commitment_scheme", "client_attestor_version"},
        )
        _digest(params["manifest_root_digest"])
        if params["commitment_scheme"] != "sha256_secret_blinded_v1":
            _schema_fail()
        _semver(params["client_attestor_version"])
    return claim_id, params, validated_refs


def _validate_typed_unsigned_manifest(manifest: dict[str, Any]) -> None:
    """Enforce the Schema's typed document and its local equality obligations."""

    _keys(manifest, set(_MANIFEST_KEYS))

    subject = manifest["subject"]
    _keys(
        subject,
        {
            "subject_kind",
            "hash_algorithm",
            "build_session_ref",
            "session_commitment_digest",
        },
    )
    if subject["subject_kind"] != "build_session" or subject["hash_algorithm"] != "v1":
        _schema_fail()
    subject_session = _opaque_ref(subject["build_session_ref"])
    _digest(subject["session_commitment_digest"])

    seal = manifest["seal"]
    _keys(
        seal,
        {
            "seal_ref",
            "chain_schema_version",
            "build_session_ref",
            "expected_stream_projection",
            "seal_statement_digest",
        },
    )
    _opaque_ref(seal["seal_ref"])
    _opaque_ref(seal["build_session_ref"])
    if seal["chain_schema_version"] != "traigent.cert_ledger.v0":
        _schema_fail()
    _digest(seal["seal_statement_digest"])
    if seal["build_session_ref"] != subject_session:
        _schema_fail()
    streams = seal["expected_stream_projection"]
    _keys(streams, {"decision_stream", "receipt_event_stream", "transition_stream"})
    stream_families = {
        "decision_stream": "decision",
        "receipt_event_stream": "receipt_event",
        "transition_stream": "transition",
    }
    for name, family in stream_families.items():
        stream = streams[name]
        _keys(stream, {"stream_family", "chain_status"}, optional={"root_commitment"})
        if (
            type(stream["stream_family"]) is not str
            or type(stream["chain_status"]) is not str
            or stream["stream_family"] != family
            or stream["chain_status"]
            not in {
                "sealed",
                "empty_sealed",
                "legacy_unsealed",
                "not_applicable",
            }
        ):
            _schema_fail()
        sealed = stream["chain_status"] in {"sealed", "empty_sealed"}
        if sealed != ("root_commitment" in stream):
            _schema_fail()
        if sealed:
            _digest(stream["root_commitment"])

    claims = manifest["claims"]
    if type(claims) is not list or len(claims) > 16:
        _schema_fail()
    _unique(claims)
    validated_claims = [_validate_claim(claim) for claim in claims]
    claim_ids = [claim[0] for claim in validated_claims]
    if len(claim_ids) != len(set(claim_ids)):
        _schema_fail()

    tiers = manifest["tiers"]
    if type(tiers) is not list or len(tiers) != len(claims) or len(tiers) > 16:
        _schema_fail()
    _unique(tiers)
    for tier in tiers:
        _keys(tier, {"claim_id", "tier"})
        if type(tier["claim_id"]) is not str or type(tier["tier"]) is not int:
            _schema_fail()
    expected_tiers = [{"claim_id": claim_id, "tier": 1} for claim_id in claim_ids]
    if tiers != expected_tiers:
        _schema_fail()

    non_claims = manifest["non_claims"]
    if type(non_claims) is not list or len(non_claims) != len(_NON_CLAIMS):
        _schema_fail()
    for item, (non_claim_id, template_id) in zip(non_claims, _NON_CLAIMS, strict=True):
        _keys(item, {"record_type", "non_claim_id", "reason_template_id"})
        if (
            item["record_type"] != "non_claim"
            or item["non_claim_id"] != non_claim_id
            or item["reason_template_id"] != template_id
        ):
            _schema_fail()

    privacy = manifest["privacy_mode"]
    _keys(privacy, {"declared_mode"})
    if type(privacy["declared_mode"]) is not str or privacy["declared_mode"] not in {
        "offline",
        "current_online",
    }:
        _schema_fail()
    sdk = manifest["sdk_identity"]
    _keys(sdk, {"sdk_ref", "sdk_version"})
    if _SHA_RE.fullmatch(_string(sdk["sdk_ref"])) is None:
        _schema_fail()
    _semver(sdk["sdk_version"])
    compiler = manifest["compiler_register_versions"]
    compiler_keys = {
        "compiler_version",
        "semantics_manifest_digest",
        "claim_template_catalog_digest",
        "prohibited_register_digest",
        "verifier_catalog_digest",
        "non_claim_reason_catalog_digest",
    }
    _keys(compiler, compiler_keys)
    _semver(compiler["compiler_version"])
    for key in compiler_keys - {"compiler_version"}:
        _digest(compiler[key])

    kri = manifest["key_ring_identifiers"]
    _keys(
        kri,
        {"issuer_key_ref", "trust_ring_ref", "issuer_signature_algorithm"},
        optional={"client_key_ref", "client_signature_algorithm"},
    )
    _opaque_ref(kri["issuer_key_ref"])
    _opaque_ref(kri["trust_ring_ref"])
    if (
        type(kri["issuer_signature_algorithm"]) is not str
        or kri["issuer_signature_algorithm"] not in _ALGORITHMS
    ):
        _schema_fail()
    if ("client_key_ref" in kri) != ("client_signature_algorithm" in kri):
        _schema_fail()
    if "client_key_ref" in kri:
        _opaque_ref(kri["client_key_ref"])
        if (
            type(kri["client_signature_algorithm"]) is not str
            or kri["client_signature_algorithm"] not in _ALGORITHMS
        ):
            _schema_fail()
    freshness = manifest["freshness"]
    _keys(freshness, {"nonce"})
    nonce = _string(freshness["nonce"])
    if re.fullmatch(r"[a-f0-9]{32,64}", nonce) is None:
        _schema_fail()

    evidence = manifest["evidence_digests"]
    if type(evidence) is not list or not 1 <= len(evidence) <= 1024:
        _schema_fail()
    validated_evidence = [_validate_evidence_ref(item) for item in evidence]
    _unique(validated_evidence)
    audit_refs = [
        ref
        for _, _, refs in validated_claims
        for ref in refs
        if ref["evidence_kind"] == "audit_report_digest"
    ]
    expected_audit = audit_refs[0]["evidence_digest"] if audit_refs else None
    if any(ref["evidence_digest"] != expected_audit for ref in audit_refs):
        _schema_fail()
    if evidence[0]["evidence_kind"] != "audit_report_digest":
        _schema_fail()
    if expected_audit is not None and evidence[0]["evidence_digest"] != expected_audit:
        _schema_fail()
    expected_evidence: list[dict[str, Any]] = [
        {
            "evidence_kind": "audit_report_digest",
            "evidence_digest": evidence[0]["evidence_digest"],
        }
    ]
    for _, _, refs in validated_claims:
        for ref in refs:
            if ref not in expected_evidence:
                expected_evidence.append(ref)
    if evidence != expected_evidence:
        _schema_fail()
    for claim_id, params, _ in validated_claims:
        if claim_id == "D2":
            if (
                params["declared_mode"] != privacy["declared_mode"]
                or params["sdk_ref"] != sdk["sdk_ref"]
            ):
                _schema_fail()


def canonical_manifest(manifest: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
    """Validate the plain manifest boundary and return a defensive copy/bytes."""

    if type(manifest) is not dict:
        raise CertificationError("MANIFEST_NOT_PLAIN_DICT")
    if any(type(key) is not str for key in manifest) or set(manifest) != _MANIFEST_KEYS:
        raise CertificationError("MANIFEST_SHAPE")
    try:
        _reject_floats(manifest)
        _validate_typed_unsigned_manifest(manifest)
        canonical = fp2.canonicalize(manifest).encode("utf-8")
    except CertificationError:
        raise
    except fp2.Fp2UnsupportedValue:
        raise CertificationError("MANIFEST_NOT_FP2") from None
    return _plain_copy(manifest), canonical


def manifest_digest(manifest: dict[str, Any]) -> str:
    """Return the role-separated digest of a canonical unsigned manifest."""

    _, canonical = canonical_manifest(manifest)
    return (
        "sha256:"
        + hashlib.sha256(UNSIGNED_MANIFEST_DOMAIN + b"\x00" + canonical).hexdigest()
    )


def _framed(domain: bytes, manifest_bytes: bytes, trailing: bytes = b"") -> bytes:
    if type(manifest_bytes) is not bytes or type(trailing) is not bytes:
        raise SignatureError("MATERIAL_NOT_BYTES")
    if len(manifest_bytes) >= 2**64 or len(trailing) >= 2**64:
        raise SignatureError("MATERIAL_TOO_LARGE")
    return (
        domain
        + b"\x00"
        + struct.pack(">Q", len(manifest_bytes))
        + manifest_bytes
        + trailing
    )


def client_signed_material(manifest_bytes: bytes) -> bytes:
    return _framed(CLIENT_DOMAIN, manifest_bytes)


def issuer_signed_material(
    manifest_bytes: bytes, co_signature_raw: bytes = b""
) -> bytes:
    if type(co_signature_raw) is not bytes:
        raise SignatureError("MATERIAL_NOT_BYTES")
    if len(co_signature_raw) >= 2**64:
        raise SignatureError("MATERIAL_TOO_LARGE")
    return _framed(
        ISSUER_DOMAIN,
        manifest_bytes,
        struct.pack(">Q", len(co_signature_raw)) + co_signature_raw,
    )


def encode_signature(raw_signature: bytes) -> str:
    if type(raw_signature) is not bytes or len(raw_signature) != 64:
        raise SignatureError("SIGNATURE_LENGTH")
    return base64.b64encode(raw_signature).decode("ascii")


def decode_signature(encoded_signature: str) -> bytes:
    if type(encoded_signature) is not str:
        raise VerificationError("SIGNATURE_BASE64")
    try:
        raw = base64.b64decode(encoded_signature.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError):
        raise VerificationError("SIGNATURE_BASE64") from None
    if len(raw) != 64 or base64.b64encode(raw).decode("ascii") != encoded_signature:
        raise VerificationError("SIGNATURE_BASE64")
    return raw


def _validate_algorithm(algorithm: str, error_type: type[CertificationError]) -> None:
    if type(algorithm) is not str or algorithm not in _ALGORITHMS:
        raise error_type("ALGORITHM")


def _sign_raw(private_key: object, algorithm: str, material: bytes) -> bytes:
    _validate_algorithm(algorithm, SignatureError)
    if type(material) is not bytes:
        raise SignatureError("MATERIAL_NOT_BYTES")
    try:
        if algorithm == "ed25519":
            if not isinstance(private_key, ed25519.Ed25519PrivateKey):
                raise SignatureError("KEY_ALGORITHM")
            return private_key.sign(material)
        if not isinstance(private_key, ec.EllipticCurvePrivateKey):
            raise SignatureError("KEY_ALGORITHM")
        if not isinstance(private_key.curve, ec.SECP256R1):
            raise SignatureError("KEY_ALGORITHM")
        der = private_key.sign(material, ec.ECDSA(hashes.SHA256()))
        r, s = utils.decode_dss_signature(der)
        if not 1 <= r < _ECDSA_ORDER or not 1 <= s < _ECDSA_ORDER:
            raise SignatureError("SIGNATURE_SCALAR")
        if s > _ECDSA_HALF_ORDER:
            s = _ECDSA_ORDER - s
        return r.to_bytes(32, "big") + s.to_bytes(32, "big")
    except SignatureError:
        raise
    except Exception:
        raise SignatureError("SIGNING_FAILED") from None


def _verify_raw(
    public_key: object, algorithm: str, material: bytes, raw_signature: bytes
) -> None:
    _validate_algorithm(algorithm, VerificationError)
    if type(material) is not bytes or type(raw_signature) is not bytes:
        raise VerificationError("MATERIAL_NOT_BYTES")
    if len(raw_signature) != 64:
        raise VerificationError("SIGNATURE_LENGTH")
    try:
        if algorithm == "ed25519":
            if not isinstance(public_key, ed25519.Ed25519PublicKey):
                raise VerificationError("KEY_ALGORITHM")
            public_key.verify(raw_signature, material)
            return
        if not isinstance(public_key, ec.EllipticCurvePublicKey):
            raise VerificationError("KEY_ALGORITHM")
        if not isinstance(public_key.curve, ec.SECP256R1):
            raise VerificationError("KEY_ALGORITHM")
        r = int.from_bytes(raw_signature[:32], "big")
        s = int.from_bytes(raw_signature[32:], "big")
        if not 1 <= r < _ECDSA_ORDER or not 1 <= s < _ECDSA_ORDER:
            raise VerificationError("SIGNATURE_SCALAR")
        if s > _ECDSA_HALF_ORDER:
            raise VerificationError("SIGNATURE_HIGH_S")
        public_key.verify(
            utils.encode_dss_signature(r, s),
            material,
            ec.ECDSA(hashes.SHA256()),
        )
    except VerificationError:
        raise
    except (InvalidSignature, ValueError):
        raise VerificationError("SIGNATURE_INVALID") from None
    except Exception:
        raise VerificationError("VERIFICATION_FAILED") from None


def verify_signature(
    public_key: object,
    algorithm: str,
    material: bytes,
    encoded_signature: str,
) -> bool:
    """Verify one already-framed role material with a resolved public key."""

    raw = decode_signature(encoded_signature)
    _verify_raw(public_key, algorithm, material, raw)
    return True


def _sign_material(private_key: object, algorithm: str, material: bytes) -> str:
    """Private test/attestor helper; no issuer signing API is exported."""

    return encode_signature(_sign_raw(private_key, algorithm, material))
