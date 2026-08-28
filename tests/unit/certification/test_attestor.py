from __future__ import annotations

import base64
import copy
from dataclasses import replace

import pytest
from cryptography.hazmat.primitives.asymmetric import ec, ed25519

from traigent.certification import (
    ClientEvidenceBuild,
    build_client_evidence_manifest,
    compute_manifest_root,
)
from traigent.certification.attestor import (
    ExpectedCertificateContext,
    create_client_co_attestation,
    verify_certificate_signatures,
    verify_client_co_attestation,
    _verify_issuer_signature_only,
)
from traigent.certification.signers import (
    CertificationError,
    VerificationError,
    canonical_manifest,
    client_signed_material,
    issuer_signed_material,
    manifest_digest,
    _sign_material,
)

_CLIENT_REF = "client:keyref12345678"
_ISSUER_REF = "issuer:keyref12345678"
_RING_REF = "ring:keyref12345678"
_NONCE = "b" * 32


def _manifest(
    *, algorithm: str = "ed25519", claims: list[dict[str, object]] | None = None
) -> dict[str, object]:
    digest = "sha256:" + "a" * 64
    witness_digest = "sha256:" + "b" * 64
    non_claims = (
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
    default_claims = [
        {
            "record_type": "claim",
            "claim_id": "D2",
            "tier": 1,
            "payload": {
                "claim_id": "D2",
                "template_id": "tmpl.cert.d2.offline_backend_egress_witness.v1",
                "params": {
                    "declared_mode": "offline",
                    "witness_kind": "strace_network_trace",
                    "sdk_ref": "a" * 40,
                    "workload_class": "mock_grid_no_integrations_no_analytics_no_langfuse",
                    "witness_bundle_digest": witness_digest,
                },
            },
            "verifier": {
                "verifier_id": "ver.cert.offline_egress_witness",
                "verifier_version": "0.1.0",
                "result": "PASS",
            },
            "evidence_refs": [
                {"evidence_kind": "audit_report_digest", "evidence_digest": digest},
                {
                    "evidence_kind": "sdk_witness_bundle",
                    "evidence_digest": witness_digest,
                },
            ],
        }
    ]
    effective_claims = default_claims if claims is None else claims
    projected_evidence = [
        {"evidence_kind": "audit_report_digest", "evidence_digest": digest}
    ]
    for claim in effective_claims:
        for ref in claim.get("evidence_refs", []):
            if ref not in projected_evidence:
                projected_evidence.append(ref)
    return {
        "subject": {
            "subject_kind": "build_session",
            "hash_algorithm": "v1",
            "build_session_ref": "bsn:abcdef0123456789",
            "session_commitment_digest": digest,
        },
        "seal": {
            "seal_ref": "seal:abcdef0123456789",
            "chain_schema_version": "traigent.cert_ledger.v0",
            "build_session_ref": "bsn:abcdef0123456789",
            "expected_stream_projection": {
                f"{family}_stream": {
                    "stream_family": family,
                    "chain_status": "empty_sealed",
                    "root_commitment": digest,
                }
                for family in ("decision", "receipt_event", "transition")
            },
            "seal_statement_digest": digest,
        },
        "claims": effective_claims,
        "tiers": [
            {"claim_id": claim["claim_id"], "tier": claim["tier"]}
            for claim in effective_claims
        ],
        "evidence_digests": projected_evidence,
        "non_claims": [
            {
                "record_type": "non_claim",
                "non_claim_id": non_claim_id,
                "reason_template_id": template_id,
            }
            for non_claim_id, template_id in non_claims
        ],
        "privacy_mode": {"declared_mode": "offline"},
        "sdk_identity": {"sdk_ref": "a" * 40, "sdk_version": "0.1.0"},
        "compiler_register_versions": {
            "compiler_version": "0.1.0",
            "semantics_manifest_digest": digest,
            "claim_template_catalog_digest": digest,
            "prohibited_register_digest": digest,
            "verifier_catalog_digest": digest,
            "non_claim_reason_catalog_digest": digest,
        },
        "key_ring_identifiers": {
            "issuer_key_ref": _ISSUER_REF,
            "trust_ring_ref": _RING_REF,
            "issuer_signature_algorithm": algorithm,
            "client_key_ref": _CLIENT_REF,
            "client_signature_algorithm": algorithm,
        },
        "freshness": {"nonce": _NONCE},
    }


def _build() -> ClientEvidenceBuild:
    return build_client_evidence_manifest(
        {"slot": "agent"},
        {"slot": "dataset"},
        {"slot": "evaluator"},
        {"slot": "evidence"},
    )


def _g1_claim(build: ClientEvidenceBuild) -> dict[str, object]:
    return {
        "record_type": "claim",
        "claim_id": "G1",
        "tier": 1,
        "payload": {
            "claim_id": "G1",
            "template_id": "tmpl.cert.g1.client_evidence_manifest_commitment.v1",
            "params": {
                "manifest_root_digest": compute_manifest_root(build.manifest),
                "commitment_scheme": "sha256_secret_blinded_v1",
                "client_attestor_version": "0.1.0",
            },
        },
        "verifier": {
            "verifier_id": "ver.cert.manifest_commitment",
            "verifier_version": "0.1.0",
            "result": "PASS",
        },
        "evidence_refs": [
            {
                "evidence_kind": "audit_report_digest",
                "evidence_digest": "sha256:" + "a" * 64,
            }
        ],
    }


def _g1_manifest(
    build: ClientEvidenceBuild, *, algorithm: str = "ed25519"
) -> dict[str, object]:
    return _manifest(algorithm=algorithm, claims=[_g1_claim(build)])


def _context() -> ExpectedCertificateContext:
    return ExpectedCertificateContext(
        expected_nonce=_NONCE,
        expected_build_session_ref="bsn:abcdef0123456789",
        expected_session_commitment_digest="sha256:" + "a" * 64,
        expected_client_key_ref=_CLIENT_REF,
        expected_issuer_key_ref=_ISSUER_REF,
        expected_trust_ring_ref=_RING_REF,
    )


def _zero_context() -> ExpectedCertificateContext:
    return ExpectedCertificateContext(
        expected_nonce=_NONCE,
        expected_build_session_ref="bsn:abcdef0123456789",
        expected_session_commitment_digest="sha256:" + "a" * 64,
        expected_client_key_ref=None,
        expected_issuer_key_ref=_ISSUER_REF,
        expected_trust_ring_ref=_RING_REF,
    )


def test_client_attestor_binds_manifest_nonce_key_algorithm_and_root() -> None:
    build = _build()
    manifest = _g1_manifest(build)
    key = ed25519.Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    co = create_client_co_attestation(
        manifest, _NONCE, _CLIENT_REF, key, "ed25519", build
    )
    assert set(co) == {
        "algorithm",
        "client_key_ref",
        "signed_manifest_digest",
        "nonce",
        "signature",
    }
    assert verify_client_co_attestation(
        manifest, co, key.public_key(), _context(), build
    )
    assert "slot" not in repr(co)
    assert "agent" not in repr(co)

    changed = copy.deepcopy(manifest)
    changed["freshness"] = {"nonce": "c" * 32}
    with pytest.raises(VerificationError) as exc_info:
        verify_client_co_attestation(changed, co, key.public_key(), _context(), build)
    assert "agent" not in str(exc_info.value)


def test_client_attestor_rejects_unsupported_claims_and_root_tamper() -> None:
    build = _build()
    key = ed25519.Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    cases = [
        [{"claim_id": "F1", "tier": 1}],
        [{"claim_id": "D2", "tier": 2}],
        [
            {
                "claim_id": "G1",
                "tier": 1,
                "params": {"manifest_root_digest": "sha256:" + "f" * 64},
            }
        ],
    ]
    for claims in cases:
        with pytest.raises(CertificationError):
            create_client_co_attestation(
                _manifest(claims=claims), _NONCE, _CLIENT_REF, key, "ed25519", build
            )


def test_d2_only_client_co_attestation_is_rejected() -> None:
    build = _build()
    key = ed25519.Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    with pytest.raises(CertificationError, match="G1_REQUIRED"):
        create_client_co_attestation(
            _manifest(), _NONCE, _CLIENT_REF, key, "ed25519", build
        )

    d2_manifest = _manifest()
    _, manifest_bytes = canonical_manifest(d2_manifest)
    co = {
        "algorithm": "ed25519",
        "client_key_ref": _CLIENT_REF,
        "signed_manifest_digest": manifest_digest(d2_manifest),
        "nonce": _NONCE,
        "signature": _sign_material(
            key,
            "ed25519",
            client_signed_material(manifest_bytes),
        ),
    }
    with pytest.raises(VerificationError, match="G1_REQUIRED"):
        verify_client_co_attestation(
            d2_manifest, co, key.public_key(), _context(), build
        )

    issuer_key = ed25519.Ed25519PrivateKey.from_private_bytes(
        bytes(reversed(range(32)))
    )
    issuer = {
        "algorithm": "ed25519",
        "issuer_key_ref": _ISSUER_REF,
        "trust_ring_ref": _RING_REF,
        "signed_payload": ["unsigned_manifest", "co_attestation"],
        "signature": _sign_material(
            issuer_key,
            "ed25519",
            issuer_signed_material(
                manifest_bytes,
                base64.b64decode(co["signature"], validate=True),
            ),
        ),
    }
    with pytest.raises(VerificationError, match="G1_REQUIRED"):
        verify_certificate_signatures(
            d2_manifest,
            co,
            issuer,
            key.public_key(),
            issuer_key.public_key(),
            _context(),
        )


def _invalid_g1_manifest(build: ClientEvidenceBuild, kind: str) -> dict[str, object]:
    manifest = copy.deepcopy(_g1_manifest(build))
    if kind == "duplicate":
        manifest["claims"].append(copy.deepcopy(manifest["claims"][0]))
        manifest["tiers"].append({"claim_id": "G1", "tier": 1})
    else:
        manifest["claims"][0]["payload"]["params"]["manifest_root_digest"] = (
            "not-a-digest"
        )
    return manifest


@pytest.mark.parametrize("kind", ["duplicate", "malformed"])
@pytest.mark.parametrize("path", ["create", "verify_client", "verify_certificate"])
def test_invalid_g1_claims_fail_closed_on_public_paths(kind: str, path: str) -> None:
    build = _build()
    manifest = _invalid_g1_manifest(build, kind)
    key = ed25519.Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    with pytest.raises(VerificationError) as exc_info:
        if path == "create":
            create_client_co_attestation(
                manifest, _NONCE, _CLIENT_REF, key, "ed25519", build
            )
        elif path == "verify_client":
            verify_client_co_attestation(
                manifest, {}, key.public_key(), _context(), build
            )
        else:
            verify_certificate_signatures(
                manifest, None, {}, None, key.public_key(), _context()
            )
    assert exc_info.value.code == "MANIFEST_SCHEMA"


def test_issuer_verifier_binds_optional_co_signature_and_signed_payload() -> None:
    build = _build()
    manifest = _g1_manifest(build)
    client_key = ed25519.Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    issuer_key = ed25519.Ed25519PrivateKey.from_private_bytes(
        bytes(reversed(range(32)))
    )
    co = create_client_co_attestation(
        manifest, _NONCE, _CLIENT_REF, client_key, "ed25519", build
    )
    _, manifest_bytes = canonical_manifest(manifest)
    co_raw = base64.b64decode(co["signature"], validate=True)
    issuer = {
        "algorithm": "ed25519",
        "issuer_key_ref": _ISSUER_REF,
        "trust_ring_ref": _RING_REF,
        "signed_payload": ["unsigned_manifest", "co_attestation"],
        "signature": _sign_material(
            issuer_key,
            "ed25519",
            issuer_signed_material(manifest_bytes, co_raw),
        ),
    }
    assert _verify_issuer_signature_only(
        manifest, issuer, issuer_key.public_key(), _context(), co
    )
    assert verify_certificate_signatures(
        manifest,
        co,
        issuer,
        client_key.public_key(),
        issuer_key.public_key(),
        _context(),
    )
    with pytest.raises(VerificationError):
        _verify_issuer_signature_only(
            manifest,
            {**issuer, "signed_payload": ["unsigned_manifest"]},
            issuer_key.public_key(),
            _context(),
            co,
        )
    with pytest.raises(VerificationError):
        _verify_issuer_signature_only(
            manifest,
            issuer,
            issuer_key.public_key(),
            _context(),
            {**co, "signature": _sign_material(client_key, "ed25519", b"wrong")},
        )


def test_verifiers_reject_wrong_key_algorithm_and_malformed_signature() -> None:
    build = _build()
    manifest = _g1_manifest(build)
    key = ed25519.Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    co = create_client_co_attestation(
        manifest, _NONCE, _CLIENT_REF, key, "ed25519", build
    )
    wrong_key = ed25519.Ed25519PrivateKey.generate()
    with pytest.raises(VerificationError):
        verify_client_co_attestation(
            manifest, co, wrong_key.public_key(), _context(), build
        )
    with pytest.raises(VerificationError):
        verify_client_co_attestation(
            manifest,
            {**co, "signature": "not-base64"},
            key.public_key(),
            _context(),
            build,
        )
    with pytest.raises(CertificationError):
        create_client_co_attestation(
            manifest, "c" * 32, _CLIENT_REF, key, "ed25519", build
        )


def test_verification_requires_immutable_context_and_rejects_zero_claim_co() -> None:
    build = _build()
    key = ed25519.Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    with pytest.raises(CertificationError, match="CO_ATTESTATION_FORBIDDEN"):
        create_client_co_attestation(
            _manifest(claims=[]), _NONCE, _CLIENT_REF, key, "ed25519", build
        )
    manifest = _g1_manifest(build)
    co = create_client_co_attestation(
        manifest,
        _NONCE,
        _CLIENT_REF,
        key,
        "ed25519",
        build,
    )
    with pytest.raises(VerificationError, match="CONTEXT"):
        verify_client_co_attestation(
            manifest,
            co,
            key.public_key(),
            replace(_context(), expected_build_session_ref="bsn:otherref123456"),
            build,
        )


def test_context_rejects_nonce_and_commitment_replay() -> None:
    build = _build()
    manifest = _g1_manifest(build)
    key = ed25519.Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    co = create_client_co_attestation(
        manifest, _NONCE, _CLIENT_REF, key, "ed25519", build
    )
    with pytest.raises(VerificationError, match="CONTEXT"):
        verify_client_co_attestation(
            manifest,
            co,
            key.public_key(),
            replace(_context(), expected_nonce="c" * 32),
            build,
        )
    with pytest.raises(VerificationError, match="CONTEXT"):
        verify_client_co_attestation(
            manifest,
            co,
            key.public_key(),
            replace(
                _context(), expected_session_commitment_digest="sha256:" + "c" * 64
            ),
            build,
        )


def test_zero_claim_issuer_only_context_has_no_client_key() -> None:
    zero = _manifest(claims=[])
    del zero["key_ring_identifiers"]["client_key_ref"]
    del zero["key_ring_identifiers"]["client_signature_algorithm"]
    _, manifest_bytes = canonical_manifest(zero)
    issuer_key = ed25519.Ed25519PrivateKey.from_private_bytes(
        bytes(reversed(range(32)))
    )
    issuer = {
        "algorithm": "ed25519",
        "issuer_key_ref": _ISSUER_REF,
        "trust_ring_ref": _RING_REF,
        "signed_payload": ["unsigned_manifest"],
        "signature": _sign_material(
            issuer_key,
            "ed25519",
            issuer_signed_material(manifest_bytes),
        ),
    }
    assert verify_certificate_signatures(
        zero,
        None,
        issuer,
        None,
        issuer_key.public_key(),
        _zero_context(),
    )
    client_key = ed25519.Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    build = _build()
    manifest = _g1_manifest(build)
    co = create_client_co_attestation(
        manifest, _NONCE, _CLIENT_REF, client_key, "ed25519", build
    )
    with pytest.raises(VerificationError, match="CONTEXT"):
        verify_certificate_signatures(
            manifest,
            co,
            issuer,
            client_key.public_key(),
            issuer_key.public_key(),
            _zero_context(),
        )


@pytest.mark.parametrize("state", ["legacy_unsealed", "not_applicable"])
def test_certificate_paths_reject_unsealed_stream_states(state: str) -> None:
    build = _build()
    manifest = _g1_manifest(build)
    manifest["seal"]["expected_stream_projection"]["decision_stream"][
        "chain_status"
    ] = state
    del manifest["seal"]["expected_stream_projection"]["decision_stream"][
        "root_commitment"
    ]
    with pytest.raises(VerificationError, match="STREAM_STATE"):
        create_client_co_attestation(
            manifest,
            _NONCE,
            _CLIENT_REF,
            ed25519.Ed25519PrivateKey.from_private_bytes(bytes(range(32))),
            "ed25519",
            build,
        )


def test_ecdsa_client_path_is_supported() -> None:
    build = _build()
    manifest = _g1_manifest(build, algorithm="ecdsa_p256_sha256")
    key = ec.generate_private_key(ec.SECP256R1())
    co = create_client_co_attestation(
        manifest, _NONCE, _CLIENT_REF, key, "ecdsa_p256_sha256", build
    )
    assert verify_client_co_attestation(
        manifest, co, key.public_key(), _context(), build
    )
