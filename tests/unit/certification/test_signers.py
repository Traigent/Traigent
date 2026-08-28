from __future__ import annotations

import base64
import copy

import pytest
from cryptography.hazmat.primitives.asymmetric import ec, ed25519

from traigent.certification.signers import (
    CertificationError,
    SignatureError,
    VerificationError,
    canonical_manifest,
    client_signed_material,
    decode_signature,
    encode_signature,
    issuer_signed_material,
    manifest_digest,
    verify_signature,
    _sign_material,
)


def _manifest() -> dict[str, object]:
    digest = "sha256:" + "a" * 64
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
        "claims": [],
        "tiers": [],
        "evidence_digests": [
            {"evidence_kind": "audit_report_digest", "evidence_digest": digest}
        ],
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
            "issuer_key_ref": "issuer:keyref12345678",
            "trust_ring_ref": "ring:keyref12345678",
            "issuer_signature_algorithm": "ed25519",
            "client_key_ref": "client:keyref12345678",
            "client_signature_algorithm": "ed25519",
        },
        "freshness": {"nonce": "a" * 32},
    }


def _d2_manifest() -> dict[str, object]:
    manifest = copy.deepcopy(_manifest())
    audit_digest = manifest["evidence_digests"][0]["evidence_digest"]
    witness_digest = "sha256:" + "b" * 64
    manifest["claims"] = [
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
                {
                    "evidence_kind": "audit_report_digest",
                    "evidence_digest": audit_digest,
                },
                {
                    "evidence_kind": "sdk_witness_bundle",
                    "evidence_digest": witness_digest,
                },
            ],
        }
    ]
    manifest["tiers"] = [{"claim_id": "D2", "tier": 1}]
    manifest["evidence_digests"] = [
        {"evidence_kind": "audit_report_digest", "evidence_digest": audit_digest},
        {"evidence_kind": "sdk_witness_bundle", "evidence_digest": witness_digest},
    ]
    return manifest


def test_framing_and_manifest_digest_known_answers() -> None:
    assert client_signed_material(b"{}") == (
        b"traigent.agent_certificate.client_co_attestation.v0\x00"
        b"\x00\x00\x00\x00\x00\x00\x00\x02{}"
    )
    assert issuer_signed_material(b"{}", b"x") == (
        b"traigent.agent_certificate.issuer_signature.v0\x00"
        b"\x00\x00\x00\x00\x00\x00\x00\x02{}"
        b"\x00\x00\x00\x00\x00\x00\x00\x01x"
    )
    assert (
        manifest_digest(_manifest())
        == "sha256:a33aaa0903178866b90d4998fd5631af639078517fec352d20b32784cada81e3"
    )


def test_ed25519_signature_known_answer_and_verification() -> None:
    key = ed25519.Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    _, manifest_bytes = canonical_manifest(_manifest())
    signature = _sign_material(key, "ed25519", client_signed_material(manifest_bytes))
    assert signature == (
        "2GWVsOXe2facfsyhWDG0bQeusteLVg09mWeEsQN7+jPaEibt0ktmKwncf5hie4NS2ZFOmZwaeSosXZW1vCH4Ag=="
    )
    assert verify_signature(
        key.public_key(),
        "ed25519",
        client_signed_material(manifest_bytes),
        signature,
    )


def test_signature_base64_is_strict_and_exactly_64_bytes() -> None:
    encoded = encode_signature(bytes(range(64)))
    assert decode_signature(encoded) == bytes(range(64))
    for malformed in (
        encoded[:-1],
        encoded[:-2] + "A=",
        encoded.replace("A", " ", 1),
        base64.b64encode(b"short").decode(),
    ):
        with pytest.raises(VerificationError):
            decode_signature(malformed)
    with pytest.raises(SignatureError):
        encode_signature(b"short")


def test_canonical_manifest_rejects_shape_and_unsafe_values_without_echo() -> None:
    bad = _manifest()
    bad["secret"] = "DO_NOT_ECHO"
    with pytest.raises(CertificationError) as exc_info:
        canonical_manifest(bad)
    assert "DO_NOT_ECHO" not in str(exc_info.value)
    bad = _manifest()
    bad["seal"] = 1.5
    with pytest.raises(CertificationError):
        canonical_manifest(bad)


def test_typed_manifest_boundary_rejects_private_and_incomplete_nested_shapes() -> None:
    private = _manifest()
    private["subject"]["private_content"] = "DO_NOT_ECHO"
    with pytest.raises(CertificationError, match="MANIFEST_SCHEMA"):
        canonical_manifest(private)

    empty_projection = _manifest()
    empty_projection["privacy_mode"] = {}
    with pytest.raises(CertificationError, match="MANIFEST_SCHEMA"):
        canonical_manifest(empty_projection)

    missing_witness = _d2_manifest()
    del missing_witness["claims"][0]["payload"]["params"]["witness_bundle_digest"]
    with pytest.raises(CertificationError, match="MANIFEST_SCHEMA"):
        canonical_manifest(missing_witness)


def test_unsigned_manifest_parser_preserves_schema_stream_vocabulary() -> None:
    for state in ("legacy_unsealed", "not_applicable"):
        manifest = _manifest()
        stream = manifest["seal"]["expected_stream_projection"]["decision_stream"]
        del stream["root_commitment"]
        stream["chain_status"] = state
        copied, _ = canonical_manifest(manifest)
        assert (
            copied["seal"]["expected_stream_projection"]["decision_stream"][
                "chain_status"
            ]
            == state
        )


def test_ecdsa_p1363_is_low_s_and_high_s_is_rejected() -> None:
    key = ec.generate_private_key(ec.SECP256R1())
    _, manifest_bytes = canonical_manifest(_manifest())
    material = client_signed_material(manifest_bytes)
    signature = _sign_material(key, "ecdsa_p256_sha256", material)
    raw = decode_signature(signature)
    order = int("FFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551", 16)
    s = int.from_bytes(raw[32:], "big")
    assert 0 < s <= order // 2
    assert verify_signature(key.public_key(), "ecdsa_p256_sha256", material, signature)
    high_s = raw[:32] + (order - s).to_bytes(32, "big")
    with pytest.raises(VerificationError, match="SIGNATURE_HIGH_S"):
        verify_signature(
            key.public_key(),
            "ecdsa_p256_sha256",
            material,
            encode_signature(high_s),
        )
