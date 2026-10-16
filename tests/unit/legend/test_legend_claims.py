"""Offline Legend claim-statement verifier against the Schema golden vectors (2244144)."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, ed25519
from cryptography.hazmat.primitives.asymmetric.utils import (
    decode_dss_signature,
    encode_dss_signature,
)

from traigent.legend.claims import (
    LEGEND_CLAIM_VERIFICATION_ERROR_CODES,
    LEGEND_SIGNATURE_DOMAIN,
    LegendClaimVerificationError,
    TrustedLegendKey,
    canonical_subject_json,
    legend_signing_material,
    verify_legend_claim_statement,
)
from traigent.utils import fp2

VECTORS = Path(__file__).parent / "vectors" / "legend_claim_statement_v0_vectors.json"
VECTORS_SHA256 = "46f1c2b3c0c748261adbc4a583ed57fe9f38bc9b4251f7714393627c21c35589"
_ORDER = int("FFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551", 16)


def _vectors():
    return json.loads(VECTORS.read_bytes())


def _trusted(v):
    pub = ed25519.Ed25519PublicKey.from_public_bytes(
        base64.b64decode(v["test_public_key_raw_b64"])
    )
    return {v["trusted_key_id"]: TrustedLegendKey("ed25519", pub)}


def _code(stmt, trusted) -> str:
    try:
        verify_legend_claim_statement(stmt, trusted)
    except LegendClaimVerificationError as e:
        assert e.code in LEGEND_CLAIM_VERIFICATION_ERROR_CODES
        return e.code
    return "VERIFIED"


def test_vector_file_is_pinned() -> None:
    assert hashlib.sha256(VECTORS.read_bytes()).hexdigest() == VECTORS_SHA256


def test_canonicalization_vectors_match_fp2() -> None:
    v = _vectors()
    assert v["digest_alg"] == "traigent-fp2-sha256"
    for case in v["canonicalization"]:
        value = json.loads(case["input_json"])
        assert fp2.canonicalize(value) == case["canonical"], case["input_json"]
        assert fp2.digest(value) == case["digest"], case["input_json"]


def test_valid_vectors_verify_and_canonical_bytes_match() -> None:
    v = _vectors()
    trusted = _trusted(v)
    cases = {"valid": v["valid"], **v["valid_evaluated"]}
    for name, case in cases.items():
        assert _code(case["statement"], trusted) == case["expect"], name
    top = verify_legend_claim_statement(v["valid"]["statement"], trusted)
    assert canonical_subject_json(top.subject) == v["canonical_subject_json"]
    assert top.canonical_bytes == legend_signing_material(top.subject)
    assert top.canonical_bytes.startswith(LEGEND_SIGNATURE_DOMAIN + b"\x00")


def test_negative_vectors_refused_with_expected_code() -> None:
    v = _vectors()
    trusted = _trusted(v)
    for name, case in v["negative"].items():
        assert _code(case["statement"], trusted) == case["expect"], name


def test_structure_and_unpinned_refusals() -> None:
    v = _vectors()
    trusted = _trusted(v)
    good = v["valid"]["statement"]
    assert _code("x", trusted) == "STRUCTURE"
    assert _code({"subject": good["subject"]}, trusted) == "STRUCTURE"
    assert _code(good, {}) == "UNKNOWN_KEY"
    wrong_alg = {v["trusted_key_id"]: TrustedLegendKey("ecdsa_p256_sha256", None)}
    assert _code(good, wrong_alg) == "ALGORITHM"
    bad_sig = copy.deepcopy(good)
    bad_sig["signature"]["value"] = "!" * 88
    assert _code(bad_sig, trusted) == "SCHEMA"
    # A key object of the wrong type for its pinned algorithm is refused.
    mismatched = {v["trusted_key_id"]: TrustedLegendKey("ed25519", object())}
    assert _code(good, mismatched) == "KEY_ALGORITHM"


def test_ecdsa_p256_low_s_enforced() -> None:
    v = _vectors()
    good = v["valid"]["statement"]
    key = ec.generate_private_key(ec.SECP256R1())
    trusted = {"p256-key": TrustedLegendKey("ecdsa_p256_sha256", key.public_key())}
    material = legend_signing_material(good["subject"])
    r, s = decode_dss_signature(key.sign(material, ec.ECDSA(hashes.SHA256())))
    low = min(s, _ORDER - s)

    def stmt(s_val: int) -> dict:
        out = copy.deepcopy(good)
        out["signature"].update(
            algorithm="ecdsa_p256_sha256",
            key_id="p256-key",
            value=base64.b64encode(
                r.to_bytes(32, "big") + s_val.to_bytes(32, "big")
            ).decode(),
        )
        return out

    assert _code(stmt(low), trusted) == "VERIFIED"
    assert _code(stmt(_ORDER - low), trusted) == "SIGNATURE_HIGH_S"
    assert _code(stmt(0), trusted) == "SIGNATURE_SCALAR"
    other = {
        "p256-key": TrustedLegendKey(
            "ecdsa_p256_sha256", ec.generate_private_key(ec.SECP256R1()).public_key()
        )
    }
    assert _code(stmt(low), other) == "SIGNATURE_INVALID"
    assert encode_dss_signature(r, low)  # sanity: DER re-encode works
