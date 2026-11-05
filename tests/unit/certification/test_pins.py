"""Issuer-algorithm handling of the trusted pin file (ed25519 and ECDSA P-256)."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519

from traigent.certification import pins as pins_module

PublicKey = ec.EllipticCurvePublicKey | ed25519.Ed25519PublicKey


def _der_b64(public: PublicKey) -> str:
    return base64.b64encode(
        public.public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    ).decode()


def _document(algorithm: str, public: PublicKey) -> dict:
    return {
        "project_ref": "prj_test",
        "issuer_algorithm": algorithm,
        "issuer_public_key_der_b64": _der_b64(public),
        "issuer_key_ref": "issuer:AAAAAAAAAAAA",
        "trust_ring_ref": "ring:BBBBBBBBBBBB",
        "compiler_register_versions": [["register", "1"]],
        "g1_verifier_id": "verifier",
        "g1_verifier_version": "1",
        "disclosure_profile_id": "profile",
    }


def _write(tmp_path: Path, document: dict) -> Path:
    source = tmp_path / "src.json"
    source.write_text(json.dumps(document))
    return source


def _p256() -> PublicKey:
    return ec.generate_private_key(ec.SECP256R1()).public_key()


def _p384() -> PublicKey:
    return ec.generate_private_key(ec.SECP384R1()).public_key()


def _ed() -> PublicKey:
    return ed25519.Ed25519PrivateKey.generate().public_key()


def test_p256_issuer_pin_is_accepted(tmp_path: Path) -> None:
    pins = pins_module.write_pins(
        _write(tmp_path, _document("ecdsa_p256_sha256", _p256())), tmp_path / "p.json"
    )
    assert pins.issuer_algorithm == "ecdsa_p256_sha256"
    assert isinstance(pins.issuer_public_key, ec.EllipticCurvePublicKey)
    reloaded = pins_module.load_pins(tmp_path / "p.json")
    assert isinstance(reloaded.issuer_public_key, ec.EllipticCurvePublicKey)


def test_ed25519_issuer_pin_still_accepted(tmp_path: Path) -> None:
    pins = pins_module.write_pins(
        _write(tmp_path, _document("ed25519", _ed())), tmp_path / "p.json"
    )
    assert isinstance(pins.issuer_public_key, ed25519.Ed25519PublicKey)


@pytest.mark.parametrize(
    "algorithm,make_public",
    [
        ("ecdsa_p256_sha256", _ed),
        ("ed25519", _p256),
        ("ecdsa_p256_sha256", _p384),
        ("rsa", _p256),
    ],
)
def test_algorithm_key_mismatch_or_unknown_is_refused(
    tmp_path: Path, algorithm: str, make_public: Callable[[], PublicKey]
) -> None:
    with pytest.raises(pins_module.PinError) as caught:
        pins_module.write_pins(
            _write(tmp_path, _document(algorithm, make_public())), tmp_path / "p.json"
        )
    assert caught.value.code == "PIN_INVALID"
