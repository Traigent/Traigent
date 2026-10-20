"""HPKE (RFC 9180) base mode, single-shot.

Suite: DHKEM(X25519, HKDF-SHA256) / HKDF-SHA256 / AES-256-GCM (0x0020, 0x0001, 0x0002).
Built on ``cryptography`` primitives (the pinned release has no HPKE module). The KEM,
key schedule and AEAD framing are proven against RFC 9180 Appendix A.1 and A.2 in
``tests/unit/labels/test_hpke_rfc9180.py``.

Base mode gives no sender authentication and no replay protection: writer identity
comes from the authenticated API call, freshness from the revision bound into the AAD.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Security FUNC-SECURITY

from __future__ import annotations

import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from ._codec import LabelCryptoError, hkdf_expand, hkdf_extract

KEM_ID = 0x0020
KDF_ID = 0x0001
AEAD_AES128 = 0x0001
AEAD_AES256 = 0x0002
AEAD_CHACHA = 0x0003
_AEAD = {
    AEAD_AES128: (16, AESGCM),
    AEAD_AES256: (32, AESGCM),
    AEAD_CHACHA: (32, ChaCha20Poly1305),
}
_NN = 12
_KEM_SUITE = b"KEM" + KEM_ID.to_bytes(2, "big")


def _le(suite: bytes, prk: bytes, label: bytes, info: bytes, length: int) -> bytes:
    return hkdf_expand(
        prk, length.to_bytes(2, "big") + b"HPKE-v1" + suite + label + info, length
    )


def _lx(suite: bytes, salt: bytes, label: bytes, ikm: bytes) -> bytes:
    return hkdf_extract(salt, b"HPKE-v1" + suite + label + ikm)


def x25519_raw_public(sk: X25519PrivateKey) -> bytes:
    return bytes(sk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw))


def derive_keypair(ikm: bytes) -> tuple[X25519PrivateKey, bytes]:
    """RFC 9180 DeriveKeyPair for X25519. ``ikm`` must be CSPRNG output or a test seed."""
    prk = _lx(_KEM_SUITE, b"", b"dkp_prk", ikm)
    sk = X25519PrivateKey.from_private_bytes(_le(_KEM_SUITE, prk, b"sk", b"", 32))
    return sk, x25519_raw_public(sk)


def _dh(sk: X25519PrivateKey, pk: bytes) -> bytes:
    try:
        shared = sk.exchange(X25519PublicKey.from_public_bytes(pk))
    except ValueError as exc:
        raise LabelCryptoError("invalid public key") from exc
    if shared == b"\x00" * 32:
        raise LabelCryptoError("low-order point")
    return bytes(shared)


def _extract_and_expand(dh: bytes, kem_context: bytes) -> bytes:
    eae_prk = _lx(_KEM_SUITE, b"", b"eae_prk", dh)
    return _le(_KEM_SUITE, eae_prk, b"shared_secret", kem_context, 32)


def _key_schedule(
    aead_id: int, shared_secret: bytes, info: bytes
) -> tuple[bytes, bytes]:
    suite = (
        b"HPKE"
        + KEM_ID.to_bytes(2, "big")
        + KDF_ID.to_bytes(2, "big")
        + aead_id.to_bytes(2, "big")
    )
    nk = _AEAD[aead_id][0]
    psk_id_hash = _lx(suite, b"", b"psk_id_hash", b"")
    info_hash = _lx(suite, b"", b"info_hash", info)
    context = b"\x00" + psk_id_hash + info_hash
    secret = _lx(suite, shared_secret, b"secret", b"")
    return (
        _le(suite, secret, b"key", context, nk),
        _le(suite, secret, b"base_nonce", context, _NN),
    )


def hpke_seal(
    pk_r: bytes,
    info: bytes,
    aad: bytes,
    plaintext: bytes,
    *,
    ikm_e: bytes | None = None,
    aead_id: int = AEAD_AES256,
) -> tuple[bytes, bytes]:
    """Single-shot base-mode seal; returns ``(enc, ciphertext)``.

    ``ikm_e`` is for conformance vectors only. Production callers leave it ``None``
    and a fresh 32 CSPRNG bytes are drawn.
    """
    if ikm_e is None:
        ikm_e = os.urandom(32)
    sk_e, enc = derive_keypair(ikm_e)
    shared = _extract_and_expand(_dh(sk_e, pk_r), enc + pk_r)
    key, nonce = _key_schedule(aead_id, shared, info)
    return enc, bytes(_AEAD[aead_id][1](key).encrypt(nonce, plaintext, aad))


def hpke_open(
    sk_r: X25519PrivateKey,
    enc: bytes,
    info: bytes,
    aad: bytes,
    ciphertext: bytes,
    aead_id: int = AEAD_AES256,
) -> bytes:
    pk_r = x25519_raw_public(sk_r)
    shared = _extract_and_expand(_dh(sk_r, enc), enc + pk_r)
    key, nonce = _key_schedule(aead_id, shared, info)
    try:
        return bytes(_AEAD[aead_id][1](key).decrypt(nonce, ciphertext, aad))
    except InvalidTag as exc:
        raise LabelCryptoError("authentication failed") from exc
