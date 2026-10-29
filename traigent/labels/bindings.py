"""Signed key-binding chain: pinned root -> admin identity -> project key -> member key.

Every client verifies the chain BEFORE encrypting or wrapping (design v2.2 section 3.6).
Mirrors ``verify_chain`` in TraigentSchema ``scripts/labels_reference.py``; the
conformance vectors are in ``tests/unit/labels``.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Security FUNC-SECURITY

from __future__ import annotations

import hashlib
import hmac
from datetime import datetime
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from ._codec import LabelCryptoError, b64u, jcs, unb64u

BINDING_DOMAIN = b"traigent.label.binding.v1\x00"


def key_id(public_key: bytes) -> str:
    """First 16 bytes of SHA-256(raw public key), base64url (22 chars)."""
    return b64u(hashlib.sha256(public_key).digest()[:16])


def root_fingerprint(public_key: bytes) -> str:
    """Full SHA-256 of the raw root public key, lowercase hex: the value customers pin."""
    return hashlib.sha256(public_key).hexdigest()


def sign_body(sk: Ed25519PrivateKey, body: dict[str, Any]) -> str:
    return b64u(sk.sign(BINDING_DOMAIN + jcs(body)))


def verify_body(public_key: bytes, body: dict[str, Any], signature: str) -> None:
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            unb64u(signature), BINDING_DOMAIN + jcs(body)
        )
    except InvalidSignature as exc:
        raise LabelCryptoError("bad signature") from exc
    except ValueError as exc:
        raise LabelCryptoError("invalid public key") from exc


def make_binding(
    signer_sk: Ed25519PrivateKey, signer_pub: bytes, body: dict[str, Any]
) -> dict[str, Any]:
    body = dict(body, signer_key_id=key_id(signer_pub))
    return dict(body, signature=sign_body(signer_sk, body))


def verify_binding(signer_pub: bytes, binding: dict[str, Any]) -> None:
    body = {k: v for k, v in binding.items() if k != "signature"}
    if body.get("signer_key_id") != key_id(signer_pub):
        raise LabelCryptoError("signer key id mismatch")
    if "signature" not in binding:
        raise LabelCryptoError("unsigned binding")
    verify_body(signer_pub, body, binding["signature"])


def _parse_ts(text: Any) -> datetime:
    try:
        ts = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError as exc:
        raise LabelCryptoError("bad timestamp") from exc
    if ts.tzinfo is None:
        raise LabelCryptoError("timestamp has no timezone")
    return ts


def verify_binding_list(
    root_pub: bytes,
    binding_list: dict[str, Any],
    *,
    tenant_ref: str,
    retained_sequence: int,
) -> set[str]:
    """Verify a root-signed KeyBindingList; return its revoked key ids. Fail closed.

    ``retained_sequence`` is the highest list sequence this client has already seen; a
    lower one is a rollback (replay of a list from before a revocation).
    """
    verify_binding(root_pub, binding_list)
    if binding_list["tenant_ref"] != tenant_ref:
        raise LabelCryptoError("tenant mismatch")
    if binding_list["sequence"] < retained_sequence:
        raise LabelCryptoError("binding list sequence below retained sequence")
    return set(binding_list["revoked_key_ids"])


def verify_chain(
    pinned_root_fingerprint: str,
    root_pub: bytes,
    admin_binding: dict[str, Any],
    project_binding: dict[str, Any],
    member_binding: dict[str, Any] | None,
    *,
    tenant_ref: str,
    project_ref: str,
    binding_list: dict[str, Any],
    retained_sequence: int,
    now: str,
) -> None:
    """root (pinned) -> admin identity -> project key (epoch) [-> member key]. Fail closed.

    ``member_binding`` may be ``None`` for a writer, which only needs the project key
    (it seals to the project public key and wraps to nobody). Any revoked key in the
    chain, any list below ``retained_sequence`` and any ``not_before`` after ``now`` is
    rejected.
    """
    if not hmac.compare_digest(
        root_fingerprint(root_pub), str(pinned_root_fingerprint).lower()
    ):
        raise LabelCryptoError("root key does not match pinned fingerprint")
    revoked = verify_binding_list(
        root_pub,
        binding_list,
        tenant_ref=tenant_ref,
        retained_sequence=retained_sequence,
    )
    verify_binding(root_pub, admin_binding)
    if admin_binding["purpose"] != "admin_identity":
        raise LabelCryptoError("wrong purpose for admin binding")
    admin_pub = unb64u(admin_binding["public_key"])
    pairs = [(project_binding, "project_key")]
    chain = [admin_binding, project_binding]
    if member_binding is not None:
        pairs.append((member_binding, "member_key"))
        chain.append(member_binding)
    for binding, purpose in pairs:
        verify_binding(admin_pub, binding)
        if binding["purpose"] != purpose:
            raise LabelCryptoError(f"wrong purpose, expected {purpose}")
    for binding in chain:
        if binding["tenant_ref"] != tenant_ref:
            raise LabelCryptoError("tenant mismatch")
    for binding in chain[1:]:
        if binding["project_ref"] != project_ref:
            raise LabelCryptoError("project mismatch")
    if (
        member_binding is not None
        and member_binding["epoch"] != project_binding["epoch"]
    ):
        raise LabelCryptoError("epoch mismatch")
    _verify_chain_activity(chain, revoked, now)


def _verify_chain_activity(
    chain: list[dict[str, Any]], revoked: set[str], now: str
) -> None:
    current = _parse_ts(now)
    for binding in chain:
        if binding["key_id"] in revoked or binding["signer_key_id"] in revoked:
            raise LabelCryptoError("revoked key in chain")
        if _parse_ts(binding["not_before"]) > current:
            raise LabelCryptoError("binding not yet active")
    for binding in chain:
        if binding["key_id"] != key_id(unb64u(binding["public_key"])):
            raise LabelCryptoError("key id does not match public key")
