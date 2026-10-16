"""Flow (b): project private key sealed (HPKE) to a member's VERIFIED public key.

The SDK writer path never needs this. It exists for machine members that an admin
explicitly approved for read access: ``open_member_wrap`` requires
``allow_machine_member_open=True`` so no code path decrypts a wrap by accident.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Security FUNC-SECURITY

from __future__ import annotations

from typing import Any

from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from ._codec import LabelCryptoError, b64u, jcs, unb64u
from .bindings import key_id
from .envelope import LABEL_VERSION
from .hpke import hpke_open, hpke_seal, x25519_raw_public


def member_wrap_info(
    tenant_ref: str, project_ref: str, epoch: int, recipient_key_id: str
) -> bytes:
    return jcs(
        {
            "tenant_ref": tenant_ref,
            "project_ref": project_ref,
            "epoch": epoch,
            "recipient_key_id": recipient_key_id,
            "purpose": "member_wrap",
            "version": LABEL_VERSION,
        }
    )


def seal_member_wrap(
    project_private_raw: bytes,
    member_pk: bytes,
    *,
    tenant_ref: str,
    project_ref: str,
    epoch: int,
    ikm_e: bytes | None = None,
) -> dict[str, Any]:
    """Seal an epoch's project private key to a member public key.

    The caller must already have verified ``member_pk`` through the binding chain.
    """
    rid = key_id(member_pk)
    enc, ct = hpke_seal(
        member_pk,
        member_wrap_info(tenant_ref, project_ref, epoch, rid),
        b"",
        project_private_raw,
        ikm_e=ikm_e,
    )
    return {
        "version": LABEL_VERSION,
        "epoch": epoch,
        "recipient_key_id": rid,
        "enc": b64u(enc),
        "ciphertext": b64u(ct),
    }


def open_member_wrap(
    member_sk: X25519PrivateKey,
    wrap: dict[str, Any],
    *,
    tenant_ref: str,
    project_ref: str,
    allow_machine_member_open: bool = False,
) -> X25519PrivateKey:
    """Open a member wrap. Off by default: machine members are write-only unless approved."""
    if not allow_machine_member_open:
        raise LabelCryptoError("member-wrap open is disabled (write-only member)")
    if wrap["recipient_key_id"] != key_id(x25519_raw_public(member_sk)):
        raise LabelCryptoError("wrap is addressed to a different key")
    raw = hpke_open(
        member_sk,
        unb64u(wrap["enc"]),
        member_wrap_info(
            tenant_ref, project_ref, wrap["epoch"], wrap["recipient_key_id"]
        ),
        b"",
        unb64u(wrap["ciphertext"]),
    )
    return X25519PrivateKey.from_private_bytes(raw)
