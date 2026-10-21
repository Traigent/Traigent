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
from .bindings import key_id, verify_chain
from .envelope import LABEL_VERSION
from .hpke import hpke_open, hpke_seal, x25519_raw_public
from .trust_anchor import TrustAnchor
from .writer import _utc_now


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
    anchor: TrustAnchor,
    root_public_key: bytes,
    admin_binding: dict[str, Any],
    project_binding: dict[str, Any],
    member_binding: dict[str, Any],
    binding_list: dict[str, Any],
    retained_sequence: int,
    now: str | None = None,
    allow_machine_member_open: bool = False,
) -> X25519PrivateKey:
    """Open a member wrap, but only against a verified chain.

    Off by default: machine members are write-only unless an admin approved them. The
    chain root(pinned) -> admin -> project key -> member key is verified first, the wrap
    must be addressed to the verified member key and epoch, and the recovered project
    private key must be the private half of the VERIFIED project public key. A hostile
    service cannot substitute a project key it chose.
    """
    if not allow_machine_member_open:
        raise LabelCryptoError("member-wrap open is disabled (write-only member)")
    verify_chain(
        anchor.root_fingerprint,
        root_public_key,
        admin_binding,
        project_binding,
        member_binding,
        tenant_ref=anchor.tenant_ref,
        project_ref=anchor.project_ref,
        binding_list=binding_list,
        retained_sequence=retained_sequence,
        now=now or _utc_now(),
    )
    member_pub = x25519_raw_public(member_sk)
    if unb64u(member_binding["public_key"]) != member_pub:
        raise LabelCryptoError("member key is not the verified member key")
    if wrap["recipient_key_id"] != key_id(member_pub):
        raise LabelCryptoError("wrap is addressed to a different key")
    if wrap["epoch"] != project_binding["epoch"]:
        raise LabelCryptoError("wrap epoch is not the verified project epoch")
    raw = hpke_open(
        member_sk,
        unb64u(wrap["enc"]),
        member_wrap_info(
            anchor.tenant_ref,
            anchor.project_ref,
            wrap["epoch"],
            wrap["recipient_key_id"],
        ),
        b"",
        unb64u(wrap["ciphertext"]),
    )
    try:
        project_sk = X25519PrivateKey.from_private_bytes(raw)
    except ValueError as exc:
        raise LabelCryptoError("wrapped key is malformed") from exc
    if x25519_raw_public(project_sk) != unb64u(project_binding["public_key"]):
        raise LabelCryptoError("wrapped key does not match the verified project key")
    return project_sk
