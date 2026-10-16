"""Label envelope: padding buckets, AAD (incl. revision), seal and open.

Wire contract: TraigentSchema ``docs/labels-v0.md`` (label_envelope_schema.json).
Packed form ``version(1) || key_epoch(4, BE) || enc(32) || ciphertext``; the JSON form
carries the same parts, binary parts as unpadded base64url.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Security FUNC-SECURITY

from __future__ import annotations

import struct
from typing import Any

from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from ._codec import LabelCryptoError, b64u, jcs, unb64u
from .hpke import hpke_open, hpke_seal

LABEL_VERSION = 1
BUCKETS = (32, 64, 128, 256)
MAX_LABEL_BYTES = 255
HPKE_INFO_LABEL = b"traigent.label.v1"
PURPOSES = ("label", "knob", "choice", "metric", "template")


def pad(text: str) -> bytes:
    """UTF-8, ISO/IEC 7816-4 padding (0x80 then zeros) to the smallest bucket that fits."""
    raw = text.encode("utf-8")
    if len(raw) > MAX_LABEL_BYTES:
        raise LabelCryptoError("label longer than 255 UTF-8 bytes")
    for bucket in BUCKETS:
        if len(raw) + 1 <= bucket:
            return raw + b"\x80" + b"\x00" * (bucket - len(raw) - 1)
    raise LabelCryptoError("unreachable")  # pragma: no cover


def unpad(padded: bytes) -> str:
    if len(padded) not in BUCKETS:
        raise LabelCryptoError("padded length is not a bucket")
    stripped = padded.rstrip(b"\x00")
    if not stripped or stripped[-1] != 0x80:
        raise LabelCryptoError("malformed padding")
    body = stripped[:-1]
    if len(padded) != next(b for b in BUCKETS if len(body) + 1 <= b):
        raise LabelCryptoError("non-minimal bucket")
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LabelCryptoError("label is not UTF-8") from exc


def label_aad(
    tenant_ref: str,
    project_ref: str,
    purpose: str,
    object_ref: str,
    key_epoch: int,
    revision: int,
) -> bytes:
    """RFC 8785 canonical AAD. Not transmitted: both sides rebuild it."""
    if purpose not in PURPOSES:
        raise LabelCryptoError("unknown purpose")
    return jcs(
        {
            "schema_version": LABEL_VERSION,
            "tenant_ref": tenant_ref,
            "project_ref": project_ref,
            "purpose": purpose,
            "object_ref": object_ref,
            "key_epoch": key_epoch,
            "revision": revision,
        }
    )


def pack_envelope(version: int, epoch: int, enc: bytes, ciphertext: bytes) -> bytes:
    return bytes([version]) + struct.pack(">I", epoch) + enc + ciphertext


def unpack_envelope(blob: bytes) -> tuple[int, int, bytes, bytes]:
    if len(blob) < 1 + 4 + 32 + 16 + min(BUCKETS):
        raise LabelCryptoError("envelope too short")
    version, epoch = blob[0], struct.unpack(">I", blob[1:5])[0]
    if version != LABEL_VERSION:
        raise LabelCryptoError("unsupported envelope version")
    return version, epoch, blob[5:37], blob[37:]


def seal_label(
    pk_project: bytes,
    text: str,
    *,
    tenant_ref: str,
    project_ref: str,
    purpose: str,
    object_ref: str,
    key_epoch: int,
    revision: int,
    ikm_e: bytes | None = None,
) -> dict[str, Any]:
    """Seal ``text`` to ``pk_project`` under the NEW ``revision`` (``base_revision + 1``).

    Low level: it does not verify that ``pk_project`` is bound to the pinned root. Use
    ``LabelWriter`` (writer.py), which verifies the binding chain first.
    """
    if revision < 1:
        raise LabelCryptoError("revision must be at least 1")
    enc, ct = hpke_seal(
        pk_project,
        HPKE_INFO_LABEL,
        label_aad(tenant_ref, project_ref, purpose, object_ref, key_epoch, revision),
        pad(text),
        ikm_e=ikm_e,
    )
    return {
        "version": LABEL_VERSION,
        "key_epoch": key_epoch,
        "enc": b64u(enc),
        "ciphertext": b64u(ct),
    }


def open_label(
    sk_project: X25519PrivateKey,
    envelope: dict[str, Any],
    *,
    tenant_ref: str,
    project_ref: str,
    purpose: str,
    object_ref: str,
    revision: int,
) -> str:
    """Open under the revision the caller REQUESTED (a replay under another revision fails)."""
    if envelope.get("version") != LABEL_VERSION:
        raise LabelCryptoError("unsupported envelope version")
    epoch = envelope["key_epoch"]
    padded = hpke_open(
        sk_project,
        unb64u(envelope["enc"]),
        HPKE_INFO_LABEL,
        label_aad(tenant_ref, project_ref, purpose, object_ref, epoch, revision),
        unb64u(envelope["ciphertext"]),
    )
    return unpad(padded)


class RevisionGuard:
    """Per-object retained maximum revision: rejects a revision below it (rollback)."""

    def __init__(self) -> None:
        self._max: dict[str, int] = {}

    def check(self, object_ref: str, revision: int) -> None:
        if revision < self._max.get(object_ref, 0):
            raise LabelCryptoError("stale revision below retained maximum")

    def accept(self, object_ref: str, revision: int) -> None:
        self.check(object_ref, revision)
        self._max[object_ref] = max(revision, self._max.get(object_ref, 0))
