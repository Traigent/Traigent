"""Encodings shared by the label service client: base64url, RFC 8785 JSON, HKDF."""

# Traceability: CONC-Layer-Core CONC-Quality-Security FUNC-SECURITY

from __future__ import annotations

import base64
import hashlib
import hmac
import json


class LabelCryptoError(ValueError):
    """Any verification, padding, decryption or format failure (fail closed).

    Messages are closed strings: they never echo ciphertext, keys, labels or refs.
    """


def b64u(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def unb64u(text: str) -> bytes:
    if not isinstance(text, str) or "=" in text or not text.isascii():
        raise LabelCryptoError("not unpadded base64url")
    try:
        raw = base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    except Exception as exc:  # noqa: BLE001
        raise LabelCryptoError("not base64url") from exc
    if b64u(raw) != text:
        raise LabelCryptoError("non-canonical base64url")
    return raw


def jcs(value: object) -> bytes:
    """RFC 8785 canonical JSON for the subset the contract uses.

    Supports str, int, bool, null, list and dict. Floats are refused: no AAD,
    info or binding field is fractional.
    """

    def enc(v: object) -> str:
        if v is None:
            return "null"
        if v is True:
            return "true"
        if v is False:
            return "false"
        if isinstance(v, int):
            return str(v)
        if isinstance(v, str):
            return json.dumps(v, ensure_ascii=False, separators=(",", ":"))
        if isinstance(v, list):
            return "[" + ",".join(enc(i) for i in v) + "]"
        if isinstance(v, dict):
            keys = sorted(v, key=lambda k: k.encode("utf-16-be"))
            return "{" + ",".join(enc(k) + ":" + enc(v[k]) for k in keys) + "}"
        raise LabelCryptoError("unsupported canonical-JSON type")

    return enc(value).encode("utf-8")


def hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    return hmac.new(salt or b"\x00" * 32, ikm, hashlib.sha256).digest()


def hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    out, block, i = b"", b"", 1
    while len(out) < length:
        block = hmac.new(prk, block + info + bytes([i]), hashlib.sha256).digest()
        out += block
        i += 1
    return out[:length]


def hkdf(ikm: bytes, info: bytes, length: int = 32, salt: bytes = b"") -> bytes:
    return hkdf_expand(hkdf_extract(salt, ikm), info, length)
