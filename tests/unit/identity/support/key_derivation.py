"""Master-key derivation for the conformance tests ONLY -- never shipped.

The SDK never holds a tenant master (content-identity-v1 section 12, ruling
D1): production code receives derived purpose keys from the Backend. The
vectors pin the derivation, so this port of the TraigentSchema reference
(``traigent_schema/example_identity.py`` at ``024d55b0``) lives here, next to
the tests that exercise it (same arrangement as traigent-js
``tests/unit/identity/support/test-key-derivation.ts``).
"""

from __future__ import annotations

import hashlib
import hmac
import re

from traigent.identity.content_identity import (
    DOMAIN_EXAMPLE_ID,
    DOMAIN_EXAMPLE_VERSION,
    DOMAIN_KEY_ID,
    HKDF_SALT,
    ContentIdentityError,
    TenantIdentityKeys,
)

_TENANT_MASTER_LENGTH = 32
_TENANT_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,128}")
_KEY_ID_BYTES = 8


def hkdf_sha256(ikm: bytes, *, salt: bytes, info: bytes, length: int) -> bytes:
    """RFC 5869 HKDF with SHA-256 (Extract then Expand)."""
    if not 0 < length <= 255 * 32:
        raise ContentIdentityError("HKDF output length out of range")
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    okm = b""
    block = b""
    counter = 1
    while len(okm) < length:
        block = hmac.new(prk, block + info + bytes([counter]), hashlib.sha256).digest()
        okm += block
        counter += 1
    return okm[:length]


def hkdf_info(domain: str, tenant_id: str) -> bytes:
    """HKDF ``info`` = UTF8(domain) || 0x00 || UTF8(tenant_id)."""
    if not isinstance(tenant_id, str) or not _TENANT_ID_RE.fullmatch(tenant_id):
        raise ContentIdentityError("tenant_id must fully match [A-Za-z0-9_-]{1,128}")
    return domain.encode("utf-8") + b"\x00" + tenant_id.encode("ascii")


def derive_tenant_keys(tenant_master: bytes, tenant_id: str) -> TenantIdentityKeys:
    """Derive the v1 purpose keys and key id for one tenant.

    ``tenant_id`` is the Backend's exact tenant id string. Binding it into
    every HKDF ``info`` means a master secret reused by mistake across two
    tenants still yields unrelated keys and unlinkable ids.
    """
    if not isinstance(tenant_master, (bytes, bytearray)):
        raise ContentIdentityError("tenant master secret must be bytes")
    if len(tenant_master) != _TENANT_MASTER_LENGTH:
        raise ContentIdentityError("tenant master secret must be exactly 32 bytes")
    ikm = bytes(tenant_master)
    hkdf_info(DOMAIN_KEY_ID, tenant_id)  # validate before deriving anything

    def expand(domain: str, length: int) -> bytes:
        return hkdf_sha256(
            ikm, salt=HKDF_SALT, info=hkdf_info(domain, tenant_id), length=length
        )

    return TenantIdentityKeys(
        key_id="k" + expand(DOMAIN_KEY_ID, _KEY_ID_BYTES).hex(),
        example_id_key=expand(DOMAIN_EXAMPLE_ID, 32),
        example_version_key=expand(DOMAIN_EXAMPLE_VERSION, 32),
    )
