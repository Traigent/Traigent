"""Content-identity purpose keys: received from the Backend, never derived here.

Spec: TraigentSchema ``docs/identity/content-identity-v1.md`` sections 3, 12
and 16 (ruling D1). Each tenant has one random 32-byte master per key version;
the Backend holds it in KMS/Vault and it never leaves the Backend. An
authenticated SDK session receives only::

    {tenant_id, kid, example_id_key, example_version_key}

(the purpose-key issuance endpoint the Backend adds in milestone M3). This
module holds exactly that grant, in memory only, and nothing else.

**Fail closed.** Until a grant has been installed, :func:`get_content_identity_keys`
returns ``None`` and every content-identity producer in the SDK emits no
``ex1``/``exv1``/``msr1`` identifiers at all. The SDK never invents, derives or
defaults a key: a made-up key would mint ids that look authoritative and that
no other party can reproduce.

Key material never appears in ``repr``, logs or exception messages.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from traigent.identity.content_identity import ContentIdentityError, TenantIdentityKeys

__all__ = [
    "ContentIdentityKeys",
    "clear_content_identity_keys",
    "get_content_identity_keys",
    "set_content_identity_keys",
]

_TENANT_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,128}")
_KID_RE = re.compile(r"k[0-9a-f]{16}")
_KEY_HEX_RE = re.compile(r"[0-9a-f]{64}")

#: Exactly PurposeKeyGrantV1's fields (TraigentSchema
#: ``datasets/purpose_key_grant_v1_schema.json``, ``additionalProperties: false``).
_GRANT_FIELDS = frozenset(
    {"tenant_id", "kid", "example_id_key", "example_version_key", "encoding"}
)


@dataclass(frozen=True)
class ContentIdentityKeys:
    """One tenant's derived content-identity purpose keys, as the Backend grants them.

    ``tenant_id`` is the Backend's exact tenant id string (never folded or
    reformatted: it is bound into the HKDF ``info`` that produced these keys).
    ``kid`` is public; the two keys are secret and hidden from ``repr``.
    """

    tenant_id: str
    kid: str
    example_id_key: bytes = field(repr=False)
    example_version_key: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.tenant_id, str) or not _TENANT_ID_RE.fullmatch(
            self.tenant_id
        ):
            raise ContentIdentityError(
                "tenant_id must fully match [A-Za-z0-9_-]{1,128}"
            )
        if not isinstance(self.kid, str) or not _KID_RE.fullmatch(self.kid):
            raise ContentIdentityError("kid must fully match k[0-9a-f]{16}")
        for name in ("example_id_key", "example_version_key"):
            value = getattr(self, name)
            if not isinstance(value, bytes) or len(value) != 32:
                raise ContentIdentityError(f"{name} must be exactly 32 bytes")
        if self.example_id_key == self.example_version_key:
            # The two purposes are derived with distinct HKDF infos; equal keys
            # mean the grant was assembled wrongly (e.g. one key sent twice).
            raise ContentIdentityError("the two purpose keys must differ")

    @classmethod
    def from_grant(cls, grant: Mapping[str, Any]) -> ContentIdentityKeys:
        """Build keys from a Backend purpose-key grant.

        Validated against ``PurposeKeyGrantV1`` (TraigentSchema
        ``datasets/purpose_key_grant_v1_schema.json``, spec section 18)::

            {"tenant_id": "...", "kid": "k<16 hex>",
             "example_id_key": "<64 lowercase hex>",
             "example_version_key": "<64 lowercase hex>",
             "encoding": "hex"}

        Closed shape: an unknown/extra field is rejected. ``encoding`` must be
        exactly ``"hex"`` -- any other value (or a missing ``encoding``) is
        rejected rather than guessed. ``tenant_id`` / ``kid`` format and the
        two purpose keys' length/casing are validated here and in
        ``__post_init__``; anything malformed fails closed with
        :class:`ContentIdentityError` so the run degrades to no content
        identity (``key_status`` unavailable) rather than minting ids from a
        grant that cannot be trusted.
        """
        if not isinstance(grant, Mapping):
            raise ContentIdentityError("purpose-key grant must be a mapping")
        extra = set(grant.keys()) - _GRANT_FIELDS
        if extra:
            raise ContentIdentityError(
                f"purpose-key grant has unknown field(s): {sorted(extra)}"
            )
        if grant.get("encoding") != "hex":
            raise ContentIdentityError('purpose-key grant encoding must be "hex"')
        keys: dict[str, bytes] = {}
        for name in ("example_id_key", "example_version_key"):
            value = grant.get(name)
            if not isinstance(value, str) or not _KEY_HEX_RE.fullmatch(value):
                raise ContentIdentityError(
                    f"{name} must be 64 lowercase hex characters"
                )
            keys[name] = bytes.fromhex(value)
        return cls(
            tenant_id=grant.get("tenant_id"),  # type: ignore[arg-type]  # validated in __post_init__
            kid=grant.get("kid"),  # type: ignore[arg-type]
            example_id_key=keys["example_id_key"],
            example_version_key=keys["example_version_key"],
        )

    def as_tenant_keys(self) -> TenantIdentityKeys:
        """The primitive-level key object the identity functions accept."""
        return TenantIdentityKeys(
            key_id=self.kid,
            example_id_key=self.example_id_key,
            example_version_key=self.example_version_key,
        )


_lock = threading.Lock()
_active: ContentIdentityKeys | None = None


def set_content_identity_keys(keys: ContentIdentityKeys | None) -> None:
    """Install (or, with ``None``, remove) the process's purpose-key grant."""
    global _active
    if keys is not None and not isinstance(keys, ContentIdentityKeys):
        raise ContentIdentityError("keys must be ContentIdentityKeys")
    with _lock:
        _active = keys


def get_content_identity_keys() -> ContentIdentityKeys | None:
    """The installed grant, or ``None`` -- in which case NO content ids are minted."""
    with _lock:
        return _active


def clear_content_identity_keys() -> None:
    """Forget the installed grant (e.g. on logout or tenant switch)."""
    set_content_identity_keys(None)
