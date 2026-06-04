"""Canonical hashing for certificate freshness contexts (RFC 0001 §3.5).

``canonical_hash`` implements the TVL canonicalization profile: sorted-key
compact JSON over NFC-normalized strings with the RFC's value restrictions
(finite numbers only, ``-0.0`` folded to ``0.0``, duplicate keys rejected at
parse time by the loaders). The profile is versioned via
``CTX_SCHEMA_VERSION``, which is itself part of every hashed context, so any
future change to this module is automatically staleness-inducing for
previously issued certificates.

Two conformant implementations hashing the same context MUST produce the
same digest — the cross-implementation known-answer fixtures live in the tvl
repo's conformance suite.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["CTX_SCHEMA_VERSION", "CanonicalizationError", "canonical_hash"]

#: Version of the freshness-context schema AND this canonicalization profile.
CTX_SCHEMA_VERSION = 1


class CanonicalizationError(ValueError):
    """Raised when a value cannot be canonically hashed (non-finite numbers,
    duplicate keys, unsupported types)."""


def _normalize(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise CanonicalizationError("non-finite numbers are rejected, never hashed")
        if value == 0.0:
            return 0.0  # fold -0.0
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalizationError("mapping keys must be strings")
            norm_key = unicodedata.normalize("NFC", key)
            if norm_key in normalized:
                raise CanonicalizationError(f"duplicate key after NFC: {norm_key!r}")
            normalized[norm_key] = _normalize(item)
        return normalized
    if isinstance(value, Sequence):
        return [_normalize(item) for item in value]
    raise CanonicalizationError(
        f"unsupported type for canonical hashing: {type(value).__name__}"
    )


def canonical_hash(value: Any) -> str:
    """SHA-256 over the canonical serialization of ``value``."""
    payload = json.dumps(
        _normalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
