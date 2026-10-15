"""Closed request bodies (``legend_requests_v0.json``). Only enums, digests and opaque refs.

Idempotency (design note section 3.4): ``idempotency_key`` is a FRESH random key per
logical request and is reused unchanged on every retry of it; it is never derived from
content. ``request_digest`` is ``sha256:`` + hex SHA-256(domain || 0x00 || jcs_v1(body))
over the body excluding ``idempotency_key`` and ``request_digest``.
"""

from __future__ import annotations

import hashlib
import secrets
from typing import Any

from traigent.identity.content_identity import canonical_bytes
from traigent.legend import _contract as c
from traigent.legend.errors import LegendError
from traigent.legend.pillars import Pillar

_EXCLUDED_FROM_DIGEST = ("idempotency_key", "request_digest")


def new_idempotency_key() -> str:
    """A fresh opaque ``idk_<32 hex>`` key (128 random bits)."""
    return "idk_" + secrets.token_hex(16)


def request_digest(body: dict[str, Any]) -> str:
    """The retry-stable digest of a request body (key and digest fields excluded)."""
    clean = {k: v for k, v in body.items() if k not in _EXCLUDED_FROM_DIGEST}
    framed = c.REQUEST_DOMAIN.encode("utf-8") + b"\x00" + canonical_bytes(clean)
    return "sha256:" + hashlib.sha256(framed).hexdigest()


def seal(body: dict[str, Any], idempotency_key: str | None = None) -> dict[str, Any]:
    """Return ``body`` with ``idempotency_key`` (fresh unless supplied) and ``request_digest``."""
    key = idempotency_key if idempotency_key is not None else new_idempotency_key()
    if not c.IDEMPOTENCY_KEY_RE.fullmatch(key):
        raise LegendError("invalid_request", "idempotency_key must be idk_<32 hex>")
    sealed = {k: v for k, v in body.items() if k not in _EXCLUDED_FROM_DIGEST}
    sealed["idempotency_key"] = key
    sealed["request_digest"] = request_digest(sealed)
    return sealed


def build_declaration_payload(
    pillar: Pillar,
    *,
    entity_ref: str | None,
    previous_version_ref: str | None,
    change_reason: str | None,
    initial_parent_version_ref: str | None = None,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """``DeclareRequestV0``.

    ``initial_parent_version_ref`` is honoured by the server only on the version's FIRST
    declaration (send ``None`` otherwise); ``previous_version_ref`` (where the declarer
    moved from) is independent of it and may be set on every declaration.
    """
    for value, what, pattern in (
        (entity_ref, "entity", c.ENTITY_REF_RE),
        (previous_version_ref, "previous_version", c.VERSION_REF_RE),
        (initial_parent_version_ref, "initial_parent", c.VERSION_REF_RE),
    ):
        if value is not None and not pattern.fullmatch(value):
            raise LegendError("invalid_request", f"{what} must be an opaque ref")
    if change_reason is not None and change_reason not in c.CHANGE_REASONS:
        raise LegendError("invalid_request", "change_reason is not in the closed list")
    body: dict[str, Any] = {
        "schema_version": c.SCHEMA_VERSION,
        "entity_ref": entity_ref,
        **pillar.wire_identity(),
        "initial_parent_version_ref": initial_parent_version_ref,
        "identity_source": pillar.identity_source,
        "provenance": c.PROVENANCE_EXPLICIT,
    }
    if previous_version_ref is not None:
        body["previous_version_ref"] = previous_version_ref
    if change_reason is not None:
        body["change_reason"] = change_reason
    return seal(body, idempotency_key)


def build_query_payload(
    entity_ref: str,
    page_size: int,
    page_cursor: str | None,
    kind: str | None = None,
) -> dict[str, Any]:
    """``QueryRequestV0`` (no ``instruction_ref``: that exists only on the Director path)."""
    if not c.ENTITY_REF_RE.fullmatch(entity_ref or ""):
        raise LegendError("invalid_request", "entity must be an opaque entity ref")
    if type(page_size) is not int or not 1 <= page_size <= c.QUERY_MAX_PAGE_SIZE:
        raise LegendError("invalid_request", "page_size out of range")
    if page_cursor is not None and not c.CURSOR_RE.fullmatch(page_cursor):
        raise LegendError("invalid_request", "page_cursor must be an opaque cursor")
    if kind is not None and kind not in c.ENTITY_KINDS:
        raise LegendError("invalid_request", "unknown pillar kind")
    return {
        "schema_version": c.SCHEMA_VERSION,
        "entity_ref": entity_ref,
        "kind": kind,
        "page_size": page_size,
        "page_cursor": page_cursor,
    }
