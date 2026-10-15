"""Closed declaration / query payloads. Only enums, digests and opaque refs -- no names, no paths."""

from __future__ import annotations

import hashlib
from typing import Any

from traigent.identity.content_identity import canonical_bytes
from traigent.legend import _contract as c
from traigent.legend.errors import LegendError
from traigent.legend.pillars import Pillar


def idempotency_key(payload: dict[str, Any]) -> str:
    """sha256 hex of the canonical payload without its own ``idempotency_key``."""
    body = {k: v for k, v in payload.items() if k != "idempotency_key"}
    return hashlib.sha256(canonical_bytes(body)).hexdigest()


def build_declaration_payload(
    pillar: Pillar,
    *,
    entity_ref: str | None,
    previous_version_ref: str | None,
    change_reason: str | None,
) -> dict[str, Any]:
    """The declare request: version identity plus the ``declared`` event's client fields."""
    if entity_ref is not None and not c.ENTITY_REF_RE.fullmatch(entity_ref):
        raise LegendError("invalid_request", "entity must be an opaque entity ref")
    if previous_version_ref is not None and not c.VERSION_REF_RE.fullmatch(
        previous_version_ref
    ):
        raise LegendError(
            "invalid_request", "previous_version must be an opaque version ref"
        )
    if change_reason is not None and change_reason not in c.CHANGE_REASONS:
        raise LegendError("invalid_request", "change_reason is not in the closed list")
    payload: dict[str, Any] = {
        "schema_version": c.SCHEMA_VERSION,
        "entity_ref": entity_ref,
        **pillar.wire_identity(),
        "initial_parent_version_ref": previous_version_ref,
        "identity_source": pillar.identity_source,
        "provenance": c.PROVENANCE_EXPLICIT,
        "previous_version_ref": previous_version_ref,
    }
    if change_reason is not None:
        payload["change_reason"] = change_reason
    payload["idempotency_key"] = idempotency_key(payload)
    return payload


def build_query_payload(
    entity_ref: str, page_size: int, page_cursor: str | None
) -> dict[str, Any]:
    """The query request (``QueryVersionsInstruction`` minus the server-issued ``instruction_ref``)."""
    if not c.ENTITY_REF_RE.fullmatch(entity_ref or ""):
        raise LegendError("invalid_request", "entity must be an opaque entity ref")
    if type(page_size) is not int or not 1 <= page_size <= c.QUERY_MAX_PAGE_SIZE:
        raise LegendError("invalid_request", "page_size out of range")
    if page_cursor is not None and not c.CURSOR_RE.fullmatch(page_cursor):
        raise LegendError("invalid_request", "page_cursor must be an opaque cursor")
    return {
        "schema_version": c.SCHEMA_VERSION,
        "entity_ref": entity_ref,
        "page_size": page_size,
        "page_cursor": page_cursor,
    }
