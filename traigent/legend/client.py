"""``declare`` and ``query``: fingerprint locally, send a closed payload through a transport."""

from __future__ import annotations

from typing import Any

from traigent.legend import payloads
from traigent.legend.pillars import compute_pillar
from traigent.legend.transport import (
    DeclareResult,
    InMemoryTransport,
    LegendTransport,
    QueryResult,
)

_default_transport: LegendTransport | None = None


def set_default_transport(transport: LegendTransport | None) -> None:
    global _default_transport
    _default_transport = transport


def _resolve(transport: LegendTransport | None) -> LegendTransport:
    if transport is not None:
        return transport
    if _default_transport is None:
        # No network by default: an unconfigured client talks to a private in-memory fake.
        set_default_transport(InMemoryTransport())
    assert _default_transport is not None  # nosec B101 - narrowed above
    return _default_transport


def declare(
    kind: str,
    source: Any,
    *,
    entity: str | None = None,
    previous_version: str | None = None,
    change_reason: str | None = None,
    transport: LegendTransport | None = None,
) -> DeclareResult:
    """Declare a pillar version.

    ``entity`` and ``previous_version`` are opaque refs (never names; use the
    local alias map to translate). Passing ``previous_version`` records where
    the declarer moved from; returning to an older fingerprint (a revert) is
    just a declare of that fingerprint with ``previous_version`` set.
    """
    pillar = compute_pillar(kind, source)
    payload = payloads.build_declaration_payload(
        pillar,
        entity_ref=entity,
        previous_version_ref=previous_version,
        change_reason=change_reason,
    )
    return _resolve(transport).declare(payload)


def query(
    entity: str,
    *,
    page_size: int = 50,
    page_cursor: str | None = None,
    transport: LegendTransport | None = None,
) -> QueryResult:
    """List version refs of an entity (paginated)."""
    payload = payloads.build_query_payload(entity, page_size, page_cursor)
    return _resolve(transport).query(payload)
