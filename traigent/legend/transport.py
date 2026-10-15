"""Transport for Legend requests: a protocol, an in-memory fake and a default-off HTTP stub."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from traigent.legend import _contract as c
from traigent.legend.errors import LegendError


@dataclass(frozen=True)
class DeclareResult:
    version_ref: str
    entity_ref: str
    created: bool
    event_ref: str


@dataclass(frozen=True)
class QueryResult:
    version_refs: tuple[str, ...]
    next_page_cursor: str | None


class CaptureTransport(Protocol):
    """Optional transport half that carries client-originated capture / restore records."""

    def record_capture(self, payload: Mapping[str, Any]) -> str: ...

    def record_restore(self, payload: Mapping[str, Any]) -> str: ...


class LegendTransport(Protocol):
    def declare(self, payload: Mapping[str, Any]) -> DeclareResult: ...

    def query(self, payload: Mapping[str, Any]) -> QueryResult: ...


def _ref(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:32]
    return f"{prefix}_{digest}"


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class InMemoryTransport:
    """A fake server for tests and offline use: dedup, append-only events, paging.

    Behaves like the contract's declare service (design note section 3.4):
    a declare of an existing identity appends an event and creates no second
    version; the same idempotency key with a different payload is refused.
    ``versions`` / ``events`` hold contract-shaped records for inspection.
    """

    def __init__(self) -> None:
        self.versions: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []
        self.entities: dict[str, dict[str, Any]] = {}
        self.requests: list[dict[str, Any]] = []
        self._by_identity: dict[tuple[Any, ...], str] = {}
        self._by_idem: dict[str, tuple[dict[str, Any], DeclareResult]] = {}
        self._entity_seq = 0

    def declare(self, payload: Mapping[str, Any]) -> DeclareResult:
        body = dict(payload)
        self.requests.append(body)
        idem = body["idempotency_key"]
        if idem in self._by_idem:
            first, result = self._by_idem[idem]
            if first != body:
                raise LegendError("conflicting_idempotency_payload")
            return result
        entity_ref = body["entity_ref"]
        if entity_ref is None:
            self._entity_seq += 1
            entity_ref = _ref("entityref", "entity", str(self._entity_seq), idem)
            self.entities[entity_ref] = {
                "schema_version": c.SCHEMA_VERSION,
                "entity_ref": entity_ref,
                "kind": body["kind"],
                "created_at": _now(),
            }
        elif entity_ref not in self.entities:
            raise LegendError("unknown_ref")
        parent = body["initial_parent_version_ref"]
        prev = body["previous_version_ref"]
        for ref in (parent, prev):
            if ref is not None and (
                ref not in self.versions
                or self.versions[ref]["entity_ref"] != entity_ref
            ):
                raise LegendError("unknown_ref")
        identity = (
            entity_ref,
            body["fingerprint_scheme"],
            body["key_id"],
            body["fingerprint"],
        )
        created = identity not in self._by_identity
        if created:
            version_ref = _ref("versionref", *map(str, identity))
            self._by_identity[identity] = version_ref
            self.versions[version_ref] = {
                "schema_version": c.SCHEMA_VERSION,
                "version_ref": version_ref,
                "entity_ref": entity_ref,
                "kind": body["kind"],
                "fingerprint_scheme": body["fingerprint_scheme"],
                "key_id": body["key_id"],
                "fingerprint": body["fingerprint"],
                "initial_parent_version_ref": parent,
                "identity_source": body["identity_source"],
                "declared_at": _now(),
            }
        version_ref = self._by_identity[identity]
        event: dict[str, Any] = {
            "schema_version": c.SCHEMA_VERSION,
            "event_ref": _ref("eventref", str(len(self.events)), idem),
            "version_ref": version_ref,
            "actor_ref": _ref("actorref", "in-memory"),
            "origin": "client",
            "evidence_source": "client_reported",
            "recorded_at": _now(),
            "idempotency_key": idem,
            "event_type": "declared",
            "provenance": body["provenance"],
            "identity_source": body["identity_source"],
            "previous_version_ref": prev,
        }
        if "change_reason" in body:
            event["change_reason"] = body["change_reason"]
        self.events.append(event)
        result = DeclareResult(version_ref, entity_ref, created, event["event_ref"])
        self._by_idem[idem] = (body, result)
        return result

    def _record(self, payload: Mapping[str, Any], event_type: str) -> str:
        body = dict(payload)
        self.requests.append(body)
        if body.get("event_type") != event_type:
            raise LegendError("invalid_request")
        if body.get("version_ref") not in self.versions:
            raise LegendError("unknown_ref")
        idem = body["idempotency_key"]
        for event in self.events:
            if event["idempotency_key"] == idem:
                if {k: event.get(k) for k in body} != body:
                    raise LegendError("conflicting_idempotency_payload")
                return str(event["event_ref"])
        event = {
            **body,
            "event_ref": _ref("eventref", str(len(self.events)), idem),
            "actor_ref": _ref("actorref", "in-memory"),
            "recorded_at": _now(),
        }
        self.events.append(event)
        return str(event["event_ref"])

    def record_capture(self, payload: Mapping[str, Any]) -> str:
        return self._record(payload, "captured")

    def record_restore(self, payload: Mapping[str, Any]) -> str:
        return self._record(payload, "restore_verified")

    def query(self, payload: Mapping[str, Any]) -> QueryResult:
        self.requests.append(dict(payload))
        entity_ref = payload["entity_ref"]
        if entity_ref not in self.entities:
            raise LegendError("unknown_ref")
        refs = [r for r, v in self.versions.items() if v["entity_ref"] == entity_ref]
        start = 0
        cursor = payload["page_cursor"]
        if cursor is not None:
            known = {
                _ref("cursor", entity_ref, str(i)): i for i in range(len(refs) + 1)
            }
            if cursor not in known:
                raise LegendError("invalid_request")
            start = known[cursor]
        size = payload["page_size"]
        page = refs[start : start + size]
        nxt = (
            _ref("cursor", entity_ref, str(start + size))
            if start + size < len(refs)
            else None
        )
        return QueryResult(tuple(page), nxt)


class HttpLegendTransport:
    """Thin stub for the future Backend routes; off unless ``TRAIGENT_LEGEND_HTTP_ENABLED=1``.

    ``send(path, payload) -> dict`` is injected by the caller (it owns auth and
    the network); there is no built-in network call, so nothing here can reach
    out by default.
    """

    def __init__(
        self, send: Callable[[str, Mapping[str, Any]], Mapping[str, Any]] | None = None
    ) -> None:
        self._send = send

    def _enabled(self) -> Callable[[str, Mapping[str, Any]], Mapping[str, Any]]:
        if os.environ.get(c.LEGEND_ENABLED_ENV) != "1":
            raise LegendError("flag_disabled")
        if self._send is None:
            raise LegendError("store_unavailable")
        return self._send

    def declare(self, payload: Mapping[str, Any]) -> DeclareResult:
        data = self._enabled()(c.ROUTE_DECLARE, payload)
        try:
            return DeclareResult(
                data["version_ref"],
                data["entity_ref"],
                bool(data["created"]),
                data["event_ref"],
            )
        except (KeyError, TypeError):
            raise LegendError("internal_error") from None

    def record_capture(self, payload: Mapping[str, Any]) -> str:
        return self._record(c.ROUTE_RECORD_CAPTURE, payload)

    def record_restore(self, payload: Mapping[str, Any]) -> str:
        return self._record(c.ROUTE_RECORD_RESTORE, payload)

    def _record(self, path: str, payload: Mapping[str, Any]) -> str:
        data = self._enabled()(path, payload)
        try:
            return str(data["event_ref"])
        except (KeyError, TypeError):
            raise LegendError("internal_error") from None

    def query(self, payload: Mapping[str, Any]) -> QueryResult:
        data = self._enabled()(c.ROUTE_QUERY, payload)
        try:
            return QueryResult(
                tuple(data["version_refs"]), data.get("next_page_cursor")
            )
        except (KeyError, TypeError):
            raise LegendError("internal_error") from None
