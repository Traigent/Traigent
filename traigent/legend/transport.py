"""Transport for Legend requests: a protocol, an in-memory fake."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, TypeVar

from traigent.legend import _contract as c
from traigent.legend.errors import LegendError
from traigent.legend.payloads import request_digest

T = TypeVar("T")
# Transient codes worth a retry; the request (and so its idempotency key) is unchanged.
RETRYABLE_CODES = frozenset({"store_unavailable", "internal_error"})
MAX_ATTEMPTS = 3


def with_retries(send: Callable[[], T], attempts: int = MAX_ATTEMPTS) -> T:
    """Run ``send`` (a closure over ONE sealed payload), retrying transient failures.

    The payload -- and therefore ``idempotency_key`` and ``request_digest`` -- is built
    once outside this call, so every attempt carries the same key and digest.
    """
    for attempt in range(attempts):
        try:
            return send()
        except LegendError as error:
            if error.code not in RETRYABLE_CODES or attempt == attempts - 1:
                raise
    raise LegendError("internal_error")  # pragma: no cover - loop always returns/raises


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

    def record_availability(self, payload: Mapping[str, Any]) -> str: ...


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
        self._position = 0  # server-assigned, strictly increasing ledger position

    def _next_position(self) -> int:
        self._position += 1
        return self._position

    @staticmethod
    def _check_sealed(body: Mapping[str, Any]) -> None:
        if body.get("request_digest") != request_digest(dict(body)):
            raise LegendError("invalid_request")
        if not c.IDEMPOTENCY_KEY_RE.fullmatch(str(body.get("idempotency_key"))):
            raise LegendError("invalid_request")

    def declare(self, payload: Mapping[str, Any]) -> DeclareResult:
        body = dict(payload)
        self.requests.append(body)
        self._check_sealed(body)
        idem = body["idempotency_key"]
        if idem in self._by_idem:
            first, result = self._by_idem[idem]
            if first["request_digest"] != body["request_digest"]:
                raise LegendError("conflicting_idempotency_payload")
            return result
        entity_ref = body["entity_ref"]
        known_entity = entity_ref is not None and entity_ref in self.entities
        if entity_ref is not None and not known_entity:
            raise LegendError("unknown_ref")
        parent = body["initial_parent_version_ref"]
        prev = body.get("previous_version_ref")
        for ref in (parent, prev):
            if ref is None:
                continue
            if ref not in self.versions:
                raise LegendError("unknown_ref")
            if known_entity and self.versions[ref]["entity_ref"] != entity_ref:
                raise LegendError("unknown_ref")
        if entity_ref is None and (parent is not None or prev is not None):
            raise LegendError("invalid_request")  # a new entity has no earlier versions
        if entity_ref is None:
            self._entity_seq += 1
            entity_ref = _ref("entityref", "entity", str(self._entity_seq), idem)
            self.entities[entity_ref] = {
                "schema_version": c.SCHEMA_VERSION,
                "entity_ref": entity_ref,
                "kind": body["kind"],
                "created_at": _now(),
            }
        identity = (
            entity_ref,
            body["fingerprint_scheme"],
            body["key_id"],
            body["fingerprint"],
        )
        created = identity not in self._by_identity
        if created:
            version_ref = _ref("versionref", *map(str, identity))
            if parent == version_ref:
                raise LegendError("invalid_request")
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
        else:
            stored = self.versions[self._by_identity[identity]]
            # the initial parent is set on the FIRST declaration only and never rewritten
            if parent is not None and parent != stored["initial_parent_version_ref"]:
                raise LegendError("invalid_request")
        version_ref = self._by_identity[identity]
        event: dict[str, Any] = {
            "schema_version": c.SCHEMA_VERSION,
            "event_ref": _ref("eventref", str(len(self.events)), idem),
            "version_ref": version_ref,
            "actor_ref": _ref("actorref", "in-memory"),
            "origin": "client",
            "evidence_source": "client_reported",
            "ledger_position": self._next_position(),
            "recorded_at": _now(),
            "idempotency_key": idem,
            "request_digest": body["request_digest"],
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
        self._check_sealed(body)
        if body.get("version_ref") not in self.versions:
            raise LegendError("unknown_ref")
        idem = body["idempotency_key"]
        for event in self.events:
            if event["idempotency_key"] == idem:
                if event["request_digest"] != body["request_digest"]:
                    raise LegendError("conflicting_idempotency_payload")
                return str(event["event_ref"])
        # origin / evidence_source are assigned by the (trusted) route, never the body
        event = {
            **body,
            "event_type": event_type,
            "origin": "client",
            "evidence_source": "client_sdk_reported",
            "event_ref": _ref("eventref", str(len(self.events)), idem),
            "actor_ref": _ref("actorref", "in-memory"),
            "ledger_position": self._next_position(),
            "recorded_at": _now(),
        }
        if event_type == "restore_verified":
            event["backend_match_result"] = (
                "not_applicable"  # a client restore carries no Backend match
            )
        self.events.append(event)
        return str(event["event_ref"])

    def record_capture(self, payload: Mapping[str, Any]) -> str:
        return self._record(payload, "captured")

    def record_restore(self, payload: Mapping[str, Any]) -> str:
        return self._record(payload, "restore_verified")

    def record_availability(self, payload: Mapping[str, Any]) -> str:
        return self._record(payload, "availability_changed")

    def query(self, payload: Mapping[str, Any]) -> QueryResult:
        self.requests.append(dict(payload))
        entity_ref = payload["entity_ref"]
        if entity_ref not in self.entities:
            raise LegendError("unknown_ref")
        kind = payload.get("kind")
        refs = [
            r
            for r, v in self.versions.items()
            if v["entity_ref"] == entity_ref and kind in (None, v["kind"])
        ]
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
