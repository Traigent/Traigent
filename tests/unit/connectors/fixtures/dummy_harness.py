"""In-memory reference harness for the dummy connector, plus broken variants.

Each variant carries exactly one defect so the suite can be shown to catch it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from traigent.connectors.privacy import (
    SummaryValidationError,
    serialize_summary,
    validate_summary,
)
from traigent.connectors.testing import (
    CANARY,
    Connection,
    ConnectorHarness,
    Fault,
    FaultKind,
)

from .dummy import Connector


class DummyHarness(ConnectorHarness):
    """Correct behaviour: idempotent keyed writes, durable cursors, per-connection state.

    The dummy connector has no privacy gate of its own, so its gate and
    serializer are the SDK's. A real connector must supply its own.
    """

    kind = Connector.kind
    # Fixture-only: the summary schema's connector-kind enum is closed and
    # "dummy" is not a member, so the fixture borrows an existing value.
    summary_kind = "git_file"

    def __init__(self) -> None:
        self._count = 0
        self._store: dict[str, list[Mapping[str, Any]]] = {}
        self._applied: dict[tuple[str, str], int] = {}
        self._seeded: dict[str, tuple[list[Mapping[str, Any]], int, int]] = {}
        self._epoch = 0
        self._fault: Fault | None = None
        self._requests: list[str] = []
        self._mem: dict[str, Any] = {}  # in-memory state; wiped by crash()

    # -- connections -----------------------------------------------------
    def _credential(self, label: str) -> str:
        return f"cred-{label}-{self._count}"

    def _minter(self, label: str):  # type: ignore[no-untyped-def]
        from traigent.connectors.models import ConnectionRef
        from traigent.connectors.privacy import CustomerSideMinter

        return CustomerSideMinter(ConnectionRef(self.kind))

    def open_connection(self, label: str) -> Connection:
        self._count += 1
        conn = Connection(label, self._minter(label), self._credential(label))
        self._store[self._store_id(conn)] = []
        return conn

    def _store_id(self, connection: Connection) -> str:
        return str(id(connection))

    def _log(self, connection: Connection) -> None:
        self._requests.append(connection.credential)

    def request_credentials(self) -> list[str]:
        return list(self._requests)

    # -- writes ----------------------------------------------------------
    def _scope(self, connection: Connection, idem_key: str) -> tuple[str, str]:
        return (self._store_id(connection), idem_key)

    def _batch(self, rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        seen: set[str] = set()
        batch = []
        for row in rows:
            if self.row_key(row) not in seen:
                seen.add(self.row_key(row))
                batch.append(row)
        return batch

    def _persist(self, row: Mapping[str, Any]) -> Mapping[str, Any]:
        return row

    def _record(self, scope: tuple[str, str], done: int, fault: Fault | None) -> None:
        self._applied[scope] = done

    def write(self, connection, rows, idem_key):  # type: ignore[no-untyped-def]
        self._log(connection)
        fault, self._fault = self._fault, None
        scope = self._scope(connection, idem_key)
        batch = self._batch(rows)
        if fault is not None and fault.kind is FaultKind.FAIL_BEFORE_APPLY:
            self._on_fail_before(scope, len(batch))
            raise ConnectionError("injected failure before apply")
        done = self._applied.get(scope, 0)
        limit = len(batch)
        if fault is not None and fault.kind is FaultKind.PARTIAL:
            limit = min(limit, fault.applied)
        store = self._store[self._store_id(connection)]
        store.extend(self._persist(row) for row in batch[done:limit])
        self._record(scope, max(done, limit), fault)
        if fault is not None and fault.kind is FaultKind.PARTIAL:
            raise ConnectionError("injected partial failure")
        if fault is not None and fault.kind is FaultKind.LOST_ACK:
            raise TimeoutError("injected lost acknowledgement")

    def _on_fail_before(self, scope: tuple[str, str], size: int) -> None:
        return None

    def committed(self, connection):  # type: ignore[no-untyped-def]
        return list(self._store[self._store_id(connection)])

    def arm_fault(self, fault: Fault) -> None:
        self._fault = fault

    # -- reads -----------------------------------------------------------
    def seed(self, connection, rows, page_size, overlap=0):  # type: ignore[no-untyped-def]
        self._log(connection)
        self._seeded[self._store_id(connection)] = (list(rows), page_size, overlap)

    def _issue(self, offset: int) -> str:
        return f"{self._epoch}:{offset}"

    def _resolve(self, cursor: str) -> int:
        epoch, _, offset = cursor.partition(":")
        if int(epoch) != self._epoch:
            raise LookupError("cursor expired")
        return int(offset)

    def _next_offset(self, end: int, overlap: int) -> int:
        return end - overlap

    def _shape(self, items: list[Mapping[str, Any]], overlap: int, resumed: bool):  # type: ignore[no-untyped-def]
        return items

    def read_page(self, connection, cursor):  # type: ignore[no-untyped-def]
        self._log(connection)
        rows, size, overlap = self._seeded[self._store_id(connection)]
        start = 0 if cursor is None else self._resolve(cursor)
        end = start + size
        nxt = self._issue(self._next_offset(end, overlap)) if end < len(rows) else None
        items = self._shape(rows[start:end], overlap, cursor is not None)
        return {"items": items, "next_cursor": nxt}

    def expire_cursors(self) -> None:
        self._epoch += 1

    def crash(self) -> bool:
        self._mem.clear()
        return True

    # -- privacy ---------------------------------------------------------
    def validate_summary(self, payload):  # type: ignore[no-untyped-def]
        return validate_summary(payload)

    def serialize_summary(self, connection, payload):  # type: ignore[no-untyped-def]
        return serialize_summary(payload, minter=connection.minter)


# -- write-path defects ---------------------------------------------------
class NoIdempotencyHarness(DummyHarness):
    """Ignores idempotency: every retry or replay appends again."""

    def _record(self, scope, done, fault):  # type: ignore[no-untyped-def]
        return None

    def _batch(self, rows):  # type: ignore[no-untyped-def]
        return list(rows)


class FailBeforeRecordsKeyHarness(DummyHarness):
    """Marks the key done even though nothing was applied (retry is skipped)."""

    def _on_fail_before(self, scope, size):  # type: ignore[no-untyped-def]
        self._applied[scope] = size


class LostAckNotRecordedHarness(DummyHarness):
    """Applies the batch on a lost ack but forgets it, so a retry duplicates."""

    def _record(self, scope, done, fault):  # type: ignore[no-untyped-def]
        if fault is None or fault.kind is not FaultKind.LOST_ACK:
            super()._record(scope, done, fault)


class PartialNotRecordedHarness(DummyHarness):
    """Forgets a partially applied prefix, so a retry duplicates it."""

    def _record(self, scope, done, fault):  # type: ignore[no-untyped-def]
        if fault is None or fault.kind is not FaultKind.PARTIAL:
            super()._record(scope, done, fault)


class SwallowedFaultHarness(DummyHarness):
    """Never injects faults, so the suite cannot be passed vacuously."""

    def arm_fault(self, fault: Fault) -> None:
        return None


class CorruptsOnWriteHarness(DummyHarness):
    """Keeps ids but drops values on write."""

    def _persist(self, row):  # type: ignore[no-untyped-def]
        return {**row, "value": None}


# -- read-path defects ----------------------------------------------------
class CorruptsOnReadHarness(DummyHarness):
    """Keeps ids but drops values on read."""

    def _shape(self, items, overlap, resumed):  # type: ignore[no-untyped-def]
        return [{**row, "value": None} for row in items]


class NoOpCrashHarness(DummyHarness):
    """crash() discards nothing."""

    def crash(self) -> bool:
        return False


class InMemoryCursorHarness(DummyHarness):
    """Cursors are handles into a table that lives only in process memory."""

    def _issue(self, offset: int) -> str:
        table = self._mem.setdefault("cursors", {})
        handle = f"h{len(table)}"
        table[handle] = (self._epoch, offset)
        return handle

    def _resolve(self, cursor: str) -> int:
        epoch, offset = self._mem["cursors"][cursor]
        if epoch != self._epoch:
            raise LookupError("cursor expired")
        return int(offset)


class StaleCursorHarness(DummyHarness):
    """Serves data for an expired cursor instead of raising."""

    def _resolve(self, cursor: str) -> int:
        return int(cursor.partition(":")[2])


class OverlapLosesRowHarness(DummyHarness):
    """With overlapping pages the next cursor skips a row."""

    def _next_offset(self, end: int, overlap: int) -> int:
        return end - overlap + (2 if overlap else 0)


class ConflictingOverlapHarness(DummyHarness):
    """A row repeated on a later page differs from its first delivery."""

    def _shape(self, items, overlap, resumed):  # type: ignore[no-untyped-def]
        if resumed and overlap:
            return [{**row, "value": "changed"} for row in items[:overlap]] + list(
                items[overlap:]
            )
        return items


# -- connection defects ---------------------------------------------------
class SharedCredentialHarness(DummyHarness):
    def _credential(self, label: str) -> str:
        return "cred-shared"


class SharedMinterHarness(DummyHarness):
    def _minter(self, label):  # type: ignore[no-untyped-def]
        if "minter" not in self.__dict__:
            self.minter = super()._minter(label)
        return self.minter


class CachedCredentialHarness(DummyHarness):
    """Requests keep using the first connection's credential."""

    def _log(self, connection: Connection) -> None:
        self._first = getattr(self, "_first", connection.credential)
        self._requests.append(self._first)


class GlobalIdempotencyHarness(DummyHarness):
    """Idempotency keys are not scoped to a connection."""

    def _scope(self, connection, idem_key):  # type: ignore[no-untyped-def]
        return ("global", idem_key)


class ForeignTokenAcceptedHarness(DummyHarness):
    """Serializer skips token-ownership checks."""

    def serialize_summary(self, connection, payload):  # type: ignore[no-untyped-def]
        from traigent.connectors.testing.suites import _wire

        return validate_summary(_wire(dict(payload)))


# -- privacy-gate defects -------------------------------------------------
class LeakyGateHarness(DummyHarness):
    def validate_summary(self, payload):  # type: ignore[no-untyped-def]
        return dict(payload)


class EchoingGateHarness(DummyHarness):
    def validate_summary(self, payload):  # type: ignore[no-untyped-def]
        try:
            return validate_summary(payload)
        except SummaryValidationError:
            raise SummaryValidationError(f"bad value {payload!r}") from None


class WrongPointerGateHarness(DummyHarness):
    def validate_summary(self, payload):  # type: ignore[no-untyped-def]
        try:
            return validate_summary(payload)
        except SummaryValidationError:
            raise SummaryValidationError("invalid summary at /") from None


class UncheckedStatusGateHarness(DummyHarness):
    """Gate that lets any value through in one field (status)."""

    def validate_summary(self, payload):  # type: ignore[no-untyped-def]
        patched = dict(payload)
        if patched.get("status") == CANARY:
            patched["status"] = "partial"
        return validate_summary(patched)
