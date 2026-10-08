"""In-memory reference harness for the dummy connector, plus deliberately broken variants."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from traigent.connectors.privacy import SummaryValidationError, validate_summary
from traigent.connectors.testing import Connection, ConnectorHarness

from .dummy import Connector


class DummyHarness(ConnectorHarness):
    """Correct behaviour: idempotent keyed writes, durable cursors, per-connection state."""

    kind = Connector.kind

    def __init__(self) -> None:
        self._count = 0
        self._applied: dict[tuple[str, str], int] = {}
        self._rows: dict[str, dict[str, Mapping[str, Any]]] = {}
        self._seeded: dict[str, tuple[list[Mapping[str, Any]], int]] = {}
        self._epoch = 0
        self._lose = False
        self._partial: int | None = None

    def open_connection(self, label: str) -> Connection:
        self._count += 1
        conn = Connection(label, self.make_minter(), f"cred-{label}-{self._count}")
        self._rows[conn.credential] = {}
        return conn

    def write(self, connection, rows, idem_key):  # type: ignore[no-untyped-def]
        store = self._rows[connection.credential]
        done = self._applied.get((connection.credential, idem_key), 0)
        limit = len(rows)
        partial, self._partial = self._partial, None
        if partial is not None:
            limit = min(limit, partial)
        for row in rows[done:limit]:
            store[self.row_key(row)] = row
        self._applied[(connection.credential, idem_key)] = max(done, limit)
        if partial is not None:
            raise ConnectionError("injected partial failure")
        if self._lose:
            self._lose = False
            raise TimeoutError("injected lost response")

    def stored(self, connection):  # type: ignore[no-untyped-def]
        return list(self._rows[connection.credential].values())

    def seed(self, connection, rows, page_size):  # type: ignore[no-untyped-def]
        self._seeded[connection.credential] = (list(rows), page_size)

    def read_page(self, connection, cursor):  # type: ignore[no-untyped-def]
        rows, size = self._seeded[connection.credential]
        start = 0
        if cursor is not None:
            epoch, _, offset = cursor.partition(":")
            if int(epoch) != self._epoch:
                raise LookupError("cursor expired")
            start = int(offset)
        end = start + size
        nxt = f"{self._epoch}:{end}" if end < len(rows) else None
        return {"items": rows[start:end], "next_cursor": nxt}

    def arm_lost_response(self) -> None:
        self._lose = True

    def arm_partial_failure(self, applied: int) -> None:
        self._partial = applied

    def expire_cursors(self) -> None:
        self._epoch += 1


class NoIdempotencyHarness(DummyHarness):
    """Appends on every write: duplicates on replay, retry and intra-batch repeats."""

    def write(self, connection, rows, idem_key):  # type: ignore[no-untyped-def]
        self._append_count = getattr(self, "_append_count", 0)
        store = self._rows[connection.credential]
        partial, self._partial = self._partial, None
        limit = len(rows) if partial is None else min(len(rows), partial)
        for row in rows[:limit]:
            self._append_count += 1
            store[f"{self.row_key(row)}#{self._append_count}"] = row
        if partial is not None:
            raise ConnectionError("injected partial failure")
        if self._lose:
            self._lose = False
            raise TimeoutError("injected lost response")

    def row_key(self, row):  # type: ignore[no-untyped-def]
        return str(row["id"])

    def stored(self, connection):  # type: ignore[no-untyped-def]
        return list(self._rows[connection.credential].values())


class NoResumeHarness(DummyHarness):
    """Ignores the cursor: every read restarts from the beginning."""

    def read_page(self, connection, cursor):  # type: ignore[no-untyped-def]
        return super().read_page(connection, None)


class SwallowedFaultHarness(DummyHarness):
    """Never surfaces injected faults, so the suite cannot be trivially passed."""

    def arm_lost_response(self) -> None:
        pass

    def arm_partial_failure(self, applied: int) -> None:
        pass


class StaleCursorHarness(DummyHarness):
    """Serves data for an expired cursor instead of raising."""

    def expire_cursors(self) -> None:
        pass


class SharedConnectionHarness(DummyHarness):
    """Every connection shares one credential, minter and row store."""

    def open_connection(self, label: str) -> Connection:
        if not hasattr(self, "_shared"):
            self._shared = super().open_connection("shared")
        return self._shared


class LeakyGateHarness(DummyHarness):
    """Privacy gate that accepts content."""

    def validate(self, payload):  # type: ignore[no-untyped-def]
        return dict(payload)


class EchoingGateHarness(DummyHarness):
    """Privacy gate that rejects but echoes the submitted value."""

    def validate(self, payload):  # type: ignore[no-untyped-def]
        try:
            return validate_summary(payload)
        except SummaryValidationError:
            raise SummaryValidationError(f"bad value {payload!r}") from None
