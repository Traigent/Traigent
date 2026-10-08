"""Stateful conformance cases, as plain functions and as a pytest-collectable class."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ..privacy import SummaryValidationError, serialize_summary
from .harness import CANARY, ConformanceFailure, Connection, ConnectorHarness

Case = Callable[[ConnectorHarness], None]


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ConformanceFailure(message)


def _rows(count: int) -> list[dict[str, Any]]:
    return [{"id": f"row-{index}", "value": index} for index in range(count)]


def _keys(harness: ConnectorHarness, rows: Any) -> list[str]:
    return [harness.row_key(row) for row in rows]


def _write_expecting_fault(
    harness: ConnectorHarness, connection: Connection, rows: Sequence[Any], key: str
) -> None:
    try:
        harness.write(connection, rows, key)
    except harness.transient_errors:
        return
    raise ConformanceFailure("injected fault was not surfaced by write")


def _read_all(
    harness: ConnectorHarness, connection: Connection, cursor: str | None
) -> list[Any]:
    items: list[Any] = []
    seen: set[str] = set()
    for _ in range(1000):
        page = harness.read_page(connection, cursor)
        items.extend(page["items"])
        cursor = page["next_cursor"]
        if cursor is None:
            return items
        _check(cursor not in seen, "pagination repeated a cursor")
        seen.add(cursor)
    raise ConformanceFailure("pagination did not terminate")


def case_lost_response(harness: ConnectorHarness) -> None:
    conn = harness.open_connection("lost")
    rows = _rows(4)
    harness.arm_lost_response()
    _write_expecting_fault(harness, conn, rows, "idem-lost")
    harness.write(conn, rows, "idem-lost")
    _check(
        sorted(_keys(harness, harness.stored(conn))) == sorted(_keys(harness, rows)),
        "retry after a lost response must store each row exactly once",
    )


def case_crash_resume(harness: ConnectorHarness) -> None:
    conn = harness.open_connection("resume")
    rows = _rows(7)
    harness.seed(conn, rows, 3)
    first = harness.read_page(conn, None)
    checkpoint = first["next_cursor"]
    _check(checkpoint is not None, "seeded data must span more than one page")
    harness.restart()
    rest = _read_all(harness, conn, checkpoint)
    got = _keys(harness, list(first["items"]) + rest)
    _check(
        got == _keys(harness, rows),
        "resume from a checkpoint must neither lose nor repeat rows",
    )


def case_duplicate_rows(harness: ConnectorHarness) -> None:
    conn = harness.open_connection("dup")
    rows = _rows(3)
    harness.write(conn, rows, "idem-dup")
    harness.write(conn, rows, "idem-dup")
    _check(
        sorted(_keys(harness, harness.stored(conn))) == sorted(_keys(harness, rows)),
        "replaying an idempotency key must not duplicate rows",
    )
    conn2 = harness.open_connection("dup-batch")
    harness.write(conn2, rows + [rows[0]], "idem-dup-batch")
    _check(
        sorted(_keys(harness, harness.stored(conn2))) == sorted(_keys(harness, rows)),
        "duplicate rows inside one batch must be stored once",
    )


def case_expired_cursor(harness: ConnectorHarness) -> None:
    conn = harness.open_connection("expired")
    rows = _rows(6)
    harness.seed(conn, rows, 2)
    cursor = harness.read_page(conn, None)["next_cursor"]
    _check(cursor is not None, "seeded data must span more than one page")
    harness.expire_cursors()
    try:
        harness.read_page(conn, cursor)
    except harness.expired_cursor_errors:
        pass
    else:
        raise ConformanceFailure("an expired cursor must raise, not return data")
    _check(
        _keys(harness, _read_all(harness, conn, None)) == _keys(harness, rows),
        "restarting from the beginning after expiry must read every row",
    )


def case_partial_batch(harness: ConnectorHarness) -> None:
    conn = harness.open_connection("partial")
    rows = _rows(5)
    harness.arm_partial_failure(2)
    _write_expecting_fault(harness, conn, rows, "idem-partial")
    harness.write(conn, rows, "idem-partial")
    _check(
        sorted(_keys(harness, harness.stored(conn))) == sorted(_keys(harness, rows)),
        "retrying a partially applied batch must converge to each row once",
    )


def case_connection_collision(harness: ConnectorHarness) -> None:
    a = harness.open_connection("conn-a")
    b = harness.open_connection("conn-b")
    _check(a.credential != b.credential, "connections must not share credentials")
    _check(a.minter is not b.minter, "connections must not share a minter")
    _check(
        a.minter.connection_token.value != b.minter.connection_token.value,
        "connections must not share a connection token",
    )
    token = a.minter.mint("run", "synthetic")
    _check(not b.minter.owns(token), "a token must belong to one connection only")
    _check(
        a.minter.digest("synthetic") != b.minter.digest("synthetic"),
        "digest keys must differ between connections",
    )
    harness.write(a, [{"id": "only-a", "value": 1}], "idem-a")
    _check(
        "only-a" not in _keys(harness, harness.stored(b)),
        "data written on one connection must not appear on another",
    )
    own = b.minter.mint("run", "synthetic")
    owned = {"run_token": own, "connection_token": b.minter.connection_token}
    serialize_summary(harness.summary_for(b, owned), minter=b.minter)  # control
    foreign = {"run_token": token, "connection_token": b.minter.connection_token}
    try:
        serialize_summary(harness.summary_for(b, foreign), minter=b.minter)
    except SummaryValidationError:
        return
    raise ConformanceFailure("a foreign connection's token must be rejected")


def case_content_rejection(harness: ConnectorHarness) -> None:
    conn = harness.open_connection("content")
    harness.validate(harness.summary_for(conn))  # control: valid summary passes
    attacks: list[dict[str, Any]] = [
        {"command": CANARY},
        {"error_code": CANARY},
        {"unexpected_field": CANARY},
        {CANARY: 1},
        {"counts": {"observations_read": CANARY}},
    ]
    for overrides in attacks:
        try:
            harness.validate(harness.summary_for(conn, overrides))
        except SummaryValidationError as exc:
            texts = [str(exc), repr(exc), *map(str, exc.args)]
            _check(
                not any(CANARY in text for text in texts),
                "rejection error must not echo submitted content",
            )
        else:
            raise ConformanceFailure("a summary carrying content must be rejected")


CASES: dict[str, Case] = {
    "lost_response": case_lost_response,
    "crash_resume": case_crash_resume,
    "duplicate_rows": case_duplicate_rows,
    "expired_cursor": case_expired_cursor,
    "partial_batch": case_partial_batch,
    "connection_collision": case_connection_collision,
    "content_rejection": case_content_rejection,
}


class ConnectorConformanceSuite:
    """Subclass (named ``Test...``) and override ``make_harness`` to run all cases."""

    @staticmethod
    def make_harness() -> ConnectorHarness:
        raise NotImplementedError("override make_harness in your suite subclass")

    def test_lost_response(self) -> None:
        case_lost_response(self.make_harness())

    def test_crash_resume(self) -> None:
        case_crash_resume(self.make_harness())

    def test_duplicate_rows(self) -> None:
        case_duplicate_rows(self.make_harness())

    def test_expired_cursor(self) -> None:
        case_expired_cursor(self.make_harness())

    def test_partial_batch(self) -> None:
        case_partial_batch(self.make_harness())

    def test_connection_collision(self) -> None:
        case_connection_collision(self.make_harness())

    def test_content_rejection(self) -> None:
        case_content_rejection(self.make_harness())
