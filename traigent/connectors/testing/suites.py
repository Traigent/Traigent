"""Stateful conformance cases, as plain functions and as a pytest-collectable class."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ..privacy import OpaqueToken, SummaryValidationError
from .harness import (
    CANARY,
    ConformanceFailure,
    Connection,
    ConnectorHarness,
    Fault,
    FaultKind,
)

Case = Callable[[ConnectorHarness], None]


def _check(condition: bool, reason: str, message: str) -> None:
    if not condition:
        raise ConformanceFailure(reason, message)


def _rows(count: int, tag: str = "") -> list[dict[str, Any]]:
    return [
        {"id": f"row{tag}-{i}", "value": {"n": i, "text": f"synthetic-{tag}{i}"}}
        for i in range(count)
    ]


def _digest(row: Mapping[str, Any]) -> str:
    blob = json.dumps(row, sort_keys=True, default=repr).encode()
    return hashlib.sha256(blob).hexdigest()


def _snapshot(harness: ConnectorHarness, rows: Sequence[Any]) -> list[tuple[str, str]]:
    """Order-insensitive identity of rows: (key, digest of full content)."""
    return sorted((harness.row_key(row), _digest(row)) for row in rows)


def _ordered(harness: ConnectorHarness, rows: Sequence[Any]) -> list[tuple[str, str]]:
    return [(harness.row_key(row), _digest(row)) for row in rows]


def _write_expecting_fault(
    harness: ConnectorHarness, connection: Connection, rows: Sequence[Any], key: str
) -> None:
    try:
        harness.write(connection, rows, key)
    except harness.transient_errors:
        return
    raise ConformanceFailure("fault_not_injected", "injected fault was not surfaced")


def _retry(
    harness: ConnectorHarness, connection: Connection, rows: Sequence[Any], key: str
) -> None:
    try:
        harness.write(connection, rows, key)
    except Exception:
        raise ConformanceFailure(
            "retry_failed", "retry with the same key failed"
        ) from None


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
        _check(cursor not in seen, "pagination_loop", "pagination repeated a cursor")
        seen.add(cursor)
    raise ConformanceFailure("pagination_loop", "pagination did not terminate")


def case_lost_response(harness: ConnectorHarness) -> None:
    """(a) failure before apply, (b) applied then acknowledgement lost."""
    for fault, applied_before_retry in (
        (FaultKind.FAIL_BEFORE_APPLY, False),
        (FaultKind.LOST_ACK, True),
    ):
        conn = harness.open_connection(f"lost-{fault}")
        rows = _rows(3, str(fault))
        key = f"idem-{fault}"
        harness.arm_fault(Fault(fault))
        _write_expecting_fault(harness, conn, rows, key)
        expected_pre = _snapshot(harness, rows) if applied_before_retry else []
        _check(
            _snapshot(harness, harness.committed(conn)) == expected_pre,
            "fault_not_injected",
            "backend state after the fault does not match the injected fault",
        )
        _retry(harness, conn, rows, key)
        _check(
            _snapshot(harness, harness.committed(conn)) == _snapshot(harness, rows),
            "rows_mismatch",
            "after retry the committed rows differ from the intended rows",
        )


def case_partial_batch(harness: ConnectorHarness) -> None:
    """(c) a strict prefix of the batch is committed, then the write fails."""
    for size, applied in ((2, 1), (5, 3)):
        conn = harness.open_connection(f"partial-{size}")
        rows = _rows(size, f"p{size}")
        key = f"idem-partial-{size}"
        harness.arm_fault(Fault(FaultKind.PARTIAL, applied))
        _write_expecting_fault(harness, conn, rows, key)
        _check(
            len(harness.committed(conn)) == applied,
            "fault_not_injected",
            "backend did not commit exactly the injected prefix",
        )
        _retry(harness, conn, rows, key)
        _check(
            _snapshot(harness, harness.committed(conn)) == _snapshot(harness, rows),
            "rows_mismatch",
            "retrying a partially applied batch must converge to each row once",
        )


def case_duplicate_rows(harness: ConnectorHarness) -> None:
    """Write-side duplicates: replayed idempotency key and repeats in one batch."""
    conn = harness.open_connection("dup")
    rows = _rows(3, "d")
    harness.write(conn, rows, "idem-dup")
    harness.write(conn, rows, "idem-dup")
    _check(
        _snapshot(harness, harness.committed(conn)) == _snapshot(harness, rows),
        "rows_mismatch",
        "replaying an idempotency key must not duplicate rows",
    )
    conn2 = harness.open_connection("dup-batch")
    harness.write(conn2, rows + [rows[0]], "idem-dup-batch")
    _check(
        _snapshot(harness, harness.committed(conn2)) == _snapshot(harness, rows),
        "rows_mismatch",
        "duplicate rows inside one batch must be stored once",
    )


def case_overlapping_pages(harness: ConnectorHarness) -> None:
    """Read-side duplicates: overlapping pages may repeat rows, never lose or alter."""
    conn = harness.open_connection("overlap")
    rows = _rows(7, "o")
    harness.seed(conn, rows, 3, overlap=1)
    first: dict[str, str] = {}
    for row in _read_all(harness, conn, None):
        key, digest = harness.row_key(row), _digest(row)
        _check(
            first.setdefault(key, digest) == digest,
            "conflicting_duplicate",
            "a row repeated across pages must be identical",
        )
    _check(
        sorted(first.items()) == _snapshot(harness, rows),
        "rows_mismatch",
        "overlapping pages must still deliver every row, unaltered",
    )


def case_crash_resume(harness: ConnectorHarness) -> None:
    conn = harness.open_connection("resume")
    rows = _rows(7, "r")
    harness.seed(conn, rows, 3)
    first = harness.read_page(conn, None)
    checkpoint = first["next_cursor"]
    _check(checkpoint is not None, "harness_incomplete", "data must span two pages")
    _check(
        harness.crash() is True,
        "crash_not_real",
        "crash() must discard in-memory connector state",
    )
    try:
        rest = _read_all(harness, conn, checkpoint)
    except ConformanceFailure:
        raise
    except Exception:
        raise ConformanceFailure(
            "resume_failed", "resume from the persisted checkpoint raised"
        ) from None
    _check(
        _ordered(harness, list(first["items"]) + rest) == _ordered(harness, rows),
        "rows_mismatch",
        "resume must neither lose, repeat nor alter rows",
    )


def case_expired_cursor(harness: ConnectorHarness) -> None:
    conn = harness.open_connection("expired")
    rows = _rows(6, "e")
    harness.seed(conn, rows, 2)
    cursor = harness.read_page(conn, None)["next_cursor"]
    _check(cursor is not None, "harness_incomplete", "data must span two pages")
    harness.expire_cursors()
    try:
        harness.read_page(conn, cursor)
    except harness.expired_cursor_errors:
        pass
    else:
        raise ConformanceFailure(
            "expiry_not_enforced", "an expired cursor must raise, not return data"
        )
    _check(
        _ordered(harness, _read_all(harness, conn, None)) == _ordered(harness, rows),
        "rows_mismatch",
        "restarting after expiry must read every row unaltered",
    )


def _token_objects_run(harness: ConnectorHarness, conn: Connection) -> dict[str, Any]:
    minter = conn.minter
    return {
        "schema_version": "1",
        "run_token": minter.mint("run", "synthetic"),
        "connector_kind": harness.summary_kind,
        "connection_token": minter.connection_token,
        "command": "check",
        "status": "queued",
        "counts": {
            "observations_read": 0,
            "scores_read": 0,
            "rows_dropped_invalid": 0,
            "pages": 0,
            "items_written": 0,
        },
        "guarantees": [],
    }


def _wire(node: Any) -> Any:
    if isinstance(node, OpaqueToken):
        return node.value
    if isinstance(node, dict):
        return {key: _wire(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_wire(value) for value in node]
    return node


def _golden_summaries(
    harness: ConnectorHarness, conn: Connection
) -> list[dict[str, Any]]:
    """Valid wire-form summaries of every schema with all optional fields set."""
    minter = conn.minter
    run = _token_objects_run(harness, conn)
    run.update(
        started_at="2026-10-08T10:00:00Z",
        finished_at="2026-10-08T10:00:42Z",
        status="partial",
        error_code="rate_limited",
        guarantees=[
            {
                "operation": "read_scores",
                "support": "emulated",
                "reason": "emulated_client_side",
            }
        ],
    )
    dataset = {
        "schema_version": "1",
        "dataset_token": minter.mint("dataset", "synthetic"),
        "revision": 2,
        "source_connector_kind": harness.summary_kind,
        "item_count": 200,
        "holdout_count": 40,
        "approved": True,
        "approval_at": "2026-10-08T11:00:00Z",
        "sampling_policy": {
            "kind": "random",
            "fraction": 0.25,
            "seed": 7,
            "sample_size": 50,
        },
        "score_semantics": [
            {
                "score_token": minter.mint("score", "synthetic"),
                "type": "numeric",
                "direction": "higher_better",
            }
        ],
    }
    correlation = {
        "schema_version": "1",
        "run_token": minter.mint("run", "synthetic"),
        "trials_total": 10,
        "trials_linked": 8,
        "trials_unknown": 2,
        "tier_counts": {
            "exact": 5,
            "commit_name": 2,
            "name_only": 1,
            "deployment": 0,
            "ambiguous": 0,
        },
        "agent_function_ref": "my_pkg.agents:answer_question",
        "agent_file_path": "src/my_pkg/agents.py",
    }
    return [_wire(run), _wire(dataset), _wire(correlation)]


def _leaf_paths(node: Any, path: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    if isinstance(node, dict):
        return [p for k, v in node.items() for p in _leaf_paths(v, path + (k,))]
    if isinstance(node, list):
        return [p for i, v in enumerate(node) for p in _leaf_paths(v, path + (i,))]
    return [path]


def _with_leaf(payload: Any, path: tuple[Any, ...], value: Any) -> Any:
    if not path:
        return value
    clone = json.loads(json.dumps(payload))
    node = clone
    for part in path[:-1]:
        node = node[part]
    node[path[-1]] = value
    return clone


def _pointer(path: tuple[Any, ...]) -> str:
    return "/" + "/".join(str(part) for part in path)


def _assert_rejected_at(
    harness: ConnectorHarness, payload: Mapping[str, Any], pointer: str
) -> None:
    try:
        harness.validate_summary(payload)
    except SummaryValidationError as exc:
        texts = [str(exc), repr(exc), *map(str, exc.args)]
        _check(
            not any(CANARY in text for text in texts),
            "gate_echoes_content",
            "rejection error must not echo submitted content",
        )
        pointers = str(exc).removeprefix("invalid summary at ").split(", ")
        _check(
            pointer in pointers,
            "gate_wrong_pointer",
            "rejection must be attributed to the canary field",
        )
    else:
        raise ConformanceFailure(
            "gate_accepts_content", "a summary carrying content must be rejected"
        )


def case_content_rejection(harness: ConnectorHarness) -> None:
    conn = harness.open_connection("content")
    summaries = _golden_summaries(harness, conn)
    for summary in summaries:  # control: otherwise-valid payloads pass
        try:
            harness.validate_summary(summary)
        except Exception:
            raise ConformanceFailure(
                "control_rejected", "a valid summary was rejected by the gate"
            ) from None
    for summary in summaries:
        for path in _leaf_paths(summary):
            _assert_rejected_at(
                harness, _with_leaf(summary, path, CANARY), _pointer(path)
            )
        _assert_rejected_at(
            harness, {**summary, "unexpected_field": CANARY}, "/<unknown>"
        )
        _assert_rejected_at(harness, {**summary, CANARY: 1}, "/<unknown>")


def case_connection_collision(harness: ConnectorHarness) -> None:
    a = harness.open_connection("conn-a")
    b = harness.open_connection("conn-b")
    _check(a.credential != b.credential, "credential_shared", "shared credential")
    _check(a.minter is not b.minter, "token_shared", "shared minter")
    _check(
        a.minter.connection_token.value != b.minter.connection_token.value,
        "token_shared",
        "connections share a connection token",
    )
    token = a.minter.mint("run", "synthetic")
    _check(not b.minter.owns(token), "token_shared", "token owned by both")
    _check(
        a.minter.digest("synthetic") != b.minter.digest("synthetic"),
        "token_shared",
        "digest keys are shared",
    )
    # Credentials actually used by requests, interleaved across connections.
    for connection in (a, b, a, b):
        mark = len(harness.request_credentials())
        harness.seed(connection, _rows(3, "c"), 2)
        harness.read_page(connection, None)
        harness.write(connection, _rows(1, connection.label), f"k-{mark}")
        used = harness.request_credentials()[mark:]
        _check(bool(used), "requests_unobserved", "no requests were recorded")
        _check(
            all(c == connection.credential for c in used),
            "credential_cross_use",
            "a request used another connection's credential",
        )
    # Identical idempotency keys on two connections must not collide.
    c = harness.open_connection("idem-a")
    d = harness.open_connection("idem-b")
    rows_c, rows_d = _rows(2, "ic"), _rows(2, "id")
    harness.write(c, rows_c, "same-key")
    harness.write(d, rows_d, "same-key")
    _check(
        _snapshot(harness, harness.committed(c)) == _snapshot(harness, rows_c)
        and _snapshot(harness, harness.committed(d)) == _snapshot(harness, rows_d),
        "idem_collision",
        "identical idempotency keys on two connections collided",
    )
    # Cross-connection data must not be visible.
    _check(
        not {harness.row_key(r) for r in rows_c}
        & {harness.row_key(r) for r in harness.committed(d)},
        "data_crossed",
        "data written on one connection appeared on another",
    )
    # The connector's serializer must refuse another connection's token.
    own = _token_objects_run(harness, b)
    try:
        harness.serialize_summary(b, own)
    except Exception:
        raise ConformanceFailure(
            "control_rejected", "own tokens were rejected"
        ) from None
    foreign = {**own, "run_token": token}
    try:
        harness.serialize_summary(b, foreign)
    except SummaryValidationError:
        return
    raise ConformanceFailure(
        "foreign_token_accepted", "a foreign connection's token must be rejected"
    )


CASES: dict[str, Case] = {
    "lost_response": case_lost_response,
    "crash_resume": case_crash_resume,
    "duplicate_rows": case_duplicate_rows,
    "overlapping_pages": case_overlapping_pages,
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

    def test_overlapping_pages(self) -> None:
        case_overlapping_pages(self.make_harness())

    def test_expired_cursor(self) -> None:
        case_expired_cursor(self.make_harness())

    def test_partial_batch(self) -> None:
        case_partial_batch(self.make_harness())

    def test_connection_collision(self) -> None:
        case_connection_collision(self.make_harness())

    def test_content_rejection(self) -> None:
        case_content_rejection(self.make_harness())
