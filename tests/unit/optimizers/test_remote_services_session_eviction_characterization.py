"""Characterization tests for the local session-cache eviction in
``RemoteOptimizationService._validate_session``.

The tests drive the method unbound against a duck-typed receiver that exposes
only the attributes and methods the method reads, and record every access to
the session cache, the session timestamps and the logger, so the exact order
and repeat counts of those accesses are pinned. No remote API or database is
used.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Reliability FUNC-OPT-ALGORITHMS FUNC-CLOUD-HYBRID

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from traigent.optimizers import remote_services
from traigent.optimizers.remote_services import (
    OptimizationSessionStatus,
    RemoteOptimizationService,
    ServiceError,
)

Status = OptimizationSessionStatus
COMPLETED = Status.COMPLETED
FAILED = Status.FAILED
ACTIVE = Status.ACTIVE
INITIALIZING = Status.INITIALIZING


class _Stamp:
    """created_at stand-in that records timestamp() calls."""

    def __init__(self, name: str, value: float, events: list[Any]) -> None:
        self._name = name
        self._value = value
        self._events = events

    def timestamp(self) -> float:
        self._events.append(("timestamp", self._name))
        return self._value


class _Session:
    """Session stand-in that records status reads."""

    def __init__(self, name: str, status: Any, stamp: float, events: list[Any]) -> None:
        self.name = name
        self._status = status
        self._events = events
        self.created_at = _Stamp(name, stamp, events)

    @property
    def status(self) -> Any:
        self._events.append(("status", self.name))
        return self._status


class _Cache(dict):
    """dict that records the operations _validate_session performs on it."""

    def bind(self, events: list[Any]) -> None:
        self._events = events

    def __contains__(self, key: object) -> bool:
        self._events.append(("contains", key))
        return super().__contains__(key)

    def __len__(self) -> int:
        self._events.append(("len",))
        return super().__len__()

    def items(self):  # type: ignore[override]
        self._events.append(("items",))
        return super().items()

    def keys(self):  # type: ignore[override]
        self._events.append(("keys",))
        return super().keys()

    def __getitem__(self, key: Any) -> Any:
        self._events.append(("getitem", key))
        return super().__getitem__(key)

    def __delitem__(self, key: Any) -> None:
        self._events.append(("delitem", key))
        super().__delitem__(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        self._events.append(("setitem", key))
        super().__setitem__(key, value)


class _Receiver:
    """Duck-typed service exposing only what _validate_session reads."""

    def __init__(
        self,
        events: list[Any],
        limit: int,
        fetched: Any = None,
        fetch_error: BaseException | None = None,
    ) -> None:
        self._events = events
        self._limit = limit
        self._fetched = fetched
        self._fetch_error = fetch_error
        self._active_sessions = _Cache()
        self._active_sessions.bind(events)

    @property
    def _max_active_sessions(self) -> int:
        self._events.append(("limit",))
        return self._limit

    async def get_session(self, session_id: str) -> Any:
        self._events.append(("fetch", session_id))
        if self._fetch_error is not None:
            raise self._fetch_error
        return self._fetched


class _Logger:
    def __init__(self, events: list[Any], error: BaseException | None = None) -> None:
        self._events = events
        self._error = error

    def debug(self, message: str) -> None:
        self._events.append(("debug", message))
        if self._error is not None:
            raise self._error


def _build(
    specs: list[tuple[str, Any, float]],
    limit: int,
    monkeypatch: pytest.MonkeyPatch,
    fetched: Any = "FETCHED",
    fetch_error: BaseException | None = None,
    log_error: BaseException | None = None,
) -> tuple[_Receiver, list[Any]]:
    events: list[Any] = []
    receiver = _Receiver(events, limit, fetched, fetch_error)
    for name, status, stamp in specs:
        dict.__setitem__(
            receiver._active_sessions, name, _Session(name, status, stamp, events)
        )
    monkeypatch.setattr(remote_services, "logger", _Logger(events, log_error))
    return receiver, events


def _validate(receiver: Any, session_id: str) -> Any:
    return asyncio.run(
        RemoteOptimizationService._validate_session(receiver, session_id)
    )


def _removed(sid: str) -> tuple[str, str]:
    return (
        "debug",
        f"Removed session {sid} to stay within active session limit",
    )


def _keys(receiver: _Receiver) -> list[str]:
    return list(dict.keys(receiver._active_sessions))


def test_miss_below_capacity_inserts_without_eviction(monkeypatch):
    receiver, events = _build([("a", COMPLETED, 1.0)], 2, monkeypatch)

    assert _validate(receiver, "new") == "FETCHED"

    assert _keys(receiver) == ["a", "new"]
    assert events == [
        ("contains", "new"),
        ("fetch", "new"),
        ("len",),
        ("limit",),
        ("setitem", "new"),
    ]


def test_miss_at_capacity_evicts_one_then_inserts(monkeypatch):
    receiver, events = _build([("a", COMPLETED, 1.0)], 1, monkeypatch)

    assert _validate(receiver, "new") == "FETCHED"

    assert _keys(receiver) == ["new"]
    assert events == [
        ("contains", "new"),
        ("fetch", "new"),
        ("len",),
        ("limit",),
        ("items",),
        ("status", "a"),
        ("timestamp", "a"),
        ("timestamp", "a"),
        ("delitem", "a"),
        _removed("a"),
        ("setitem", "new"),
    ]


def test_oldest_inactive_preferred_over_older_active(monkeypatch):
    receiver, events = _build(
        [
            ("act_old", ACTIVE, 1.0),
            ("done_new", COMPLETED, 5.0),
            ("failed_mid", FAILED, 3.0),
            ("init_older", INITIALIZING, 0.5),
        ],
        4,
        monkeypatch,
    )

    assert _validate(receiver, "new") == "FETCHED"

    assert _keys(receiver) == ["act_old", "done_new", "init_older", "new"]
    assert events == [
        ("contains", "new"),
        ("fetch", "new"),
        ("len",),
        ("limit",),
        ("items",),
        ("status", "act_old"),
        ("status", "done_new"),
        ("timestamp", "done_new"),
        ("timestamp", "done_new"),
        ("status", "failed_mid"),
        ("timestamp", "failed_mid"),
        ("timestamp", "failed_mid"),
        ("status", "init_older"),
        ("delitem", "failed_mid"),
        _removed("failed_mid"),
        ("setitem", "new"),
    ]


def test_inactive_that_is_not_older_costs_one_timestamp_call(monkeypatch):
    receiver, events = _build(
        [("first", COMPLETED, 1.0), ("later", FAILED, 9.0)], 2, monkeypatch
    )

    _validate(receiver, "new")

    assert _keys(receiver) == ["later", "new"]
    assert events[5:11] == [
        ("status", "first"),
        ("timestamp", "first"),
        ("timestamp", "first"),
        ("status", "later"),
        ("timestamp", "later"),
        ("delitem", "first"),
    ]


def test_all_active_falls_back_to_oldest(monkeypatch):
    receiver, events = _build(
        [("b", ACTIVE, 7.0), ("a", INITIALIZING, 2.0), ("c", ACTIVE, 4.0)],
        3,
        monkeypatch,
    )

    assert _validate(receiver, "new") == "FETCHED"

    assert _keys(receiver) == ["b", "c", "new"]
    assert events == [
        ("contains", "new"),
        ("fetch", "new"),
        ("len",),
        ("limit",),
        ("items",),
        ("status", "b"),
        ("status", "a"),
        ("status", "c"),
        ("keys",),
        ("getitem", "b"),
        ("timestamp", "b"),
        ("getitem", "a"),
        ("timestamp", "a"),
        ("getitem", "c"),
        ("timestamp", "c"),
        ("delitem", "a"),
        _removed("a"),
        ("setitem", "new"),
    ]


@pytest.mark.parametrize(
    "status", [Status.CANCELLED, Status.PAUSED, Status.PENDING, Status.UNKNOWN]
)
def test_only_completed_and_failed_count_as_inactive(monkeypatch, status):
    receiver, _ = _build(
        [("odd", status, 1.0), ("done", COMPLETED, 9.0)], 2, monkeypatch
    )

    _validate(receiver, "new")

    assert _keys(receiver) == ["odd", "new"]


def test_equal_timestamps_among_inactive_pick_first_seen(monkeypatch):
    receiver, events = _build(
        [("x", FAILED, 3.0), ("y", COMPLETED, 3.0), ("z", COMPLETED, 3.0)],
        3,
        monkeypatch,
    )

    _validate(receiver, "new")

    assert _keys(receiver) == ["y", "z", "new"]
    assert ("timestamp", "y") in events
    assert events.count(("timestamp", "y")) == 1
    assert events.count(("timestamp", "z")) == 1
    assert events.count(("timestamp", "x")) == 2


def test_equal_timestamps_among_active_pick_first_seen(monkeypatch):
    receiver, _ = _build(
        [("x", ACTIVE, 3.0), ("y", ACTIVE, 3.0), ("z", INITIALIZING, 3.0)],
        3,
        monkeypatch,
    )

    _validate(receiver, "new")

    assert _keys(receiver) == ["y", "z", "new"]


def test_zero_capacity_with_empty_cache_is_translated_value_error(monkeypatch):
    receiver, events = _build([], 0, monkeypatch)
    with pytest.raises(ValueError) as expected:
        min([], key=lambda s: s)

    with pytest.raises(ServiceError) as excinfo:
        _validate(receiver, "new")

    assert str(excinfo.value) == f"Invalid or inactive session new: {expected.value}"
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__suppress_context__ is True
    assert _keys(receiver) == []
    assert events == [
        ("contains", "new"),
        ("fetch", "new"),
        ("len",),
        ("limit",),
        ("items",),
        ("keys",),
    ]


def test_zero_capacity_with_entries_evicts_before_insert(monkeypatch):
    receiver, _ = _build([("a", ACTIVE, 1.0)], 0, monkeypatch)

    assert _validate(receiver, "new") == "FETCHED"

    assert _keys(receiver) == ["new"]


def test_cached_initializing_and_active_returned_without_fetch(monkeypatch):
    for status in (INITIALIZING, ACTIVE):
        receiver, events = _build([("s", status, 1.0)], 1, monkeypatch)
        session = dict.__getitem__(receiver._active_sessions, "s")
        del events[:]

        assert _validate(receiver, "s") is session

        assert events == [("contains", "s"), ("getitem", "s"), ("status", "s")]
        assert ("fetch", "s") not in events
        assert ("limit",) not in events


@pytest.mark.parametrize(
    "status",
    [
        Status.COMPLETED,
        Status.FAILED,
        Status.CANCELLED,
        Status.PAUSED,
        Status.PENDING,
        Status.CREATED,
        Status.UNKNOWN,
    ],
)
def test_cached_rejected_statuses_raise_unwrapped_service_error(monkeypatch, status):
    receiver, events = _build([("s", status, 1.0)], 1, monkeypatch)
    del events[:]

    with pytest.raises(ServiceError) as excinfo:
        _validate(receiver, "s")

    assert str(excinfo.value) == f"Session s is not active (status: {status})"
    assert excinfo.value.__cause__ is None
    assert ("fetch", "s") not in events
    assert _keys(receiver) == ["s"]


def test_fetched_session_without_status_is_stored_and_returned(monkeypatch):
    fetched = object()
    receiver, _ = _build([], 5, monkeypatch, fetched=fetched)

    assert _validate(receiver, "new") is fetched
    assert dict.__getitem__(receiver._active_sessions, "new") is fetched


def test_fetched_none_is_stored_then_fails_status_check_on_next_call(monkeypatch):
    receiver, _ = _build([], 5, monkeypatch, fetched=None)

    assert _validate(receiver, "new") is None
    assert _keys(receiver) == ["new"]

    with pytest.raises(AttributeError):
        _validate(receiver, "new")


def test_empty_string_session_id_is_fetched_like_any_miss(monkeypatch):
    receiver, _ = _build([], 5, monkeypatch)

    assert _validate(receiver, "") == "FETCHED"
    assert _keys(receiver) == [""]


def test_fetch_error_is_wrapped_from_none_and_cache_untouched(monkeypatch):
    receiver, events = _build(
        [("a", COMPLETED, 1.0)], 1, monkeypatch, fetch_error=RuntimeError("boom")
    )

    with pytest.raises(ServiceError) as excinfo:
        _validate(receiver, "new")

    assert str(excinfo.value) == "Invalid or inactive session new: boom"
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__suppress_context__ is True
    assert _keys(receiver) == ["a"]
    assert events == [("contains", "new"), ("fetch", "new")]


def test_service_error_from_fetch_is_wrapped_again(monkeypatch):
    receiver, _ = _build([], 5, monkeypatch, fetch_error=ServiceError("gone"))

    with pytest.raises(ServiceError) as excinfo:
        _validate(receiver, "new")

    assert str(excinfo.value) == "Invalid or inactive session new: gone"
    assert excinfo.value.__cause__ is None


def test_limit_read_error_is_wrapped(monkeypatch):
    receiver, _ = _build([], 5, monkeypatch)

    class _BadLimit(_Receiver):
        @property
        def _max_active_sessions(self) -> int:
            raise RuntimeError("limit unavailable")

    bad = _BadLimit(receiver._events, 5, "FETCHED")
    bad._active_sessions = receiver._active_sessions

    with pytest.raises(ServiceError) as excinfo:
        _validate(bad, "new")

    assert str(excinfo.value) == "Invalid or inactive session new: limit unavailable"
    assert excinfo.value.__cause__ is None
    assert _keys(bad) == []


def test_timestamp_error_during_eviction_is_wrapped_and_nothing_changes(monkeypatch):
    receiver, _ = _build([("a", COMPLETED, 1.0)], 1, monkeypatch)

    def _boom() -> float:
        raise OverflowError("bad stamp")

    dict.__getitem__(receiver._active_sessions, "a").created_at.timestamp = _boom

    with pytest.raises(ServiceError) as excinfo:
        _validate(receiver, "new")

    assert str(excinfo.value) == "Invalid or inactive session new: bad stamp"
    assert excinfo.value.__cause__ is None
    assert _keys(receiver) == ["a"]


def test_status_error_during_eviction_is_wrapped(monkeypatch):
    receiver, _ = _build([("a", COMPLETED, 1.0)], 1, monkeypatch)

    class _NoStatus:
        @property
        def status(self) -> Any:
            raise LookupError("no status")

    dict.__setitem__(receiver._active_sessions, "a", _NoStatus())

    with pytest.raises(ServiceError) as excinfo:
        _validate(receiver, "new")

    assert str(excinfo.value) == "Invalid or inactive session new: no status"
    assert _keys(receiver) == ["a"]


def test_logging_error_is_wrapped_after_deletion_and_blocks_insert(monkeypatch):
    receiver, events = _build(
        [("a", COMPLETED, 1.0)], 1, monkeypatch, log_error=RuntimeError("log down")
    )

    with pytest.raises(ServiceError) as excinfo:
        _validate(receiver, "new")

    assert str(excinfo.value) == "Invalid or inactive session new: log down"
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__suppress_context__ is True
    assert _keys(receiver) == []
    assert events[-2:] == [("delitem", "a"), _removed("a")]
    assert ("setitem", "new") not in events


def test_insertion_error_is_wrapped(monkeypatch):
    receiver, _ = _build([], 5, monkeypatch)

    class _FullCache(_Cache):
        def __setitem__(self, key: Any, value: Any) -> None:
            raise RuntimeError("cannot store")

    cache = _FullCache()
    cache.bind(receiver._events)
    receiver._active_sessions = cache

    with pytest.raises(ServiceError) as excinfo:
        _validate(receiver, "new")

    assert str(excinfo.value) == "Invalid or inactive session new: cannot store"


def test_duck_typed_receiver_needs_only_old_attributes(monkeypatch):
    monkeypatch.setattr(remote_services, "logger", _Logger([]))

    class _Bare:
        _max_active_sessions = 1

        def __init__(self) -> None:
            self._active_sessions: dict[str, Any] = {
                "a": _Session("a", COMPLETED, 1.0, [])
            }

        async def get_session(self, session_id: str) -> str:
            return f"got-{session_id}"

    bare = _Bare()

    assert _validate(bare, "n") == "got-n"
    assert list(bare._active_sessions) == ["n"]
