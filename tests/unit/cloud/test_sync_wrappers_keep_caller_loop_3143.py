"""Issue #3143: the session sync wrappers must leave the calling thread's
event loop exactly as they found it (never install, close or clear it)."""

from __future__ import annotations

import asyncio
import warnings
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from traigent.cloud import session_operations as session_ops_module
from traigent.cloud.session_operations import SessionOperations


@pytest.fixture
def user_loop() -> Iterator[asyncio.AbstractEventLoop]:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield loop
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def _current_loop() -> asyncio.AbstractEventLoop:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return asyncio.get_event_loop()


def _ops(monkeypatch) -> SessionOperations:
    ops = SessionOperations(SimpleNamespace(cloud_egress_intent=True))

    async def delete_session(session_id: str, cascade: bool = True) -> bool:
        return True

    async def finalize_session(session_id: str, *args: Any, **kwargs: Any) -> str:
        return "finalized"

    monkeypatch.setattr(ops, "delete_session", delete_session)
    monkeypatch.setattr(ops, "finalize_session", finalize_session)
    return ops


def _assert_user_loop_intact(loop: asyncio.AbstractEventLoop) -> None:
    assert _current_loop() is loop
    assert not loop.is_closed()

    async def ping() -> str:
        return "pong"

    assert loop.run_until_complete(ping()) == "pong"


def test_delete_session_sync_keeps_caller_loop(monkeypatch, user_loop) -> None:
    ops = _ops(monkeypatch)
    assert ops.delete_session_sync("s1") is True
    _assert_user_loop_intact(user_loop)


def test_finalize_session_sync_keeps_caller_loop(monkeypatch, user_loop) -> None:
    ops = _ops(monkeypatch)
    assert ops.finalize_session_sync("s1") == "finalized"
    _assert_user_loop_intact(user_loop)


def test_create_session_sync_keeps_caller_loop(monkeypatch, user_loop) -> None:
    ops = _ops(monkeypatch)
    sentinel = object()

    async def fake_create(self: Any, context: dict[str, Any]) -> object:
        return sentinel

    monkeypatch.setattr(session_ops_module, "is_backend_offline", lambda: False)
    monkeypatch.setattr(session_ops_module, "_create_session_async", fake_create)

    assert ops.create_session("fn", {"x": [1, 2]}) is sentinel
    _assert_user_loop_intact(user_loop)


@pytest.mark.parametrize("call", ["delete", "finalize"])
def test_wrappers_still_work_inside_a_running_loop(monkeypatch, call: str) -> None:
    ops = _ops(monkeypatch)
    invoke: Callable[[], Any] = (
        (lambda: ops.delete_session_sync("s1"))
        if call == "delete"
        else (lambda: ops.finalize_session_sync("s1"))
    )

    async def main() -> Any:
        return invoke()

    assert asyncio.run(main()) in (True, "finalized")
