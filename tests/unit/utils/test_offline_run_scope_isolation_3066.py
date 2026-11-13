"""offline_run_scope is scoped to its own run, not the whole process (#3066)."""

from __future__ import annotations

import contextvars
import threading

import pytest

from traigent.utils.env_config import is_backend_offline, offline_run_scope


@pytest.fixture(autouse=True)
def _clear_offline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "TRAIGENT_OFFLINE_MODE",
        "TRAIGENT_OFFLINE",
        "TRAIGENT_PRIVATE_DATA",
        "TRAIGENT_COMPLIANCE_MODE",
    ):
        monkeypatch.delenv(name, raising=False)


def _run_in_thread(fn, *, copy: bool = False):
    out: list[bool] = []
    ctx = contextvars.copy_context() if copy else contextvars.Context()
    t = threading.Thread(target=lambda: out.append(ctx.run(fn)))
    t.start()
    t.join()
    return out[0]


def test_online_run_is_not_offline_while_another_run_is_offline() -> None:
    inside = threading.Event()
    release = threading.Event()

    def offline_run() -> None:
        with offline_run_scope(True):
            inside.set()
            release.wait(5)

    a = threading.Thread(target=offline_run)
    a.start()
    assert inside.wait(5)
    try:

        def online_run() -> bool:
            with offline_run_scope(False):
                return is_backend_offline()

        assert _run_in_thread(online_run) is False
    finally:
        release.set()
        a.join()
    assert is_backend_offline() is False


def test_offline_run_is_offline_in_its_own_context_and_copied_threads() -> None:
    with offline_run_scope(True):
        assert is_backend_offline() is True
        assert _run_in_thread(is_backend_offline, copy=True) is True


def test_unscoped_thread_fails_closed_while_an_offline_run_is_in_flight() -> None:
    with offline_run_scope(True):
        assert _run_in_thread(is_backend_offline) is True
    assert _run_in_thread(is_backend_offline) is False


def test_nested_online_scope_does_not_lift_enclosing_offline_scope() -> None:
    with offline_run_scope(True):
        with offline_run_scope(False):
            assert is_backend_offline() is True


def test_env_offline_still_wins_inside_an_online_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    with offline_run_scope(False):
        assert is_backend_offline() is True


def test_optimize_sync_worker_inherits_the_callers_offline_scope() -> None:
    import asyncio
    from types import SimpleNamespace

    from traigent.core.optimized_function import OptimizedFunction

    seen: list[bool] = []

    async def fake_optimize(**_: object) -> str:
        # What the real run does first: enter its own scope, then egress checks.
        with offline_run_scope(False):
            seen.append(is_backend_offline())
        return "done"

    stub = SimpleNamespace(optimize=lambda **kw: fake_optimize(**kw))

    async def caller() -> str:
        with offline_run_scope(True):
            return OptimizedFunction.optimize_sync(stub)

    assert asyncio.run(caller()) == "done"
    assert seen == [True]


def test_compare_sync_worker_inherits_the_callers_offline_scope() -> None:
    import asyncio
    from types import SimpleNamespace

    from traigent.core.optimized_function import OptimizedFunction

    seen: list[bool] = []

    async def fake_compare(*_: object, **__: object) -> str:
        with offline_run_scope(False):
            seen.append(is_backend_offline())
        return "done"

    stub = SimpleNamespace(compare_async=fake_compare)

    async def caller() -> str:
        with offline_run_scope(True):
            return OptimizedFunction.compare(stub, [{"a": 1}])

    assert asyncio.run(caller()) == "done"
    assert seen == [True]
