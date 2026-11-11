"""A shared observe() factory entered by concurrent tasks ends each task's own context (#2984)."""

from __future__ import annotations

import asyncio
import itertools

import pytest

from traigent.observability import decorators
from traigent.observability.client import ObservabilityClient
from traigent.observability.config import ObservabilityConfig


def test_shared_factory_ends_the_context_its_task_entered(monkeypatch) -> None:
    counter = itertools.count(1)
    ended: list[tuple[str, int]] = []

    class FakeContext:
        def __init__(self, **_kwargs) -> None:
            self.ident = next(counter)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc) -> bool:
            ended.append((asyncio.current_task().get_name(), self.ident))
            return False

    monkeypatch.setattr(decorators, "ObserveContext", FakeContext)
    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000", enable_atexit_flush=False
        ),
        sender=lambda _batch: None,
    )
    factory = decorators.observe("handle", client=client)

    async def worker(first: asyncio.Event, second: asyncio.Event) -> int:
        async with factory as ctx:
            first.set()
            await second.wait()
            return ctx.ident

    async def main() -> list[int]:
        a_in, b_in = asyncio.Event(), asyncio.Event()
        task_a = asyncio.create_task(worker(a_in, b_in), name="a")
        await a_in.wait()
        task_b = asyncio.create_task(worker(b_in, b_in), name="b")
        return list(await asyncio.gather(task_a, task_b))

    idents = asyncio.run(main())
    assert sorted(idents) == [1, 2]
    assert sorted(ended) == [("a", idents[0]), ("b", idents[1])]


def test_exit_without_enter_still_raises() -> None:
    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000", enable_atexit_flush=False
        ),
        sender=lambda _batch: None,
    )
    factory = decorators.observe("handle", client=client)
    with pytest.raises(RuntimeError):
        factory.__exit__(None, None, None)


def test_entered_mapping_is_empty_after_exit_across_many_factories(monkeypatch) -> None:
    class FakeContext:
        def __init__(self, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc) -> bool:
            return False

    monkeypatch.setattr(decorators, "ObserveContext", FakeContext)
    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000", enable_atexit_flush=False
        ),
        sender=lambda _batch: None,
    )
    for _ in range(50):
        factory = decorators.observe("handle", client=client)
        with factory:
            with factory:
                pass
    assert not decorators._ENTERED_CONTEXTS.get()
