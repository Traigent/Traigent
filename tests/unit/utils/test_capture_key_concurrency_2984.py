"""Concurrent examples keep their own capture key on one event loop (#2984)."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from traigent.utils.langchain_interceptor import (
    capture_key,
    capture_langchain_response,
    capture_scope,
    get_captured_response_by_key,
)


def test_concurrent_examples_store_responses_under_their_own_key() -> None:
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    responses = {"ex1": MagicMock(name="r1"), "ex2": MagicMock(name="r2")}

    async def example(key: str, wait_for: asyncio.Event, announce: asyncio.Event):
        with capture_key(key):
            announce.set()
            await wait_for.wait()
            capture_langchain_response(responses[key])

    async def main():
        with capture_scope():
            await asyncio.gather(
                example("ex1", second_started, first_started),
                example("ex2", first_started, second_started),
            )
            return (
                get_captured_response_by_key("ex1"),
                get_captured_response_by_key("ex2"),
            )

    first, second = asyncio.run(main())
    assert first is responses["ex1"]
    assert second is responses["ex2"]
