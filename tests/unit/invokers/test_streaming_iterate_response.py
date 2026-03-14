"""Tests for uncovered lines in StreamingInvoker._iterate_response.

Covers:
  1. Async iterator with chunk_timeout=None (the ``else`` branch at line ~201)
  2. Async iterator with TimeoutError (the error handler at line ~205)
"""

from __future__ import annotations

import asyncio

import pytest

from traigent.invokers.streaming import StreamingInvoker
from traigent.utils.exceptions import InvocationError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _collect(async_iter):
    """Collect all items from an async iterator."""
    items = []
    async for item in async_iter:
        items.append(item)
    return items


class _AsyncIter:
    """Async iterator that yields a sequence of items."""

    def __init__(self, items):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration


class _SlowAsyncIter:
    """Async iterator that sleeps longer than the chunk timeout."""

    def __init__(self, delay: float):
        self._delay = delay
        self._called = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._called:
            raise StopAsyncIteration
        self._called = True
        await asyncio.sleep(self._delay)
        return "never_reached"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_iterate_response_async_without_chunk_timeout():
    """Async iterator path with chunk_timeout=None (line ~201)."""
    invoker = StreamingInvoker(timeout=60.0, chunk_timeout=30.0)
    # Override chunk_timeout to None after construction validation
    invoker.chunk_timeout = None

    response = _AsyncIter(["chunk_a", "chunk_b", "chunk_c"])
    result = await _collect(invoker._iterate_response(response))

    assert result == ["chunk_a", "chunk_b", "chunk_c"]


@pytest.mark.asyncio
async def test_iterate_response_async_timeout_raises_invocation_error():
    """TimeoutError from wait_for must become InvocationError (line ~205)."""
    # Use a very small chunk_timeout so it triggers quickly
    invoker = StreamingInvoker(timeout=60.0, chunk_timeout=0.01)

    response = _SlowAsyncIter(delay=5.0)

    with pytest.raises(InvocationError, match="Chunk timeout"):
        await _collect(invoker._iterate_response(response))
