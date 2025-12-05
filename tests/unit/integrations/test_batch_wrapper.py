"""Tests for batch wrapper utilities."""

import asyncio

import pytest

from traigent.integrations.batch_wrapper import BatchWrapper, attach_batch_methods


def test_batch_wrapper_sync_with_mappings():
    def multiply(**kwargs):
        return kwargs["value"] * 2

    wrapper = BatchWrapper(multiply)
    results = wrapper.batch([{"value": 1}, {"value": 3}])

    assert results == [2, 6]


def test_batch_wrapper_sync_with_item_param():
    def square(value: int) -> int:
        return value * value

    wrapper = BatchWrapper(square, item_param="value")
    results = wrapper.batch([2, 4, 6])

    assert results == [4, 16, 36]


@pytest.mark.asyncio
async def test_batch_wrapper_async_with_concurrency():
    call_order: list[int] = []

    async def add_one(value: int) -> int:
        call_order.append(value)
        await asyncio.sleep(0)  # allow context switch
        return value + 1

    wrapper = BatchWrapper(add_one, item_param="value", concurrency=2)
    results = await wrapper.abatch([1, 2, 3])

    assert results == [2, 3, 4]
    assert call_order == [1, 2, 3]  # Preserves input ordering


@pytest.mark.asyncio
async def test_attach_batch_methods():
    class Dummy:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def invoke(self, value: int) -> int:
            self.calls.append(value)
            return value * 10

    dummy = Dummy()
    attach_batch_methods(dummy, method_name="invoke", item_param="value")

    assert dummy.batch([1, 2]) == [10, 20]
    async_results = await dummy.abatch([3, 4])

    assert async_results == [30, 40]
    assert dummy.calls == [1, 2, 3, 4]
