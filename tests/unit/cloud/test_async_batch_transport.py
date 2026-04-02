from __future__ import annotations

import asyncio

import pytest

from traigent.cloud.async_batch_transport import AsyncBatchTransport


@pytest.mark.asyncio
async def test_transport_remains_reusable_after_completed_flush():
    sent_payloads: list[list[dict[str, int]]] = []

    async def sender(payloads: list[dict[str, int]]) -> dict[str, list[str]]:
        sent_payloads.append(list(payloads))
        await asyncio.sleep(0.01)
        return {"warnings": []}

    transport = AsyncBatchTransport(sender, batch_size=2, max_buffer_age=10.0)

    await transport.submit("first", {"value": 1})
    await transport.submit("second", {"value": 2})
    first_result = await transport.flush()

    await transport.submit("third", {"value": 3})
    second_result = await transport.flush()

    assert first_result.items_sent == 2
    assert first_result.items_pending == 0
    assert second_result.items_sent == 3
    assert second_result.items_pending == 0
    assert sent_payloads == [
        [{"value": 1}, {"value": 2}],
        [{"value": 3}],
    ]


@pytest.mark.asyncio
async def test_transport_close_completes_during_concurrent_submit_and_flush():
    async def sender(payloads: list[dict[str, int]]) -> dict[str, list[str]]:
        await asyncio.sleep(0.005)
        return {"warnings": []}

    transport = AsyncBatchTransport(
        sender,
        batch_size=3,
        max_buffer_age=0.01,
        max_queue_size=100,
    )

    async def producer(prefix: str) -> None:
        for index in range(10):
            await asyncio.sleep(0.001)
            await transport.submit(f"{prefix}-{index}", {"value": index})

    async def flusher() -> None:
        for _ in range(5):
            await asyncio.sleep(0.002)
            await transport.flush()

    producer_tasks = [
        asyncio.create_task(producer("a")),
        asyncio.create_task(producer("b")),
    ]
    flush_task = asyncio.create_task(flusher())

    await asyncio.sleep(0.004)
    close_task = asyncio.create_task(transport.close())

    await asyncio.wait_for(
        asyncio.gather(*producer_tasks, flush_task, return_exceptions=True),
        timeout=2,
    )
    close_result = await asyncio.wait_for(close_task, timeout=2)

    assert close_result.items_pending == 0
    assert close_result.failed_batches == 0
    assert close_result.items_dropped == 0
