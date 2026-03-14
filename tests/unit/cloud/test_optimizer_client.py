import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from traigent.cloud.optimizer_client import OptimizerDirectClient


@pytest.mark.asyncio
async def test_submit_metrics_flushes_with_lock(monkeypatch):
    client = OptimizerDirectClient("https://api.example.com", "secret-token")
    client._batch_size = 5
    flushed_batches: list[list[tuple[str, dict]]] = []

    async def fake_flush_buffer():
        assert client._buffer_lock.locked()
        flushed_batches.append(list(client._metric_buffer))
        client._metric_buffer.clear()
        return {"status": "flushed", "count": len(flushed_batches[-1])}

    monkeypatch.setattr(client, "_flush_buffer", fake_flush_buffer)

    result = await client.submit_metrics("session-1", "trial-1", {"score": 0.5}, 1.2)
    assert result["status"] == "flushed"
    assert flushed_batches and len(flushed_batches[0]) == 1

    client._batch_size = 100  # keep batching off for the next calls
    flushed_batches.clear()

    async def submit(metric_value: float):
        return await client.submit_metrics(
            "session-2",
            f"trial-{metric_value}",
            {"score": metric_value},
            0.5,
        )

    await asyncio.gather(*(submit(v) for v in (0.1, 0.2, 0.3)))
    assert len(flushed_batches) == 3
    assert sum(len(batch) for batch in flushed_batches) == 3


def test_optimizer_client_requires_endpoint_and_token():
    with pytest.raises(ValueError):
        OptimizerDirectClient("", "token")
    with pytest.raises(ValueError):
        OptimizerDirectClient("https://api.example.com", " ")


@pytest.mark.asyncio
async def test_submit_metrics_validates_identifiers(monkeypatch):
    client = OptimizerDirectClient("https://api.example.com", "secret-token")

    with pytest.raises(ValueError):
        await client.submit_metrics(
            " ",
            "trial-1",
            {"score": 0.9},
            0.4,
        )

    with pytest.raises(ValueError):
        await client.submit_metrics(
            "session-1",
            "",
            {"score": 0.9},
            0.4,
        )


@pytest.mark.asyncio
async def test_context_exit_suppresses_cancelled_flush_task() -> None:
    client = OptimizerDirectClient("https://api.example.com", "secret-token")
    client.flush = AsyncMock()
    client.session = MagicMock()
    client.session.close = AsyncMock()

    async def wait_forever() -> None:
        await asyncio.sleep(3600)

    client._flush_task = asyncio.create_task(wait_forever())

    await client.__aexit__(None, None, None)

    assert client._flush_task.cancelled()
    client.flush.assert_awaited_once()
    client.session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_periodic_flush_reraises_cancelled_error(monkeypatch) -> None:
    client = OptimizerDirectClient("https://api.example.com", "secret-token")

    async def raise_cancelled(*_args, **_kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", raise_cancelled)

    with pytest.raises(asyncio.CancelledError):
        await client._periodic_flush()
