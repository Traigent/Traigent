"""Issue #2988: a session's buffered metrics are posted in batches of at most
200 submissions (the service's per-request cap), and each chunk's outcome is
handled on its own so accepted chunks are never re-sent."""

from __future__ import annotations

import pytest

from traigent.cloud import optimizer_client as optimizer_client_module
from traigent.cloud.optimizer_client import OptimizerDirectClient


def _client() -> OptimizerDirectClient:
    return OptimizerDirectClient("https://api.example.com", "secret-token")


def _fill(client: OptimizerDirectClient, session_id: str, count: int) -> None:
    client._metric_buffer.extend(
        (session_id, {"trial_id": f"{session_id}-t{i}"}) for i in range(count)
    )


class _Transient(OSError):
    pass


def test_cap_matches_the_service_limit() -> None:
    assert optimizer_client_module._MAX_SUBMISSIONS_PER_BATCH == 200


@pytest.mark.asyncio
async def test_large_backlog_is_split_into_capped_batches(monkeypatch) -> None:
    client = _client()
    _fill(client, "s1", 450)
    sizes: list[int] = []

    async def fake_batch(session_id, submissions):
        if len(submissions) > 200:
            raise AssertionError("batch exceeds the service cap")
        sizes.append(len(submissions))
        return {"status": "ok", "count": len(submissions)}

    monkeypatch.setattr(client, "_submit_batch", fake_batch)

    response = await client._flush_buffer()

    assert sizes == [200, 200, 50]
    assert client._metric_buffer == []
    assert response["status"] == "batch_flushed"
    assert response["sessions_processed"] == 1


@pytest.mark.asyncio
async def test_transient_failure_retains_only_unsent_chunks(monkeypatch) -> None:
    client = _client()
    _fill(client, "s1", 450)
    sent: list[str] = []

    async def fake_batch(session_id, submissions):
        if sent:  # the second chunk hits an outage
            raise _Transient("backend blip")
        sent.extend(s["trial_id"] for s in submissions)
        return {"status": "ok"}

    monkeypatch.setattr(client, "_submit_batch", fake_batch)

    response = await client._flush_buffer()

    assert len(sent) == 200
    retained = [s["trial_id"] for _, s in client._metric_buffer]
    # Accepted chunk is gone; the failed chunk and the unsent one stay, in order.
    assert retained == [f"s1-t{i}" for i in range(200, 450)]
    assert response["retained_for_retry"] == 250
    assert response["abandoned_permanent"] == 0


@pytest.mark.asyncio
async def test_permanent_rejection_abandons_only_its_chunk(monkeypatch) -> None:
    client = _client()
    _fill(client, "s1", 401)
    calls: list[int] = []

    async def fake_batch(session_id, submissions):
        calls.append(len(submissions))
        if len(calls) == 1:
            raise ValueError("rejected")
        return {"status": "ok"}

    async def fake_single(session_id, submission):
        raise AssertionError("a multi-submission session must use the batch route")

    monkeypatch.setattr(client, "_submit_batch", fake_batch)
    monkeypatch.setattr(client, "_submit_single", fake_single)

    response = await client._flush_buffer()

    assert calls == [200, 200, 1]
    assert client._metric_buffer == []
    assert response["abandoned_permanent"] == 200
    assert response["retained_for_retry"] == 0
