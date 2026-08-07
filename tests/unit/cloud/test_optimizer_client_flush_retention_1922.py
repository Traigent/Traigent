"""Issue #1922: a failed flush must not discard the metrics it failed to send.

`_flush_buffer` grouped the buffer by session, POSTed each group, caught any
exception into an `errors` list -- and then cleared the WHOLE buffer, including
the metrics of the session that had just failed. One transient 5xx or timeout
therefore silently dropped up to `batch_size` (default 100) trials' worth of
optimizer signal.

Nothing upstream caught it either: the `@backoff` decorator sits on
`submit_metrics`, but `_flush_buffer` swallows the POST exception and returns a
`partial_success` dict instead of raising, so backoff never re-fires; and the
periodic flush and the `__aexit__` final flush bypass the decorated method
entirely. The run still reported "completed", with the optimizer's frontier
computed from a silently truncated set.

Scope: this pins the retention only. Whether hybrid submission should gain a
durable local outbox with retry-on-reconnect is the open owner question in the
issue and is deliberately NOT decided here -- retained metrics ride the next
periodic flush or the final flush, in memory, and are lost if the process dies.
"""

from __future__ import annotations

import logging

import pytest

from traigent.cloud.optimizer_client import OptimizerDirectClient


def _client() -> OptimizerDirectClient:
    return OptimizerDirectClient("https://api.example.com", "secret-token")


def _buffered_sessions(client: OptimizerDirectClient) -> list[str]:
    return [session_id for session_id, _ in client._metric_buffer]


@pytest.mark.asyncio
async def test_failed_session_metrics_stay_buffered(monkeypatch) -> None:
    """The whole point: a transient failure must not consume the metrics."""
    client = _client()
    client._metric_buffer = [
        ("session-ok", {"trial_id": "t1"}),
        ("session-bad", {"trial_id": "t2"}),
        ("session-bad", {"trial_id": "t3"}),
    ]

    async def fake_batch(session_id, submissions):
        raise TimeoutError("backend blip")

    async def fake_single(session_id, submission):
        if session_id == "session-bad":
            raise TimeoutError("backend blip")
        return {"status": "ok", "session_id": session_id}

    monkeypatch.setattr(client, "_submit_single", fake_single)
    monkeypatch.setattr(client, "_submit_batch", fake_batch)

    response = await client._flush_buffer()

    assert response["status"] == "partial_success"
    assert response["retained_for_retry"] == 2
    assert response["dropped_over_cap"] == 0

    # The successful session is gone; the failed one is still here, intact.
    assert _buffered_sessions(client) == ["session-bad", "session-bad"]
    assert [s["trial_id"] for _, s in client._metric_buffer] == ["t2", "t3"]


@pytest.mark.asyncio
async def test_retained_metrics_are_resubmitted_on_the_next_flush(
    monkeypatch,
) -> None:
    """Retention is only worth anything if the next flush actually sends them."""
    client = _client()
    client._metric_buffer = [("session-bad", {"trial_id": "t1"})]

    attempts: list[str] = []
    fail = True

    async def fake_single(session_id, submission):
        attempts.append(submission["trial_id"])
        if fail:
            raise TimeoutError("backend blip")
        return {"status": "ok"}

    monkeypatch.setattr(client, "_submit_single", fake_single)

    await client._flush_buffer()
    assert client._metric_buffer, "metrics were dropped, so nothing can retry"

    fail = False
    response = await client._flush_buffer()

    assert attempts == ["t1", "t1"], "the retained metric was never re-sent"
    assert response["status"] == "ok"
    assert client._metric_buffer == [], "a successful flush must drain the buffer"


@pytest.mark.asyncio
async def test_a_fully_successful_flush_still_drains_the_buffer(monkeypatch) -> None:
    """Guards the obvious regression: retention must not leak on success."""
    client = _client()
    client._metric_buffer = [
        ("session-a", {"trial_id": "t1"}),
        ("session-b", {"trial_id": "t2"}),
    ]

    async def fake_single(session_id, submission):
        return {"status": "ok", "session_id": session_id}

    monkeypatch.setattr(client, "_submit_single", fake_single)

    await client._flush_buffer()

    assert client._metric_buffer == []


@pytest.mark.asyncio
async def test_retention_is_bounded_and_the_loss_is_logged(monkeypatch, caplog) -> None:
    """A backend that stays down must not grow the buffer without limit.

    Dropping is still loss, so it is logged at ERROR and counted in the
    response -- bounded, but never silent.
    """
    client = _client()
    client._max_retained_metrics = 5
    client._metric_buffer = [("session-bad", {"trial_id": f"t{i}"}) for i in range(8)]

    async def fake_batch(session_id, submissions):
        raise TimeoutError("backend down")

    monkeypatch.setattr(client, "_submit_batch", fake_batch)

    with caplog.at_level(logging.ERROR):
        response = await client._flush_buffer()

    assert len(client._metric_buffer) == 5
    assert response["retained_for_retry"] == 5
    assert response["dropped_over_cap"] == 3
    assert "discarded the 3 oldest buffered metrics" in caplog.text

    # Oldest dropped, newest kept: the freshest signal is the useful one.
    assert [s["trial_id"] for _, s in client._metric_buffer] == [
        "t3",
        "t4",
        "t5",
        "t6",
        "t7",
    ]


@pytest.mark.asyncio
async def test_partial_failure_keeps_only_the_failed_session(monkeypatch) -> None:
    """Three sessions, one fails -- the other two must not be re-sent later."""
    client = _client()
    client._metric_buffer = [
        ("s1", {"trial_id": "a"}),
        ("s2", {"trial_id": "b"}),
        ("s3", {"trial_id": "c"}),
    ]

    async def fake_single(session_id, submission):
        if session_id == "s2":
            raise ConnectionError("reset")
        return {"status": "ok", "session_id": session_id}

    monkeypatch.setattr(client, "_submit_single", fake_single)

    response = await client._flush_buffer()

    assert response["successful"] == 2
    assert response["failed"] == 1
    assert _buffered_sessions(client) == ["s2"]
