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


# ---------------------------------------------------------------------------
# Through the REAL entry points (red-team: the tests above all call
# _flush_buffer directly, which is exactly why the two regressions below
# slipped through the first revision)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.backend_online  # goes through submit_metrics; transports still mocked
async def test_a_backlog_does_not_re_arm_the_batch_size_trigger(monkeypatch) -> None:
    """Retained metrics must not make every later trial re-POST synchronously.

    The first revision left the buffer sitting at `batch_size` after a failed
    flush, so `submit_metrics` fired a full flush on EVERY subsequent trial --
    a growing payload, POSTed while holding the lock. With a 30s client
    timeout that injects minutes of blocking into the hybrid trial loop.
    """
    client = _client()
    client._batch_size = 10
    posts = 0

    async def fake(session_id, *args):
        nonlocal posts
        posts += 1
        raise TimeoutError("backend down")

    monkeypatch.setattr(client, "_submit_single", fake)
    monkeypatch.setattr(client, "_submit_batch", fake)

    for i in range(20):
        await client.submit_metrics("s1", f"t{i}", {"a": 1.0}, 0.1)

    assert posts == 2, (
        f"expected one flush per full batch of 10 (2 total), got {posts} -- "
        f"the retained backlog is re-triggering the size-based flush"
    )
    assert len(client._metric_buffer) == 20, "nothing should have been dropped"


@pytest.mark.asyncio
@pytest.mark.backend_online  # goes through submit_metrics; transports still mocked
async def test_a_permanently_rejected_batch_does_not_poison_the_session(
    monkeypatch,
) -> None:
    """A 4xx must be discarded, not retained.

    `_submit_batch` is all-or-nothing and retained entries regroup with later
    ones under the same session_id, so retaining a permanent rejection would
    block every subsequent metric for that session forever -- losing strictly
    more than the unconditional clear this fix replaced.
    """
    client = _client()
    client._batch_size = 10
    delivered: list[str] = []

    async def fake_batch(session_id, submissions):
        if any(s["trial_id"] == "poison" for s in submissions):
            raise ValueError("422 unprocessable entity")
        delivered.extend(s["trial_id"] for s in submissions)
        return {"status": "ok"}

    monkeypatch.setattr(client, "_submit_batch", fake_batch)
    monkeypatch.setattr(client, "_submit_single", fake_batch)

    await client.submit_metrics("s1", "poison", {"a": 1.0}, 0.1)
    for i in range(20):
        await client.submit_metrics("s1", f"g{i}", {"a": 1.0}, 0.1)

    assert delivered, "the permanent rejection blocked every later metric"
    assert "poison" not in delivered
    assert len(client._metric_buffer) < 5, (
        "the rejected batch is still buffered and will be retried forever"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exc", "retryable"),
    [
        (TimeoutError("timeout"), True),
        (ConnectionResetError("reset"), True),
        (ValueError("403 forbidden: token expired"), False),
        (RuntimeError("Session not initialized"), False),
    ],
    ids=["timeout", "conn-reset", "permanent-403", "permanent-runtime"],
)
async def test_only_retryable_failures_are_retained(
    monkeypatch, exc, retryable
) -> None:
    client = _client()
    client._metric_buffer = [("s1", {"trial_id": "t1"})]

    async def fake_single(session_id, submission):
        raise exc

    monkeypatch.setattr(client, "_submit_single", fake_single)

    response = await client._flush_buffer()

    assert response["errors"][0]["retryable"] is retryable
    assert bool(client._metric_buffer) is retryable
    assert response["abandoned_permanent"] == (0 if retryable else 1)


@pytest.mark.asyncio
async def test_the_cap_drops_by_age_not_by_session_grouping(monkeypatch) -> None:
    """The cap slices the buffer in chronological order.

    Grouping by session happens before submission, so slicing the grouped
    order would have discarded the newest metric in the buffer while keeping
    older ones, and wiped whole sessions rather than trimming uniformly.
    """
    client = _client()
    client._max_retained_metrics = 3
    client._metric_buffer = [
        ("s1", {"trial_id": "oldest"}),
        ("s2", {"trial_id": "b"}),
        ("s2", {"trial_id": "c"}),
        ("s2", {"trial_id": "d"}),
        ("s1", {"trial_id": "newest"}),
    ]

    async def fake(session_id, *args):
        raise TimeoutError("down")

    monkeypatch.setattr(client, "_submit_single", fake)
    monkeypatch.setattr(client, "_submit_batch", fake)

    await client._flush_buffer()

    kept = [s["trial_id"] for _, s in client._metric_buffer]
    assert kept == ["c", "d", "newest"], f"cap did not slice by age: {kept}"
