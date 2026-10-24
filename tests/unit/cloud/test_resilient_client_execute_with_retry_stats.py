"""Characterization tests for ResilientClient.execute_with_retry bookkeeping."""

import asyncio

import pytest

from traigent.cloud.resilient_client import ResilientClient
from traigent.utils.exceptions import RetryableError

pytestmark = pytest.mark.backend_online


def _client(max_retries: int = 2) -> ResilientClient:
    return ResilientClient(
        max_retries=max_retries, base_delay=0.1, max_delay=0.1, jitter_factor=0.0
    )


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    async def _instant(_delay):
        return None

    monkeypatch.setattr(asyncio, "sleep", _instant)


def test_first_attempt_success_records_no_retries():
    client = _client()

    async def op():
        return "ok"

    assert asyncio.run(client.execute_with_retry(op)) == "ok"
    stats = client._retry_stats
    assert stats["total_requests"] == 1
    assert stats["successful_requests"] == 1
    assert stats["total_retries"] == 0
    assert stats["retry_successes"] == 0
    assert stats["failed_requests"] == 0


def test_success_after_retries_counts_each_retry_and_one_retry_success():
    client = _client(max_retries=3)
    calls = {"n": 0}

    async def op():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RetryableError("transient")
        return "done"

    assert asyncio.run(client.execute_with_retry(op)) == "done"
    stats = client._retry_stats
    assert calls["n"] == 3
    assert stats["total_retries"] == 2
    assert stats["retry_successes"] == 1
    assert stats["successful_requests"] == 1
    assert stats["failed_requests"] == 0


def test_exhausted_retries_count_failure_and_reraise_last_error():
    client = _client(max_retries=2)
    calls = {"n": 0}

    async def op():
        calls["n"] += 1
        raise RetryableError("still failing")

    with pytest.raises(RetryableError, match="still failing"):
        asyncio.run(client.execute_with_retry(op))
    stats = client._retry_stats
    assert calls["n"] == 3
    assert stats["total_retries"] == 2
    assert stats["retry_successes"] == 0
    assert stats["successful_requests"] == 0
    assert stats["failed_requests"] == 1
