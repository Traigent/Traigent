"""Bridge timeouts must not be held back by joining a slow worker (#3142)."""

from __future__ import annotations

import asyncio
import concurrent.futures
import time
from collections.abc import Iterator
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from traigent.cloud.backend_client import BackendIntegratedClient
from traigent.cloud.session_operations import SessionOperations
from traigent.cloud.trial_operations import TrialOperations
from traigent.integrations.bedrock_client import BedrockChatClient
from traigent.utils.thread_timeout import run_in_thread_with_timeout

# Opt into the connected code paths (see pyproject markers).
pytestmark = pytest.mark.backend_online

SLOW = 1.5
FAST_TIMEOUT = 0.3
BOUND = 1.0  # well under SLOW, well over FAST_TIMEOUT


@pytest.fixture
def short_future_timeouts() -> Iterator[None]:
    """Shrink the 60 s / 30 s ``Future.result`` timeouts the bridges use."""
    original = concurrent.futures.Future.result

    def result(self: Any, timeout: float | None = None) -> Any:
        if timeout in (60, 30.0):
            timeout = FAST_TIMEOUT
        return original(self, timeout)

    with patch.object(concurrent.futures.Future, "result", result):
        yield


async def _slow_async(*_a: Any, **_k: Any) -> Any:
    await asyncio.sleep(SLOW)
    return True


def _in_loop(fn: Any) -> Any:
    async def main() -> Any:
        started = time.perf_counter()
        result = fn()
        return result, time.perf_counter() - started

    return asyncio.run(main())


def test_helper_returns_at_timeout_not_after_worker() -> None:
    started = time.perf_counter()
    with pytest.raises(TimeoutError):
        run_in_thread_with_timeout(lambda: time.sleep(SLOW), FAST_TIMEOUT)
    assert time.perf_counter() - started < BOUND


def test_register_trial_start_sync_timeout_is_enforced(
    short_future_timeouts: None,
) -> None:
    ops = TrialOperations(cast(Any, MagicMock()))
    with patch.object(ops, "register_trial_start", _slow_async):
        result, elapsed = _in_loop(lambda: ops.register_trial_start_sync("s", "t", {}))
    assert result is False
    assert elapsed < BOUND


def test_finalize_session_sync_timeout_is_enforced(short_future_timeouts: None) -> None:
    ops = SessionOperations(cast(Any, MagicMock()))
    with patch.object(ops, "finalize_session", _slow_async):
        result, elapsed = _in_loop(lambda: ops.finalize_session_sync("sess"))
    assert result is None
    assert elapsed < BOUND


def test_delete_session_sync_timeout_is_enforced(short_future_timeouts: None) -> None:
    ops = SessionOperations(cast(Any, MagicMock()))
    with patch.object(ops, "delete_session", _slow_async):
        result, elapsed = _in_loop(lambda: ops.delete_session_sync("sess"))
    assert result is False
    assert elapsed < BOUND


def test_sync_auth_headers_timeout_is_enforced(short_future_timeouts: None) -> None:
    client = object.__new__(BackendIntegratedClient)
    client.timeout = 60.0

    async def slow_headers(target: Any = None) -> Any:
        await asyncio.sleep(SLOW)
        return {}

    client.auth_manager = MagicMock(auth=MagicMock(get_headers=slow_headers))

    def call() -> Any:
        try:
            client._get_sync_auth_headers()
        except Exception as exc:  # noqa: BLE001 - fail-closed error expected
            return exc
        return None

    raised, elapsed = _in_loop(call)
    assert raised is not None
    assert elapsed < BOUND


class _SlowBedrock(BedrockChatClient):
    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        time.sleep(SLOW)
        return MagicMock()

    def invoke_stream(self, *args: Any, **kwargs: Any) -> Any:
        time.sleep(SLOW)
        return iter(())


async def _cancel_elapsed(coro: Any) -> float:
    started = time.perf_counter()
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(coro, timeout=FAST_TIMEOUT)
    return time.perf_counter() - started


def test_bedrock_ainvoke_cancel_does_not_join_worker() -> None:
    client = _SlowBedrock(client=MagicMock())

    async def main() -> float:
        with (
            patch.dict("sys.modules", {"aioboto3": None}),
            patch.object(client, "_is_mock_enabled", return_value=False),
        ):
            return await _cancel_elapsed(
                client.ainvoke(model_id="anthropic.claude-3", messages="hi")
            )

    assert asyncio.run(main()) < BOUND


def test_bedrock_ainvoke_stream_cancel_does_not_join_worker() -> None:
    client = _SlowBedrock(client=MagicMock())

    async def consume() -> None:
        async for _ in client.ainvoke_stream(
            model_id="anthropic.claude-3", messages="hi"
        ):
            pass

    async def main() -> float:
        with (
            patch.dict("sys.modules", {"aioboto3": None}),
            patch.object(client, "_is_mock_enabled", return_value=False),
        ):
            return await _cancel_elapsed(consume())

    assert asyncio.run(main()) < BOUND


def test_auth_no_aiohttp_validation_cancel_does_not_join_worker() -> None:
    from traigent.cloud import auth as auth_module
    from traigent.cloud.auth import AuthManager

    manager = object.__new__(AuthManager)
    manager.no_egress = False
    manager.config = MagicMock(backend_base_url="https://backend.example.test")

    def slow_sync(*_a: Any, **_k: Any) -> None:
        time.sleep(SLOW)

    async def allow_url(url: str, **_k: Any) -> str:
        return url

    async def main() -> float:
        with (
            patch.object(auth_module, "AIOHTTP_AVAILABLE", False),
            patch.object(auth_module, "validate_cloud_base_url_async", allow_url),
            patch(
                "traigent.cloud.client.cloud_backend_egress_disabled",
                return_value=False,
            ),
            patch.object(manager, "_validate_api_key_with_backend_sync", slow_sync),
        ):
            return await _cancel_elapsed(
                manager._validate_api_key_with_backend("tg_" + "x" * 61)
            )

    assert asyncio.run(main()) < BOUND
