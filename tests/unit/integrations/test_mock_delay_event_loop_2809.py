"""#2809: a configured mock latency (TRAIGENT_MOCK_DELAY_MS) must not block the
asyncio event loop. Async LiteLLM, LangChain and Bedrock mock paths await the
delay; synchronous callers still observe it."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any

import pytest

from traigent.integrations.utils.mock_adapter import MockAdapter

DELAY_MS = 60


@pytest.fixture(autouse=True)
def _delay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRAIGENT_MOCK_DELAY_MS", str(DELAY_MS))
    monkeypatch.setattr(
        MockAdapter,
        "is_mock_enabled",
        classmethod(lambda cls, provider: True),
    )


async def _heartbeat_ran_before_return(call: Callable[[], Awaitable[Any]]) -> bool:
    """True when an unrelated coroutine (5 ms timer) ran while ``call`` waited."""
    beat: dict[str, float] = {}

    async def heartbeat() -> None:
        await asyncio.sleep(0.005)
        beat["t"] = time.perf_counter()

    task = asyncio.create_task(heartbeat())
    await call()
    ran = "t" in beat
    await task
    return ran


async def _litellm_call() -> Any:
    from traigent.utils.litellm_interceptor import _acompletion_once

    async def original(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise AssertionError("provider must not be called in mock mode")

    return await _acompletion_once(original, (), {"model": "gpt-4o-mini"})


async def _langchain_ainvoke() -> Any:
    pytest.importorskip("langchain_core")
    from traigent.utils.langchain_interceptor import _create_ainvoke_wrapper

    async def original(self: Any, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise AssertionError("provider must not be called in mock mode")

    wrapper = _create_ainvoke_wrapper(original, "openai")
    return await wrapper(SimpleNamespace(model_name="gpt-4o-mini"), "hi")


async def _langchain_astream() -> Any:
    pytest.importorskip("langchain_core")
    from traigent.utils.langchain_interceptor import _create_astream_wrapper

    def original(self: Any, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise AssertionError("provider must not be called in mock mode")

    wrapper = _create_astream_wrapper(original, "openai")
    return [c async for c in wrapper(SimpleNamespace(model_name="gpt-4o-mini"), "hi")]


async def _bedrock_ainvoke() -> Any:
    from traigent.integrations.bedrock_client import BedrockChatClient

    return await BedrockChatClient(client=object()).ainvoke(
        model_id="anthropic.claude-3-haiku", messages="hi"
    )


ASYNC_PATHS = [_litellm_call, _langchain_ainvoke, _langchain_astream, _bedrock_ainvoke]


@pytest.mark.parametrize("path", ASYNC_PATHS, ids=lambda f: f.__name__)
def test_other_coroutines_progress_while_an_async_mock_response_waits(path) -> None:
    assert asyncio.run(_heartbeat_ran_before_return(path)) is True


@pytest.mark.parametrize("path", ASYNC_PATHS, ids=lambda f: f.__name__)
def test_concurrent_async_mock_responses_overlap_their_delay(path) -> None:
    async def run() -> float:
        start = time.perf_counter()
        await asyncio.gather(*(path() for _ in range(6)))
        return time.perf_counter() - start

    elapsed = asyncio.run(run())
    serial = 6 * DELAY_MS / 1000
    assert elapsed < serial * 0.6, f"{elapsed:.3f}s is not overlapped (serial={serial})"


@pytest.mark.parametrize("path", ASYNC_PATHS, ids=lambda f: f.__name__)
def test_async_mock_response_still_waits_the_configured_delay(path) -> None:
    start = time.perf_counter()
    asyncio.run(path())
    assert time.perf_counter() - start >= DELAY_MS / 1000 * 0.9


def test_synchronous_mock_calls_still_observe_the_delay() -> None:
    start = time.perf_counter()
    MockAdapter.get_mock_response("openai", model="m")
    assert time.perf_counter() - start >= DELAY_MS / 1000 * 0.9


def test_apply_delay_false_skips_the_delay() -> None:
    start = time.perf_counter()
    MockAdapter.get_mock_response("openai", model="m", apply_delay=False)
    assert time.perf_counter() - start < DELAY_MS / 1000 * 0.5


def test_no_detached_sleep_task_is_scheduled_in_a_running_loop() -> None:
    async def run() -> int:
        before = len(asyncio.all_tasks())
        MockAdapter.get_mock_response("openai", model="m")
        return len(asyncio.all_tasks()) - before

    assert asyncio.run(run()) == 0
