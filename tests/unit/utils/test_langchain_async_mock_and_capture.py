"""Mock coverage and cost capture for LangChain async/streaming paths.

Regression tests for #2424 (ainvoke/stream/astream not mocked) and #2445
(async calls not cost-captured). No network: a socket spy blocks any connect
and the transport targets a dead local port.
"""

from __future__ import annotations

import asyncio
import socket
from typing import Any

import pytest

pytest.importorskip("langchain_openai")

from langchain_openai import ChatOpenAI  # noqa: E402

from traigent.utils.langchain_interceptor import (  # noqa: E402
    clear_captured_responses,
    get_all_captured_responses,
    patch_langchain_for_metadata_capture,
)


@pytest.fixture()
def connects(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    seen: list[Any] = []

    def _spy(self: Any, addr: Any) -> None:
        seen.append(addr)
        raise OSError("blocked")

    monkeypatch.setattr(socket.socket, "connect", _spy)
    clear_captured_responses()
    patch_langchain_for_metadata_capture()
    yield seen
    clear_captured_responses()


def _llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key="sk-fake",
        base_url="http://127.0.0.1:9/v1",
        model="gpt-4o-mini",
        max_retries=0,
    )


@pytest.fixture()
def mock_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "traigent.integrations.utils.mock_adapter.MockAdapter.is_mock_enabled",
        classmethod(lambda cls, provider: True),
    )


def test_mock_mode_covers_ainvoke_stream_astream_abatch(
    connects: list[Any], mock_on: None
) -> None:
    llm = _llm()

    async def run() -> tuple[Any, list[Any], list[Any]]:
        one = await llm.ainvoke("hi")
        chunks = [c async for c in llm.astream("hi")]
        batch = await llm.abatch(["a", "b"])
        return one, chunks, batch

    one, achunks, batch = asyncio.run(run())
    schunks = list(llm.stream("hi"))

    assert one.content and achunks and schunks and len(batch) == 2
    assert connects == []


def test_async_calls_are_cost_captured_when_not_mocked(
    connects: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from langchain_core.messages import AIMessage

    monkeypatch.setattr(
        "traigent.integrations.utils.mock_adapter.MockAdapter.is_mock_enabled",
        classmethod(lambda cls, provider: False),
    )

    async def fake_agenerate_ainvoke(self: Any, *a: Any, **k: Any) -> AIMessage:
        return AIMessage(
            content="ok",
            usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        )

    # Replace the *original* underneath the traigent wrapper (no network).
    from traigent.utils import langchain_interceptor as li

    wrapped = li._create_ainvoke_wrapper(fake_agenerate_ainvoke, "openai")
    llm = _llm()
    asyncio.run(wrapped(llm, "hi"))

    captured = get_all_captured_responses()
    assert len(captured) == 1
    assert captured[0].usage_metadata["total_tokens"] == 5
    assert connects == []


def test_real_ainvoke_attribute_is_patched() -> None:
    patch_langchain_for_metadata_capture()
    assert "_traigent_patched_ainvoke" in vars(ChatOpenAI)


@pytest.fixture()
def mock_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "traigent.integrations.utils.mock_adapter.MockAdapter.is_mock_enabled",
        classmethod(lambda cls, provider: False),
    )


def _fallback_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key="sk-fake",
        base_url="http://127.0.0.1:9/v1",
        model="gpt-4o-mini",
        max_retries=0,
        disable_streaming=True,
    )


def _canned_result() -> Any:
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    message = AIMessage(
        content="ok",
        response_metadata={"model_name": "gpt-4o-mini"},
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )
    return ChatResult(generations=[ChatGeneration(message=message)])


def test_async_stream_fallback_to_ainvoke_captured_once(
    monkeypatch: pytest.MonkeyPatch, connects: list[Any], mock_off: None
) -> None:
    """Sol r4 finding 2: astream->ainvoke fallback must record one call."""

    async def _agenerate(self: Any, *a: Any, **k: Any) -> Any:
        return _canned_result()

    monkeypatch.setattr(ChatOpenAI, "_agenerate", _agenerate)

    async def run() -> list[Any]:
        return [c async for c in _fallback_llm().astream("hi")]

    chunks = asyncio.run(run())
    assert chunks
    assert len(get_all_captured_responses()) == 1


def test_sync_stream_fallback_to_invoke_captured_once(
    monkeypatch: pytest.MonkeyPatch, connects: list[Any], mock_off: None
) -> None:
    def _generate(self: Any, *a: Any, **k: Any) -> Any:
        return _canned_result()

    monkeypatch.setattr(ChatOpenAI, "_generate", _generate)

    chunks = list(_fallback_llm().stream("hi"))
    assert chunks
    assert len(get_all_captured_responses()) == 1


def test_plain_ainvoke_and_invoke_still_captured_once(
    monkeypatch: pytest.MonkeyPatch, connects: list[Any], mock_off: None
) -> None:
    async def _agenerate(self: Any, *a: Any, **k: Any) -> Any:
        return _canned_result()

    def _generate(self: Any, *a: Any, **k: Any) -> Any:
        return _canned_result()

    monkeypatch.setattr(ChatOpenAI, "_agenerate", _agenerate)
    monkeypatch.setattr(ChatOpenAI, "_generate", _generate)

    asyncio.run(_fallback_llm().ainvoke("hi"))
    assert len(get_all_captured_responses()) == 1
    clear_captured_responses()
    _fallback_llm().invoke("hi")
    assert len(get_all_captured_responses()) == 1
