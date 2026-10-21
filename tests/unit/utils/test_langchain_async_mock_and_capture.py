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


def test_sync_stream_close_after_first_chunk_still_captures_once(
    monkeypatch: pytest.MonkeyPatch, connects: list[Any], mock_off: None
) -> None:
    """Sol r5: close() after the fallback's only chunk must not lose usage."""

    def _generate(self: Any, *a: Any, **k: Any) -> Any:
        return _canned_result()

    monkeypatch.setattr(ChatOpenAI, "_generate", _generate)

    stream = _fallback_llm().stream("hi")
    assert next(stream) is not None
    stream.close()
    captured = get_all_captured_responses()
    assert len(captured) == 1
    assert captured[0].usage_metadata["total_tokens"] == 15


def test_async_stream_aclose_after_first_chunk_still_captures_once(
    monkeypatch: pytest.MonkeyPatch, connects: list[Any], mock_off: None
) -> None:
    async def _agenerate(self: Any, *a: Any, **k: Any) -> Any:
        return _canned_result()

    monkeypatch.setattr(ChatOpenAI, "_agenerate", _agenerate)

    async def run() -> None:
        stream = _fallback_llm().astream("hi")
        assert await stream.__anext__() is not None
        await stream.aclose()

    asyncio.run(run())
    captured = get_all_captured_responses()
    assert len(captured) == 1
    assert captured[0].usage_metadata["total_tokens"] == 15


def _named_llm(name: str) -> ChatOpenAI:
    return ChatOpenAI(
        api_key="sk-fake",
        base_url="http://127.0.0.1:9/v1",
        model=name,
        max_retries=0,
        disable_streaming=True,
    )


def test_nested_call_to_a_different_model_is_captured_separately(
    monkeypatch: pytest.MonkeyPatch, connects: list[Any], mock_off: None
) -> None:
    """Sol r5: a callback invoking a second model is an independent call."""
    inner = _named_llm("inner-model")

    def _generate(self: Any, *a: Any, **k: Any) -> Any:
        if self.model_name == "outer-model":
            inner.invoke("nested")  # e.g. a callback calling another model
        return _canned_result()

    async def _agenerate(self: Any, *a: Any, **k: Any) -> Any:
        if self.model_name == "outer-model":
            await inner.ainvoke("nested")
        return _canned_result()

    monkeypatch.setattr(ChatOpenAI, "_generate", _generate)
    monkeypatch.setattr(ChatOpenAI, "_agenerate", _agenerate)

    _named_llm("outer-model").invoke("hi")
    assert len(get_all_captured_responses()) == 2
    clear_captured_responses()
    asyncio.run(_named_llm("outer-model").ainvoke("hi"))
    assert len(get_all_captured_responses()) == 2
    clear_captured_responses()

    # The outer stream's fallback still counts the outer call once, plus the
    # independent nested call.
    chunks = list(_named_llm("outer-model").stream("hi"))
    assert chunks
    assert len(get_all_captured_responses()) == 2


def test_same_instance_reentry_is_captured_separately(
    monkeypatch: pytest.MonkeyPatch, connects: list[Any], mock_off: None
) -> None:
    """Sol r6: a callback re-invoking the SAME instance is a billed call too."""
    llm = _fallback_llm()
    state = {"depth": 0}

    def _generate(self: Any, *a: Any, **k: Any) -> Any:
        if state["depth"] == 0:
            state["depth"] = 1
            llm.invoke("nested")
        return _canned_result()

    async def _agenerate(self: Any, *a: Any, **k: Any) -> Any:
        if state["depth"] == 0:
            state["depth"] = 1
            await llm.ainvoke("nested")
        return _canned_result()

    monkeypatch.setattr(ChatOpenAI, "_generate", _generate)
    monkeypatch.setattr(ChatOpenAI, "_agenerate", _agenerate)

    llm.invoke("hi")
    assert len(get_all_captured_responses()) == 2
    clear_captured_responses()
    state["depth"] = 0
    asyncio.run(llm.ainvoke("hi"))
    assert len(get_all_captured_responses()) == 2
    clear_captured_responses()
    state["depth"] = 0
    assert list(llm.stream("hi"))
    assert len(get_all_captured_responses()) == 2  # fallback call + nested call


def _streaming_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key="sk-fake",
        base_url="http://127.0.0.1:9/v1",
        model="gpt-4o-mini",
        max_retries=0,
    )


def _canned_chunks() -> list[Any]:
    from langchain_core.messages import AIMessageChunk
    from langchain_core.outputs import ChatGenerationChunk

    first = AIMessageChunk(content="a")
    last = AIMessageChunk(
        content="b",
        response_metadata={"model_name": "gpt-4o-mini"},
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )
    return [ChatGenerationChunk(message=first), ChatGenerationChunk(message=last)]


def test_real_stream_records_once_and_nested_invoke_is_not_suppressed(
    monkeypatch: pytest.MonkeyPatch, connects: list[Any], mock_off: None
) -> None:
    llm = _streaming_llm()
    state = {"nested": False}

    def _stream(self: Any, *a: Any, **k: Any) -> Any:
        yield _canned_chunks()[0]
        if state["nested"]:
            _fallback_llm().invoke("nested")
        yield _canned_chunks()[1]

    monkeypatch.setattr(ChatOpenAI, "_stream", _stream)
    monkeypatch.setattr(ChatOpenAI, "_generate", lambda self, *a, **k: _canned_result())

    assert list(llm.stream("hi"))
    assert len(get_all_captured_responses()) == 1
    clear_captured_responses()
    state["nested"] = True
    assert list(llm.stream("hi"))
    assert len(get_all_captured_responses()) == 2


def test_real_stream_close_after_first_chunk_records_once(
    monkeypatch: pytest.MonkeyPatch, connects: list[Any], mock_off: None
) -> None:
    def _stream(self: Any, *a: Any, **k: Any) -> Any:
        yield from _canned_chunks()

    async def _astream(self: Any, *a: Any, **k: Any) -> Any:
        for chunk in _canned_chunks():
            yield chunk

    monkeypatch.setattr(ChatOpenAI, "_stream", _stream)
    monkeypatch.setattr(ChatOpenAI, "_astream", _astream)

    stream = _streaming_llm().stream("hi")
    assert next(stream) is not None
    stream.close()
    assert len(get_all_captured_responses()) == 1
    clear_captured_responses()

    async def run() -> None:
        astream = _streaming_llm().astream("hi")
        assert await astream.__anext__() is not None
        await astream.aclose()

    asyncio.run(run())
    assert len(get_all_captured_responses()) == 1


def test_stream_falls_back_helper_degrades_when_check_unavailable() -> None:
    """If LangChain's private ``_should_stream`` is gone or fails, report unknown."""
    from traigent.utils.langchain_interceptor import _stream_falls_back

    class NoCheck:
        pass

    class Raises:
        def _should_stream(self, **_kw: Any) -> bool:
            raise TypeError("signature changed")

    assert _stream_falls_back(NoCheck(), async_api=False, kwargs={}) is None
    assert _stream_falls_back(Raises(), async_api=True, kwargs={}) is None
    assert _stream_falls_back(_fallback_llm(), async_api=False, kwargs={}) is True
    assert _stream_falls_back(_streaming_llm(), async_api=False, kwargs={}) is False


@pytest.mark.parametrize(
    ("key", "package"),
    [
        ("openai", "langchain_openai"),
        ("anthropic", "langchain_anthropic"),
        ("bedrock", "langchain_aws"),
        ("litellm", "litellm"),
    ],
)
def test_import_error_outcome_distinguishes_absent_from_broken(
    monkeypatch: pytest.MonkeyPatch, key: str, package: str
) -> None:
    """Sol r6: only a provably absent package is not-applicable."""
    import builtins
    import importlib.util

    from traigent.utils.langchain_interceptor import (
        patch_langchain_for_metadata_capture,
    )
    from traigent.utils.litellm_interceptor import patch_litellm_for_metadata_capture

    real_import = builtins.__import__

    def fake_import(name: str, *a: Any, **k: Any) -> Any:
        if name == package:
            raise ImportError("transitive dependency broke")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    patch = (
        patch_litellm_for_metadata_capture
        if key == "litellm"
        else patch_langchain_for_metadata_capture
    )

    real_find_spec = importlib.util.find_spec
    # Package present on disk but failing to import -> failed.
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda n, *a: object() if n == package else real_find_spec(n, *a),
    )
    broken: dict[str, str] = {}
    patch(broken)
    assert broken[key] == "failed"

    # Package provably absent -> not-applicable.
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda n, *a: None if n == package else real_find_spec(n, *a),
    )
    absent: dict[str, str] = {}
    patch(absent)
    assert absent[key] == "not-applicable"
