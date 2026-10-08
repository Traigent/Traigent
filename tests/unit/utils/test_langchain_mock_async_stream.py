"""LangChain ``ainvoke`` / ``stream`` / ``astream`` are mocked under mock mode (#2424).

Before the fix only ``invoke`` had a mock branch: ``ainvoke`` was never
patched and the stream wrappers only recorded usage around the original call,
so async or streaming LangChain agents reached the provider under mock mode.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from traigent.utils.langchain_interceptor import (
    _create_ainvoke_wrapper,
    _create_astream_wrapper,
    _create_stream_wrapper,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]

_PROBE = """
import asyncio, socket
SEEN = []
def _gai(host, *a, **k):
    SEEN.append(("dns", host))
    raise OSError("blocked by test spy")
def _conn(self, addr):
    SEEN.append(("connect", addr))
    raise OSError("blocked by test spy")
socket.getaddrinfo = _gai
socket.socket.connect = _conn

from traigent.testing import enable_mock_mode_for_quickstart
from traigent.evaluators.local import _ensure_metadata_capture_patches
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

enable_mock_mode_for_quickstart()
_ensure_metadata_capture_patches()

clients = {
    "openai": ChatOpenAI(api_key="sk-fake", base_url="http://127.0.0.1:9/v1",  # pragma: allowlist secret
                         model="gpt-4o-mini", max_retries=0),
    "anthropic": ChatAnthropic(api_key="sk-ant-fake", base_url="http://127.0.0.1:9",  # pragma: allowlist secret
                               model="claude-3-5-haiku-latest", max_retries=0),
}

async def _astream(llm):
    return "".join([c.content async for c in llm.astream("hi")])

for name, llm in clients.items():
    for method, call in (
        ("invoke", lambda: llm.invoke("hi").content),
        ("ainvoke", lambda: asyncio.run(llm.ainvoke("hi")).content),
        ("stream", lambda: "".join(c.content for c in llm.stream("hi"))),
        ("astream", lambda: asyncio.run(_astream(llm))),
    ):
        try:
            print("RESULT", name, method, repr(call()), len(SEEN))
        except Exception as exc:
            print("RESULT", name, method, "ERROR", type(exc).__name__, len(SEEN))
print("SEEN", SEEN)
"""


def test_all_langchain_entry_points_are_mocked() -> None:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.endswith("_API_KEY") and k != "TRAIGENT_MOCK_LLM"
    }
    env.update(
        PYTHONPATH=str(_REPO_ROOT),
        TRAIGENT_OFFLINE_MODE="true",
        TRAIGENT_SKIP_DOTENV="1",
        LITELLM_MODE="PRODUCTION",
        LITELLM_LOCAL_MODEL_COST_MAP="True",
        ENVIRONMENT="development",
    )
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_PROBE)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(_REPO_ROOT),
        timeout=180,
    )
    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    mock_text = repr("This is a mock response for testing.")
    for provider in ("openai", "anthropic"):
        for method in ("invoke", "ainvoke", "stream", "astream"):
            assert f"RESULT {provider} {method} {mock_text} 0" in result.stdout, (
                combined
            )
    assert "SEEN []" in result.stdout, combined


@pytest.fixture
def mock_off() -> Any:
    with patch(
        "traigent.integrations.utils.mock_adapter.MockAdapter.is_mock_enabled",
        return_value=False,
    ):
        yield


@pytest.mark.usefixtures("mock_off")
@pytest.mark.asyncio
async def test_ainvoke_wrapper_calls_original_when_mock_off() -> None:
    calls: list[tuple[Any, ...]] = []
    sentinel = object()

    async def original(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append((self, args, kwargs))
        return sentinel

    wrapper = _create_ainvoke_wrapper(original, provider="openai")
    client = object()
    assert await wrapper(client, "hi", stop=["x"]) is sentinel
    assert calls == [(client, ("hi",), {"stop": ["x"]})]


@pytest.mark.usefixtures("mock_off")
@pytest.mark.asyncio
async def test_stream_wrappers_call_original_when_mock_off() -> None:
    class _Chunk:
        def __init__(self, content: str) -> None:
            self.content = content
            self.response_metadata: dict[str, Any] = {}

    def original_stream(self: Any, *args: Any, **kwargs: Any) -> Any:
        yield _Chunk("a")
        yield _Chunk("b")

    async def original_astream(self: Any, *args: Any, **kwargs: Any) -> Any:
        yield _Chunk("c")

    stream = _create_stream_wrapper(original_stream, provider="openai")
    astream = _create_astream_wrapper(original_astream, provider="openai")
    assert [c.content for c in stream(object())] == ["a", "b"]
    assert [c.content async for c in astream(object())] == ["c"]
