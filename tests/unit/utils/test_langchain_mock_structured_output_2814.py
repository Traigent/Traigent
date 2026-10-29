"""LangChain structured output under mock mode never yields None or a crash (#2814).

The canned reply is plain text, so ``with_structured_output`` either crashed
(OpenAI) or returned ``None`` (Anthropic). It now raises an error that names
mock mode. Plain calls and plain ``bind_tools`` are unchanged. Offline: no key
is real and the call never leaves the process.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest

pydantic = pytest.importorskip("pydantic")

from traigent import testing as traigent_testing  # noqa: E402
from traigent.utils.langchain_interceptor import (  # noqa: E402
    MockModeStructuredOutputError,
)


class Answer(pydantic.BaseModel):
    verdict: str


def _openai() -> object:
    langchain_openai = pytest.importorskip("langchain_openai")
    return langchain_openai.ChatOpenAI(
        model="gpt-4o-mini", api_key="sk-mock"  # pragma: allowlist secret
    )


def _anthropic() -> object:
    langchain_anthropic = pytest.importorskip("langchain_anthropic")
    return langchain_anthropic.ChatAnthropic(
        model="claude-haiku-4-5", api_key="sk-mock"  # pragma: allowlist secret
    )


@pytest.fixture(autouse=True)
def _mock_on(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("TRAIGENT_MOCK_LLM", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "development")
    traigent_testing._reset_for_tests()
    traigent_testing.enable_mock_mode_for_quickstart()
    yield
    traigent_testing._reset_for_tests()


@pytest.mark.parametrize("make", [_openai, _anthropic])
def test_with_structured_output_raises_naming_mock_mode_2814(make) -> None:
    chain = make().with_structured_output(Answer)
    with pytest.raises(MockModeStructuredOutputError, match="mock mode"):
        chain.invoke("hi")


def test_openai_json_mode_structured_output_raises_2814() -> None:
    chain = _openai().with_structured_output(Answer, method="json_mode")
    with pytest.raises(MockModeStructuredOutputError):
        chain.invoke("hi")


@pytest.mark.parametrize("make", [_openai, _anthropic])
def test_async_structured_output_raises_2814(make) -> None:
    chain = make().with_structured_output(Answer)
    with pytest.raises(MockModeStructuredOutputError):
        asyncio.run(chain.ainvoke("hi"))


@pytest.mark.parametrize("make", [_openai, _anthropic])
def test_plain_invoke_and_bind_tools_are_unchanged_2814(make) -> None:
    model = make()
    assert model.invoke("hi").content == "This is a mock response for testing."
    bound = model.bind_tools([Answer])
    assert bound.invoke("hi").content == "This is a mock response for testing."


@pytest.mark.parametrize("make", [_openai, _anthropic])
def test_structured_output_is_untouched_when_mock_is_off_2814(make, monkeypatch) -> None:
    """With mock off the interceptor must not raise its own error."""
    traigent_testing.disable_mock_mode()
    # Unreachable proxy, set BEFORE the client is built (httpx reads it at
    # construction), so nothing can leave the machine.
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    chain = make().with_structured_output(Answer)
    with pytest.raises(Exception) as excinfo:  # noqa: PT011 - any non-mock failure
        chain.invoke("hi")
    assert not isinstance(excinfo.value, MockModeStructuredOutputError)
