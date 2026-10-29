"""Mock mode honours the shape of a LiteLLM request (#2798, #2801).

``mock_response=`` is LiteLLM's standard way to give a dry run a structured
reply, ``n=`` asks for several choices, and ``stream=True`` needs something the
caller can iterate. Every test is offline: the call never leaves the process.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest

litellm = pytest.importorskip("litellm")

from traigent import testing as traigent_testing  # noqa: E402

CANNED = "This is a mock response for testing."
MESSAGES = [{"role": "user", "content": "hi"}]


@pytest.fixture(autouse=True)
def _mock_on(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("TRAIGENT_MOCK_LLM", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "development")
    traigent_testing._reset_for_tests()
    traigent_testing.enable_mock_mode_for_quickstart()
    yield
    traigent_testing._reset_for_tests()


def _complete(**kwargs: object) -> object:
    return litellm.completion(model="gpt-4o-mini", messages=MESSAGES, **kwargs)


def _acomplete(**kwargs: object) -> object:
    return asyncio.run(
        litellm.acompletion(model="gpt-4o-mini", messages=MESSAGES, **kwargs)
    )


@pytest.mark.parametrize("call", [_complete, _acomplete])
def test_mock_response_is_returned_instead_of_the_canned_text_2798(call) -> None:
    response = call(mock_response='{"checks": []}')
    assert response.choices[0].message.content == '{"checks": []}'
    # The synthetic usage block is unchanged, so cost capture still has input.
    assert response.usage.total_tokens == 30


@pytest.mark.parametrize("call", [_complete, _acomplete])
def test_empty_mock_response_is_a_reply_2798(call) -> None:
    assert call(mock_response="").choices[0].message.content == ""


@pytest.mark.parametrize("call", [_complete, _acomplete])
def test_without_mock_response_the_canned_text_is_unchanged_2798(call) -> None:
    assert call().choices[0].message.content == CANNED


def test_mock_response_exception_is_raised_2798() -> None:
    with pytest.raises(RuntimeError, match="boom"):
        _complete(mock_response=RuntimeError("boom"))


def test_unsupported_mock_response_type_is_refused_not_replaced_2798() -> None:
    with pytest.raises(TypeError, match="mock_response"):
        _complete(mock_response={"not": "supported"})


def test_n_choices_are_returned_2798() -> None:
    response = _complete(n=3, mock_response="x")
    assert [c.index for c in response.choices] == [0, 1, 2]
    assert {c.message.content for c in response.choices} == {"x"}


def test_sync_stream_yields_chunks_that_join_to_the_reply_2801() -> None:
    stream = _complete(stream=True)
    chunks = list(stream)
    assert "".join(c.choices[0].delta.content or "" for c in chunks) == CANNED
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert chunks[-1].usage.total_tokens == 30


def test_async_stream_yields_chunks_that_join_to_the_reply_2801() -> None:
    async def run() -> tuple[str, str]:
        stream = await litellm.acompletion(
            model="gpt-4o-mini", messages=MESSAGES, stream=True
        )
        chunks = [c async for c in stream]
        return (
            "".join(c.choices[0].delta.content or "" for c in chunks),
            chunks[-1].choices[0].finish_reason,
        )

    text, finish = asyncio.run(run())
    assert text == CANNED
    assert finish == "stop"


def test_stream_honours_mock_response_and_n_2801() -> None:
    chunks = list(_complete(stream=True, mock_response='{"a": 1}', n=2))
    for index in (0, 1):
        text = "".join(
            c.choices[0].delta.content or ""
            for c in chunks
            if c.choices[0].index == index
        )
        assert text == '{"a": 1}'


def test_stream_false_is_a_plain_response_2801() -> None:
    assert _complete(stream=False).choices[0].message.content == CANNED
