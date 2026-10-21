"""Extra raw ``openai`` call paths must reach cost capture (issue #2444, item 4).

Covers ``chat.completions.parse`` (also via ``beta.chat.completions``),
``with_raw_response.create``/``parse`` and the Responses API, sync and async.
The REAL ``openai`` client runs against ``httpx.MockTransport``: no network,
no spend. The caller must still receive the normal return value.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

openai = pytest.importorskip("openai")

from pydantic import BaseModel  # noqa: E402

from traigent.integrations.framework_override import (  # noqa: E402
    FrameworkOverrideManager,
)
from traigent.utils.langchain_interceptor import (  # noqa: E402
    clear_captured_responses,
    get_all_captured_responses,
)

MODEL = "gpt-4o-mini"
IN_TOK = 11
OUT_TOK = 7


class Answer(BaseModel):
    text: str


def _chat_body(model: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": '{"text": "ok"}'},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": IN_TOK,
            "completion_tokens": OUT_TOK,
            "total_tokens": IN_TOK + OUT_TOK,
        },
    }


def _responses_body(model: str) -> dict[str, Any]:
    return {
        "id": "resp-test",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": model,
        "output": [
            {
                "id": "msg-1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": '{"text": "ok"}',
                        "annotations": [],
                    }
                ],
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": IN_TOK,
            "output_tokens": OUT_TOK,
            "total_tokens": IN_TOK + OUT_TOK,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def _handle(request: httpx.Request) -> httpx.Response:
    payload = json.loads(request.content)
    if request.url.path.endswith("/responses"):
        return httpx.Response(200, json=_responses_body(payload["model"]))
    return httpx.Response(200, json=_chat_body(payload["model"]))


def _sync_client() -> Any:
    return openai.OpenAI(
        api_key="test-key",
        base_url="https://gateway.invalid/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(_handle)),
        max_retries=0,
    )


def _async_client() -> Any:
    return openai.AsyncOpenAI(
        api_key="test-key",
        base_url="https://gateway.invalid/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(_handle)),
        max_retries=0,
    )


@pytest.fixture(autouse=True)
def _overrides():
    manager = FrameworkOverrideManager()
    manager.activate_overrides(["openai.OpenAI", "openai.AsyncOpenAI"])
    clear_captured_responses()
    try:
        yield manager
    finally:
        manager.deactivate_overrides()
        clear_captured_responses()


MESSAGES = [{"role": "user", "content": "hi"}]


def _assert_one_capture() -> None:
    from traigent.evaluators.metrics_tracker import extract_llm_metrics

    captured = get_all_captured_responses()
    assert len(captured) == 1
    metrics = extract_llm_metrics(captured[0], MODEL)
    assert metrics.tokens.input_tokens == IN_TOK
    assert metrics.tokens.output_tokens == OUT_TOK
    assert metrics.tokens.total_tokens == IN_TOK + OUT_TOK
    assert metrics.cost.total_cost > 0
    assert getattr(captured[0], "model", None) == MODEL


def test_chat_parse_sync():
    result = _sync_client().chat.completions.parse(
        model=MODEL, messages=MESSAGES, response_format=Answer
    )
    assert result.choices[0].message.parsed == Answer(text="ok")
    _assert_one_capture()


def test_beta_chat_parse_sync():
    result = _sync_client().beta.chat.completions.parse(
        model=MODEL, messages=MESSAGES, response_format=Answer
    )
    assert result.choices[0].message.parsed == Answer(text="ok")
    _assert_one_capture()


@pytest.mark.asyncio
async def test_chat_parse_async():
    result = await _async_client().chat.completions.parse(
        model=MODEL, messages=MESSAGES, response_format=Answer
    )
    assert result.choices[0].message.parsed == Answer(text="ok")
    _assert_one_capture()


def test_with_raw_response_create_sync_returns_raw_and_parses_once():
    raw = _sync_client().chat.completions.with_raw_response.create(
        model=MODEL, messages=MESSAGES
    )
    _assert_one_capture()
    assert raw.http_response.status_code == 200
    completion = raw.parse()
    assert completion.model == MODEL
    # The caller's parse() is the instance Traigent captured: body read once.
    assert completion is get_all_captured_responses()[0]


@pytest.mark.asyncio
async def test_with_raw_response_create_async():
    raw = await _async_client().chat.completions.with_raw_response.create(
        model=MODEL, messages=MESSAGES
    )
    _assert_one_capture()
    assert raw.http_response.status_code == 200
    assert raw.parse().choices[0].message.content == '{"text": "ok"}'


def test_with_raw_response_parse_sync():
    raw = _sync_client().chat.completions.with_raw_response.parse(
        model=MODEL, messages=MESSAGES, response_format=Answer
    )
    _assert_one_capture()
    assert raw.parse().choices[0].message.parsed == Answer(text="ok")


def test_responses_create_sync():
    result = _sync_client().responses.create(model=MODEL, input="hi")
    assert result.output_text == '{"text": "ok"}'
    _assert_one_capture()


@pytest.mark.asyncio
async def test_responses_create_async():
    result = await _async_client().responses.create(model=MODEL, input="hi")
    assert result.output_text == '{"text": "ok"}'
    _assert_one_capture()


def test_responses_parse_sync():
    result = _sync_client().responses.parse(model=MODEL, input="hi", text_format=Answer)
    assert result.output_parsed == Answer(text="ok")
    _assert_one_capture()


@pytest.mark.asyncio
async def test_responses_parse_async():
    result = await _async_client().responses.parse(
        model=MODEL, input="hi", text_format=Answer
    )
    assert result.output_parsed == Answer(text="ok")
    _assert_one_capture()


def test_with_raw_response_responses_create_sync():
    raw = _sync_client().responses.with_raw_response.create(model=MODEL, input="hi")
    _assert_one_capture()
    assert raw.parse().output_text == '{"text": "ok"}'


def test_new_paths_do_not_capture_when_overrides_inactive(_overrides):
    _overrides.deactivate_overrides()
    clear_captured_responses()
    _sync_client().responses.create(model=MODEL, input="hi")
    _sync_client().chat.completions.parse(
        model=MODEL, messages=MESSAGES, response_format=Answer
    )
    assert get_all_captured_responses() == []
