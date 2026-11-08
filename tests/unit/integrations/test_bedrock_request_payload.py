"""Per-family request bodies for BedrockChatClient across sync/async/stream (#2931).

Field names: Anthropic Messages on Bedrock (anthropic_version, max_tokens,
messages, temperature, top_p) and AI21 Jamba chat (messages, max_tokens,
temperature, top_p) per the AWS Bedrock model-parameters documentation.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility
# Traceability: CONC-Quality-Reliability FUNC-INTEGRATIONS REQ-INT-008

from __future__ import annotations

import io
import json
import sys
import types
from unittest.mock import MagicMock

import pytest

from traigent.integrations.bedrock_client import BedrockChatClient

AI21 = "ai21.jamba-1-5-mini-v1:0"
CLAUDE = "anthropic.claude-3-haiku-20240307-v1:0"
TUNED = {"max_tokens": 7, "temperature": 0.9, "top_p": 0.3}

AI21_BODY = {
    "messages": [{"role": "user", "content": "hi"}],
    "max_tokens": 7,
    "temperature": 0.9,
    "top_p": 0.3,
}
CLAUDE_BODY = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 7,
    "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    "temperature": 0.9,
    "top_p": 0.3,
}


@pytest.fixture(autouse=True)
def _no_mock_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "false")
    monkeypatch.delenv("BEDROCK_MOCK", raising=False)


def _chunk(payload: dict) -> dict:
    return {"chunk": {"bytes": json.dumps(payload).encode()}}


def _sync_boto(response_payload=None, stream_events=None) -> MagicMock:
    boto = MagicMock()
    boto.invoke_model.return_value = {
        "body": io.BytesIO(json.dumps(response_payload or {}).encode())
    }
    boto.invoke_model_with_response_stream.return_value = {
        "body": iter(stream_events or [])
    }
    return boto


def _body(call) -> dict:
    return json.loads(call.kwargs["body"])


@pytest.mark.parametrize(
    ("model_id", "expected"), [(AI21, AI21_BODY), (CLAUDE, CLAUDE_BODY)]
)
def test_sync_invoke_body(model_id: str, expected: dict) -> None:
    boto = _sync_boto({"choices": [{"message": {"content": "ok"}}]})
    BedrockChatClient(client=boto).invoke(model_id=model_id, messages="hi", **TUNED)
    assert _body(boto.invoke_model.call_args) == expected


def test_sync_ai21_extra_params_merged() -> None:
    boto = _sync_boto({"choices": [{"message": {"content": "ok"}}]})
    BedrockChatClient(client=boto).invoke(
        model_id=AI21, messages="hi", extra_params={"stop": ["X"]}, **TUNED
    )
    assert _body(boto.invoke_model.call_args) == {**AI21_BODY, "stop": ["X"]}


def test_sync_ai21_omits_top_p_when_unset() -> None:
    boto = _sync_boto({"choices": [{"message": {"content": "ok"}}]})
    BedrockChatClient(client=boto).invoke(
        model_id=AI21, messages="hi", max_tokens=7, temperature=0.9
    )
    assert "top_p" not in _body(boto.invoke_model.call_args)


@pytest.mark.parametrize(
    ("model_id", "expected"), [(AI21, AI21_BODY), (CLAUDE, CLAUDE_BODY)]
)
def test_sync_stream_body(model_id: str, expected: dict) -> None:
    boto = _sync_boto()
    gen = BedrockChatClient(client=boto).invoke_stream(
        model_id=model_id, messages="hi", **TUNED
    )
    assert list(gen) == []
    assert _body(boto.invoke_model_with_response_stream.call_args) == expected


def test_sync_stream_ai21_yields_delta_text() -> None:
    boto = _sync_boto(
        stream_events=[
            _chunk({"choices": [{"delta": {"content": "Hel"}}]}),
            _chunk({"choices": [{"delta": {"content": "lo"}}]}),
            _chunk({"choices": [{"delta": {}}]}),
        ]
    )
    chunks = list(
        BedrockChatClient(client=boto).invoke_stream(
            model_id=AI21, messages="hi", **TUNED
        )
    )
    assert chunks == ["Hel", "lo"]


@pytest.mark.parametrize("model_id", ["meta.llama3-8b-instruct-v1:0", "amazon.titan-x"])
def test_unsupported_family_rejected_before_any_request(model_id: str) -> None:
    boto = _sync_boto()
    client = BedrockChatClient(client=boto)
    with pytest.raises(ValueError, match="does not support model family"):
        client.invoke(model_id=model_id, messages="hi", **TUNED)
    with pytest.raises(ValueError, match="does not support model family"):
        list(client.invoke_stream(model_id=model_id, messages="hi", **TUNED))
    boto.invoke_model.assert_not_called()
    boto.invoke_model_with_response_stream.assert_not_called()


class _AsyncBody:
    def __init__(self, payload: dict) -> None:
        self._raw = json.dumps(payload).encode()

    async def read(self) -> bytes:
        return self._raw


class _AsyncEvents:
    def __init__(self, events: list[dict]) -> None:
        self._it = iter(events)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration from None


def _install_fake_aioboto3(
    monkeypatch: pytest.MonkeyPatch, response_payload=None, stream_events=None
) -> MagicMock:
    """Fake aioboto3 whose bedrock client records calls; returns the recorder."""
    recorder = MagicMock()

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def invoke_model(self, **kwargs):
            recorder.invoke_model(**kwargs)
            return {"body": _AsyncBody(response_payload or {})}

        async def invoke_model_with_response_stream(self, **kwargs):
            recorder.invoke_model_with_response_stream(**kwargs)
            return {"body": _AsyncEvents(stream_events or [])}

    class _Session:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def client(self, *args, **kwargs):
            return _Client()

    fake = types.ModuleType("aioboto3")
    fake.Session = _Session  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "aioboto3", fake)
    return recorder


@pytest.mark.parametrize(
    ("model_id", "expected"), [(AI21, AI21_BODY), (CLAUDE, CLAUDE_BODY)]
)
async def test_async_invoke_body(
    monkeypatch: pytest.MonkeyPatch, model_id: str, expected: dict
) -> None:
    rec = _install_fake_aioboto3(
        monkeypatch, {"choices": [{"message": {"content": "async ok"}}]}
    )
    resp = await BedrockChatClient().ainvoke(model_id=model_id, messages="hi", **TUNED)
    assert _body(rec.invoke_model.call_args) == expected
    if model_id == AI21:
        assert resp.text == "async ok"


@pytest.mark.parametrize(
    ("model_id", "expected"), [(AI21, AI21_BODY), (CLAUDE, CLAUDE_BODY)]
)
async def test_async_stream_body(
    monkeypatch: pytest.MonkeyPatch, model_id: str, expected: dict
) -> None:
    rec = _install_fake_aioboto3(
        monkeypatch,
        stream_events=[_chunk({"choices": [{"delta": {"content": "Hi"}}]})],
    )
    chunks = [
        c
        async for c in BedrockChatClient().ainvoke_stream(
            model_id=model_id, messages="hi", **TUNED
        )
    ]
    assert _body(rec.invoke_model_with_response_stream.call_args) == expected
    assert chunks == (["Hi"] if model_id == AI21 else [])


async def test_async_rejects_unsupported_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec = _install_fake_aioboto3(monkeypatch)
    with pytest.raises(ValueError, match="does not support model family"):
        await BedrockChatClient().ainvoke(
            model_id="meta.llama3-8b-instruct-v1:0", messages="hi", **TUNED
        )
    rec.invoke_model.assert_not_called()
