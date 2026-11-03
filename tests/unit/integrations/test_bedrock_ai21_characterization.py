"""Characterization tests for BedrockChatClient._invoke_ai21 edge shapes."""

import io
import json
from unittest.mock import MagicMock

import pytest

from traigent.integrations.bedrock_client import BedrockChatClient


def _client(payload, *, stream=False):
    body = json.dumps(payload)
    raw = io.BytesIO(body.encode()) if stream else body
    mock = MagicMock()
    mock.invoke_model.return_value = {"body": raw}
    return BedrockChatClient(client=mock), mock


def _call(client, messages="hi"):
    return client._invoke_ai21(
        model_id="ai21.jamba",
        messages=messages,
        max_tokens=8,
        temperature=0.1,
        top_p=None,
    )


def _sent(mock):
    kwargs = mock.invoke_model.call_args.kwargs
    return json.loads(kwargs["body"]), kwargs


def test_stream_body_is_read():
    client, _ = _client({"choices": [{"message": {"content": " hey "}}]}, stream=True)
    assert _call(client).text == "hey"


def test_request_envelope_fields():
    client, mock = _client({"choices": [{"message": {"content": "x"}}]})
    _call(client)
    body, kwargs = _sent(mock)
    assert set(body) == {"messages"}
    assert kwargs["modelId"] == "ai21.jamba"
    assert kwargs["accept"] == kwargs["contentType"] == "application/json"


def test_list_content_blocks_are_joined_and_non_mapping_blocks_dropped():
    client, mock = _client({"choices": [{"message": {"content": "x"}}]})
    _call(
        client,
        [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "a"},
                    "skipped",
                    {"type": "text"},
                    {"type": "text", "text": "b"},
                ],
            },
            {"content": "plain"},
            {"role": "user", "content": 5},
        ],
    )
    body, _ = _sent(mock)
    assert body["messages"] == [
        {"role": "assistant", "content": "a\n\nb"},
        {"role": "user", "content": "plain"},
        {"role": "user", "content": "5"},
    ]


def test_iterable_of_strings_become_user_messages():
    client, mock = _client({"choices": [{"message": {"content": "x"}}]})
    _call(client, ["p1", "p2"])
    body, _ = _sent(mock)
    assert body["messages"] == [
        {"role": "user", "content": "p1"},
        {"role": "user", "content": "p2"},
    ]


@pytest.mark.parametrize(
    ("payload", "text", "usage"),
    [
        (
            {"choices": [{"message": {"content": "  a  "}}], "usage": {"u": 1}},
            "a",
            {"u": 1},
        ),
        ({"choices": [{"message": {"content": ""}}], "outputText": " fb "}, "fb", None),
        ({"choices": [], "outputText": "ot", "tokenUsage": {"t": 2}}, "ot", {"t": 2}),
        ({"choices": None, "usage": {}, "tokenUsage": {"t": 3}}, "", {"t": 3}),
        ({"choices": ["notmapping"], "outputText": "ot2"}, "ot2", None),
        ({"choices": [{"message": "str"}], "outputText": "ot3"}, "ot3", None),
        ({"choices": [{"message": {}}]}, "", None),
        ({}, "", None),
        ({"choices": [{"message": {"content": None}}]}, "None", None),
    ],
)
def test_response_text_and_usage_extraction(payload, text, usage):
    client, _ = _client(payload)
    resp = _call(client)
    assert resp.text == text
    assert resp.usage == usage
    assert resp.raw == payload
    assert resp.model == "ai21.jamba"


def test_non_mapping_response_is_stringified():
    client, _ = _client(["a", 1])
    resp = _call(client)
    assert resp.text == str(["a", 1])
    assert resp.usage is None
