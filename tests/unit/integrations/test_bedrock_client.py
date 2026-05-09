"""Unit tests for AWS Bedrock client integration.

Tests for BedrockChatClient and related helper functions, covering both
normalized response parsing and mocked boto3 interactions.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility
# Traceability: CONC-Quality-Reliability FUNC-INTEGRATIONS
# Traceability: REQ-INT-008 SYNC-IntegrationHook

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from traigent.integrations.bedrock_client import (
    BedrockChatClient,
    BedrockChatResponse,
    _coerce_user_messages,
    _default_anthropic_version,
    _extract_text_from_messages_response,
    _require_boto3,
    resolve_default_bedrock_model_id,
)


def _mock_invoke_client(payload: dict) -> MagicMock:
    mock_client = MagicMock()
    mock_client.invoke_model.return_value = {
        "body": MagicMock(read=lambda: json.dumps(payload).encode())
    }
    return mock_client


def _mock_stream_client(*payloads: dict) -> MagicMock:
    mock_client = MagicMock()
    mock_client.invoke_model_with_response_stream.return_value = {
        "body": iter(
            {"chunk": {"bytes": json.dumps(payload).encode()}} for payload in payloads
        )
    }
    return mock_client


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_require_boto3_raises_on_missing_import(self) -> None:
        """Test _require_boto3 raises ImportError when boto3 is not available."""
        with patch(
            "builtins.__import__", side_effect=ImportError("No module named boto3")
        ):
            with pytest.raises(ImportError, match="boto3 is required"):
                _require_boto3()

    def test_require_boto3_returns_boto3_when_available(self) -> None:
        """Test _require_boto3 returns boto3 module when available."""
        mock_boto3 = MagicMock()
        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            result = _require_boto3()
            assert result is mock_boto3

    def test_default_anthropic_version(self) -> None:
        """Test _default_anthropic_version returns expected version string."""
        version = _default_anthropic_version()
        assert version == "bedrock-2023-05-31"
        assert isinstance(version, str)

    def test_coerce_user_messages_with_string(self) -> None:
        """Test _coerce_user_messages converts string to message format."""
        result = _coerce_user_messages("Hello world")
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["text"] == "Hello world"

    def test_coerce_user_messages_with_list_of_strings(self) -> None:
        """Test _coerce_user_messages converts list of strings to messages."""
        result = _coerce_user_messages(["First", "Second", "Third"])
        assert len(result) == 3
        assert all(msg["role"] == "user" for msg in result)
        assert result[0]["content"][0]["text"] == "First"
        assert result[1]["content"][0]["text"] == "Second"
        assert result[2]["content"][0]["text"] == "Third"

    def test_coerce_user_messages_with_list_of_dicts(self) -> None:
        """Test _coerce_user_messages passes through list of message dicts."""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
        ]
        result = _coerce_user_messages(messages)
        assert result == messages

    def test_coerce_user_messages_with_empty_list(self) -> None:
        """Test _coerce_user_messages handles empty list gracefully."""
        result = _coerce_user_messages([])
        assert result == []

    def test_extract_text_from_messages_response_with_text_blocks(self) -> None:
        """Test _extract_text_from_messages_response extracts text from content blocks."""
        response = {
            "content": [
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world!"},
            ]
        }
        result = _extract_text_from_messages_response(response)
        assert result == "Hello world!"

    def test_extract_text_from_messages_response_with_non_text_blocks(self) -> None:
        """Test _extract_text_from_messages_response ignores non-text blocks."""
        response = {
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "image", "data": "base64..."},
                {"type": "text", "text": "world"},
            ]
        }
        result = _extract_text_from_messages_response(response)
        assert result == "Helloworld"

    def test_extract_text_from_messages_response_with_empty_content(self) -> None:
        """Test _extract_text_from_messages_response handles empty content."""
        assert _extract_text_from_messages_response({"content": []}) == ""
        assert _extract_text_from_messages_response({"content": None}) == ""
        assert _extract_text_from_messages_response({}) == ""

    def test_extract_text_from_messages_response_strips_whitespace(self) -> None:
        """Test _extract_text_from_messages_response strips trailing whitespace."""
        response = {"content": [{"type": "text", "text": "  Hello  "}]}
        result = _extract_text_from_messages_response(response)
        assert result == "Hello"


class TestBedrockChatResponseDataclass:
    """Tests for BedrockChatResponse dataclass."""

    def test_bedrock_chat_response_creation(self) -> None:
        """Test BedrockChatResponse can be created with required fields."""
        response = BedrockChatResponse(text="test", raw={"data": "value"})
        assert response.text == "test"
        assert response.raw == {"data": "value"}
        assert response.usage is None

    def test_bedrock_chat_response_with_usage(self) -> None:
        """Test BedrockChatResponse stores usage information."""
        usage = {"input_tokens": 10, "output_tokens": 20}
        response = BedrockChatResponse(text="test", raw={}, usage=usage)
        assert response.usage == usage


class TestBedrockChatClientInit:
    """Tests for BedrockChatClient initialization."""

    def test_init_with_defaults(self) -> None:
        """Test BedrockChatClient initializes with default values."""
        client = BedrockChatClient()
        assert client._client is None
        assert client._region_name is None
        assert client._profile_name is None

    def test_init_with_region_and_profile(self) -> None:
        """Test BedrockChatClient stores region and profile name."""
        client = BedrockChatClient(region_name="us-east-1", profile_name="test-profile")
        assert client._region_name == "us-east-1"
        assert client._profile_name == "test-profile"

    def test_init_with_custom_client(self) -> None:
        """Test BedrockChatClient accepts pre-configured client."""
        mock_client = MagicMock()
        client = BedrockChatClient(client=mock_client)
        assert client._client is mock_client


class TestBedrockChatClientEnsureClient:
    """Tests for BedrockChatClient._ensure_client method."""

    def test_ensure_client_returns_existing_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test _ensure_client returns existing client if already set."""
        mock_client = MagicMock()
        client = BedrockChatClient(client=mock_client)
        result = client._ensure_client()
        assert result is mock_client

    @patch("traigent.integrations.bedrock_client._require_boto3")
    def test_ensure_client_uses_boto3_when_bedrock_mock_is_set(
        self, mock_require: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test BEDROCK_MOCK does not bypass boto3 client creation."""
        monkeypatch.setenv("BEDROCK_MOCK", "true")
        mock_boto3 = MagicMock()
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_boto3.session.Session.return_value = mock_session
        mock_require.return_value = mock_boto3

        client = BedrockChatClient(region_name="us-west-2")
        result = client._ensure_client()

        assert result is mock_client
        mock_session.client.assert_called_once_with(
            "bedrock-runtime", region_name="us-west-2"
        )


class TestBedrockChatClientInvoke:
    """Tests for BedrockChatClient.invoke method."""

    def test_invoke_uses_configured_client_when_bedrock_mock_is_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test BEDROCK_MOCK does not fabricate invoke responses."""
        monkeypatch.setenv("BEDROCK_MOCK", "true")
        mock_client = _mock_invoke_client(
            {
                "content": [{"type": "text", "text": "Response text"}],
                "usage": {"input_tokens": 5, "output_tokens": 10},
            }
        )
        client = BedrockChatClient(client=mock_client)

        response = client.invoke(
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            messages="Hello world",
            max_tokens=16,
        )

        assert response.text == "Response text"
        assert "[MOCK:" not in response.text
        assert response.usage is not None
        assert response.usage["output_tokens"] == 10
        mock_client.invoke_model.assert_called_once()

    def test_invoke_with_list_messages(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test invoke sends list messages to Bedrock payload."""
        mock_client = _mock_invoke_client(
            {"content": [{"type": "text", "text": "List response"}]}
        )
        client = BedrockChatClient(client=mock_client)

        messages = [
            {"role": "user", "content": [{"type": "text", "text": "First"}]},
            {"role": "user", "content": [{"type": "text", "text": "Last message"}]},
        ]
        response = client.invoke(model_id="test-model", messages=messages)

        assert response.text == "List response"
        body = json.loads(mock_client.invoke_model.call_args.kwargs["body"])
        assert body["messages"] == messages

    def test_invoke_with_temperature_and_top_p(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test invoke accepts temperature and top_p parameters."""
        mock_client = _mock_invoke_client(
            {"content": [{"type": "text", "text": "Temperature response"}]}
        )
        client = BedrockChatClient(client=mock_client)

        response = client.invoke(
            model_id="test-model",
            messages="test",
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
        )

        assert response.text == "Temperature response"
        body = json.loads(mock_client.invoke_model.call_args.kwargs["body"])
        assert body["max_tokens"] == 100
        assert body["temperature"] == 0.7
        assert body["top_p"] == 0.9

    def test_invoke_with_extra_params(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test invoke accepts extra_params."""
        mock_client = _mock_invoke_client(
            {"content": [{"type": "text", "text": "Extra params response"}]}
        )
        client = BedrockChatClient(client=mock_client)

        extra = {"stop_sequences": ["\n"], "custom_param": "value"}
        response = client.invoke(
            model_id="test-model",
            messages="test",
            extra_params=extra,
        )

        assert response.text == "Extra params response"
        body = json.loads(mock_client.invoke_model.call_args.kwargs["body"])
        assert body["stop_sequences"] == ["\n"]
        assert body["custom_param"] == "value"

    @patch("traigent.integrations.bedrock_client._require_boto3")
    def test_invoke_with_real_boto3_client(
        self, mock_require: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test invoke with mocked boto3 client (non-mock mode)."""
        monkeypatch.delenv("BEDROCK_MOCK", raising=False)

        # Setup mock boto3
        mock_boto3 = MagicMock()
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_response = {
            "body": MagicMock(
                read=lambda: json.dumps(
                    {
                        "content": [{"type": "text", "text": "Response text"}],
                        "usage": {"input_tokens": 5, "output_tokens": 10},
                    }
                ).encode()
            )
        }
        mock_client.invoke_model.return_value = mock_response
        mock_session.client.return_value = mock_client
        mock_boto3.session.Session.return_value = mock_session
        mock_require.return_value = mock_boto3

        client = BedrockChatClient(region_name="us-west-2")
        response = client.invoke(
            model_id="anthropic.claude-3-sonnet-20240229-v1:0", messages="test"
        )

        assert response.text == "Response text"
        assert response.usage["input_tokens"] == 5
        mock_client.invoke_model.assert_called_once()

    @patch("traigent.integrations.bedrock_client._require_boto3")
    def test_invoke_with_profile_name(
        self, mock_require: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test invoke uses profile_name when provided."""
        monkeypatch.delenv("BEDROCK_MOCK", raising=False)

        mock_boto3 = MagicMock()
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_response = {
            "body": MagicMock(
                read=lambda: json.dumps(
                    {
                        "content": [{"type": "text", "text": "Test"}],
                    }
                ).encode()
            )
        }
        mock_client.invoke_model.return_value = mock_response
        mock_session.client.return_value = mock_client
        mock_boto3.session.Session.return_value = mock_session
        mock_require.return_value = mock_boto3

        client = BedrockChatClient(profile_name="my-profile")
        response = client.invoke(model_id="test-model", messages="test")

        assert response.text == "Test"
        mock_boto3.session.Session.assert_called_once_with(profile_name="my-profile")

    @patch("traigent.integrations.bedrock_client._require_boto3")
    def test_invoke_with_json_string_body(
        self, mock_require: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test invoke handles response body as JSON string."""
        monkeypatch.delenv("BEDROCK_MOCK", raising=False)

        mock_boto3 = MagicMock()
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_response = {
            "body": json.dumps({"content": [{"type": "text", "text": "String body"}]})
        }
        mock_client.invoke_model.return_value = mock_response
        mock_session.client.return_value = mock_client
        mock_boto3.session.Session.return_value = mock_session
        mock_require.return_value = mock_boto3

        client = BedrockChatClient()
        response = client.invoke(model_id="test-model", messages="test")

        assert response.text == "String body"

    def test_invoke_delegates_to_ai21_for_jamba_models(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test invoke delegates to _invoke_ai21 for AI21 models."""
        mock_client = _mock_invoke_client(
            {
                "choices": [{"message": {"content": "AI21 response"}}],
                "usage": {"input_tokens": 3, "output_tokens": 4},
            }
        )
        client = BedrockChatClient(client=mock_client)

        response = client.invoke(
            model_id="ai21.jamba-1-5-mini-v1:0",
            messages="test prompt",
            max_tokens=50,
        )

        assert response.text == "AI21 response"
        mock_client.invoke_model.assert_called_once()


class TestBedrockChatClientInvokeAI21:
    """Tests for BedrockChatClient._invoke_ai21 method."""

    def test_invoke_ai21_with_string_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test _invoke_ai21 sends string messages to Bedrock."""
        mock_client = _mock_invoke_client(
            {
                "choices": [{"message": {"content": "AI21 string response"}}],
                "usage": {"input_tokens": 4, "output_tokens": 8},
            }
        )
        client = BedrockChatClient(client=mock_client)

        response = client._invoke_ai21(
            model_id="ai21.jamba-1-5-large-v1:0",
            messages="AI21 test",
            max_tokens=32,
            temperature=0.5,
            top_p=None,
        )

        assert response.text == "AI21 string response"
        assert response.usage is not None
        assert response.usage["output_tokens"] == 8
        body = json.loads(mock_client.invoke_model.call_args.kwargs["body"])
        assert body["messages"] == [{"role": "user", "content": "AI21 test"}]

    def test_invoke_ai21_with_list_messages(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test _invoke_ai21 sends list messages to Bedrock."""
        mock_client = _mock_invoke_client(
            {"choices": [{"message": {"content": "AI21 list response"}}]}
        )
        client = BedrockChatClient(client=mock_client)

        messages = [
            {"role": "user", "content": [{"type": "text", "text": "First"}]},
            {"role": "user", "content": [{"type": "text", "text": "Second"}]},
        ]
        response = client._invoke_ai21(
            model_id="ai21.jamba-1-5-mini-v1:0",
            messages=messages,
            max_tokens=64,
            temperature=0.7,
            top_p=0.9,
        )

        assert response.text == "AI21 list response"
        body = json.loads(mock_client.invoke_model.call_args.kwargs["body"])
        assert body["messages"] == [
            {"role": "user", "content": "First"},
            {"role": "user", "content": "Second"},
        ]

    @patch("traigent.integrations.bedrock_client._require_boto3")
    def test_invoke_ai21_real_mode_with_choices(
        self, mock_require: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test _invoke_ai21 extracts text from choices in response."""
        monkeypatch.delenv("BEDROCK_MOCK", raising=False)

        mock_boto3 = MagicMock()
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_response = {
            "body": MagicMock(
                read=lambda: json.dumps(
                    {
                        "choices": [{"message": {"content": "AI21 response"}}],
                        "usage": {"input_tokens": 8, "output_tokens": 12},
                    }
                ).encode()
            )
        }
        mock_client.invoke_model.return_value = mock_response
        mock_session.client.return_value = mock_client
        mock_boto3.session.Session.return_value = mock_session
        mock_require.return_value = mock_boto3

        client = BedrockChatClient()
        response = client._invoke_ai21(
            model_id="ai21.jamba-1-5-large-v1:0",
            messages="test",
            max_tokens=50,
            temperature=0.5,
            top_p=None,
        )

        assert response.text == "AI21 response"
        assert response.usage["input_tokens"] == 8

    @patch("traigent.integrations.bedrock_client._require_boto3")
    def test_invoke_ai21_real_mode_with_output_text(
        self, mock_require: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test _invoke_ai21 falls back to outputText field."""
        monkeypatch.delenv("BEDROCK_MOCK", raising=False)

        mock_boto3 = MagicMock()
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_response = {
            "body": MagicMock(
                read=lambda: json.dumps(
                    {
                        "outputText": "Fallback text",
                        "tokenUsage": {"input": 5, "output": 7},
                    }
                ).encode()
            )
        }
        mock_client.invoke_model.return_value = mock_response
        mock_session.client.return_value = mock_client
        mock_boto3.session.Session.return_value = mock_session
        mock_require.return_value = mock_boto3

        client = BedrockChatClient()
        response = client._invoke_ai21(
            model_id="ai21.jamba-1-5-mini-v1:0",
            messages="test",
            max_tokens=50,
            temperature=0.5,
            top_p=None,
        )

        assert response.text == "Fallback text"

    def test_invoke_ai21_coerces_list_content(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test _invoke_ai21 handles list content in messages."""
        mock_client = _mock_invoke_client(
            {"choices": [{"message": {"content": "Coerced response"}}]}
        )
        client = BedrockChatClient(client=mock_client)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Part 1"},
                    {"type": "text", "text": "Part 2"},
                ],
            }
        ]
        response = client._invoke_ai21(
            model_id="ai21.jamba-1-5-mini-v1:0",
            messages=messages,
            max_tokens=32,
            temperature=0.5,
            top_p=None,
        )

        assert response.text == "Coerced response"
        body = json.loads(mock_client.invoke_model.call_args.kwargs["body"])
        assert body["messages"] == [{"role": "user", "content": "Part 1\nPart 2"}]


class TestBedrockChatClientInvokeStream:
    """Tests for BedrockChatClient.invoke_stream method."""

    def test_invoke_stream_uses_configured_client_when_bedrock_mock_is_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test BEDROCK_MOCK does not fabricate streaming chunks."""
        monkeypatch.setenv("BEDROCK_MOCK", "true")
        mock_client = _mock_stream_client(
            {"content": [{"type": "text", "text": "Stream"}]},
            {"content": [{"type": "text", "text": " response"}]},
        )
        client = BedrockChatClient(client=mock_client)

        gen = client.invoke_stream(
            model_id="anthropic.claude-3-opus-20240229-v1:0",
            messages="Stream test",
            max_tokens=512,
        )

        chunks = list(gen)

        assert chunks == ["Stream", "response"]
        assert not any("[MOCK:" in c for c in chunks)
        mock_client.invoke_model_with_response_stream.assert_called_once()

    def test_invoke_stream_sends_coerced_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test invoke_stream sends the message in the Bedrock payload."""
        mock_client = _mock_stream_client()
        client = BedrockChatClient(client=mock_client)

        long_message = "This is a longer message for testing"
        gen = client.invoke_stream(model_id="test-model", messages=long_message)

        chunks = list(gen)

        assert chunks == []
        body = json.loads(
            mock_client.invoke_model_with_response_stream.call_args.kwargs["body"]
        )
        assert body["messages"][0]["content"][0]["text"] == long_message

    @patch("traigent.integrations.bedrock_client._require_boto3")
    def test_invoke_stream_with_real_client(
        self, mock_require: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test invoke_stream with mocked boto3 streaming client."""
        monkeypatch.delenv("BEDROCK_MOCK", raising=False)

        # Mock streaming response
        mock_boto3 = MagicMock()
        mock_session = MagicMock()
        mock_client = MagicMock()

        # Create mock stream events
        chunk_data = [
            {"content": [{"type": "text", "text": "First"}]},
            {"content": [{"type": "text", "text": "second"}]},
            {"content": [{"type": "text", "text": "third"}]},
        ]
        mock_events = [
            {"chunk": {"bytes": json.dumps(data).encode()}} for data in chunk_data
        ]
        mock_stream = iter(mock_events)

        mock_response = {"body": mock_stream}
        mock_client.invoke_model_with_response_stream.return_value = mock_response
        mock_session.client.return_value = mock_client
        mock_boto3.session.Session.return_value = mock_session
        mock_require.return_value = mock_boto3

        client = BedrockChatClient()
        gen = client.invoke_stream(model_id="test-model", messages="test")

        chunks = []
        for chunk in gen:
            if isinstance(chunk, str):
                chunks.append(chunk)

        # Verify all expected chunks are present
        assert "First" in chunks
        assert "second" in chunks
        assert "third" in chunks

    @patch("traigent.integrations.bedrock_client._require_boto3")
    def test_invoke_stream_handles_malformed_chunks(
        self, mock_require: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test invoke_stream gracefully handles malformed stream chunks."""
        monkeypatch.delenv("BEDROCK_MOCK", raising=False)

        mock_boto3 = MagicMock()
        mock_session = MagicMock()
        mock_client = MagicMock()

        # Mix of valid and invalid chunks
        mock_events = [
            {
                "chunk": {
                    "bytes": json.dumps(
                        {"content": [{"type": "text", "text": "Valid"}]}
                    ).encode()
                }
            },
            {"chunk": {"bytes": b"invalid json"}},  # Will be skipped
            {"no_chunk": "data"},  # Will be skipped
        ]
        mock_stream = iter(mock_events)

        mock_response = {"body": mock_stream}
        mock_client.invoke_model_with_response_stream.return_value = mock_response
        mock_session.client.return_value = mock_client
        mock_boto3.session.Session.return_value = mock_session
        mock_require.return_value = mock_boto3

        client = BedrockChatClient()
        gen = client.invoke_stream(model_id="test-model", messages="test")

        chunks = []
        for chunk in gen:
            if isinstance(chunk, str):
                chunks.append(chunk)

        # Should only get valid chunk
        assert chunks == ["Valid"]


class TestBedrockChatClientAsyncInvoke:
    """Tests for BedrockChatClient.ainvoke method."""

    @pytest.mark.asyncio
    async def test_ainvoke_uses_sync_fallback_when_bedrock_mock_is_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test BEDROCK_MOCK does not fabricate async invoke responses."""
        monkeypatch.setenv("BEDROCK_MOCK", "true")
        mock_response = BedrockChatResponse(
            text="Fallback response",
            raw={"fallback": True},
            usage={"input_tokens": 5, "output_tokens": 10},
        )
        client = BedrockChatClient()

        with patch.object(client, "invoke", return_value=mock_response) as mock_invoke:
            with patch("builtins.__import__", side_effect=ImportError("No aioboto3")):
                response = await client.ainvoke(
                    model_id="anthropic.claude-3-haiku-20240307-v1:0",
                    messages="Async test",
                    max_tokens=32,
                )

        assert response.text == "Fallback response"
        assert "[MOCK:" not in response.text
        mock_invoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_ainvoke_falls_back_to_sync_without_aioboto3(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test ainvoke falls back to sync invoke when aioboto3 not available."""
        monkeypatch.delenv("BEDROCK_MOCK", raising=False)

        # Mock sync invoke to return a response
        mock_response = BedrockChatResponse(
            text="Fallback response",
            raw={"fallback": True},
            usage={"input_tokens": 5, "output_tokens": 10},
        )

        client = BedrockChatClient()

        with patch.object(client, "invoke", return_value=mock_response):
            with patch("builtins.__import__", side_effect=ImportError("No aioboto3")):
                response = await client.ainvoke(model_id="test-model", messages="test")

        assert response.text == "Fallback response"
        assert response.raw["fallback"] is True

    @pytest.mark.asyncio
    async def test_ainvoke_with_aioboto3(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test ainvoke uses aioboto3 when available."""
        monkeypatch.delenv("BEDROCK_MOCK", raising=False)

        # Mock aioboto3
        mock_aioboto3 = MagicMock()
        mock_session = MagicMock()
        mock_client = AsyncMock()

        # Mock async context manager
        mock_client_context = AsyncMock()
        mock_client_context.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_context.__aexit__ = AsyncMock(return_value=None)
        mock_session.client.return_value = mock_client_context

        # Mock invoke response
        mock_body = AsyncMock()
        mock_body.read = AsyncMock(
            return_value=json.dumps(
                {
                    "content": [{"type": "text", "text": "Async response"}],
                    "usage": {"input_tokens": 7, "output_tokens": 14},
                }
            ).encode()
        )

        mock_client.invoke_model = AsyncMock(return_value={"body": mock_body})
        mock_aioboto3.Session.return_value = mock_session

        client = BedrockChatClient()

        with patch.dict("sys.modules", {"aioboto3": mock_aioboto3}):
            response = await client.ainvoke(model_id="test-model", messages="test")

        assert response.text == "Async response"
        assert response.usage["input_tokens"] == 7


class TestBedrockChatClientAsyncInvokeStream:
    """Tests for BedrockChatClient.ainvoke_stream method."""

    @pytest.mark.asyncio
    async def test_ainvoke_stream_uses_sync_fallback_when_bedrock_mock_is_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test BEDROCK_MOCK does not fabricate async streaming chunks."""
        monkeypatch.setenv("BEDROCK_MOCK", "true")
        client = BedrockChatClient()

        def mock_sync_stream(*args, **kwargs):
            yield "chunk1"
            yield "chunk2"
            return BedrockChatResponse(text="chunk1chunk2", raw={"streamed": True})

        with patch.object(client, "invoke_stream", side_effect=mock_sync_stream):
            with patch("builtins.__import__", side_effect=ImportError("No aioboto3")):
                chunks = []
                async for chunk in client.ainvoke_stream(
                    model_id="anthropic.claude-3-opus-20240229-v1:0",
                    messages="Async stream test",
                ):
                    chunks.append(chunk)

        assert chunks == ["chunk1", "chunk2"]
        assert not any("[MOCK:" in c for c in chunks)

    @pytest.mark.asyncio
    async def test_ainvoke_stream_falls_back_to_sync(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test ainvoke_stream falls back to sync streaming without aioboto3."""
        monkeypatch.delenv("BEDROCK_MOCK", raising=False)

        client = BedrockChatClient()

        # Mock sync streaming generator
        def mock_sync_stream(*args, **kwargs):
            yield "chunk1"
            yield "chunk2"
            return BedrockChatResponse(text="chunk1chunk2", raw={"streamed": True})

        with patch.object(client, "invoke_stream", side_effect=mock_sync_stream):
            with patch("builtins.__import__", side_effect=ImportError("No aioboto3")):
                chunks = []
                async for chunk in client.ainvoke_stream(
                    model_id="test-model", messages="test"
                ):
                    chunks.append(chunk)

        assert chunks == ["chunk1", "chunk2"]

    @pytest.mark.asyncio
    async def test_ainvoke_stream_with_aioboto3(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test ainvoke_stream uses aioboto3 when available."""
        monkeypatch.delenv("BEDROCK_MOCK", raising=False)

        # Create async generator for stream events
        async def mock_stream():
            chunks_data = [
                {"content": [{"type": "text", "text": "First"}]},
                {"content": [{"type": "text", "text": "second"}]},
            ]
            for data in chunks_data:
                yield {"chunk": {"bytes": json.dumps(data).encode()}}

        # Mock aioboto3
        mock_aioboto3 = MagicMock()
        mock_session = MagicMock()
        mock_client = AsyncMock()

        mock_client_context = AsyncMock()
        mock_client_context.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_context.__aexit__ = AsyncMock(return_value=None)
        mock_session.client.return_value = mock_client_context

        mock_client.invoke_model_with_response_stream = AsyncMock(
            return_value={"body": mock_stream()}
        )
        mock_aioboto3.Session.return_value = mock_session

        client = BedrockChatClient()

        with patch.dict("sys.modules", {"aioboto3": mock_aioboto3}):
            chunks = []
            async for chunk in client.ainvoke_stream(
                model_id="test-model", messages="test"
            ):
                chunks.append(chunk)

        # Verify all expected chunks are present
        assert "First" in chunks
        assert "second" in chunks


class TestResolveDefaultBedrockModelId:
    """Tests for resolve_default_bedrock_model_id function."""

    def test_resolve_uses_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test resolve_default_bedrock_model_id respects BEDROCK_MODEL_ID env var."""
        monkeypatch.setenv("BEDROCK_MODEL_ID", "custom.model.id")
        result = resolve_default_bedrock_model_id("sonnet")
        assert result == "custom.model.id"

    def test_resolve_sonnet_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test resolve_default_bedrock_model_id maps sonnet hint correctly."""
        monkeypatch.delenv("BEDROCK_MODEL_ID", raising=False)
        result = resolve_default_bedrock_model_id("sonnet")
        assert result == "anthropic.claude-3-sonnet-20240229-v1:0"

    def test_resolve_haiku_4_5_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test resolve_default_bedrock_model_id maps haiku 4.5 hint."""
        monkeypatch.delenv("BEDROCK_MODEL_ID", raising=False)
        result = resolve_default_bedrock_model_id("haiku 4.5")
        assert result == "anthropic.claude-3-5-haiku-20241022-v1:0"

    def test_resolve_haiku_3_5_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test resolve_default_bedrock_model_id maps haiku 3.5 hint."""
        monkeypatch.delenv("BEDROCK_MODEL_ID", raising=False)
        result = resolve_default_bedrock_model_id("haiku 3.5")
        assert result == "anthropic.claude-3-5-haiku-20241022-v1:0"

    def test_resolve_haiku_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test resolve_default_bedrock_model_id maps haiku hint."""
        monkeypatch.delenv("BEDROCK_MODEL_ID", raising=False)
        result = resolve_default_bedrock_model_id("haiku")
        assert result == "anthropic.claude-3-haiku-20240307-v1:0"

    def test_resolve_opus_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test resolve_default_bedrock_model_id maps opus hint."""
        monkeypatch.delenv("BEDROCK_MODEL_ID", raising=False)
        result = resolve_default_bedrock_model_id("opus")
        assert result == "anthropic.claude-3-opus-20240229-v1:0"

    def test_resolve_jamba_mini_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test resolve_default_bedrock_model_id maps jamba mini hint."""
        monkeypatch.delenv("BEDROCK_MODEL_ID", raising=False)
        result = resolve_default_bedrock_model_id("jamba mini")
        assert result == "ai21.jamba-1-5-mini-v1:0"

    def test_resolve_jamba_large_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test resolve_default_bedrock_model_id maps jamba large hint."""
        monkeypatch.delenv("BEDROCK_MODEL_ID", raising=False)
        result = resolve_default_bedrock_model_id("jamba large")
        assert result == "ai21.jamba-1-5-large-v1:0"

    def test_resolve_jamba_instruct_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test resolve_default_bedrock_model_id maps jamba instruct hint."""
        monkeypatch.delenv("BEDROCK_MODEL_ID", raising=False)
        result = resolve_default_bedrock_model_id("jamba instruct")
        assert result == "ai21.jamba-1-5-large-v1:0"

    def test_resolve_jamba_default_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test resolve_default_bedrock_model_id defaults jamba to mini."""
        monkeypatch.delenv("BEDROCK_MODEL_ID", raising=False)
        result = resolve_default_bedrock_model_id("jamba")
        assert result == "ai21.jamba-1-5-mini-v1:0"

    def test_resolve_ai21_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test resolve_default_bedrock_model_id maps ai21 hint."""
        monkeypatch.delenv("BEDROCK_MODEL_ID", raising=False)
        result = resolve_default_bedrock_model_id("ai21")
        assert result == "ai21.jamba-1-5-mini-v1:0"

    def test_resolve_none_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test resolve_default_bedrock_model_id with None hint uses default."""
        monkeypatch.delenv("BEDROCK_MODEL_ID", raising=False)
        result = resolve_default_bedrock_model_id(None)
        assert result == "anthropic.claude-3-sonnet-20240229-v1:0"

    def test_resolve_empty_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test resolve_default_bedrock_model_id with empty hint uses default."""
        monkeypatch.delenv("BEDROCK_MODEL_ID", raising=False)
        result = resolve_default_bedrock_model_id("")
        assert result == "anthropic.claude-3-sonnet-20240229-v1:0"

    def test_resolve_unknown_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test resolve_default_bedrock_model_id with unknown hint uses default."""
        monkeypatch.delenv("BEDROCK_MODEL_ID", raising=False)
        result = resolve_default_bedrock_model_id("unknown-model")
        assert result == "anthropic.claude-3-sonnet-20240229-v1:0"

    def test_resolve_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test resolve_default_bedrock_model_id is case insensitive."""
        monkeypatch.delenv("BEDROCK_MODEL_ID", raising=False)
        result = resolve_default_bedrock_model_id("SONNET")
        assert result == "anthropic.claude-3-sonnet-20240229-v1:0"
