"""Unit tests for Azure OpenAI client integration.

Tests for AzureOpenAIChatClient and related helper functions, covering both
mock mode and mocked openai client interactions.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility CONC-Quality-Reliability FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from traigent.integrations.azure_openai_client import (
    AzureOpenAIChatClient,
    AzureOpenAIChatResponse,
    _coerce_messages,
)


class TestCoerceMessagesHelper:
    """Tests for _coerce_messages helper function."""

    def test_coerce_messages_with_string(self) -> None:
        """Test _coerce_messages converts string to message format."""
        result = _coerce_messages("Hello world")
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello world"

    def test_coerce_messages_with_list_of_strings(self) -> None:
        """Test _coerce_messages converts list of strings to messages."""
        result = _coerce_messages(["First message", "Second message"])
        assert len(result) == 2
        assert all(msg["role"] == "user" for msg in result)
        assert result[0]["content"] == "First message"
        assert result[1]["content"] == "Second message"

    def test_coerce_messages_with_list_of_dicts(self) -> None:
        """Test _coerce_messages passes through list of message dicts."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]
        result = _coerce_messages(messages)
        assert result == messages

    def test_coerce_messages_with_empty_list(self) -> None:
        """Test _coerce_messages handles empty list gracefully."""
        result = _coerce_messages([])
        assert result == []

    def test_coerce_messages_with_empty_string(self) -> None:
        """Test _coerce_messages handles empty string."""
        result = _coerce_messages("")
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == ""

    def test_coerce_messages_with_tuple_of_strings(self) -> None:
        """Test _coerce_messages handles tuple of strings."""
        result = _coerce_messages(("First", "Second"))
        assert len(result) == 2
        assert result[0]["content"] == "First"
        assert result[1]["content"] == "Second"


class TestAzureOpenAIChatResponseDataclass:
    """Tests for AzureOpenAIChatResponse dataclass."""

    def test_response_creation_with_required_fields(self) -> None:
        """Test AzureOpenAIChatResponse can be created with required fields."""
        response = AzureOpenAIChatResponse(text="test response", raw={"data": "value"})
        assert response.text == "test response"
        assert response.raw == {"data": "value"}
        assert response.usage is None

    def test_response_creation_with_usage(self) -> None:
        """Test AzureOpenAIChatResponse stores usage information."""
        usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        response = AzureOpenAIChatResponse(
            text="test", raw={"data": "value"}, usage=usage
        )
        assert response.usage == usage
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 20

    def test_response_with_none_usage(self) -> None:
        """Test AzureOpenAIChatResponse handles None usage explicitly."""
        response = AzureOpenAIChatResponse(text="test", raw={}, usage=None)
        assert response.usage is None


class TestAzureOpenAIChatClientInit:
    """Tests for AzureOpenAIChatClient initialization."""

    def test_init_with_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test AzureOpenAIChatClient initializes with default empty values."""
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)

        client = AzureOpenAIChatClient()
        assert client._endpoint == ""
        assert client._api_key == ""
        assert client._api_version == "2024-02-15-preview"

    def test_init_with_explicit_params(self) -> None:
        """Test AzureOpenAIChatClient accepts explicit initialization parameters."""
        client = AzureOpenAIChatClient(
            endpoint="https://test.openai.azure.com",
            api_key="test-key-123",
            api_version="2024-01-01",
        )
        assert client._endpoint == "https://test.openai.azure.com"
        assert client._api_key == "test-key-123"
        assert client._api_version == "2024-01-01"

    def test_init_with_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test AzureOpenAIChatClient reads from environment variables."""
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://env.openai.azure.com")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-key-456")
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2023-12-01")

        client = AzureOpenAIChatClient()
        assert client._endpoint == "https://env.openai.azure.com"
        assert client._api_key == "env-key-456"
        assert client._api_version == "2023-12-01"

    def test_init_explicit_params_override_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test explicit parameters override environment variables."""
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://env.openai.azure.com")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-key")

        client = AzureOpenAIChatClient(
            endpoint="https://explicit.openai.azure.com", api_key="explicit-key"
        )
        assert client._endpoint == "https://explicit.openai.azure.com"
        assert client._api_key == "explicit-key"


class TestAzureOpenAIChatClientInvokeMockMode:
    """Tests for AzureOpenAIChatClient.invoke in mock mode."""

    def test_invoke_mock_mode_with_string_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test invoke in mock mode with string message."""
        monkeypatch.setenv("AZURE_OPENAI_MOCK", "true")
        client = AzureOpenAIChatClient()

        response = client.invoke(deployment="gpt-4o-mini", messages="Hello AI")
        assert response.text.startswith("[MOCK_AZURE:gpt-4o-mini]")
        assert "Hello AI" in response.text
        assert response.raw["mock"] is True
        assert response.usage is not None
        assert response.usage["prompt_tokens"] == 0

    def test_invoke_mock_mode_with_list_messages(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test invoke in mock mode with list of messages."""
        monkeypatch.setenv("AZURE_OPENAI_MOCK", "true")
        client = AzureOpenAIChatClient()

        messages = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Last message"},
        ]
        response = client.invoke(deployment="gpt-4", messages=messages, max_tokens=64)
        assert "[MOCK_AZURE:gpt-4]" in response.text
        assert "Last message" in response.text
        assert response.usage["completion_tokens"] == 32  # min(max_tokens, 32)

    def test_invoke_mock_mode_case_insensitive(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test AZURE_OPENAI_MOCK environment variable is case insensitive."""
        monkeypatch.setenv("AZURE_OPENAI_MOCK", "TrUe")
        client = AzureOpenAIChatClient()

        response = client.invoke(deployment="test-deployment", messages="test")
        assert response.raw["mock"] is True

    def test_invoke_mock_mode_whitespace_handling(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test AZURE_OPENAI_MOCK handles whitespace correctly."""
        monkeypatch.setenv("AZURE_OPENAI_MOCK", "  true  ")
        client = AzureOpenAIChatClient()

        response = client.invoke(deployment="test", messages="test")
        assert response.raw["mock"] is True

    def test_invoke_mock_mode_with_empty_messages(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test invoke in mock mode with empty messages list."""
        monkeypatch.setenv("AZURE_OPENAI_MOCK", "true")
        client = AzureOpenAIChatClient()

        response = client.invoke(deployment="gpt-4", messages=[])
        assert "[MOCK_AZURE:gpt-4]" in response.text
        assert response.raw["mock"] is True


class TestAzureOpenAIChatClientInvokeRealMode:
    """Tests for AzureOpenAIChatClient.invoke with mocked OpenAI client."""

    def test_invoke_real_mode_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test invoke successfully calls OpenAI client."""
        monkeypatch.setenv("AZURE_OPENAI_MOCK", "false")

        # Mock the openai module
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "  AI response text  "
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        # Mock usage
        mock_usage = MagicMock()
        mock_usage.model_dump.return_value = {
            "prompt_tokens": 15,
            "completion_tokens": 25,
        }
        mock_response.usage = mock_usage

        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai}):
            client = AzureOpenAIChatClient(
                endpoint="https://test.openai.azure.com", api_key="test-key"
            )
            response = client.invoke(
                deployment="gpt-4o",
                messages="Hello",
                max_tokens=100,
                temperature=0.7,
            )

        assert response.text == "AI response text"
        assert response.usage["prompt_tokens"] == 15
        assert response.usage["completion_tokens"] == 25

        # Verify OpenAI client was called correctly
        mock_openai.OpenAI.assert_called_once()
        call_kwargs = mock_openai.OpenAI.call_args[1]
        assert call_kwargs["api_key"] == "test-key"
        assert "gpt-4o" in call_kwargs["base_url"]
        assert call_kwargs["default_query"]["api-version"] == "2024-02-15-preview"

    def test_invoke_real_mode_with_top_p(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test invoke includes top_p when specified."""
        monkeypatch.setenv("AZURE_OPENAI_MOCK", "false")

        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "response"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_response.usage = None

        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai}):
            client = AzureOpenAIChatClient(endpoint="https://test.com", api_key="key")
            client.invoke(
                deployment="gpt-4", messages="test", top_p=0.95, max_tokens=50
            )

        # Verify top_p was passed
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["top_p"] == 0.95
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 50

    def test_invoke_real_mode_with_extra_params(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test invoke includes extra_params in payload."""
        monkeypatch.setenv("AZURE_OPENAI_MOCK", "false")

        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "response"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_response.usage = None

        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai}):
            client = AzureOpenAIChatClient(endpoint="https://test.com", api_key="key")
            client.invoke(
                deployment="gpt-4",
                messages="test",
                extra_params={"frequency_penalty": 0.5, "presence_penalty": 0.2},
            )

        # Verify extra params were passed
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["frequency_penalty"] == 0.5
        assert call_kwargs["presence_penalty"] == 0.2

    def test_invoke_real_mode_usage_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test invoke handles usage when model_dump fails."""
        monkeypatch.setenv("AZURE_OPENAI_MOCK", "false")

        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "response"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        # Mock usage that raises on model_dump but works with dict()
        mock_usage = MagicMock()
        mock_usage.model_dump.side_effect = Exception("model_dump not available")
        mock_usage.__iter__ = lambda self: iter([("prompt_tokens", 5)])
        mock_usage.__getitem__ = lambda self, key: 5 if key == "prompt_tokens" else 0
        mock_response.usage = mock_usage

        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai}):
            client = AzureOpenAIChatClient(endpoint="https://test.com", api_key="key")
            response = client.invoke(deployment="gpt-4", messages="test")

        # Should fallback to dict() conversion
        assert response.usage is not None

    def test_invoke_raises_import_error_when_openai_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test invoke raises ImportError when openai is not installed."""
        monkeypatch.setenv("AZURE_OPENAI_MOCK", "false")

        # Remove openai from sys.modules if present
        with patch.dict("sys.modules", {"openai": None}):
            with patch(
                "builtins.__import__",
                side_effect=ImportError("No module named 'openai'"),
            ):
                client = AzureOpenAIChatClient()
                with pytest.raises(
                    ImportError, match="Install 'openai' for Azure OpenAI usage"
                ):
                    client.invoke(deployment="gpt-4", messages="test")


class TestAzureOpenAIChatClientAInvokeMockMode:
    """Tests for AzureOpenAIChatClient.ainvoke in mock mode."""

    @pytest.mark.asyncio
    async def test_ainvoke_mock_mode_with_string_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test ainvoke in mock mode with string message."""
        monkeypatch.setenv("AZURE_OPENAI_MOCK", "true")
        client = AzureOpenAIChatClient()

        response = await client.ainvoke(
            deployment="gpt-4o-mini", messages="Async hello"
        )
        assert response.text.startswith("[MOCK_AZURE:gpt-4o-mini]")
        assert "Async hello" in response.text
        assert response.raw["mock"] is True

    @pytest.mark.asyncio
    async def test_ainvoke_mock_mode_respects_max_tokens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test ainvoke mock mode usage respects max_tokens limit."""
        monkeypatch.setenv("AZURE_OPENAI_MOCK", "true")
        client = AzureOpenAIChatClient()

        response = await client.ainvoke(
            deployment="gpt-4", messages="test", max_tokens=16
        )
        assert response.usage["completion_tokens"] == 16


class TestAzureOpenAIChatClientAInvokeRealMode:
    """Tests for AzureOpenAIChatClient.ainvoke with mocked AsyncOpenAI client."""

    @pytest.mark.asyncio
    async def test_ainvoke_real_mode_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test ainvoke successfully calls AsyncOpenAI client."""
        monkeypatch.setenv("AZURE_OPENAI_MOCK", "false")

        # Mock the openai module
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "  Async AI response  "
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_usage = MagicMock()
        mock_usage.model_dump.return_value = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
        }
        mock_response.usage = mock_usage

        # Make create return an awaitable
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai.AsyncOpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai}):
            client = AzureOpenAIChatClient(
                endpoint="https://test.openai.azure.com", api_key="test-key"
            )
            response = await client.ainvoke(
                deployment="gpt-4o", messages="Hello async", temperature=0.8
            )

        assert response.text == "Async AI response"
        assert response.usage["prompt_tokens"] == 10
        mock_openai.AsyncOpenAI.assert_called_once()

    @pytest.mark.asyncio
    async def test_ainvoke_real_mode_with_top_p(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test ainvoke includes top_p when specified."""
        monkeypatch.setenv("AZURE_OPENAI_MOCK", "false")

        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "response"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_response.usage = None

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai.AsyncOpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai}):
            client = AzureOpenAIChatClient(endpoint="https://test.com", api_key="key")
            await client.ainvoke(deployment="gpt-4", messages="test", top_p=0.9)

        # Verify top_p was passed
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["top_p"] == 0.9

    @pytest.mark.asyncio
    async def test_ainvoke_real_mode_with_extra_params(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test ainvoke includes extra_params in payload."""
        monkeypatch.setenv("AZURE_OPENAI_MOCK", "false")

        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "response"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_response.usage = None

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai.AsyncOpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai}):
            client = AzureOpenAIChatClient(endpoint="https://test.com", api_key="key")
            await client.ainvoke(
                deployment="gpt-4",
                messages="test",
                extra_params={"stop": ["END"], "n": 2},
            )

        # Verify extra params were passed
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["stop"] == ["END"]
        assert call_kwargs["n"] == 2

    @pytest.mark.asyncio
    async def test_ainvoke_real_mode_usage_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test ainvoke handles usage when model_dump fails."""
        monkeypatch.setenv("AZURE_OPENAI_MOCK", "false")

        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "response"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        # Mock usage that raises on model_dump but works with dict()
        mock_usage = MagicMock()
        mock_usage.model_dump.side_effect = Exception("model_dump not available")
        mock_usage.__iter__ = lambda self: iter([("prompt_tokens", 7)])
        mock_usage.__getitem__ = lambda self, key: 7 if key == "prompt_tokens" else 0
        mock_response.usage = mock_usage

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai.AsyncOpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai}):
            client = AzureOpenAIChatClient(endpoint="https://test.com", api_key="key")
            response = await client.ainvoke(deployment="gpt-4", messages="test")

        # Should fallback to dict() conversion
        assert response.usage is not None

    @pytest.mark.asyncio
    async def test_ainvoke_raises_import_error_when_openai_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test ainvoke raises ImportError when openai is not installed."""
        monkeypatch.setenv("AZURE_OPENAI_MOCK", "false")

        with patch.dict("sys.modules", {"openai": None}):
            with patch(
                "builtins.__import__",
                side_effect=ImportError("No module named 'openai'"),
            ):
                client = AzureOpenAIChatClient()
                with pytest.raises(
                    ImportError, match="Install 'openai' for Azure OpenAI async usage"
                ):
                    await client.ainvoke(deployment="gpt-4", messages="test")


class TestAzureOpenAIChatClientInvokeStreamMockMode:
    """Tests for AzureOpenAIChatClient.invoke_stream in mock mode."""

    def test_invoke_stream_mock_mode_yields_chunks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test invoke_stream in mock mode yields text chunks."""
        monkeypatch.setenv("AZURE_OPENAI_MOCK", "true")
        client = AzureOpenAIChatClient()

        generator = client.invoke_stream(
            deployment="gpt-4o", messages="Stream test message"
        )
        chunks = []
        try:
            while True:
                chunk = next(generator)
                chunks.append(chunk)
        except StopIteration as e:
            response = e.value

        assert len(chunks) == 3
        assert "[MOCK_AZURE:gpt-4o]" in chunks[0]
        assert response is not None
        assert response.text == "".join(chunks)
        assert response.raw["mock"] is True

    def test_invoke_stream_mock_mode_with_empty_messages(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test invoke_stream in mock mode with empty messages."""
        monkeypatch.setenv("AZURE_OPENAI_MOCK", "true")
        client = AzureOpenAIChatClient()

        generator = client.invoke_stream(deployment="test", messages=[])
        chunks = list(generator)

        assert len(chunks) > 0
        assert any("[MOCK_AZURE:test]" in chunk for chunk in chunks)


class TestAzureOpenAIChatClientInvokeStreamRealMode:
    """Tests for AzureOpenAIChatClient.invoke_stream with mocked OpenAI client."""

    def test_invoke_stream_real_mode_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test invoke_stream successfully streams from OpenAI client."""
        monkeypatch.setenv("AZURE_OPENAI_MOCK", "false")

        # Mock the openai module
        mock_openai = MagicMock()
        mock_client = MagicMock()

        # Create mock streaming chunks
        mock_chunks = []
        for text in ["Hello", " there", "!"]:
            chunk = MagicMock()
            delta = MagicMock()
            delta.content = text
            choice = MagicMock()
            choice.delta = delta
            chunk.choices = [choice]
            mock_chunks.append(chunk)

        mock_client.chat.completions.create.return_value = iter(mock_chunks)
        mock_openai.OpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai}):
            client = AzureOpenAIChatClient(
                endpoint="https://test.openai.azure.com", api_key="test-key"
            )
            generator = client.invoke_stream(deployment="gpt-4o", messages="Stream me")
            chunks = []
            try:
                while True:
                    chunk = next(generator)
                    chunks.append(chunk)
            except StopIteration as e:
                response = e.value

        assert chunks == ["Hello", " there", "!"]
        assert response is not None
        assert response.text == "Hello there!"
        assert response.raw["streamed"] is True
        assert response.usage is None

        # Verify stream=True was passed
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["stream"] is True

    def test_invoke_stream_real_mode_skips_empty_chunks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test invoke_stream skips chunks with no content."""
        monkeypatch.setenv("AZURE_OPENAI_MOCK", "false")

        mock_openai = MagicMock()
        mock_client = MagicMock()

        # Create chunks where some have no content
        mock_chunks = []

        # Chunk with content
        chunk1 = MagicMock()
        delta1 = MagicMock()
        delta1.content = "Hello"
        choice1 = MagicMock()
        choice1.delta = delta1
        chunk1.choices = [choice1]
        mock_chunks.append(chunk1)

        # Empty chunk (no content)
        chunk2 = MagicMock()
        delta2 = MagicMock()
        delta2.content = None
        choice2 = MagicMock()
        choice2.delta = delta2
        chunk2.choices = [choice2]
        mock_chunks.append(chunk2)

        # Chunk with content
        chunk3 = MagicMock()
        delta3 = MagicMock()
        delta3.content = "World"
        choice3 = MagicMock()
        choice3.delta = delta3
        chunk3.choices = [choice3]
        mock_chunks.append(chunk3)

        mock_client.chat.completions.create.return_value = iter(mock_chunks)
        mock_openai.OpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai}):
            client = AzureOpenAIChatClient(endpoint="https://test.com", api_key="key")
            generator = client.invoke_stream(deployment="gpt-4", messages="test")
            chunks = list(generator)

        # Should only yield non-empty chunks
        assert chunks == ["Hello", "World"]

    def test_invoke_stream_real_mode_with_top_p(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test invoke_stream includes top_p when specified."""
        monkeypatch.setenv("AZURE_OPENAI_MOCK", "false")

        mock_openai = MagicMock()
        mock_client = MagicMock()

        # Create mock streaming chunks
        chunk = MagicMock()
        delta = MagicMock()
        delta.content = "text"
        choice = MagicMock()
        choice.delta = delta
        chunk.choices = [choice]

        mock_client.chat.completions.create.return_value = iter([chunk])
        mock_openai.OpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai}):
            client = AzureOpenAIChatClient(endpoint="https://test.com", api_key="key")
            generator = client.invoke_stream(
                deployment="gpt-4", messages="test", top_p=0.85
            )
            list(generator)

        # Verify top_p was passed
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["top_p"] == 0.85

    def test_invoke_stream_real_mode_with_extra_params(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test invoke_stream includes extra_params in payload."""
        monkeypatch.setenv("AZURE_OPENAI_MOCK", "false")

        mock_openai = MagicMock()
        mock_client = MagicMock()

        chunk = MagicMock()
        delta = MagicMock()
        delta.content = "text"
        choice = MagicMock()
        choice.delta = delta
        chunk.choices = [choice]

        mock_client.chat.completions.create.return_value = iter([chunk])
        mock_openai.OpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai}):
            client = AzureOpenAIChatClient(endpoint="https://test.com", api_key="key")
            generator = client.invoke_stream(
                deployment="gpt-4",
                messages="test",
                extra_params={"stop": ["STOP"], "seed": 42},
            )
            list(generator)

        # Verify extra params were passed
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["stop"] == ["STOP"]
        assert call_kwargs["seed"] == 42

    def test_invoke_stream_raises_import_error_when_openai_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test invoke_stream raises ImportError when openai is not installed."""
        monkeypatch.setenv("AZURE_OPENAI_MOCK", "false")

        with patch.dict("sys.modules", {"openai": None}):
            with patch(
                "builtins.__import__",
                side_effect=ImportError("No module named 'openai'"),
            ):
                client = AzureOpenAIChatClient()
                with pytest.raises(
                    ImportError,
                    match="Install 'openai' for Azure OpenAI streaming usage",
                ):
                    generator = client.invoke_stream(
                        deployment="gpt-4", messages="test"
                    )
                    list(generator)


class TestAzureOpenAIChatClientAInvokeStreamMockMode:
    """Tests for AzureOpenAIChatClient.ainvoke_stream in mock mode."""

    @pytest.mark.asyncio
    async def test_ainvoke_stream_mock_mode_yields_chunks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test ainvoke_stream in mock mode yields text chunks."""
        monkeypatch.setenv("AZURE_OPENAI_MOCK", "true")
        client = AzureOpenAIChatClient()

        chunks = []
        async for chunk in client.ainvoke_stream(
            deployment="gpt-4o", messages="Async stream test"
        ):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert "[MOCK_AZURE:gpt-4o]" in chunks[0]
        assert "Async stream test" in "".join(chunks)

    @pytest.mark.asyncio
    async def test_ainvoke_stream_mock_mode_with_empty_messages(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test ainvoke_stream in mock mode with empty messages."""
        monkeypatch.setenv("AZURE_OPENAI_MOCK", "true")
        client = AzureOpenAIChatClient()

        chunks = []
        async for chunk in client.ainvoke_stream(deployment="test", messages=[]):
            chunks.append(chunk)

        assert len(chunks) > 0


class TestAzureOpenAIChatClientAInvokeStreamRealMode:
    """Tests for AzureOpenAIChatClient.ainvoke_stream with mocked AsyncOpenAI client."""

    @pytest.mark.asyncio
    async def test_ainvoke_stream_real_mode_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test ainvoke_stream successfully streams from AsyncOpenAI client."""
        monkeypatch.setenv("AZURE_OPENAI_MOCK", "false")

        # Mock the openai module
        mock_openai = MagicMock()
        mock_client = MagicMock()

        # Create async iterator for streaming
        class AsyncChunkIterator:
            def __init__(self):
                self.chunks = [
                    self._create_chunk("Async"),
                    self._create_chunk(" stream"),
                    self._create_chunk("!"),
                ]
                self.index = 0

            def _create_chunk(self, text: str) -> MagicMock:
                chunk = MagicMock()
                delta = MagicMock()
                delta.content = text
                choice = MagicMock()
                choice.delta = delta
                chunk.choices = [choice]
                return chunk

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.chunks):
                    raise StopAsyncIteration
                chunk = self.chunks[self.index]
                self.index += 1
                return chunk

        mock_response = AsyncChunkIterator()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai.AsyncOpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai}):
            client = AzureOpenAIChatClient(
                endpoint="https://test.openai.azure.com", api_key="test-key"
            )
            chunks = []
            async for chunk in client.ainvoke_stream(
                deployment="gpt-4o", messages="Stream async"
            ):
                chunks.append(chunk)

        assert chunks == ["Async", " stream", "!"]

        # Verify stream=True was passed
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["stream"] is True

    @pytest.mark.asyncio
    async def test_ainvoke_stream_real_mode_with_top_p(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test ainvoke_stream includes top_p when specified."""
        monkeypatch.setenv("AZURE_OPENAI_MOCK", "false")

        mock_openai = MagicMock()
        mock_client = MagicMock()

        # Create async iterator
        class AsyncChunkIterator:
            def __init__(self):
                self.chunks = [self._create_chunk("test")]
                self.index = 0

            def _create_chunk(self, text: str) -> MagicMock:
                chunk = MagicMock()
                delta = MagicMock()
                delta.content = text
                choice = MagicMock()
                choice.delta = delta
                chunk.choices = [choice]
                return chunk

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.chunks):
                    raise StopAsyncIteration
                chunk = self.chunks[self.index]
                self.index += 1
                return chunk

        mock_response = AsyncChunkIterator()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai.AsyncOpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai}):
            client = AzureOpenAIChatClient(endpoint="https://test.com", api_key="key")
            chunks = []
            async for chunk in client.ainvoke_stream(
                deployment="gpt-4", messages="test", top_p=0.95
            ):
                chunks.append(chunk)

        # Verify top_p was passed
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["top_p"] == 0.95

    @pytest.mark.asyncio
    async def test_ainvoke_stream_real_mode_with_extra_params(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test ainvoke_stream includes extra_params in payload."""
        monkeypatch.setenv("AZURE_OPENAI_MOCK", "false")

        mock_openai = MagicMock()
        mock_client = MagicMock()

        # Create async iterator
        class AsyncChunkIterator:
            def __init__(self):
                self.chunks = [self._create_chunk("test")]
                self.index = 0

            def _create_chunk(self, text: str) -> MagicMock:
                chunk = MagicMock()
                delta = MagicMock()
                delta.content = text
                choice = MagicMock()
                choice.delta = delta
                chunk.choices = [choice]
                return chunk

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.chunks):
                    raise StopAsyncIteration
                chunk = self.chunks[self.index]
                self.index += 1
                return chunk

        mock_response = AsyncChunkIterator()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai.AsyncOpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai}):
            client = AzureOpenAIChatClient(endpoint="https://test.com", api_key="key")
            chunks = []
            async for chunk in client.ainvoke_stream(
                deployment="gpt-4",
                messages="test",
                extra_params={"logprobs": True, "top_logprobs": 5},
            ):
                chunks.append(chunk)

        # Verify extra params were passed
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["logprobs"] is True
        assert call_kwargs["top_logprobs"] == 5

    @pytest.mark.asyncio
    async def test_ainvoke_stream_raises_import_error_when_openai_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test ainvoke_stream raises ImportError when openai is not installed."""
        monkeypatch.setenv("AZURE_OPENAI_MOCK", "false")

        with patch.dict("sys.modules", {"openai": None}):
            with patch(
                "builtins.__import__",
                side_effect=ImportError("No module named 'openai'"),
            ):
                client = AzureOpenAIChatClient()
                with pytest.raises(
                    ImportError,
                    match="Install 'openai' for Azure OpenAI async streaming usage",
                ):
                    async for _ in client.ainvoke_stream(
                        deployment="gpt-4", messages="test"
                    ):
                        pass


class TestAzureOpenAIChatClientEndpointConstruction:
    """Tests for Azure endpoint URL construction."""

    def test_endpoint_construction_with_trailing_slash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test endpoint URL is correctly constructed when endpoint has trailing slash."""
        monkeypatch.setenv("AZURE_OPENAI_MOCK", "false")

        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "response"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_response.usage = None

        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai}):
            client = AzureOpenAIChatClient(
                endpoint="https://test.openai.azure.com/", api_key="key"
            )
            client.invoke(deployment="my-deployment", messages="test")

        call_kwargs = mock_openai.OpenAI.call_args[1]
        # Should not have double slash
        assert (
            call_kwargs["base_url"]
            == "https://test.openai.azure.com/openai/deployments/my-deployment"
        )

    def test_endpoint_construction_without_trailing_slash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test endpoint URL is correctly constructed when endpoint has no trailing slash."""
        monkeypatch.setenv("AZURE_OPENAI_MOCK", "false")

        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "response"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_response.usage = None

        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai}):
            client = AzureOpenAIChatClient(
                endpoint="https://test.openai.azure.com", api_key="key"
            )
            client.invoke(deployment="my-deployment", messages="test")

        call_kwargs = mock_openai.OpenAI.call_args[1]
        assert (
            call_kwargs["base_url"]
            == "https://test.openai.azure.com/openai/deployments/my-deployment"
        )
