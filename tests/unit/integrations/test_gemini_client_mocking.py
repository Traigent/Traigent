import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from traigent.integrations.google_gemini_client import GeminiChatClient


@pytest.fixture
def mock_genai_module():
    """Fixture that creates a mock google.generativeai module.

    Returns the mock module so tests can configure it as needed.
    """
    mock_genai = MagicMock()
    mock_model_instance = MagicMock()
    mock_genai.GenerativeModel.return_value = mock_model_instance

    # Store a reference to the model instance for test configuration
    mock_genai._mock_model = mock_model_instance

    # Save original modules to restore later
    saved_modules = {}
    modules_to_patch = ["google", "google.ai", "google.generativeai"]
    for mod in modules_to_patch:
        if mod in sys.modules:
            saved_modules[mod] = sys.modules[mod]

    # Create a mock "google" module that returns our mock_genai when .generativeai is accessed
    mock_google = MagicMock()
    mock_google.generativeai = mock_genai

    # Replace with our mocks - the key is that google.generativeai points to same mock
    sys.modules["google"] = mock_google
    sys.modules["google.ai"] = MagicMock()
    sys.modules["google.generativeai"] = mock_genai

    try:
        yield mock_genai
    finally:
        # Restore original modules
        for mod in modules_to_patch:
            if mod in saved_modules:
                sys.modules[mod] = saved_modules[mod]
            elif mod in sys.modules:
                del sys.modules[mod]


@pytest.mark.unit
def test_gemini_client_init_no_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    client = GeminiChatClient()
    assert client._api_key == ""


@pytest.mark.unit
def test_gemini_client_init_with_env_var(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "env_key")
    client = GeminiChatClient()
    assert client._api_key == "env_key"


@pytest.mark.unit
def test_gemini_invoke_real_path(mock_genai_module, monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "false")

    # Setup mock response
    mock_response = MagicMock()
    mock_response.text = "Real response"
    mock_response.usage_metadata.to_dict.return_value = {"tokens": 10}
    mock_genai_module._mock_model.generate_content.return_value = mock_response

    client = GeminiChatClient(api_key="test_key")
    response = client.invoke(model="gemini-pro", messages="Hello")

    assert response.text == "Real response"
    assert response.usage == {"tokens": 10}
    mock_genai_module.configure.assert_called_with(api_key="test_key")
    mock_genai_module._mock_model.generate_content.assert_called_once()

    # Verify generation_config is used correctly
    call_args = mock_genai_module._mock_model.generate_content.call_args
    assert call_args[0][0] == "Hello"  # prompt
    assert "generation_config" in call_args[1]
    assert call_args[1]["generation_config"]["temperature"] == 0.5


@pytest.mark.unit
def test_gemini_invoke_stream_real_path(mock_genai_module, monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "false")

    # Setup mock chunks
    mock_chunk1 = MagicMock()
    mock_chunk1.text = "Chunk 1"
    mock_chunk2 = MagicMock()
    mock_chunk2.text = "Chunk 2"

    # Mock response as an iterable
    mock_response = MagicMock()
    mock_response.__iter__ = MagicMock(return_value=iter([mock_chunk1, mock_chunk2]))
    mock_response.usage_metadata.to_dict.return_value = {"tokens": 20}

    mock_genai_module._mock_model.generate_content.return_value = mock_response

    client = GeminiChatClient(api_key="test_key")
    chunks = list(client.invoke_stream(model="gemini-pro", messages="Hello"))

    assert chunks == ["Chunk 1", "Chunk 2"]
    mock_genai_module._mock_model.generate_content.assert_called_once()

    # Verify generation_config and stream=True
    call_args = mock_genai_module._mock_model.generate_content.call_args
    assert call_args[1]["stream"] is True
    assert "generation_config" in call_args[1]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_gemini_ainvoke_real_path(mock_genai_module, monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "false")

    # Setup async mock response
    mock_response = MagicMock()
    mock_response.text = "Async response"
    mock_response.usage_metadata.to_dict.return_value = {"tokens": 15}

    # Make generate_content_async return an awaitable
    mock_genai_module._mock_model.generate_content_async = AsyncMock(
        return_value=mock_response
    )

    client = GeminiChatClient(api_key="test_key")
    response = await client.ainvoke(model="gemini-pro", messages="Hello")

    assert response.text == "Async response"
    mock_genai_module._mock_model.generate_content_async.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_gemini_ainvoke_stream_real_path(mock_genai_module, monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "false")

    # Setup mock chunk
    mock_chunk = MagicMock()
    mock_chunk.text = "Async Chunk"

    # Create async iterator for streaming response
    class AsyncChunkIterator:
        def __init__(self):
            self.chunks = [mock_chunk]
            self.index = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.index >= len(self.chunks):
                raise StopAsyncIteration
            chunk = self.chunks[self.index]
            self.index += 1
            return chunk

    mock_response = AsyncChunkIterator()

    # Make generate_content_async return an awaitable that yields our iterator
    mock_genai_module._mock_model.generate_content_async = AsyncMock(
        return_value=mock_response
    )

    client = GeminiChatClient(api_key="test_key")
    chunks = []
    async for chunk in client.ainvoke_stream(model="gemini-pro", messages="Hello"):
        chunks.append(chunk)

    assert chunks == ["Async Chunk"]


@pytest.mark.unit
def test_gemini_import_error(monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "false")
    with patch.dict(sys.modules):
        if "google.generativeai" in sys.modules:
            del sys.modules["google.generativeai"]
        # Ensure import fails
        # We need to patch builtins.__import__ but only for google.generativeai
        original_import = __import__

        def mock_import(name, *args, **kwargs):
            if name == "google.generativeai":
                raise ImportError("No module named google.generativeai")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            client = GeminiChatClient()
            with pytest.raises(ImportError, match="Install 'google-generativeai'"):
                client.invoke(model="gemini-pro", messages="Hello")


@pytest.mark.unit
def test_gemini_coerce_messages_complex():
    from traigent.integrations.google_gemini_client import _coerce_messages

    # Test list of dicts with 'parts'
    messages = [
        {"role": "user", "content": [{"text": "Hello"}, {"text": "World"}]},
        {"role": "model", "content": "Hi there"},
    ]
    texts = _coerce_messages(messages)
    assert texts == ["Hello", "World", "Hi there"]


@pytest.mark.unit
def test_gemini_invoke_extra_params(mock_genai_module, monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "false")

    mock_response = MagicMock()
    mock_response.text = "Response"
    mock_genai_module._mock_model.generate_content.return_value = mock_response

    client = GeminiChatClient(api_key="key")
    client.invoke(model="gemini-pro", messages="Hi", extra_params={"top_k": 1})

    mock_genai_module._mock_model.generate_content.assert_called()
    call_kwargs = mock_genai_module._mock_model.generate_content.call_args[1]
    # extra_params are now merged into generation_config
    assert call_kwargs["generation_config"]["top_k"] == 1


@pytest.mark.unit
def test_gemini_invoke_error_handling(mock_genai_module, monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "false")
    from unittest.mock import PropertyMock

    mock_response = MagicMock()

    # Simulate text property error
    type(mock_response).text = PropertyMock(side_effect=Exception("No text"))

    # Simulate usage.to_dict error
    mock_usage = MagicMock()
    mock_usage.to_dict.side_effect = Exception("No usage dict")
    type(mock_response).usage_metadata = PropertyMock(return_value=mock_usage)

    mock_genai_module._mock_model.generate_content.return_value = mock_response

    client = GeminiChatClient(api_key="key")
    response = client.invoke(model="gemini-pro", messages="Hi")

    assert response.text == ""
    assert response.usage is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_gemini_ainvoke_error_handling(mock_genai_module, monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "false")
    from unittest.mock import PropertyMock

    mock_response = MagicMock()

    # Simulate text property error
    type(mock_response).text = PropertyMock(side_effect=Exception("No text"))

    # Simulate usage.to_dict error
    mock_usage = MagicMock()
    mock_usage.to_dict.side_effect = Exception("No usage dict")
    type(mock_response).usage_metadata = PropertyMock(return_value=mock_usage)

    mock_genai_module._mock_model.generate_content_async = AsyncMock(
        return_value=mock_response
    )

    client = GeminiChatClient(api_key="key")
    response = await client.ainvoke(model="gemini-pro", messages="Hi")

    assert response.text == ""
    assert response.usage is None
