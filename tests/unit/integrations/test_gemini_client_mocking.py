import sys
from unittest.mock import MagicMock, patch

import pytest

from traigent.integrations.google_gemini_client import GeminiChatClient


@pytest.fixture
def mock_genai():
    with patch.dict(sys.modules, {"google.generativeai": MagicMock()}):
        mock = sys.modules["google.generativeai"]
        yield mock


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
def test_gemini_invoke_real_path(mock_genai, monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "false")

    # Setup mock
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Real response"
    mock_response.usage_metadata.to_dict.return_value = {"tokens": 10}
    mock_model.generate_content.return_value = mock_response
    mock_genai.GenerativeModel.return_value = mock_model

    client = GeminiChatClient(api_key="test_key")
    response = client.invoke(model="gemini-pro", messages="Hello")

    assert response.text == "Real response"
    assert response.usage == {"tokens": 10}
    mock_genai.configure.assert_called_with(api_key="test_key")
    mock_model.generate_content.assert_called()


@pytest.mark.unit
def test_gemini_invoke_stream_real_path(mock_genai, monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "false")

    # Setup mock
    mock_model = MagicMock()
    mock_chunk1 = MagicMock()
    mock_chunk1.text = "Chunk 1"
    mock_chunk2 = MagicMock()
    mock_chunk2.text = "Chunk 2"

    # Mock iterator
    mock_response = MagicMock()
    mock_response.__iter__.return_value = [mock_chunk1, mock_chunk2]
    mock_response.usage_metadata.to_dict.return_value = {"tokens": 20}

    mock_model.generate_content.return_value = mock_response
    mock_genai.GenerativeModel.return_value = mock_model

    client = GeminiChatClient(api_key="test_key")
    chunks = list(client.invoke_stream(model="gemini-pro", messages="Hello"))

    assert chunks == ["Chunk 1", "Chunk 2"]
    mock_model.generate_content.assert_called_with(
        "Hello", temperature=0.5, stream=True
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_gemini_ainvoke_real_path(mock_genai, monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "false")

    # Setup mock
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Async response"
    mock_response.usage_metadata.to_dict.return_value = {"tokens": 15}

    # Async mock
    async def async_generate(*args, **kwargs):
        return mock_response

    mock_model.generate_content_async = async_generate
    mock_genai.GenerativeModel.return_value = mock_model

    client = GeminiChatClient(api_key="test_key")
    response = await client.ainvoke(model="gemini-pro", messages="Hello")

    assert response.text == "Async response"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_gemini_ainvoke_stream_real_path(mock_genai, monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "false")

    # Setup mock
    mock_model = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.text = "Async Chunk"

    # Async iterator mock
    class AsyncIter:
        def __aiter__(self):
            return self

        async def __anext__(self):
            if hasattr(self, "done"):
                raise StopAsyncIteration
            self.done = True
            return mock_chunk

    mock_response = AsyncIter()

    async def async_generate(*args, **kwargs):
        return mock_response

    mock_model.generate_content_async = async_generate
    mock_genai.GenerativeModel.return_value = mock_model

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
def test_gemini_invoke_extra_params(mock_genai, monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "false")

    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Response"
    mock_model.generate_content.return_value = mock_response
    mock_genai.GenerativeModel.return_value = mock_model

    client = GeminiChatClient(api_key="key")
    client.invoke(model="gemini-pro", messages="Hi", extra_params={"top_k": 1})

    mock_model.generate_content.assert_called()
    call_kwargs = mock_model.generate_content.call_args[1]
    assert call_kwargs["top_k"] == 1


@pytest.mark.unit
def test_gemini_invoke_error_handling(mock_genai, monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "false")
    from unittest.mock import PropertyMock

    mock_model = MagicMock()
    mock_response = MagicMock()

    # Simulate text property error
    type(mock_response).text = PropertyMock(side_effect=Exception("No text"))

    # Simulate usage.to_dict error
    mock_usage = MagicMock()
    mock_usage.to_dict.side_effect = Exception("No usage dict")
    type(mock_response).usage_metadata = PropertyMock(return_value=mock_usage)

    mock_model.generate_content.return_value = mock_response
    mock_genai.GenerativeModel.return_value = mock_model

    client = GeminiChatClient(api_key="key")
    response = client.invoke(model="gemini-pro", messages="Hi")

    assert response.text == ""
    assert response.usage is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_gemini_ainvoke_error_handling(mock_genai, monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "false")
    from unittest.mock import PropertyMock

    mock_model = MagicMock()
    mock_response = MagicMock()

    # Simulate text property error
    type(mock_response).text = PropertyMock(side_effect=Exception("No text"))

    # Simulate usage.to_dict error
    mock_usage = MagicMock()
    mock_usage.to_dict.side_effect = Exception("No usage dict")
    type(mock_response).usage_metadata = PropertyMock(return_value=mock_usage)

    async def async_generate(*args, **kwargs):
        return mock_response

    mock_model.generate_content_async = async_generate
    mock_genai.GenerativeModel.return_value = mock_model

    client = GeminiChatClient(api_key="key")
    response = await client.ainvoke(model="gemini-pro", messages="Hi")

    assert response.text == ""
    assert response.usage is None
