import os

import pytest

from traigent.integrations.google_gemini_client import GeminiChatClient


@pytest.mark.unit
def test_gemini_invoke_stream_mock():
    os.environ["GEMINI_MOCK"] = "true"
    client = GeminiChatClient()

    messages = [{"role": "user", "content": "Hello"}]

    # Test simple iteration
    stream = client.invoke_stream(model="gemini-pro", messages=messages)
    chunks = list(stream)
    assert len(chunks) > 0
    assert "".join(chunks).startswith("[MOCK_GEMINI:gemini-pro]")

    # Test return value capture
    gen = client.invoke_stream(model="gemini-pro", messages=messages)
    collected_chunks = []
    response = None
    while True:
        try:
            chunk = next(gen)
            collected_chunks.append(chunk)
        except StopIteration as e:
            response = e.value
            break

    assert response is not None
    assert response.text == "".join(collected_chunks)
    assert response.raw["mock"] is True
