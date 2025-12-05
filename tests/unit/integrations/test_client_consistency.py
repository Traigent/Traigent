"""Tests for consistency across all client implementations.

This module ensures that Gemini, Bedrock, and Azure OpenAI clients
have consistent APIs and behavior patterns.
"""

import pytest


@pytest.mark.unit
def test_gemini_top_p_parameter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Gemini client supports top_p parameter."""
    monkeypatch.setenv("GEMINI_MOCK", "true")
    from traigent.integrations.google_gemini_client import GeminiChatClient

    client = GeminiChatClient()

    # Test invoke with top_p
    resp = client.invoke(
        model="gemini-pro",
        messages="test message",
        max_tokens=50,
        temperature=0.7,
        top_p=0.9,
    )
    assert resp.text.startswith("[MOCK_GEMINI:gemini-pro]")
    assert resp.raw["mock"] is True


@pytest.mark.unit
def test_gemini_streaming_top_p_parameter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Gemini streaming supports top_p parameter."""
    monkeypatch.setenv("GEMINI_MOCK", "true")
    from traigent.integrations.google_gemini_client import GeminiChatClient

    client = GeminiChatClient()

    # Test invoke_stream with top_p and capture return value
    gen = client.invoke_stream(
        model="gemini-pro",
        messages="test streaming",
        max_tokens=50,
        temperature=0.7,
        top_p=0.9,
    )

    chunks = []
    response = None
    while True:
        try:
            chunk = next(gen)
            chunks.append(chunk)
        except StopIteration as e:
            response = e.value
            break

    assert response is not None
    assert response.text == "".join(chunks)
    assert response.raw["mock"] is True


@pytest.mark.unit
async def test_gemini_async_top_p_parameter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Gemini async methods support top_p parameter."""
    monkeypatch.setenv("GEMINI_MOCK", "true")
    from traigent.integrations.google_gemini_client import GeminiChatClient

    client = GeminiChatClient()

    # Test ainvoke with top_p
    resp = await client.ainvoke(
        model="gemini-pro",
        messages="test async",
        max_tokens=50,
        temperature=0.7,
        top_p=0.9,
    )
    assert resp.text.startswith("[MOCK_GEMINI:gemini-pro]")
    assert resp.raw["mock"] is True

    # Test ainvoke_stream with top_p
    chunks = []
    async for chunk in client.ainvoke_stream(
        model="gemini-pro",
        messages="test async streaming",
        max_tokens=50,
        temperature=0.7,
        top_p=0.9,
    ):
        chunks.append(chunk)

    assert len(chunks) > 0
    assert "".join(chunks).startswith("[MOCK_GEMINI:gemini-pro]")


@pytest.mark.unit
def test_azure_openai_top_p_parameter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Azure OpenAI client supports top_p parameter."""
    monkeypatch.setenv("AZURE_OPENAI_MOCK", "true")
    from traigent.integrations.azure_openai_client import AzureOpenAIChatClient

    client = AzureOpenAIChatClient()

    # Test invoke with top_p
    resp = client.invoke(
        deployment="gpt-4",
        messages="test message",
        max_tokens=50,
        temperature=0.7,
        top_p=0.9,
    )
    assert resp.text.startswith("[MOCK_AZURE:gpt-4]")
    assert resp.raw["mock"] is True


@pytest.mark.unit
def test_azure_openai_streaming_return_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Azure OpenAI streaming returns a response object like Gemini and Bedrock."""
    monkeypatch.setenv("AZURE_OPENAI_MOCK", "true")
    from traigent.integrations.azure_openai_client import AzureOpenAIChatClient

    client = AzureOpenAIChatClient()

    # Test invoke_stream returns response object
    gen = client.invoke_stream(
        deployment="gpt-4",
        messages="test streaming",
        max_tokens=50,
        temperature=0.7,
        top_p=0.9,
    )

    chunks = []
    response = None
    while True:
        try:
            chunk = next(gen)
            chunks.append(chunk)
        except StopIteration as e:
            response = e.value
            break

    # Verify response object is returned (like Gemini and Bedrock)
    assert response is not None
    assert response.text == "".join(chunks)
    assert response.raw.get("mock") is True or response.raw.get("streamed") is True


@pytest.mark.unit
async def test_azure_openai_async_top_p_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that Azure OpenAI async methods support top_p parameter."""
    monkeypatch.setenv("AZURE_OPENAI_MOCK", "true")
    from traigent.integrations.azure_openai_client import AzureOpenAIChatClient

    client = AzureOpenAIChatClient()

    # Test ainvoke with top_p
    resp = await client.ainvoke(
        deployment="gpt-4",
        messages="test async",
        max_tokens=50,
        temperature=0.7,
        top_p=0.9,
    )
    assert resp.text.startswith("[MOCK_AZURE:gpt-4]")
    assert resp.raw["mock"] is True

    # Test ainvoke_stream with top_p
    chunks = []
    async for chunk in client.ainvoke_stream(
        deployment="gpt-4",
        messages="test async streaming",
        max_tokens=50,
        temperature=0.7,
        top_p=0.9,
    ):
        chunks.append(chunk)

    assert len(chunks) > 0
    assert "".join(chunks).startswith("[MOCK_AZURE:gpt-4]")


@pytest.mark.unit
def test_bedrock_streaming_return_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Bedrock streaming returns response object (verify existing behavior)."""
    monkeypatch.setenv("BEDROCK_MOCK", "true")
    from traigent.integrations.bedrock_client import BedrockChatClient

    client = BedrockChatClient()

    # Test invoke_stream returns response object
    gen = client.invoke_stream(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        messages="test streaming",
        max_tokens=50,
        temperature=0.7,
        top_p=0.9,
    )

    chunks = []
    response = None
    while True:
        try:
            chunk = next(gen)
            chunks.append(chunk)
        except StopIteration as e:
            response = e.value
            break

    # Verify response object is returned
    assert response is not None
    assert response.text == "".join(chunks)
