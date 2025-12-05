"""Unit tests for async cloud client methods.

These tests validate that when mock mode is enabled, the async client methods
return deterministic responses without requiring real API credentials.
"""

from __future__ import annotations

import pytest


# =============================================================================
# Bedrock Async Tests
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
async def test_bedrock_client_mock_ainvoke(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test async invocation returns valid response in mock mode."""
    monkeypatch.setenv("BEDROCK_MOCK", "true")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)

    from traigent.integrations.bedrock_client import BedrockChatClient

    client = BedrockChatClient(region_name=None, profile_name=None)
    resp = await client.ainvoke(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        messages="hello async",
        max_tokens=16,
    )

    assert resp.text.startswith("[MOCK:anthropic.claude-3-sonnet-20240229-v1:0]")
    assert "hello async" in resp.text
    assert resp.raw.get("mock") is True


@pytest.mark.asyncio
@pytest.mark.unit
async def test_bedrock_client_mock_ainvoke_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test async streaming returns chunks in mock mode."""
    monkeypatch.setenv("BEDROCK_MOCK", "true")
    from traigent.integrations.bedrock_client import BedrockChatClient

    client = BedrockChatClient()
    chunks = []
    async for chunk in client.ainvoke_stream(
        model_id="anthropic.claude-3-opus-20240229-v1:0",
        messages="world async",
    ):
        chunks.append(chunk)

    assert len(chunks) > 0
    full_text = "".join(chunks)
    assert "[MOCK:anthropic.claude-3-opus-20240229-v1:0]" in full_text


# =============================================================================
# Azure OpenAI Async Tests
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
async def test_azure_client_mock_ainvoke(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Azure async invocation returns valid response in mock mode."""
    monkeypatch.setenv("AZURE_OPENAI_MOCK", "true")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")

    from traigent.integrations.azure_openai_client import AzureOpenAIChatClient

    client = AzureOpenAIChatClient()
    resp = await client.ainvoke(
        deployment="gpt-4",
        messages="hello azure async",
        max_tokens=32,
    )

    assert "[MOCK_AZURE:gpt-4]" in resp.text
    assert "hello azure async" in resp.text
    assert resp.raw.get("mock") is True


@pytest.mark.asyncio
@pytest.mark.unit
async def test_azure_client_mock_ainvoke_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test Azure async streaming returns chunks in mock mode."""
    monkeypatch.setenv("AZURE_OPENAI_MOCK", "true")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")

    from traigent.integrations.azure_openai_client import AzureOpenAIChatClient

    client = AzureOpenAIChatClient()
    chunks = []
    async for chunk in client.ainvoke_stream(
        deployment="gpt-4",
        messages="world azure async",
    ):
        chunks.append(chunk)

    assert len(chunks) > 0
    full_text = "".join(chunks)
    assert "[MOCK_AZURE:gpt-4]" in full_text


@pytest.mark.unit
def test_azure_client_mock_invoke_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Azure sync streaming returns chunks in mock mode."""
    monkeypatch.setenv("AZURE_OPENAI_MOCK", "true")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")

    from traigent.integrations.azure_openai_client import AzureOpenAIChatClient

    client = AzureOpenAIChatClient()
    chunks = list(
        client.invoke_stream(
            deployment="gpt-4",
            messages="sync stream test",
        )
    )

    assert len(chunks) > 0
    full_text = "".join(chunks)
    assert "[MOCK_AZURE:gpt-4]" in full_text


# =============================================================================
# Gemini Async Tests
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
async def test_gemini_client_mock_ainvoke(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Gemini async invocation returns valid response in mock mode."""
    monkeypatch.setenv("GEMINI_MOCK", "true")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    from traigent.integrations.google_gemini_client import GeminiChatClient

    client = GeminiChatClient()
    resp = await client.ainvoke(
        model="gemini-pro",
        messages="hello gemini async",
        max_tokens=32,
    )

    assert "[MOCK_GEMINI:gemini-pro]" in resp.text
    assert "hello gemini async" in resp.text
    assert resp.raw.get("mock") is True


@pytest.mark.asyncio
@pytest.mark.unit
async def test_gemini_client_mock_ainvoke_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test Gemini async streaming returns chunks in mock mode."""
    monkeypatch.setenv("GEMINI_MOCK", "true")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    from traigent.integrations.google_gemini_client import GeminiChatClient

    client = GeminiChatClient()
    chunks = []
    async for chunk in client.ainvoke_stream(
        model="gemini-pro",
        messages="world gemini async",
    ):
        chunks.append(chunk)

    assert len(chunks) > 0
    full_text = "".join(chunks)
    assert "[MOCK_GEMINI:gemini-pro]" in full_text


# =============================================================================
# Concurrent Async Tests
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
async def test_concurrent_async_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test multiple async calls can run concurrently."""
    import asyncio

    monkeypatch.setenv("BEDROCK_MOCK", "true")
    monkeypatch.setenv("AZURE_OPENAI_MOCK", "true")
    monkeypatch.setenv("GEMINI_MOCK", "true")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    from traigent.integrations.bedrock_client import BedrockChatClient
    from traigent.integrations.azure_openai_client import AzureOpenAIChatClient
    from traigent.integrations.google_gemini_client import GeminiChatClient

    bedrock = BedrockChatClient()
    azure = AzureOpenAIChatClient()
    gemini = GeminiChatClient()

    results = await asyncio.gather(
        bedrock.ainvoke(model_id="anthropic.claude-3-sonnet", messages="test1"),
        azure.ainvoke(deployment="gpt-4", messages="test2"),
        gemini.ainvoke(model="gemini-pro", messages="test3"),
    )

    assert len(results) == 3
    assert all(r is not None for r in results)
    assert "[MOCK:" in results[0].text
    assert "[MOCK_AZURE:" in results[1].text
    assert "[MOCK_GEMINI:" in results[2].text
