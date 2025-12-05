"""Unit tests for the Bedrock client mock mode.

These tests validate that when `BEDROCK_MOCK=true`, the client returns
deterministic responses without requiring `boto3` or AWS credentials.
"""

from __future__ import annotations

import pytest


@pytest.mark.unit
def test_bedrock_client_mock_invoke(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BEDROCK_MOCK", "true")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)

    from traigent.integrations.bedrock_client import BedrockChatClient

    client = BedrockChatClient(region_name=None, profile_name=None)
    resp = client.invoke(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        messages="hello",
        max_tokens=16,
    )

    assert resp.text.startswith("[MOCK:anthropic.claude-3-sonnet-20240229-v1:0]")
    assert resp.raw.get("mock") is True


@pytest.mark.unit
def test_bedrock_client_mock_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BEDROCK_MOCK", "true")
    from traigent.integrations.bedrock_client import BedrockChatClient

    client = BedrockChatClient()
    gen = client.invoke_stream(
        model_id="anthropic.claude-3-opus-20240229-v1:0", messages="world"
    )
    chunks = list(gen)
    assert any("[MOCK:anthropic.claude-3-opus-20240229-v1:0]" in c for c in chunks)
