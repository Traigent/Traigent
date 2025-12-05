"""Unit tests for the Azure OpenAI client mock mode."""

from __future__ import annotations

import pytest


@pytest.mark.unit
def test_azure_client_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_MOCK", "true")
    from traigent.integrations.azure_openai_client import AzureOpenAIChatClient

    client = AzureOpenAIChatClient()
    resp = client.invoke(deployment="gpt-4o-mini", messages="hi", max_tokens=16)
    assert resp.text.startswith("[MOCK_AZURE:gpt-4o-mini]")
