"""Unit tests for the Gemini client mock mode."""

from __future__ import annotations

import pytest


@pytest.mark.unit
def test_gemini_client_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_MOCK", "true")
    from traigent.integrations.google_gemini_client import GeminiChatClient

    client = GeminiChatClient()
    resp = client.invoke(model="gemini-1.5-pro", messages="ping", max_tokens=8)
    assert resp.text.startswith("[MOCK_GEMINI:gemini-1.5-pro]")
