"""Regression tests for native Bedrock mock interception."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from traigent.integrations.bedrock_client import BedrockChatClient
from traigent.utils.langchain_interceptor import (
    clear_captured_responses,
    get_all_captured_responses,
)


@pytest.fixture(autouse=True)
def _reset_mock_mode(monkeypatch: pytest.MonkeyPatch):
    from traigent.testing import _reset_for_tests

    _reset_for_tests()
    monkeypatch.delenv("TRAIGENT_MOCK_LLM", raising=False)
    clear_captured_responses()
    yield
    _reset_for_tests()
    clear_captured_responses()


def test_traigent_mock_llm_short_circuits_bedrock_invoke(monkeypatch) -> None:
    """TRAIGENT_MOCK_LLM returns a Bedrock-shaped mock without boto3/AWS."""
    clear_captured_responses()
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")

    with patch("traigent.integrations.bedrock_client._require_boto3") as mock_require:
        response = BedrockChatClient(region_name="us-east-1").invoke(
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            messages="hello",
        )

    mock_require.assert_not_called()
    assert response.text == "This is a mock response for testing."
    assert response.raw["mock"] is True
    assert response.raw["provider"] == "bedrock"
    assert response.usage is not None
    assert response.usage["prompt_tokens"] == 10
    assert response.usage["completion_tokens"] == 20

    captured = get_all_captured_responses()
    assert captured == [response]
    clear_captured_responses()
