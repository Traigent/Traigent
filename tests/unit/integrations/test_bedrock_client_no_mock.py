"""Regression tests proving BEDROCK_MOCK is ignored by production code."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from traigent.integrations.bedrock_client import BedrockChatClient


def test_bedrock_mock_env_does_not_short_circuit_invoke(monkeypatch) -> None:
    """Setting BEDROCK_MOCK still uses the boto3 Bedrock Runtime client."""
    monkeypatch.setenv("BEDROCK_MOCK", "true")

    mock_boto3 = MagicMock()
    mock_session = MagicMock()
    mock_client = MagicMock()
    mock_client.invoke_model.return_value = {
        "body": MagicMock(
            read=lambda: json.dumps(
                {"content": [{"type": "text", "text": "from mocked boto3"}]}
            ).encode()
        )
    }
    mock_session.client.return_value = mock_client
    mock_boto3.session.Session.return_value = mock_session

    with patch(
        "traigent.integrations.bedrock_client._require_boto3",
        return_value=mock_boto3,
    ) as mock_require_boto3:
        response = BedrockChatClient(region_name="us-east-1").invoke(
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            messages="hello",
        )

    mock_require_boto3.assert_called_once()
    mock_session.client.assert_called_once_with(
        "bedrock-runtime", region_name="us-east-1"
    )
    mock_client.invoke_model.assert_called_once()
    assert response.text == "from mocked boto3"
    assert "[MOCK:" not in response.text
    assert response.raw.get("mock") is None
