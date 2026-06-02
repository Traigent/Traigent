"""Tests for learned TVAR priors client support."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from traigent.cloud.client import PriorsBundle, TraigentCloudClient


class _Response:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self.status = status
        self._payload = payload
        self.headers: dict[str, str] = {}

    async def json(self) -> dict:
        return self._payload

    async def text(self) -> str:
        return "error"


class _ResponseContext:
    def __init__(self, response: _Response) -> None:
        self._response = response

    async def __aenter__(self) -> _Response:
        return self._response

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


@pytest.mark.asyncio
async def test_fetch_tvar_priors_parses_contract(monkeypatch) -> None:
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    monkeypatch.delenv("TRAIGENT_EDGE_ANALYTICS_MODE", raising=False)
    monkeypatch.delenv("TRAIGENT_EXECUTION_MODE", raising=False)

    payload = {
        "value_priors": [
            {
                "schema_version": "1.0.0",
                "tvar_name": "temperature",
                "metric": "quality_score",
                "value_priors": [
                    {
                        "value": 0.2,
                        "score": 0.87,
                        "support_n": 96,
                        "confidence": 0.82,
                    }
                ],
                "support_n": 96,
                "confidence": 0.82,
            }
        ],
        "correlations": [
            {
                "schema_version": "1.0.0",
                "kind": "agent_type_tvar",
                "a": {"type": "agent_type", "name": "rag"},
                "b": {"type": "tvar", "name": "temperature", "value": 0.2},
                "metric": "quality_score",
                "strength": 0.41,
                "support_n": 96,
                "confidence": 0.73,
            }
        ],
        "generated_at": "2026-06-02T12:00:00Z",
    }
    response = _Response(payload)
    session = Mock()
    session.closed = False
    session.get = Mock(return_value=_ResponseContext(response))

    client = TraigentCloudClient(
        api_key="test-key",  # pragma: allowlist secret
        base_url="https://backend.example.com",
    )
    client._aio_session = session
    client.auth.get_headers = AsyncMock(return_value={"X-API-Key": "test-key"})

    bundle = await client.fetch_tvar_priors(
        agent_type="rag",
        metric="quality_score",
        tvar_names=["temperature", "top_p"],
    )

    assert bundle == PriorsBundle.from_payload(payload)
    session.get.assert_called_once()
    assert session.get.call_args.args[0] == (
        "https://backend.example.com/api/v1/optimization/priors"
    )
    assert session.get.call_args.kwargs["params"] == {
        "agent_type": "rag",
        "metric": "quality_score",
        "tvar_names": "temperature,top_p",
    }


@pytest.mark.asyncio
async def test_fetch_tvar_priors_offline_returns_empty_without_network(
    monkeypatch,
) -> None:
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")

    session = Mock()
    session.closed = False
    session.get = Mock(side_effect=AssertionError("network must not be used"))

    client = TraigentCloudClient(
        api_key="test-key",  # pragma: allowlist secret
        base_url="https://backend.example.com",
    )
    client._aio_session = session
    client.auth.get_headers = AsyncMock(
        side_effect=AssertionError("auth headers must not be used")
    )

    bundle = await client.fetch_tvar_priors(agent_type="rag")

    assert bundle == PriorsBundle.empty()
    session.get.assert_not_called()
    client.auth.get_headers.assert_not_awaited()
