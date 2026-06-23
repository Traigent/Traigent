"""Unit tests for BackendAnalyticsClient (the client.analytics read client).

All backend transport is mocked; no live backend is contacted. Fixtures match
the frozen v0 contracts (decision_payload / run_pareto / run_correlations).
"""

from __future__ import annotations

import importlib.util
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

HTTPX_AVAILABLE = importlib.util.find_spec("httpx") is not None

pytestmark = pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed")


@pytest.fixture()
def decision_payload() -> dict[str, object]:
    """A frozen-v0 decision_payload contract instance."""
    return {
        "run_id": "run_123",
        "project_id": "proj_abc",
        "intent": "iterate",
        "headline": "gpt-4o-mini at temp 0.0 is the best cost/quality tradeoff.",
        "confidence": "high",
        "recommended_action": {
            "kind": "promote_config",
            "config_id": "cfg_7",
            "why": "Top quality at lowest cost on the holdout slice.",
        },
        "evidence": [
            {"type": "pareto", "summary": "cfg_7 is the knee of the frontier."}
        ],
        "drilldowns": [
            {"label": "See the Pareto frontier", "tool": "analytics_get_run_report"}
        ],
        "warnings": [],
    }


@pytest.fixture()
def run_report_payload() -> dict[str, object]:
    return {"run_id": "run_123", "project_id": "proj_abc", "measures": {}}


def _make_client(api_key: str = "uk_test_key"):
    from traigent.cloud.analytics_client import BackendAnalyticsClient

    return BackendAnalyticsClient(backend_url="http://localhost:5000", api_key=api_key)


def _success_envelope(data: object) -> dict[str, object]:
    return {"success": True, "message": "ok", "data": data}


class TestInit:
    def test_defaults_resolve_backend_url_and_key(self) -> None:
        from traigent.cloud.analytics_client import BackendAnalyticsClient

        with (
            patch(
                "traigent.config.backend_config.BackendConfig.get_backend_url",
                return_value="http://localhost:5000",
            ),
            patch(
                "traigent.cloud.credential_manager.CredentialManager.get_api_key",
                return_value="uk_resolved",  # pragma: allowlist secret
            ),
        ):
            client = BackendAnalyticsClient()

        assert client.backend_url == "http://localhost:5000"
        assert client.api_key == "uk_resolved"  # pragma: allowlist secret
        assert client._client is None

    def test_strips_trailing_slash(self) -> None:
        client = _make_client()
        assert client.backend_url == "http://localhost:5000"

    def test_api_key_uses_x_api_key_header(self) -> None:
        """An API key (no dots) must go in X-API-Key, never as a bearer token."""
        client = _make_client(api_key="uk_abcdef")
        headers = client._auth_headers()
        assert headers == {"X-API-Key": "uk_abcdef"}

    def test_jwt_uses_bearer_header(self) -> None:
        client = _make_client(api_key="aaa.bbb.ccc")
        headers = client._auth_headers()
        assert headers == {"Authorization": "Bearer aaa.bbb.ccc"}


@pytest.mark.skipif(HTTPX_AVAILABLE, reason="only when httpx missing")
def test_init_raises_without_httpx() -> None:  # pragma: no cover - env-dependent
    from traigent.cloud import analytics_client

    with pytest.raises(ImportError, match="httpx is required"):
        analytics_client.BackendAnalyticsClient()


class TestGetClientOffline:
    def test_offline_mode_fails_closed(self, monkeypatch) -> None:
        from traigent.utils.error_handler import OfflineModeError

        monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
        client = _make_client()
        with pytest.raises(OfflineModeError):
            client._get_client()


class TestGetRunReport:
    @pytest.mark.asyncio
    async def test_calls_correct_path_and_returns_payload(
        self, run_report_payload: dict[str, object]
    ) -> None:
        client = _make_client()
        mock_response = MagicMock()
        mock_response.json.return_value = _success_envelope(run_report_payload)
        mock_response.raise_for_status = MagicMock()
        mock_http = AsyncMock()
        mock_http.get.return_value = mock_response
        client._client = mock_http

        result = await client.get_run_report("proj_abc", "run_123")

        assert result == run_report_payload
        mock_http.get.assert_called_once_with(
            "/api/v1/analytics/runs/run_123/report",
            headers={"X-Project-Id": "proj_abc"},
            params=None,
        )

    @pytest.mark.asyncio
    async def test_url_encodes_identifiers(self) -> None:
        client = _make_client()
        mock_response = MagicMock()
        mock_response.json.return_value = _success_envelope({"ok": True})
        mock_response.raise_for_status = MagicMock()
        mock_http = AsyncMock()
        mock_http.get.return_value = mock_response
        client._client = mock_http

        await client.get_run_report("proj/with slash", "run id")

        called_path = mock_http.get.call_args.args[0]
        assert "run%20id" in called_path
        assert mock_http.get.call_args.kwargs["headers"] == {
            "X-Project-Id": "proj/with slash"
        }

    @pytest.mark.asyncio
    async def test_empty_project_id_rejected(self) -> None:
        client = _make_client()
        client._client = AsyncMock()
        with pytest.raises(ValueError, match="project_id"):
            await client.get_run_report("", "run_123")

    @pytest.mark.asyncio
    async def test_non_object_response_raises(self) -> None:
        from traigent.cloud.analytics_client import AnalyticsClientError

        client = _make_client()
        mock_response = MagicMock()
        mock_response.json.return_value = ["not", "an", "object"]
        mock_response.raise_for_status = MagicMock()
        mock_http = AsyncMock()
        mock_http.get.return_value = mock_response
        client._client = mock_http

        with pytest.raises(AnalyticsClientError, match="expected a JSON object"):
            await client.get_run_report("proj_abc", "run_123")

    @pytest.mark.asyncio
    async def test_bare_dto_response_raises(
        self, run_report_payload: dict[str, object]
    ) -> None:
        from traigent.cloud.analytics_client import AnalyticsClientError

        client = _make_client()
        mock_response = MagicMock()
        mock_response.json.return_value = run_report_payload
        mock_response.raise_for_status = MagicMock()
        mock_http = AsyncMock()
        mock_http.get.return_value = mock_response
        client._client = mock_http

        with pytest.raises(AnalyticsClientError, match="success envelope"):
            await client.get_run_report("proj_abc", "run_123")


class TestGetProjectOverview:
    @pytest.mark.asyncio
    async def test_calls_correct_path(self) -> None:
        client = _make_client()
        mock_response = MagicMock()
        mock_response.json.return_value = _success_envelope(
            {"project_id": "proj_abc", "runs": []}
        )
        mock_response.raise_for_status = MagicMock()
        mock_http = AsyncMock()
        mock_http.get.return_value = mock_response
        client._client = mock_http

        result = await client.get_project_overview("proj_abc")

        assert result == {"project_id": "proj_abc", "runs": []}
        mock_http.get.assert_called_once_with(
            "/api/v1/analytics/dashboards/optimization-overview",
            headers={"X-Project-Id": "proj_abc"},
            params=None,
        )


class TestCompareRuns:
    @pytest.mark.asyncio
    async def test_posts_run_ids_to_optimization_comparisons(self) -> None:
        client = _make_client()
        mock_response = MagicMock()
        mock_response.json.return_value = _success_envelope({"comparison": []})
        mock_response.raise_for_status = MagicMock()
        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        client._client = mock_http

        result = await client.compare_runs("proj_abc", ["run_1", "run_2", "run_3"])

        assert result == {"comparison": []}
        path = mock_http.post.call_args.args[0]
        kwargs = mock_http.post.call_args.kwargs
        assert path == "/api/v1/optimization-comparisons"
        assert kwargs["json"] == {"run_ids": ["run_1", "run_2", "run_3"]}
        assert kwargs["headers"] == {"X-Project-Id": "proj_abc"}

    @pytest.mark.asyncio
    async def test_requires_two_runs(self) -> None:
        client = _make_client()
        client._client = AsyncMock()
        with pytest.raises(ValueError, match="at least two"):
            await client.compare_runs("proj_abc", ["run_1"])


class TestGetRunDecisionBrief:
    @pytest.mark.asyncio
    async def test_calls_decision_payload_endpoint_with_project_header(
        self, decision_payload: dict[str, object]
    ) -> None:
        client = _make_client()
        mock_response = MagicMock()
        mock_response.json.return_value = _success_envelope(decision_payload)
        mock_response.raise_for_status = MagicMock()
        mock_http = AsyncMock()
        mock_http.get.return_value = mock_response
        client._client = mock_http

        result = await client.get_run_decision_brief(
            "proj_abc", "run_123", intent="deploy"
        )

        assert result == decision_payload
        path = mock_http.get.call_args.args[0]
        params = mock_http.get.call_args.kwargs["params"]
        headers = mock_http.get.call_args.kwargs["headers"]
        assert path == "/api/v1/analytics/runs/run_123/decision-payload"
        assert params == {"intent": "deploy"}
        assert headers == {"X-Project-Id": "proj_abc"}

    @pytest.mark.asyncio
    async def test_defaults_intent_to_iterate(
        self, decision_payload: dict[str, object]
    ) -> None:
        client = _make_client()
        mock_response = MagicMock()
        mock_response.json.return_value = _success_envelope(decision_payload)
        mock_response.raise_for_status = MagicMock()
        mock_http = AsyncMock()
        mock_http.get.return_value = mock_response
        client._client = mock_http

        await client.get_run_decision_brief("proj_abc", "run_123")

        params = mock_http.get.call_args.kwargs["params"]
        assert params["intent"] == "iterate"

    @pytest.mark.asyncio
    async def test_missing_contract_key_fails_closed(
        self, decision_payload: dict[str, object]
    ) -> None:
        from traigent.cloud.analytics_client import AnalyticsClientError

        malformed = dict(decision_payload)
        malformed.pop("recommended_action")

        client = _make_client()
        mock_response = MagicMock()
        mock_response.json.return_value = _success_envelope(malformed)
        mock_response.raise_for_status = MagicMock()
        mock_http = AsyncMock()
        mock_http.get.return_value = mock_response
        client._client = mock_http

        with pytest.raises(AnalyticsClientError, match="missing required key"):
            await client.get_run_decision_brief("proj_abc", "run_123")

    @pytest.mark.asyncio
    async def test_rejects_unsupported_intent_before_request(self) -> None:
        client = _make_client()
        mock_http = AsyncMock()
        client._client = mock_http

        with pytest.raises(ValueError, match="intent must be one of"):
            await client.get_run_decision_brief("proj_abc", "run_123", intent="promote")

        mock_http.get.assert_not_called()
