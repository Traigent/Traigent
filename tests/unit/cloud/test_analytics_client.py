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


def _mock_get_response(client, data: object):
    mock_response = MagicMock()
    mock_response.json.return_value = _success_envelope(data)
    mock_response.raise_for_status = MagicMock()
    mock_http = AsyncMock()
    mock_http.get.return_value = mock_response
    client._client = mock_http
    return mock_http, mock_response


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


class TestWave2SingleRunAnalytics:
    @pytest.fixture()
    def pareto_payload(self) -> dict[str, object]:
        return {
            "run_id": "run_123",
            "project_id": "proj_abc",
            "measures": {
                "quality": "quality",
                "cost": "cost",
                "latency": "latency",
            },
            "frontier": [],
            "dominated": [],
            "shape": "flat",
            "warnings": [],
        }

    @pytest.fixture()
    def correlations_payload(self) -> dict[str, object]:
        return {
            "run_id": "run_123",
            "method": "spearman",
            "sample_size": 12,
            "measure_correlations": [],
            "parameter_correlations": [],
            "warnings": [],
        }

    @pytest.fixture()
    def leaderboard_payload(self) -> dict[str, object]:
        return {
            "run_id": "run_123",
            "ranking_basis": {
                "objective": "weighted",
                "weights": {"quality": 0.8},
                "constraints": {"cost": 1.0},
            },
            "configs": [],
        }

    @pytest.fixture()
    def parameter_insights_payload(self) -> dict[str, object]:
        return {
            "run_id": "run_123",
            "target_measure": "quality",
            "min_trials": 10,
            "drivers": [],
            "interactions": [],
            "warnings": [],
        }

    @pytest.fixture()
    def example_insights_payload(self) -> dict[str, object]:
        return {
            "run_id": "run_123",
            "privacy_mode": "safe_agent_projection",
            "summary": {
                "example_count": 10,
                "weak_example_count": 2,
                "unstable_example_count": 1,
                "dataset_quality": "medium",
            },
            "cohorts": [
                {
                    "kind": "weak_examples",
                    "count": 2,
                    "impact": "quality_risk",
                    "safe_example_refs": ["exref_abc123"],
                    "recommendation": "Review weak examples before promotion.",
                }
            ],
            "recommendations": [
                {
                    "action": "rebalance_dataset",
                    "reason": "Weak example cohort is non-empty.",
                }
            ],
            "redactions": {
                "raw_proprietary_signals_hidden": True,
                "raw_prompt_text_hidden_by_default": True,
            },
        }

    @pytest.mark.asyncio
    async def test_get_single_run_pareto_calls_endpoint(
        self, pareto_payload: dict[str, object]
    ) -> None:
        client = _make_client()
        mock_http, mock_response = _mock_get_response(client, pareto_payload)

        result = await client.get_single_run_pareto(
            "proj_abc",
            "run 123",
            x_measure="cost",
            y_measure="quality",
            request_count=5,
        )

        assert result == pareto_payload
        mock_response.raise_for_status.assert_called_once()
        mock_http.get.assert_called_once_with(
            "/api/v1/analytics/runs/run%20123/pareto",
            headers={"X-Project-Id": "proj_abc"},
            params={
                "x_measure": "cost",
                "y_measure": "quality",
                "request_count": "5",
            },
        )

    @pytest.mark.asyncio
    async def test_get_correlation_matrix_calls_endpoint(
        self, correlations_payload: dict[str, object]
    ) -> None:
        client = _make_client()
        mock_http, mock_response = _mock_get_response(client, correlations_payload)

        result = await client.get_correlation_matrix(
            "proj_abc", "run_123", method="spearman", min_sample=4
        )

        assert result == correlations_payload
        mock_response.raise_for_status.assert_called_once()
        mock_http.get.assert_called_once_with(
            "/api/v1/analytics/runs/run_123/correlations",
            headers={"X-Project-Id": "proj_abc"},
            params={"method": "spearman", "min_sample": "4"},
        )

    @pytest.mark.asyncio
    async def test_get_run_leaderboard_calls_endpoint_with_json_query(
        self, leaderboard_payload: dict[str, object]
    ) -> None:
        client = _make_client()
        mock_http, mock_response = _mock_get_response(client, leaderboard_payload)

        result = await client.get_run_leaderboard(
            "proj_abc",
            "run_123",
            objective="weighted",
            weights={"quality": 0.8},
            constraints='{"cost":1.0}',
            request_count=7,
            limit=10,
        )

        assert result == leaderboard_payload
        mock_response.raise_for_status.assert_called_once()
        mock_http.get.assert_called_once_with(
            "/api/v1/analytics/runs/run_123/leaderboard",
            headers={"X-Project-Id": "proj_abc"},
            params={
                "objective": "weighted",
                "weights": '{"quality":0.8}',
                "constraints": '{"cost":1.0}',
                "request_count": "7",
                "limit": "10",
            },
        )

    @pytest.mark.asyncio
    async def test_get_parameter_insights_calls_endpoint(
        self, parameter_insights_payload: dict[str, object]
    ) -> None:
        client = _make_client()
        mock_http, mock_response = _mock_get_response(
            client, parameter_insights_payload
        )

        result = await client.get_parameter_insights(
            "proj_abc",
            "run_123",
            target_measure="accuracy",
            min_trials=12,
            top_k=5,
        )

        assert result == parameter_insights_payload
        mock_response.raise_for_status.assert_called_once()
        mock_http.get.assert_called_once_with(
            "/api/v1/analytics/runs/run_123/parameter-insights",
            headers={"X-Project-Id": "proj_abc"},
            params={
                "target_measure": "accuracy",
                "min_trials": "12",
                "top_k": "5",
            },
        )

    @pytest.mark.asyncio
    async def test_get_example_insights_calls_endpoint(
        self, example_insights_payload: dict[str, object]
    ) -> None:
        client = _make_client()
        mock_http, mock_response = _mock_get_response(client, example_insights_payload)

        result = await client.get_example_insights("proj_abc", "run 123")

        assert result == example_insights_payload
        mock_response.raise_for_status.assert_called_once()
        mock_http.get.assert_called_once_with(
            "/api/v1/analytics/runs/run%20123/example-insights",
            headers={"X-Project-Id": "proj_abc"},
            params=None,
        )

    @pytest.mark.asyncio
    async def test_get_example_insights_rejects_empty_ids(
        self, example_insights_payload: dict[str, object]
    ) -> None:
        client = _make_client()
        mock_http, _ = _mock_get_response(client, example_insights_payload)

        with pytest.raises(ValueError, match="project_id"):
            await client.get_example_insights("", "run_123")
        with pytest.raises(ValueError, match="run_id"):
            await client.get_example_insights("proj_abc", " ")

        mock_http.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_defaults_match_backend_query_schema(
        self,
        pareto_payload: dict[str, object],
        correlations_payload: dict[str, object],
        leaderboard_payload: dict[str, object],
        parameter_insights_payload: dict[str, object],
    ) -> None:
        cases = [
            (
                lambda client: client.get_single_run_pareto("proj_abc", "run_123"),
                pareto_payload,
                {
                    "x_measure": "cost",
                    "y_measure": "quality",
                    "request_count": "1",
                },
            ),
            (
                lambda client: client.get_correlation_matrix("proj_abc", "run_123"),
                correlations_payload,
                {"method": "pearson", "min_sample": "3"},
            ),
            (
                lambda client: client.get_run_leaderboard("proj_abc", "run_123"),
                leaderboard_payload,
                {"objective": "weighted", "request_count": "1", "limit": "50"},
            ),
            (
                lambda client: client.get_parameter_insights("proj_abc", "run_123"),
                parameter_insights_payload,
                {"target_measure": "quality", "min_trials": "10", "top_k": "10"},
            ),
        ]

        for call, payload, expected_params in cases:
            client = _make_client()
            mock_http, _ = _mock_get_response(client, payload)

            await call(client)

            assert mock_http.get.call_args.kwargs["params"] == expected_params

    @pytest.mark.asyncio
    async def test_missing_required_dto_keys_fail_closed(
        self,
        pareto_payload: dict[str, object],
        correlations_payload: dict[str, object],
        leaderboard_payload: dict[str, object],
        parameter_insights_payload: dict[str, object],
        example_insights_payload: dict[str, object],
    ) -> None:
        from traigent.cloud.analytics_client import AnalyticsClientError

        cases = [
            (
                "shape",
                pareto_payload,
                lambda client: client.get_single_run_pareto("proj_abc", "run_123"),
            ),
            (
                "sample_size",
                correlations_payload,
                lambda client: client.get_correlation_matrix("proj_abc", "run_123"),
            ),
            (
                "ranking_basis",
                leaderboard_payload,
                lambda client: client.get_run_leaderboard("proj_abc", "run_123"),
            ),
            (
                "drivers",
                parameter_insights_payload,
                lambda client: client.get_parameter_insights("proj_abc", "run_123"),
            ),
            (
                "redactions",
                example_insights_payload,
                lambda client: client.get_example_insights("proj_abc", "run_123"),
            ),
        ]

        for missing_key, payload, call in cases:
            malformed = dict(payload)
            malformed.pop(missing_key)
            client = _make_client()
            _mock_get_response(client, malformed)

            with pytest.raises(AnalyticsClientError, match=missing_key):
                await call(client)

    @pytest.mark.asyncio
    async def test_malformed_envelope_fails_closed(self) -> None:
        from traigent.cloud.analytics_client import AnalyticsClientError

        client = _make_client()
        mock_response = MagicMock()
        mock_response.json.return_value = {"success": True}
        mock_response.raise_for_status = MagicMock()
        mock_http = AsyncMock()
        mock_http.get.return_value = mock_response
        client._client = mock_http

        with pytest.raises(AnalyticsClientError, match="success envelope"):
            await client.get_parameter_insights("proj_abc", "run_123")

    @pytest.mark.asyncio
    async def test_transport_error_surfaces_before_unwrap(
        self, pareto_payload: dict[str, object]
    ) -> None:
        client = _make_client()
        _, mock_response = _mock_get_response(client, pareto_payload)
        mock_response.raise_for_status.side_effect = RuntimeError("http error")

        with pytest.raises(RuntimeError, match="http error"):
            await client.get_single_run_pareto("proj_abc", "run_123")
