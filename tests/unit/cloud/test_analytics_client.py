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


def _make_client(api_key: str | None = "uk_test_key", jwt_token: str | None = None):
    from traigent.cloud.analytics_client import BackendAnalyticsClient

    return BackendAnalyticsClient(
        backend_url="http://localhost:5000",
        api_key=api_key,
        jwt_token=jwt_token,
    )


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
        fake_key = "uk_abcdef"
        client = _make_client(api_key=fake_key)
        headers = client._auth_headers()
        assert headers == {"X-API-Key": fake_key}

    def test_dotted_api_key_uses_x_api_key_header(self) -> None:
        """Regression: a three-segment API key is still an API key, not a JWT."""
        client = _make_client(api_key="uk_test.segment.withdots")
        headers = client._auth_headers()
        assert headers == {"X-API-Key": "uk_test.segment.withdots"}
        assert "Authorization" not in headers

    def test_jwt_uses_bearer_header(self) -> None:
        client = _make_client(api_key=None, jwt_token="aaa.bbb.ccc")
        headers = client._auth_headers()
        assert headers == {"Authorization": "Bearer aaa.bbb.ccc"}

    def test_api_key_wins_when_api_key_and_jwt_are_both_set(self) -> None:
        """Match JS buildTraigentHeaders: apiKey wins over jwtToken."""
        fake_key = "uk_abcdef"
        client = _make_client(
            api_key=fake_key,
            jwt_token="aaa.bbb.ccc",
        )

        headers = client._auth_headers()

        assert headers == {"X-API-Key": fake_key}
        assert "Authorization" not in headers


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


class TestGetClientHeaders:
    def test_default_headers_include_versioned_user_agent(self, monkeypatch) -> None:
        from traigent.cloud import analytics_client as analytics_client_mod

        captured: dict[str, object] = {}

        class _FakeAsyncClient:
            def __init__(self, *args, **kwargs) -> None:
                captured["headers"] = kwargs["headers"]

        monkeypatch.setattr(analytics_client_mod.httpx, "AsyncClient", _FakeAsyncClient)

        fake_key = "uk_abcdef"
        client = _make_client(api_key=fake_key)
        client._get_client()

        headers = captured["headers"]
        assert headers["X-API-Key"] == fake_key
        assert headers["User-Agent"].startswith("traigent-sdk/")


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
                "suspicious_example_count": 1,
                "notable_example_count": 3,
                "stable_example_count": 4,
                "dataset_quality": "medium",
            },
            "example_rows": [
                {
                    "safe_example_ref": "exref_0123456789abcdef",
                    "review_priority": "high",
                    "difficulty_bucket": "medium",
                    "suspicious_flags": ["possible_mislabel"],
                    "recommended_action": "review_label",
                }
            ],
            "cohorts": [
                {
                    "kind": "weak_examples",
                    "count": 2,
                    "impact": "quality_risk",
                    "safe_example_refs": ["exref_abcdef0123456789"],
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
        assert result is not example_insights_payload
        rows = result["example_rows"]
        assert isinstance(rows, list)
        assert set(rows[0]) == {
            "safe_example_ref",
            "review_priority",
            "difficulty_bucket",
            "suspicious_flags",
            "recommended_action",
        }
        mock_response.raise_for_status.assert_called_once()
        mock_http.get.assert_called_once_with(
            "/api/v1/analytics/runs/run%20123/example-insights",
            headers={"X-Project-Id": "proj_abc"},
            params=None,
        )

    @pytest.mark.asyncio
    async def test_get_example_insights_projects_extra_raw_row_fields(
        self, example_insights_payload: dict[str, object]
    ) -> None:
        client = _make_client()
        extended_payload = dict(example_insights_payload)
        rows = example_insights_payload["example_rows"]
        assert isinstance(rows, list)
        assert isinstance(rows[0], dict)
        extended_payload["example_rows"] = [
            {
                **rows[0],
                "composite_score": 0.9,
                "success_rate": 0.1,
                "example_id": "raw-123",
            }
        ]
        _, _ = _mock_get_response(client, extended_payload)

        result = await client.get_example_insights("proj_abc", "run_123")

        result_rows = result["example_rows"]
        assert isinstance(result_rows, list)
        row = result_rows[0]
        assert isinstance(row, dict)
        assert set(row) == {
            "safe_example_ref",
            "review_priority",
            "difficulty_bucket",
            "suspicious_flags",
            "recommended_action",
        }
        assert "composite_score" not in row
        assert "success_rate" not in row
        assert "example_id" not in row

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("field", "bad_value", "message"),
        [
            ("review_priority", "urgent", "review_priority"),
            ("difficulty_bucket", "hard", "difficulty_bucket"),
            ("recommended_action", "review_before_promotion", "recommended_action"),
            ("suspicious_flags", ["format_drift"], "suspicious_flags"),
            ("safe_example_ref", "exref_bad", "safe_example_ref"),
        ],
    )
    async def test_get_example_insights_rejects_invalid_row_fields(
        self,
        example_insights_payload: dict[str, object],
        field: str,
        bad_value: object,
        message: str,
    ) -> None:
        from traigent.cloud.analytics_client import AnalyticsClientError

        client = _make_client()
        payload = dict(example_insights_payload)
        rows = example_insights_payload["example_rows"]
        assert isinstance(rows, list)
        assert isinstance(rows[0], dict)
        payload["example_rows"] = [{**rows[0], field: bad_value}]
        _mock_get_response(client, payload)

        with pytest.raises(AnalyticsClientError, match=message):
            await client.get_example_insights("proj_abc", "run_123")

    @pytest.mark.asyncio
    async def test_get_example_insights_rejects_too_many_rows(
        self, example_insights_payload: dict[str, object]
    ) -> None:
        from traigent.cloud.analytics_client import AnalyticsClientError

        client = _make_client()
        payload = dict(example_insights_payload)
        rows = example_insights_payload["example_rows"]
        assert isinstance(rows, list)
        assert isinstance(rows[0], dict)
        payload["example_rows"] = [
            {**rows[0], "safe_example_ref": f"exref_{index:016x}"}
            for index in range(101)
        ]
        _mock_get_response(client, payload)

        with pytest.raises(AnalyticsClientError, match="at most 100"):
            await client.get_example_insights("proj_abc", "run_123")

    @pytest.mark.asyncio
    async def test_get_example_insights_conforming_summary_and_cohorts_pass_unchanged(
        self, example_insights_payload: dict[str, object]
    ) -> None:
        """A conforming payload's summary/cohort values pass through unchanged."""
        client = _make_client()
        _mock_get_response(client, example_insights_payload)

        result = await client.get_example_insights("proj_abc", "run_123")

        assert result["summary"] == example_insights_payload["summary"]
        assert result["cohorts"] == example_insights_payload["cohorts"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("dataset_quality", ["low", "medium", "high"])
    async def test_get_example_insights_accepts_all_dataset_quality_enum_values(
        self,
        example_insights_payload: dict[str, object],
        dataset_quality: str,
    ) -> None:
        client = _make_client()
        payload = dict(example_insights_payload)
        summary = payload["summary"]
        assert isinstance(summary, dict)
        payload["summary"] = {**summary, "dataset_quality": dataset_quality}
        _mock_get_response(client, payload)

        result = await client.get_example_insights("proj_abc", "run_123")

        assert result["summary"]["dataset_quality"] == dataset_quality

    @pytest.mark.asyncio
    async def test_get_example_insights_rejects_raw_dataset_quality_value(
        self, example_insights_payload: dict[str, object]
    ) -> None:
        """A backend regression emitting a raw value in dataset_quality must fail closed."""
        from traigent.cloud.analytics_client import AnalyticsClientError

        client = _make_client()
        payload = dict(example_insights_payload)
        summary = payload["summary"]
        assert isinstance(summary, dict)
        payload["summary"] = {
            **summary,
            "dataset_quality": "raw prompt text leaked from backend",
        }
        _mock_get_response(client, payload)

        with pytest.raises(AnalyticsClientError, match=r"summary\.dataset_quality"):
            await client.get_example_insights("proj_abc", "run_123")

    @pytest.mark.asyncio
    async def test_get_example_insights_rejects_missing_dataset_quality(
        self, example_insights_payload: dict[str, object]
    ) -> None:
        from traigent.cloud.analytics_client import AnalyticsClientError

        client = _make_client()
        payload = dict(example_insights_payload)
        summary = payload["summary"]
        assert isinstance(summary, dict)
        payload["summary"] = {
            key: value for key, value in summary.items() if key != "dataset_quality"
        }
        _mock_get_response(client, payload)

        with pytest.raises(AnalyticsClientError, match=r"summary\.dataset_quality"):
            await client.get_example_insights("proj_abc", "run_123")

    @pytest.mark.asyncio
    async def test_get_example_insights_rejects_non_exref_cohort_safe_example_refs(
        self, example_insights_payload: dict[str, object]
    ) -> None:
        """A backend regression placing raw content in safe_example_refs must fail closed."""
        from traigent.cloud.analytics_client import AnalyticsClientError

        client = _make_client()
        payload = dict(example_insights_payload)
        cohorts = payload["cohorts"]
        assert isinstance(cohorts, list)
        assert isinstance(cohorts[0], dict)
        payload["cohorts"] = [
            {**cohorts[0], "safe_example_refs": ["please summarize this raw prompt"]}
        ]
        _mock_get_response(client, payload)

        with pytest.raises(
            AnalyticsClientError, match=r"cohorts\[0\]\.safe_example_refs"
        ):
            await client.get_example_insights("proj_abc", "run_123")

    @pytest.mark.asyncio
    async def test_get_example_insights_rejects_non_list_cohort_safe_example_refs(
        self, example_insights_payload: dict[str, object]
    ) -> None:
        """A nested object/scalar in place of the refs list must fail closed."""
        from traigent.cloud.analytics_client import AnalyticsClientError

        client = _make_client()
        payload = dict(example_insights_payload)
        cohorts = payload["cohorts"]
        assert isinstance(cohorts, list)
        assert isinstance(cohorts[0], dict)
        payload["cohorts"] = [{**cohorts[0], "safe_example_refs": {"nested": "object"}}]
        _mock_get_response(client, payload)

        with pytest.raises(
            AnalyticsClientError,
            match=r"cohorts\[0\]\.safe_example_refs must be a list",
        ):
            await client.get_example_insights("proj_abc", "run_123")

    @pytest.mark.asyncio
    async def test_get_example_insights_rejects_too_many_cohort_safe_example_refs(
        self, example_insights_payload: dict[str, object]
    ) -> None:
        from traigent.cloud.analytics_client import AnalyticsClientError

        client = _make_client()
        payload = dict(example_insights_payload)
        cohorts = payload["cohorts"]
        assert isinstance(cohorts, list)
        assert isinstance(cohorts[0], dict)
        payload["cohorts"] = [
            {
                **cohorts[0],
                "safe_example_refs": [f"exref_{index:016x}" for index in range(51)],
            }
        ]
        _mock_get_response(client, payload)

        with pytest.raises(AnalyticsClientError, match="at most 50 refs"):
            await client.get_example_insights("proj_abc", "run_123")

    @pytest.mark.asyncio
    async def test_get_example_insights_allows_cohort_without_safe_example_refs(
        self, example_insights_payload: dict[str, object]
    ) -> None:
        """safe_example_refs is optional per cohort; its absence is not a violation."""
        client = _make_client()
        payload = dict(example_insights_payload)
        cohorts = payload["cohorts"]
        assert isinstance(cohorts, list)
        assert isinstance(cohorts[0], dict)
        payload["cohorts"] = [
            {
                key: value
                for key, value in cohorts[0].items()
                if key != "safe_example_refs"
            }
        ]
        _mock_get_response(client, payload)

        result = await client.get_example_insights("proj_abc", "run_123")

        assert "safe_example_refs" not in result["cohorts"][0]

    @pytest.mark.asyncio
    async def test_get_example_insights_drops_unknown_top_level_keys(
        self, example_insights_payload: dict[str, object]
    ) -> None:
        client = _make_client()
        extended_payload = dict(example_insights_payload)
        extended_payload["raw_dataset_scores"] = {"composite_score": 0.9}
        _mock_get_response(client, extended_payload)

        result = await client.get_example_insights("proj_abc", "run_123")

        assert list(result) == [
            "run_id",
            "privacy_mode",
            "summary",
            "example_rows",
            "cohorts",
            "recommendations",
            "redactions",
        ]
        assert "raw_dataset_scores" not in result

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


# --- SDK #1893: W3C traceparent injection on analytics backend calls ---------
# Follow-up to #1882/#1892. BackendAnalyticsClient uses a *cached, long-lived*
# httpx.AsyncClient, so trace context must be injected per-request (never frozen
# into the cached client's default headers, which would go stale).

import re  # noqa: E402
import sys  # noqa: E402
from contextlib import contextmanager  # noqa: E402

from opentelemetry.sdk.trace import TracerProvider as _SdkTracerProvider  # noqa: E402

_TRACEPARENT_RE = re.compile(r"^00-[0-9a-f]{32}-[0-9a-f]{16}-0[0-9a-f]$")


@contextmanager
def _recording_span():
    """Attach a real, recording OTel span to the current context.

    Uses a *local* SDK ``TracerProvider`` (never the global one) so the test
    does not pollute global tracing state.
    """
    provider = _SdkTracerProvider()
    tracer = provider.get_tracer("test-sdk-1893-analytics")
    with tracer.start_as_current_span("test-span") as span:
        yield span


class TestAnalyticsTraceparentInjection:
    """SDK #1893: analytics backend calls carry a per-request W3C traceparent."""

    @pytest.mark.asyncio
    async def test_get_request_carries_traceparent_matching_active_span(self) -> None:
        client = _make_client()
        mock_response = MagicMock()
        mock_response.json.return_value = _success_envelope({"ok": True})
        mock_response.raise_for_status = MagicMock()
        mock_http = AsyncMock()
        mock_http.get.return_value = mock_response
        client._client = mock_http

        with _recording_span() as span:
            await client.get_run_report("proj_abc", "run_123")
            expected_trace_id = format(span.get_span_context().trace_id, "032x")

        headers = mock_http.get.call_args.kwargs["headers"]
        assert _TRACEPARENT_RE.match(headers["traceparent"]), headers["traceparent"]
        assert headers["traceparent"].split("-")[1] == expected_trace_id
        # Caller-supplied project header still rides alongside.
        assert headers["X-Project-Id"] == "proj_abc"

    @pytest.mark.asyncio
    async def test_post_request_carries_traceparent_matching_active_span(self) -> None:
        client = _make_client()
        mock_response = MagicMock()
        mock_response.json.return_value = _success_envelope({"comparison": []})
        mock_response.raise_for_status = MagicMock()
        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        client._client = mock_http

        with _recording_span() as span:
            await client.compare_runs("proj_abc", ["run_1", "run_2"])
            expected_trace_id = format(span.get_span_context().trace_id, "032x")

        headers = mock_http.post.call_args.kwargs["headers"]
        assert headers["traceparent"].split("-")[1] == expected_trace_id
        assert headers["X-Project-Id"] == "proj_abc"

    @pytest.mark.asyncio
    async def test_post_json_site_carries_traceparent_matching_active_span(
        self,
    ) -> None:
        """Direct coverage of the _post_json injection site (used by e.g.
        observability cohort compare). It builds NO per-request headers of its
        own, so this proves the empty-headers -> per-request-injection path."""
        client = _make_client()
        mock_response = MagicMock()
        mock_response.json.return_value = _success_envelope({"ok": True})
        mock_response.raise_for_status = MagicMock()
        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        client._client = mock_http

        with _recording_span() as span:
            await client._post_json(
                "/api/v1/observability/x/analysis/cohorts/compare",
                what="observability cohort comparison",
                json_body={"a": 1},
            )
            expected_trace_id = format(span.get_span_context().trace_id, "032x")

        headers = mock_http.post.call_args.kwargs["headers"]
        assert _TRACEPARENT_RE.match(headers["traceparent"]), headers["traceparent"]
        assert headers["traceparent"].split("-")[1] == expected_trace_id

    @pytest.mark.asyncio
    async def test_cached_client_default_headers_never_carry_traceparent(
        self, monkeypatch
    ) -> None:
        """Staleness guard: even when a span is active at client-construction
        time, the cached client's DEFAULT headers must carry no trace context.
        Trace context rides per-request only (see #1892 rev-3)."""
        from traigent.cloud import analytics_client as analytics_client_mod

        captured: dict[str, object] = {}

        class _FakeAsyncClient:
            def __init__(self, *args, **kwargs) -> None:
                captured["headers"] = kwargs["headers"]

        monkeypatch.setattr(analytics_client_mod.httpx, "AsyncClient", _FakeAsyncClient)

        client = _make_client(api_key="uk_abcdef")
        with _recording_span():
            client._get_client()

        default_headers = captured["headers"]
        assert not any(
            name.lower() in ("traceparent", "tracestate") for name in default_headers
        ), default_headers

    @pytest.mark.asyncio
    async def test_no_span_headers_are_byte_identical(self) -> None:
        """No active span -> per-request headers are byte-identical to the
        pre-#1893 headers (the injection is a pure no-op)."""
        client = _make_client()
        mock_response = MagicMock()
        mock_response.json.return_value = _success_envelope({"ok": True})
        mock_response.raise_for_status = MagicMock()
        mock_http = AsyncMock()
        mock_http.get.return_value = mock_response
        client._client = mock_http

        await client.get_run_report("proj_abc", "run_123")

        headers = mock_http.get.call_args.kwargs["headers"]
        assert headers == {"X-Project-Id": "proj_abc"}

    @pytest.mark.asyncio
    async def test_no_span_post_headers_byte_identical(self) -> None:
        client = _make_client()
        mock_response = MagicMock()
        mock_response.json.return_value = _success_envelope({"ok": True})
        mock_response.raise_for_status = MagicMock()
        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        client._client = mock_http

        # _post_json path (no per-request headers pre-#1893) -> empty dict now.
        await client.compare_runs("proj_abc", ["run_1", "run_2"])
        assert mock_http.post.call_args.kwargs["headers"] == {
            "X-Project-Id": "proj_abc"
        }

    def test_helper_is_noop_without_opentelemetry(self) -> None:
        """Degrade path: with opentelemetry unimportable, _request_headers is a
        silent no-op returning the caller's headers unchanged."""
        client = _make_client()
        with patch.dict(
            sys.modules,
            {
                "opentelemetry": None,
                "opentelemetry.trace.propagation.tracecontext": None,
            },
        ):
            result = client._request_headers({"X-Project-Id": "p"})
        assert result == {"X-Project-Id": "p"}

    @pytest.mark.asyncio
    async def test_caller_supplied_traceparent_not_overridden(self) -> None:
        caller_tp = "00-" + "b" * 32 + "-" + "c" * 16 + "-01"
        client = _make_client()
        with _recording_span():
            result = client._request_headers({"traceparent": caller_tp})
        assert result == {"traceparent": caller_tp}
