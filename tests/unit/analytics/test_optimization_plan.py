"""Unit tests for OptimizationPlanClient."""

from __future__ import annotations

import importlib.util
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from traigent.analytics import optimization_plan as op_module

HTTPX_AVAILABLE = importlib.util.find_spec("httpx") is not None


@pytest.fixture(autouse=True)
def _online_backend(jwt_development_mode, monkeypatch):
    """These tests mock transport but exercise the online request path."""
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")


@pytest.fixture()
def valid_plan_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "phase": "P1_STATIC",
        "plan": {
            "objectives": [
                {"name": "accuracy", "weight": 1.0, "orientation": "maximize"}
            ],
            "models": ["gpt-4o-mini"],
            "knobs": [{"name": "temperature", "values": ["0.0", "0.3"]}],
            "algorithm": "auto",
            "max_trials": 4,
            "cost_limit_usd": 5.0,
            "offline": False,
        },
        "steps": [
            {
                "id": "review_plan",
                "label": "Review plan",
                "command_template": "traigent plan --task-description ...",
            }
        ],
        "evidence_level": "medium",
        "caveat": "Plans are advisory until a run starts.",
        "advisory": True,
    }


class TestOptimizationPlanClientInit:
    """Tests for OptimizationPlanClient initialization."""

    @pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed")
    def test_init_defaults(self) -> None:
        from traigent.analytics.optimization_plan import OptimizationPlanClient

        with patch(
            "traigent.utils.env_config.get_api_key",
            return_value="test_key",  # pragma: allowlist secret
        ):
            client = OptimizationPlanClient()

            assert client.backend_url == "http://localhost:5000"
            assert client.timeout == 30.0
            assert client.api_key == "test_key"  # pragma: allowlist secret
            assert client._client is None

    @pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed")
    def test_init_with_custom_values(self) -> None:
        from traigent.analytics.optimization_plan import OptimizationPlanClient

        client = OptimizationPlanClient(
            backend_url="https://custom.api.com",
            api_key="custom_key",  # pragma: allowlist secret
            timeout=60.0,
        )

        assert client.backend_url == "https://custom.api.com"
        assert client.api_key == "custom_key"  # pragma: allowlist secret
        assert client.timeout == 60.0

    @pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed")
    def test_init_strips_trailing_slash(self) -> None:
        from traigent.analytics.optimization_plan import OptimizationPlanClient

        client = OptimizationPlanClient(
            backend_url="http://localhost:5000/",
            api_key="key",  # pragma: allowlist secret
        )

        assert client.backend_url == "http://localhost:5000"

    @pytest.mark.skipif(HTTPX_AVAILABLE, reason="Test for httpx not available")
    def test_init_raises_without_httpx(self) -> None:
        with pytest.raises(ImportError, match="httpx is required"):
            op_module.OptimizationPlanClient()


@pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed")
class TestOptimizationPlanClientGetClient:
    """Tests for _get_client method."""

    def test_get_client_creates_client_with_x_api_key_header_for_api_key(self) -> None:
        """uk_-style (and other non-JWT) keys must travel via X-API-Key, never Bearer.

        See auth.py:_get_api_key_headers: Bearer is reserved for JWTs; sending
        an API key as Bearer causes a 401 INVALID_TOKEN before X-API-Key is
        considered.
        """
        from traigent.analytics.optimization_plan import OptimizationPlanClient

        api_key = "uk_testkey1234567890"  # pragma: allowlist secret
        client = OptimizationPlanClient(api_key=api_key)

        with patch("httpx.AsyncClient") as mock_async_client:
            mock_instance = MagicMock()
            mock_async_client.return_value = mock_instance

            result = client._get_client()

            assert result == mock_instance
            mock_async_client.assert_called_once()
            call_kwargs = mock_async_client.call_args.kwargs
            assert call_kwargs["headers"]["X-API-Key"] == api_key
            # Must NOT send the API key as a Bearer token
            assert "Authorization" not in call_kwargs["headers"]

    def test_get_client_uses_x_api_key_for_jwt_shaped_api_key(self) -> None:
        """JWT-shaped API-key strings must not be inferred as Bearer tokens."""
        from traigent.analytics.optimization_plan import OptimizationPlanClient

        api_key = "eyJnotjwt.with.dots"  # pragma: allowlist secret
        client = OptimizationPlanClient(api_key=api_key)

        with patch("httpx.AsyncClient") as mock_async_client:
            mock_instance = MagicMock()
            mock_async_client.return_value = mock_instance

            client._get_client()

            call_kwargs = mock_async_client.call_args.kwargs
            assert call_kwargs["headers"] == {"X-API-Key": api_key}
            assert "Authorization" not in call_kwargs["headers"]

    @pytest.mark.parametrize("api_key", [None, ""])
    def test_get_client_without_api_key(self, api_key: str | None) -> None:
        from traigent.analytics.optimization_plan import OptimizationPlanClient

        with patch("traigent.utils.env_config.get_api_key", return_value=None):
            client = OptimizationPlanClient(api_key=api_key)

        with patch("httpx.AsyncClient") as mock_async_client:
            client._get_client()

            call_kwargs = mock_async_client.call_args.kwargs
            assert call_kwargs["headers"] == {}


@pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed")
class TestGetOptimizationPlan:
    """Tests for get_optimization_plan method."""

    @pytest.mark.asyncio
    async def test_get_optimization_plan_success(
        self,
        valid_plan_payload: dict[str, object],
    ) -> None:
        from traigent.analytics.optimization_plan import OptimizationPlanClient

        client = OptimizationPlanClient(api_key="test")

        mock_response = MagicMock()
        mock_response.json.return_value = valid_plan_payload
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        client._client = mock_http

        result = await client.get_optimization_plan(
            task_description="Tune a support chatbot.",
            dataset_size=20,
            dataset_has_holdout=True,
            objectives=["accuracy", "latency"],
            max_trials=8,
            cost_limit_usd=12.5,
            task_type="chatbot",
            agent_shape="tool_agent",
            weights={"accuracy": 0.8, "latency": 0.2},
            offline=False,
        )

        assert result == valid_plan_payload
        mock_http.post.assert_called_once_with(
            "/api/v1/optimization/plan",
            json={
                "task_description": "Tune a support chatbot.",
                "dataset": {"size": 20, "has_holdout": True},
                "objectives": ["accuracy", "latency"],
                "budget": {"max_trials": 8, "cost_limit_usd": 12.5},
                "task_type": "chatbot",
                "agent_shape": "tool_agent",
                "weights": {"accuracy": 0.8, "latency": 0.2},
                "offline": False,
            },
        )

    @pytest.mark.asyncio
    async def test_get_optimization_plan_omits_absent_optional_fields(
        self,
        valid_plan_payload: dict[str, object],
    ) -> None:
        from traigent.analytics.optimization_plan import OptimizationPlanClient

        client = OptimizationPlanClient(api_key="test")
        mock_response = MagicMock()
        mock_response.json.return_value = valid_plan_payload
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        client._client = mock_http

        await client.get_optimization_plan(
            task_description="Tune a support chatbot.",
            dataset_size=20,
            dataset_has_holdout=False,
            objectives=["accuracy"],
            max_trials=4,
            cost_limit_usd=5.0,
        )

        kwargs = mock_http.post.call_args.kwargs
        assert kwargs["json"] == {
            "task_description": "Tune a support chatbot.",
            "dataset": {"size": 20, "has_holdout": False},
            "objectives": ["accuracy"],
            "budget": {"max_trials": 4, "cost_limit_usd": 5.0},
        }

    @pytest.mark.asyncio
    async def test_get_optimization_plan_requires_json_object(self) -> None:
        from traigent.analytics.optimization_plan import OptimizationPlanClient

        client = OptimizationPlanClient(api_key="test")
        mock_response = MagicMock()
        mock_response.json.return_value = ["not", "an", "object"]
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        client._client = mock_http

        with pytest.raises(ValueError, match="expected a JSON object"):
            await client.get_optimization_plan(
                task_description="Tune a support chatbot.",
                dataset_size=20,
                dataset_has_holdout=False,
                objectives=["accuracy"],
                max_trials=4,
                cost_limit_usd=5.0,
            )

    @pytest.mark.parametrize("missing_key", ["plan", "steps", "advisory"])
    async def test_get_optimization_plan_requires_full_contract_keys(
        self,
        valid_plan_payload: dict[str, object],
        missing_key: str,
    ) -> None:
        from traigent.analytics.optimization_plan import OptimizationPlanClient

        client = OptimizationPlanClient(api_key="test")
        malformed = dict(valid_plan_payload)
        malformed.pop(missing_key)

        mock_response = MagicMock()
        mock_response.json.return_value = malformed
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        client._client = mock_http

        with pytest.raises(ValueError, match="missing required key"):
            await client.get_optimization_plan(
                task_description="Tune a support chatbot.",
                dataset_size=20,
                dataset_has_holdout=False,
                objectives=["accuracy"],
                max_trials=4,
                cost_limit_usd=5.0,
            )


class TestBuildAuthHeaders:
    """Unit tests for the _build_auth_headers module-level helper.

    This helper delegates API-key header construction to cloud/auth.py.
    """

    def test_uk_api_key_uses_x_api_key_header(self) -> None:
        from traigent.analytics.optimization_plan import _build_auth_headers

        key = "uk_testkey1234567890"  # pragma: allowlist secret
        headers = _build_auth_headers(key)

        assert headers == {"X-API-Key": key}
        assert "Authorization" not in headers

    def test_generic_api_key_uses_x_api_key_header(self) -> None:
        """Any API-key credential string uses X-API-Key."""
        from traigent.analytics.optimization_plan import _build_auth_headers

        for key in (
            "tg_abc123",  # pragma: allowlist secret
            "sk_livekey",  # pragma: allowlist secret
            "ak_somekey",  # pragma: allowlist secret
            "plain_secret",  # pragma: allowlist secret
            "eyJnotjwt.with.dots",  # pragma: allowlist secret
        ):
            headers = _build_auth_headers(key)
            assert "X-API-Key" in headers, f"expected X-API-Key for {key!r}"
            assert "Authorization" not in headers, f"must not set Bearer for {key!r}"

    def test_jwt_shaped_api_key_uses_x_api_key_header(self) -> None:
        from traigent.analytics.optimization_plan import _build_auth_headers

        key = "eyJnotjwt.with.dots"  # pragma: allowlist secret
        headers = _build_auth_headers(key)

        assert headers == {"X-API-Key": key}
        assert "Authorization" not in headers

    def test_real_jwt_string_passed_as_api_key_uses_x_api_key_header(self) -> None:
        """The API-key-only client must not infer JWT mode from token shape."""
        from traigent.analytics.optimization_plan import _build_auth_headers

        jwt = "eyJhbGciOiJSUzI1NiJ9.eyJpc3MiOiJ0cmFpZ2VudCJ9.RSASIG"  # pragma: allowlist secret
        headers = _build_auth_headers(jwt)

        assert headers == {"X-API-Key": jwt}
        assert "Authorization" not in headers

    @pytest.mark.parametrize("credential", [None, ""])
    def test_empty_key_returns_no_auth_headers(self, credential: str | None) -> None:
        from traigent.analytics.optimization_plan import _build_auth_headers

        assert _build_auth_headers(credential) == {}

    def test_api_key_not_sent_as_bearer(self) -> None:
        """Regression guard: uk_ API key must never appear in Authorization header."""
        from traigent.analytics.optimization_plan import _build_auth_headers

        key = "uk_regressionkey00000"  # pragma: allowlist secret
        headers = _build_auth_headers(key)

        assert "Authorization" not in headers
        # And the Bearer value must not contain the key at all
        bearer = headers.get("Authorization", "")
        assert key not in bearer
