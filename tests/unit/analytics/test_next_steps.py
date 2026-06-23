"""Unit tests for NextStepsClient."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from traigent.analytics import next_steps as ns_module


@pytest.fixture(autouse=True)
def _online_backend(jwt_development_mode, monkeypatch):
    """These tests mock transport but exercise the online request path."""
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")


def _schema_path() -> Path:
    # Vendored from TraigentSchema PR #120 so tests run without a sibling
    # checkout (CI); swap to the packaged traigent-schema contract once
    # #120 ships in a release.
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "tests" / "fixtures" / "contracts" / "next_steps_schema.json"


def _schema_required_fields() -> set[str]:
    with _schema_path().open(encoding="utf-8") as schema_file:
        schema = json.load(schema_file)
    return set(schema["required"])


@pytest.fixture()
def valid_next_steps_payload() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "experiment_run_id": "run_123",
        "caveat": "Recommendations are category-level and should be reviewed.",
        "summary": {
            "winner_config_ref": "config_7",
            "confidence_label": "medium",
            "trade_off_note": "Latency and quality should be reviewed together.",
        },
        "next_steps": [
            {
                "id": "step_1",
                "category": "rerun_larger_sample",
                "priority": 1,
                "rationale": "Run more trials to improve confidence.",
                "action": {
                    "kind": "cli",
                    "command_template": "traigent optimize --trials 50",
                },
                "evidence_level": "medium",
            }
        ],
    }
    assert _schema_required_fields().issubset(payload.keys())
    return payload


class TestNextStepsClientInit:
    """Tests for NextStepsClient initialization."""

    @pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed")
    def test_init_defaults(self) -> None:
        from traigent.analytics.next_steps import NextStepsClient

        with patch(
            "traigent.utils.env_config.get_api_key",
            return_value="test_key",  # pragma: allowlist secret
        ):
            client = NextStepsClient()

            assert client.backend_url == "http://localhost:5000"
            assert client.timeout == 30.0
            assert client.api_key == "test_key"  # pragma: allowlist secret
            assert client._client is None

    @pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed")
    def test_init_with_custom_values(self) -> None:
        from traigent.analytics.next_steps import NextStepsClient

        client = NextStepsClient(
            backend_url="https://custom.api.com",
            api_key="custom_key",  # pragma: allowlist secret
            timeout=60.0,
        )

        assert client.backend_url == "https://custom.api.com"
        assert client.api_key == "custom_key"  # pragma: allowlist secret
        assert client.timeout == 60.0

    @pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed")
    def test_init_strips_trailing_slash(self) -> None:
        from traigent.analytics.next_steps import NextStepsClient

        client = NextStepsClient(
            backend_url="http://localhost:5000/",
            api_key="key",  # pragma: allowlist secret
        )

        assert client.backend_url == "http://localhost:5000"

    @pytest.mark.skipif(HTTPX_AVAILABLE, reason="Test for httpx not available")
    def test_init_raises_without_httpx(self) -> None:
        with pytest.raises(ImportError, match="httpx is required"):
            ns_module.NextStepsClient()


@pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed")
class TestNextStepsClientGetClient:
    """Tests for _get_client method."""

    def test_get_client_creates_client_with_x_api_key_header(self) -> None:
        """API keys must travel via X-API-Key, not Authorization: Bearer."""
        from traigent.analytics.next_steps import NextStepsClient

        client = NextStepsClient(api_key="test_token")

        with patch("httpx.AsyncClient") as mock_async_client:
            mock_instance = MagicMock()
            mock_async_client.return_value = mock_instance

            result = client._get_client()

            assert result == mock_instance
            mock_async_client.assert_called_once()
            call_kwargs = mock_async_client.call_args.kwargs
            assert call_kwargs["headers"]["X-API-Key"] == "test_token"
            assert "Authorization" not in call_kwargs["headers"]

    def test_get_client_without_api_key(self) -> None:
        from traigent.analytics.next_steps import NextStepsClient

        client = NextStepsClient(api_key=None)
        client.api_key = None

        with patch("httpx.AsyncClient") as mock_async_client:
            client._get_client()

            call_kwargs = mock_async_client.call_args.kwargs
            assert "Authorization" not in call_kwargs["headers"]

    def test_get_client_uk_api_key_uses_x_api_key_header(self) -> None:
        """uk_-style API keys must use X-API-Key only, never Authorization: Bearer."""
        from traigent.analytics.next_steps import NextStepsClient

        api_key = "uk_testkey1234567890"  # pragma: allowlist secret
        client = NextStepsClient(api_key=api_key)

        with patch("httpx.AsyncClient") as mock_async_client:
            mock_instance = MagicMock()
            mock_async_client.return_value = mock_instance

            client._get_client()

            call_kwargs = mock_async_client.call_args.kwargs
            assert call_kwargs["headers"]["X-API-Key"] == api_key
            assert "Authorization" not in call_kwargs["headers"]

    def test_get_client_jwt_string_uses_x_api_key_header(self) -> None:
        """JWT-shaped API-key strings must not be inferred as Bearer tokens."""
        from traigent.analytics.next_steps import NextStepsClient

        api_key = "eyJnotjwt.with.dots"  # pragma: allowlist secret
        client = NextStepsClient(api_key=api_key)

        with patch("httpx.AsyncClient") as mock_async_client:
            mock_instance = MagicMock()
            mock_async_client.return_value = mock_instance

            client._get_client()

            call_kwargs = mock_async_client.call_args.kwargs
            assert call_kwargs["headers"] == {"X-API-Key": api_key}
            assert "Authorization" not in call_kwargs["headers"]


@pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed")
class TestGetNextSteps:
    """Tests for get_next_steps method."""

    @pytest.mark.asyncio
    async def test_get_next_steps_success(
        self,
        valid_next_steps_payload: dict[str, object],
    ) -> None:
        from traigent.analytics.next_steps import NextStepsClient

        client = NextStepsClient(api_key="test")

        mock_response = MagicMock()
        mock_response.json.return_value = valid_next_steps_payload
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get.return_value = mock_response
        client._client = mock_http

        result = await client.get_next_steps("run_123")

        assert result == valid_next_steps_payload
        mock_http.get.assert_called_once_with(
            "/api/v1/analytics/experiments/run_123/next-steps"
        )

    @pytest.mark.asyncio
    async def test_get_next_steps_404_mentions_backend_feature(self) -> None:
        from traigent.analytics.next_steps import NextStepsClient

        client = NextStepsClient(api_key="test")
        mock_error_response = MagicMock()
        mock_error_response.status_code = 404

        error = httpx.HTTPStatusError(
            "Not Found",
            request=MagicMock(),
            response=mock_error_response,
        )

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = error

        mock_http = AsyncMock()
        mock_http.get.return_value = mock_response
        client._client = mock_http

        with pytest.raises(httpx.HTTPStatusError, match="may predate"):
            await client.get_next_steps("run_123")

    @pytest.mark.asyncio
    async def test_get_next_steps_malformed_payload_loud_error(
        self,
        valid_next_steps_payload: dict[str, object],
    ) -> None:
        from traigent.analytics.next_steps import NextStepsClient

        client = NextStepsClient(api_key="test")
        malformed = dict(valid_next_steps_payload)
        malformed.pop("next_steps")

        mock_response = MagicMock()
        mock_response.json.return_value = malformed
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get.return_value = mock_response
        client._client = mock_http

        with pytest.raises(ValueError, match="missing required key"):
            await client.get_next_steps("run_123")

    @pytest.mark.parametrize("missing_key", ["summary", "experiment_run_id"])
    async def test_get_next_steps_requires_full_contract_keys(
        self,
        valid_next_steps_payload: dict[str, object],
        missing_key: str,
    ) -> None:
        from traigent.analytics.next_steps import NextStepsClient

        client = NextStepsClient(api_key="test")
        malformed = dict(valid_next_steps_payload)
        malformed.pop(missing_key)

        mock_response = MagicMock()
        mock_response.json.return_value = malformed
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get.return_value = mock_response
        client._client = mock_http

        with pytest.raises(ValueError, match="missing required key"):
            await client.get_next_steps("run_123")
