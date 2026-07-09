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
            "/api/v1/analytics/experiments/run_123/next-steps",
            headers=None,
        )

    @pytest.mark.asyncio
    async def test_get_next_steps_preserves_optional_posture(
        self,
        valid_next_steps_payload: dict[str, object],
    ) -> None:
        from traigent.analytics.next_steps import NextStepsClient

        client = NextStepsClient(api_key="test")
        payload_with_posture = {
            **valid_next_steps_payload,
            "posture": {
                "summary_text": "Evidence is sufficient for a cautious promotion.",
                "generated_at": "2026-06-27T09:30:00Z",
            },
        }

        mock_response = MagicMock()
        mock_response.json.return_value = payload_with_posture
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get.return_value = mock_response
        client._client = mock_http

        result = await client.get_next_steps("run_123")

        assert result == payload_with_posture
        assert result["posture"] == payload_with_posture["posture"]

    @pytest.mark.asyncio
    async def test_get_next_steps_sends_guidance_variant_header_when_env_set(
        self,
        valid_next_steps_payload: dict[str, object],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from traigent.analytics.next_steps import NextStepsClient

        monkeypatch.setenv("TRAIGENT_GUIDANCE_VARIANT", "  policy  ")

        client = NextStepsClient(api_key="test")

        mock_response = MagicMock()
        decision = {
            "id": "decision_123",
            "category": "run_optimization",
            "source_engine": "policy",
            "evidence_snapshot_hash": "abc123",
            "rationale": "Run optimization.",
            "action": {"kind": "skill", "command_template": "traigent-optimize-run"},
            "evidence_level": "high",
        }
        mock_response.json.return_value = {
            **valid_next_steps_payload,
            "decision": decision,
            "next_steps": [],
            "guidance_meta": {
                "requested_variant": "policy",
                "served_variant": "policy",
                "engine": "policy",
                "fallback_reason": None,
                "decision_id": "decision_123",
                "evidence_snapshot_hash": "abc123",
            },
        }
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get.return_value = mock_response
        client._client = mock_http

        await client.get_next_steps("run_123")

        mock_http.get.assert_called_once_with(
            "/api/v1/analytics/experiments/run_123/next-steps",
            headers={"X-Traigent-Guidance-Variant": "policy"},
        )

    @pytest.mark.asyncio
    async def test_get_next_steps_omits_guidance_variant_header_when_env_unset(
        self,
        valid_next_steps_payload: dict[str, object],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from traigent.analytics.next_steps import NextStepsClient

        monkeypatch.delenv("TRAIGENT_GUIDANCE_VARIANT", raising=False)

        client = NextStepsClient(api_key="test")

        mock_response = MagicMock()
        mock_response.json.return_value = valid_next_steps_payload
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get.return_value = mock_response
        client._client = mock_http

        await client.get_next_steps("run_123")

        mock_http.get.assert_called_once_with(
            "/api/v1/analytics/experiments/run_123/next-steps",
            headers=None,
        )

    @pytest.mark.asyncio
    async def test_get_next_steps_omits_guidance_variant_header_when_env_blank(
        self,
        valid_next_steps_payload: dict[str, object],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from traigent.analytics.next_steps import NextStepsClient

        monkeypatch.setenv("TRAIGENT_GUIDANCE_VARIANT", "   ")

        client = NextStepsClient(api_key="test")

        mock_response = MagicMock()
        mock_response.json.return_value = valid_next_steps_payload
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get.return_value = mock_response
        client._client = mock_http

        await client.get_next_steps("run_123")

        mock_http.get.assert_called_once_with(
            "/api/v1/analytics/experiments/run_123/next-steps",
            headers=None,
        )

    @pytest.mark.asyncio
    async def test_get_next_steps_preserves_optional_guidance_meta(
        self,
        valid_next_steps_payload: dict[str, object],
    ) -> None:
        from traigent.analytics.next_steps import NextStepsClient

        client = NextStepsClient(api_key="test")
        decision = {
            "id": "decision_123",
            "category": "run_optimization",
            "source_engine": "rules",
            "evidence_snapshot_hash": "abc123",
            "rationale": "Run optimization.",
            "action": {"kind": "skill", "command_template": "traigent-optimize-run"},
            "evidence_level": "high",
        }
        payload_with_guidance_meta = {
            **valid_next_steps_payload,
            "decision": decision,
            "next_steps": [],
            "guidance_meta": {
                "requested_variant": "rules",
                "served_variant": "rules",
                "engine": "rules",
                "policy_table_sha": "abc123",
                "smartopt_version": "0.90.0",
                "fallback_reason": None,
                "decision_id": "decision_123",
                "evidence_snapshot_hash": "abc123",
            },
        }

        mock_response = MagicMock()
        mock_response.json.return_value = payload_with_guidance_meta
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get.return_value = mock_response
        client._client = mock_http

        result = await client.get_next_steps("run_123")

        assert result == payload_with_guidance_meta
        assert result["guidance_meta"] == payload_with_guidance_meta["guidance_meta"]

    @pytest.mark.asyncio
    async def test_explicit_guidance_variant_overrides_env_and_validates_response(
        self,
        valid_next_steps_payload: dict[str, object],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from traigent.analytics.next_steps import NextStepsClient

        monkeypatch.setenv("TRAIGENT_GUIDANCE_VARIANT", "rules")
        decision = {
            "id": "decision_123",
            "category": "run_optimization",
            "source_engine": "policy",
            "evidence_snapshot_hash": "abc123",
            "rationale": "Run optimization.",
            "action": {"kind": "skill", "command_template": "traigent-optimize-run"},
            "evidence_level": "high",
        }
        payload = {
            **valid_next_steps_payload,
            "decision": decision,
            "next_steps": [],
            "guidance_meta": {
                "requested_variant": "policy",
                "served_variant": "policy",
                "engine": "policy",
                "fallback_reason": None,
                "decision_id": "decision_123",
                "evidence_snapshot_hash": "abc123",
            },
        }
        client = NextStepsClient(api_key="test")
        response = MagicMock()
        response.json.return_value = payload
        response.raise_for_status = MagicMock()
        client._client = AsyncMock()
        client._client.get.return_value = response

        result = await client.get_next_steps("run_123", guidance_variant="policy")

        assert result == payload
        client._client.get.assert_awaited_once_with(
            "/api/v1/analytics/experiments/run_123/next-steps",
            headers={"X-Traigent-Guidance-Variant": "policy"},
        )

    @pytest.mark.asyncio
    async def test_invalid_guidance_variant_fails_before_http(
        self,
    ) -> None:
        from traigent.analytics.next_steps import NextStepsClient

        client = NextStepsClient(api_key="test")
        client._client = AsyncMock()

        with pytest.raises(ValueError, match=r"rules\|policy"):
            await client.get_next_steps("run_123", guidance_variant="variant_b")

        client._client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_explicit_blank_guidance_variant_fails_before_http(self) -> None:
        from traigent.analytics.next_steps import NextStepsClient

        client = NextStepsClient(api_key="test")
        client._client = AsyncMock()

        with pytest.raises(ValueError, match=r"rules\|policy"):
            await client.get_next_steps("run_123", guidance_variant="   ")

        client._client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_next_steps_rejects_unsafe_run_id_before_http(self) -> None:
        from traigent.analytics.next_steps import NextStepsClient

        client = NextStepsClient(api_key="test")
        client._client = AsyncMock()

        with pytest.raises(ValueError, match="experiment_run_id"):
            await client.get_next_steps("run/unsafe")

        client._client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_requested_variant_rejects_mismatched_served_variant(
        self,
        valid_next_steps_payload: dict[str, object],
    ) -> None:
        from traigent.analytics.next_steps import NextStepsClient

        client = NextStepsClient(api_key="test")
        response = MagicMock()
        response.json.return_value = {
            **valid_next_steps_payload,
            "guidance_meta": {
                "requested_variant": "policy",
                "served_variant": "rules",
                "engine": "rules",
                "fallback_reason": None,
            },
        }
        response.raise_for_status = MagicMock()
        client._client = AsyncMock()
        client._client.get.return_value = response

        with pytest.raises(ValueError, match="does not match requested"):
            await client.get_next_steps("run_123", guidance_variant="policy")

    @pytest.mark.asyncio
    async def test_strict_experiment_validates_engine_and_decision_joins(
        self,
        valid_next_steps_payload: dict[str, object],
    ) -> None:
        from traigent.analytics.next_steps import NextStepsClient

        decision = {
            "id": "decision_123",
            "category": "run_optimization",
            "source_engine": "policy",
            "evidence_snapshot_hash": "abc123",
            "rationale": "Run the next optimization.",
            "action": {
                "kind": "skill",
                "command_template": "traigent-optimize-run",
            },
            "evidence_level": "high",
        }
        payload = {
            **valid_next_steps_payload,
            "decision": decision,
            "next_steps": [],
            "guidance_meta": {
                "requested_variant": "policy",
                "served_variant": "policy",
                "engine": "policy",
                "policy_table_sha": "table-1",
                "smartopt_version": "1.0.0",
                "fallback_reason": None,
                "decision_id": "decision_123",
                "evidence_snapshot_hash": "abc123",
            },
        }
        client = NextStepsClient(api_key="test")
        response = MagicMock()
        response.json.return_value = payload
        response.raise_for_status = MagicMock()
        client._client = AsyncMock()
        client._client.get.return_value = response

        result = await client.get_next_steps(
            "run_123", guidance_variant="policy", strict_experiment=True
        )

        assert result["decision"] == decision

    @pytest.mark.asyncio
    async def test_authoritative_decision_rejects_nonempty_compatibility_steps(
        self,
        valid_next_steps_payload: dict[str, object],
    ) -> None:
        from traigent.analytics.next_steps import NextStepsClient

        decision = {
            "id": "decision_123",
            "category": "run_optimization",
            "source_engine": "policy",
            "evidence_snapshot_hash": "abc123",
            "rationale": "Run optimization.",
            "action": {"kind": "skill", "command_template": "traigent-optimize-run"},
            "evidence_level": "high",
        }
        payload = {
            **valid_next_steps_payload,
            "decision": decision,
            "next_steps": [
                {**decision, "category": "improve_evaluator", "priority": 1}
            ],
            "guidance_meta": {
                "requested_variant": "policy",
                "served_variant": "policy",
                "engine": "policy",
                "fallback_reason": None,
                "decision_id": "decision_123",
                "evidence_snapshot_hash": "abc123",
            },
        }
        client = NextStepsClient(api_key="test")
        response = MagicMock()
        response.json.return_value = payload
        response.raise_for_status = MagicMock()
        client._client = AsyncMock()
        client._client.get.return_value = response

        with pytest.raises(ValueError, match="empty next_steps"):
            await client.get_next_steps("run_123", guidance_variant="policy")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("engine", "decision_extra", "message"),
        [
            ("none", {}, "cannot produce"),
            ("rules", {"internal_state": "secret"}, "decision fields"),
        ],
    )
    async def test_authoritative_decision_rejects_engine_none_and_extra_fields(
        self,
        valid_next_steps_payload: dict[str, object],
        engine: str,
        decision_extra: dict[str, object],
        message: str,
    ) -> None:
        from traigent.analytics.next_steps import NextStepsClient

        decision = {
            "id": "decision_123",
            "category": "run_optimization",
            "source_engine": engine,
            "evidence_snapshot_hash": "abc123",
            "rationale": "Run optimization.",
            "action": {"kind": "skill", "command_template": "traigent-optimize-run"},
            "evidence_level": "high",
            **decision_extra,
        }
        payload = {
            **valid_next_steps_payload,
            "decision": decision,
            "next_steps": [],
            "guidance_meta": {
                "requested_variant": "rules",
                "served_variant": "rules",
                "engine": engine,
                "fallback_reason": "lane_error" if engine == "none" else None,
                "decision_id": "decision_123",
                "evidence_snapshot_hash": "abc123",
            },
        }
        client = NextStepsClient(api_key="test")
        response = MagicMock()
        response.json.return_value = payload
        response.raise_for_status = MagicMock()
        client._client = AsyncMock()
        client._client.get.return_value = response

        with pytest.raises(ValueError, match=message):
            await client.get_next_steps("run_123", guidance_variant="rules")

    @pytest.mark.asyncio
    async def test_authoritative_decision_requires_guidance_metadata(
        self, valid_next_steps_payload: dict[str, object]
    ) -> None:
        from traigent.analytics.next_steps import NextStepsClient

        payload = {
            **valid_next_steps_payload,
            "decision": {
                "id": "decision_123",
                "category": "run_optimization",
                "source_engine": "rules",
                "evidence_snapshot_hash": "abc123",
                "rationale": "Run optimization.",
                "action": {
                    "kind": "skill",
                    "command_template": "traigent-optimize-run",
                },
                "evidence_level": "high",
            },
            "next_steps": [],
        }
        client = NextStepsClient(api_key="test")
        response = MagicMock()
        response.json.return_value = payload
        response.raise_for_status = MagicMock()
        client._client = AsyncMock()
        client._client.get.return_value = response

        with pytest.raises(ValueError, match="omitted guidance_meta"):
            await client.get_next_steps("run_123")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("payload_override", "meta_override", "message"),
        [
            ({"internal_signal": 0.991}, {}, "top-level"),
            ({}, {"internal_state": "secret"}, "guidance_meta"),
            ({}, {"requested_variant": "__missing__"}, "requested_variant"),
            ({}, {"requested_variant": None}, "requested_variant"),
            (
                {"summary": {"confidence_label": "medium", "internal_signal": 0.99}},
                {},
                "summary",
            ),
            (
                {
                    "posture": {
                        "summary_text": "Opaque posture.",
                        "generated_at": "2026-07-10T00:00:00Z",
                        "internal_state": "secret",
                    }
                },
                {},
                "posture",
            ),
            ({}, {"policy_table_sha": {"secret": 1}}, "policy_table_sha"),
            ({"schema_version": 1}, {}, "schema_version"),
            ({"caveat": {"secret": 1}}, {}, "caveat"),
            ({"posture": None}, {}, "posture"),
            ({"decision": None}, {}, "decision"),
        ],
    )
    async def test_guidance_response_rejects_non_public_or_missing_metadata(
        self,
        valid_next_steps_payload: dict[str, object],
        payload_override: dict[str, object],
        meta_override: dict[str, object],
        message: str,
    ) -> None:
        from traigent.analytics.next_steps import NextStepsClient

        meta = {
            "requested_variant": "rules",
            "served_variant": "rules",
            "engine": "none",
            "policy_table_sha": None,
            "smartopt_version": "test",
            "fallback_reason": "lane_error",
            "decision_id": None,
            "evidence_snapshot_hash": None,
            **meta_override,
        }
        if meta_override.get("requested_variant") == "__missing__":
            meta.pop("requested_variant")
        payload = {
            **valid_next_steps_payload,
            "next_steps": [],
            "guidance_meta": meta,
            **payload_override,
        }
        client = NextStepsClient(api_key="test")
        response = MagicMock()
        response.json.return_value = payload
        response.raise_for_status = MagicMock()
        client._client = AsyncMock()
        client._client.get.return_value = response

        with pytest.raises(ValueError, match=message):
            await client.get_next_steps("run_123")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("response_run_id", ["other_run", "../admin", ""])
    async def test_response_run_id_must_be_safe_and_join_request(
        self,
        valid_next_steps_payload: dict[str, object],
        response_run_id: str,
    ) -> None:
        from traigent.analytics.next_steps import NextStepsClient

        payload = {**valid_next_steps_payload, "experiment_run_id": response_run_id}
        client = NextStepsClient(api_key="test")
        response = MagicMock()
        response.json.return_value = payload
        response.raise_for_status = MagicMock()
        client._client = AsyncMock()
        client._client.get.return_value = response

        with pytest.raises(ValueError, match="experiment_run_id"):
            await client.get_next_steps("run_123")

    @pytest.mark.asyncio
    async def test_strict_experiment_accepts_non_executable_wait(
        self,
        valid_next_steps_payload: dict[str, object],
    ) -> None:
        from traigent.analytics.next_steps import NextStepsClient

        decision = {
            "id": "decision_wait",
            "category": "wait",
            "source_engine": "rules",
            "evidence_snapshot_hash": "abc123",
            "rationale": "Wait for more evidence.",
            "action": {"kind": "none", "command_template": ""},
            "evidence_level": "low",
        }
        payload = {
            **valid_next_steps_payload,
            "decision": decision,
            "next_steps": [],
            "guidance_meta": {
                "requested_variant": "rules",
                "served_variant": "rules",
                "engine": "rules",
                "fallback_reason": None,
                "decision_id": "decision_wait",
                "evidence_snapshot_hash": "abc123",
            },
        }
        client = NextStepsClient(api_key="test")
        response = MagicMock()
        response.json.return_value = payload
        response.raise_for_status = MagicMock()
        client._client = AsyncMock()
        client._client.get.return_value = response

        result = await client.get_next_steps(
            "run_123", guidance_variant="rules", strict_experiment=True
        )

        assert result["decision"] == decision

    @pytest.mark.asyncio
    async def test_strict_experiment_rejects_policy_fallback(
        self,
        valid_next_steps_payload: dict[str, object],
    ) -> None:
        from traigent.analytics.next_steps import NextStepsClient

        client = NextStepsClient(api_key="test")
        response = MagicMock()
        response.json.return_value = {
            **valid_next_steps_payload,
            "guidance_meta": {
                "requested_variant": "policy",
                "served_variant": "policy",
                "engine": "rules",
                "fallback_reason": "unmapped_state",
            },
        }
        response.raise_for_status = MagicMock()
        client._client = AsyncMock()
        client._client.get.return_value = response

        with pytest.raises(ValueError, match="actual guidance engine"):
            await client.get_next_steps(
                "run_123", guidance_variant="policy", strict_experiment=True
            )

    @pytest.mark.asyncio
    async def test_record_decision_receipt_posts_idempotent_outcome_contract(
        self,
    ) -> None:
        from traigent.analytics.next_steps import NextStepsClient

        client = NextStepsClient(api_key="test")
        response = MagicMock()
        response.json.return_value = {
            "receipt_id": "receipt_1",
            "decision_id": "decision_1",
            "attempt_id": "attempt_1",
            "status": "completed",
            "successor_run_id": "run_456",
            "outcomes": {"holdout_status": "passed"},
            "created_at": "2026-07-09T11:59:00Z",
            "updated_at": "2026-07-09T12:00:00Z",
        }
        response.raise_for_status = MagicMock()
        client._client = AsyncMock()
        client._client.post.return_value = response

        result = await client.record_decision_receipt(
            "run_123",
            decision_id="decision_1",
            attempt_id="attempt_1",
            status="completed",
            successor_run_id="run_456",
            holdout_status="passed",
        )

        assert result["receipt_id"] == "receipt_1"
        client._client.post.assert_awaited_once_with(
            "/api/v1/analytics/experiments/run_123/next-steps/decision_1/receipt",
            json={
                "status": "completed",
                "attempt_id": "attempt_1",
                "successor_run_id": "run_456",
                "outcomes": {"holdout_status": "passed"},
            },
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("override", "message"),
        [
            ({"receipt_id": "x" * 201}, "receipt_id"),
            ({"decision_id": "decision_other"}, "decision_id"),
            ({"attempt_id": "attempt_other"}, "attempt_id"),
            ({"status": "failed"}, "status"),
            ({"successor_run_id": "run_other"}, "successor_run_id"),
            ({"outcomes": {"holdout_status": "passed"}}, "outcomes"),
            ({"created_at": "not-a-date"}, "created_at"),
            ({"updated_at": "2026-07-09T12:00:00"}, "timezone"),
            ({"unexpected": "field"}, "unexpected key"),
        ],
    )
    async def test_receipt_response_fails_closed_on_invalid_or_unjoined_fields(
        self, override: dict[str, object], message: str
    ) -> None:
        from traigent.analytics.next_steps import NextStepsClient

        payload = {
            "receipt_id": "receipt_1",
            "decision_id": "decision_1",
            "attempt_id": "attempt_1",
            "status": "completed",
            "successor_run_id": None,
            "outcomes": {},
            "created_at": "2026-07-09T11:59:00Z",
            "updated_at": "2026-07-09T12:00:00Z",
            **override,
        }
        client = NextStepsClient(api_key="test")
        response = MagicMock()
        response.json.return_value = payload
        response.raise_for_status = MagicMock()
        client._client = AsyncMock()
        client._client.post.return_value = response

        with pytest.raises(ValueError, match=message):
            await client.record_decision_receipt(
                "run_123",
                decision_id="decision_1",
                attempt_id="attempt_1",
                status="completed",
            )

    @pytest.mark.asyncio
    async def test_receipt_rejects_outcomes_before_completion(self) -> None:
        from traigent.analytics.next_steps import NextStepsClient

        client = NextStepsClient(api_key="test")
        client._client = AsyncMock()

        with pytest.raises(ValueError, match="require status='completed'"):
            await client.record_decision_receipt(
                "run_123",
                decision_id="decision_1",
                attempt_id="attempt_1",
                status="started",
                safety_gate_status="passed",
            )
        client._client.post.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("run_id", "decision_id", "message"),
        [
            ("run/unsafe", "decision_1", "experiment_run_id"),
            ("run_123", "decision/unsafe", "decision_id"),
        ],
    )
    async def test_receipt_rejects_unsafe_path_ids(
        self, run_id: str, decision_id: str, message: str
    ) -> None:
        from traigent.analytics.next_steps import NextStepsClient

        client = NextStepsClient(api_key="test")
        client._client = AsyncMock()

        with pytest.raises(ValueError, match=message):
            await client.record_decision_receipt(
                run_id,
                decision_id=decision_id,
                attempt_id="attempt_1",
                status="started",
            )
        client._client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_next_steps_without_guidance_meta_is_unchanged(
        self,
        valid_next_steps_payload: dict[str, object],
    ) -> None:
        """Backward compat: payloads without guidance_meta behave as before."""
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
        assert "guidance_meta" not in result

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
    async def test_get_next_steps_surfaces_variant_assignment_conflict(self) -> None:
        from traigent.analytics.next_steps import (
            GuidanceVariantConflictError,
            NextStepsClient,
        )

        client = NextStepsClient(api_key="test")
        conflict_response = MagicMock()
        conflict_response.status_code = 409
        error = httpx.HTTPStatusError(
            "Conflict",
            request=MagicMock(),
            response=conflict_response,
        )
        response = MagicMock()
        response.raise_for_status.side_effect = error
        client._client = AsyncMock()
        client._client.get.return_value = response

        with pytest.raises(GuidanceVariantConflictError, match="already pinned"):
            await client.get_next_steps("run_123", guidance_variant="policy")

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
