"""Tests for trial_operations.py - particularly new code paths."""

import json
import logging
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from traigent.cloud.trial_operations import (
    TrialOperations,
    TrialSlotResult,
    TrialSubmissionResult,
)
from traigent.core.metadata_helpers import build_backend_metadata


class TestRedactSensitiveFields:
    """Tests for _redact_sensitive_fields method."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_client = Mock()
        mock_client.backend_config = Mock()
        mock_client.backend_config.backend_base_url = "https://api.example.com"
        mock_client.auth_manager = Mock()
        self.ops = TrialOperations(mock_client)

    def test_redact_string_api_key(self):
        """Test that string API keys are redacted."""
        data = {
            "api_key": "sk-1234567890abcdef"  # pragma: allowlist secret
        }  # noqa: S105
        result = TrialOperations._redact_sensitive_fields(data)
        assert "[REDACTED:" in result["api_key"]
        assert "chars]" in result["api_key"]

    def test_redact_list_api_key(self):
        """Test that list API keys are redacted."""
        data = {"apikey": ["key1", "key2", "key3"]}
        result = TrialOperations._redact_sensitive_fields(data)
        assert result["apikey"] == "[REDACTED:list]"

    def test_redact_dict_password(self):
        """Test that dict passwords are redacted."""
        data = {"password": {"nested": "secret"}}
        result = TrialOperations._redact_sensitive_fields(data)
        assert result["password"] == "[REDACTED:dict]"

    def test_redact_tuple_secret(self):
        """Test that tuple secrets are redacted."""
        data = {"secret": ("a", "b", "c")}
        result = TrialOperations._redact_sensitive_fields(data)
        assert result["secret"] == "[REDACTED:tuple]"

    def test_redact_set_credentials(self):
        """Test that set credentials are redacted."""
        data = {"credentials": {"cred1", "cred2"}}
        result = TrialOperations._redact_sensitive_fields(data)
        assert result["credentials"] == "[REDACTED:set]"

    def test_redact_other_sensitive_type(self):
        """Test that other sensitive types are redacted with generic message."""
        data = {"token": 12345}  # Numeric token
        result = TrialOperations._redact_sensitive_fields(data)
        assert result["token"] == "[REDACTED]"

    def test_non_sensitive_fields_preserved(self):
        """Test that non-sensitive fields are preserved."""
        data = {"name": "test", "count": 42, "enabled": True}
        result = TrialOperations._redact_sensitive_fields(data)
        assert result["name"] == "test"
        assert result["count"] == 42
        assert result["enabled"] is True

    def test_nested_redaction(self):
        """Test that nested structures are processed."""
        data = {
            "config": {
                "api_key": "secret123",  # pragma: allowlist secret
                "name": "test",
            }
        }
        result = TrialOperations._redact_sensitive_fields(data)
        assert "[REDACTED:" in result["config"]["api_key"]
        assert result["config"]["name"] == "test"


class TestCreateLocalhostConnector:
    """Tests for _create_localhost_connector method."""

    def test_returns_none(self):
        """Test that connector creation returns None (simplified implementation)."""
        mock_client = Mock()
        mock_client.backend_config = Mock()
        mock_client.backend_config.backend_base_url = "http://localhost:8000"
        mock_client.auth_manager = Mock()

        ops = TrialOperations(mock_client)
        connector = ops._create_localhost_connector()
        assert connector is None

    def test_returns_none_for_remote_url(self):
        """Test that connector returns None for remote URLs too."""
        mock_client = Mock()
        mock_client.backend_config = Mock()
        mock_client.backend_config.backend_base_url = "https://api.example.com"
        mock_client.auth_manager = Mock()

        ops = TrialOperations(mock_client)
        connector = ops._create_localhost_connector()
        assert connector is None


class TestBackendDescription:
    """Tests for backend context emitted in failure logs."""

    def test_describe_backend_omits_api_key_preview(self):
        """Backend context must not include masked or raw credential material."""
        mock_client = Mock()
        mock_client.backend_config = Mock()
        mock_client.backend_config.backend_base_url = "https://api.example.com"
        mock_client.auth_manager = Mock()
        mock_client.auth_manager.auth = Mock()
        mock_client.auth_manager.auth.get_api_key_preview = Mock(
            return_value="tg_a****xyz1"
        )

        ops = TrialOperations(mock_client)

        description = ops._describe_backend()
        fields = dict(part.split("=", 1) for part in description.split(", "))

        assert fields["backend_url"] == "https://api.example.com"
        assert "api_key" not in description
        assert "tg_a" not in description


class TestMeasuresDictValidationInSubmission:
    """Test MeasuresDict validation warning path in submit_trial_result_via_session."""

    def test_trial_result_metadata_uses_sdk_version_and_drops_execution_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Trial result metadata uses SDK version and does not emit raw execution_mode."""
        monkeypatch.setenv("TRAIGENT_FORCE_VERSION", "9.8.6")
        mock_client = Mock()
        mock_client.backend_config = Mock()
        mock_client.auth_manager = Mock()
        ops = TrialOperations(mock_client)

        result_data = ops._build_trial_result_data(
            trial_id="trial_001",
            config={"temperature": 0.2},
            clean_metrics={"accuracy": 0.95},
            backend_status="COMPLETED",
            mode="local",
            metadata={"execution_mode": "local", "custom": "value"},
        )

        metadata = result_data["metadata"]
        assert metadata["mode"] == "local"
        assert metadata["sdk_version"] == "9.8.6"
        assert metadata["custom"] == "value"
        assert "execution_mode" not in metadata

    def test_trial_result_data_snapshots_live_submission_dicts(self) -> None:
        """Submission payload dicts are detached before serialization."""
        mock_client = Mock()
        mock_client.backend_config = Mock()
        mock_client.auth_manager = Mock()
        ops = TrialOperations(mock_client)
        config = {"temperature": 0.2, "routing": {"top_p": 0.8}}
        trial_result = SimpleNamespace(
            trial_id="trial_001",
            config=config,
            duration=1.5,
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            metrics={
                "accuracy": 0.95,
                "surrogate_score": 0.5,
                "nested_metric": {"before": 1},
            },
            metadata={
                "comparability": {
                    "per_metric_coverage": {"accuracy": {"present": 1}},
                },
                "surrogate_evaluator": {
                    "config": {"thresholds": {"minimum": 0.2}},
                },
            },
            summary_stats=None,
        )
        traigent_config = SimpleNamespace(
            execution_mode="edge_analytics",
            minimal_logging=False,
            privacy_enabled=False,
            execution_mode_enum=SimpleNamespace(value="local"),
        )
        backend_metadata = build_backend_metadata(
            trial_result,
            "accuracy",
            traigent_config,
        )
        clean_metrics = {"accuracy": 0.95, "nested": {"before": 1}}

        result_data = ops._build_trial_result_data(
            trial_id="trial_001",
            config=config,
            clean_metrics=clean_metrics,
            backend_status="COMPLETED",
            mode="local",
            metadata=backend_metadata,
        )

        config["routing"]["top_p"] = 0.1
        trial_result.metrics["surrogate_score"] = 0.9
        trial_result.metrics["nested_metric"]["before"] = 2
        trial_result.metadata["comparability"]["per_metric_coverage"]["accuracy"][
            "present"
        ] = 2
        trial_result.metadata["surrogate_evaluator"]["config"]["thresholds"][
            "minimum"
        ] = 0.8
        backend_metadata["all_metrics"]["accuracy"] = 0.1
        clean_metrics["nested"]["before"] = 2

        assert result_data["config"] == {
            "temperature": 0.2,
            "routing": {"top_p": 0.8},
        }
        assert result_data["metrics"] == {
            "accuracy": 0.95,
            "nested": {"before": 1},
        }
        assert result_data["metadata"]["all_metrics"] == {
            "accuracy": 0.95,
            "surrogate_score": 0.5,
            "nested_metric": {"before": 1},
        }
        assert (
            result_data["metadata"]["comparability"]["per_metric_coverage"]["accuracy"][
                "present"
            ]
            == 1
        )
        assert (
            result_data["metadata"]["surrogate_evaluator"]["config"]["thresholds"][
                "minimum"
            ]
            == 0.2
        )

    @pytest.mark.asyncio
    async def test_hybrid_trial_submission_uses_session_results_endpoint(self) -> None:
        """Hybrid/backend tracking posts trial metrics to the session results API."""
        mock_client = Mock()
        mock_client.backend_config = Mock()
        mock_client.backend_config.backend_base_url = (
            "https://api.example.com"  # pragma: allowlist secret
        )
        mock_client.backend_config.api_base_url = "https://api.example.com/api/v1"
        mock_client.auth_manager = AsyncMock()
        mock_client.auth_manager.augment_headers = AsyncMock(return_value={})
        mock_client._map_to_backend_status = Mock(return_value="COMPLETED")
        mock_client._normalize_execution_mode = Mock(return_value="local")
        mock_client._sanitize_error_message = Mock(return_value="")

        ops = TrialOperations(mock_client)
        ops._handle_trial_success_response = AsyncMock(return_value=True)

        mock_response = Mock()
        mock_response.status = 201

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = Mock(return_value=mock_post_ctx)

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "traigent.cloud.trial_operations.is_backend_offline",
                return_value=False,
            ),
            patch("traigent.cloud.trial_operations.AIOHTTP_AVAILABLE", True),
            patch(
                "traigent.cloud.trial_operations.validate_configuration_run_submission",
            ),
            patch("traigent.cloud.trial_operations.aiohttp") as mock_aiohttp,
        ):
            mock_aiohttp.ClientSession = Mock(return_value=mock_session_ctx)
            mock_aiohttp.ClientTimeout = Mock()

            result = await ops.submit_trial_result_via_session(
                session_id="session_123",
                trial_id="trial_001",
                config={"temperature": 0.2},
                metrics={"accuracy": 0.95},
                status="completed",
                execution_mode="hybrid",
            )

        assert result is True
        assert (
            mock_session.post.call_args.args[0]
            == "https://api.example.com/api/v1/sessions/session_123/results"
        )
        mock_client._normalize_execution_mode.assert_called_once_with("hybrid")

    @pytest.mark.asyncio
    async def test_failed_trial_error_message_wire_key_is_error_message(self) -> None:
        """Traigent#1724: a failed trial's error must ride under "error_message".

        The backend route and TraigentCloudClient both read "error_message";
        a wire key of "error" is silently dropped, so every failed-trial
        message was lost before this fix.
        """
        mock_client = Mock()
        mock_client.backend_config = Mock()
        mock_client.backend_config.backend_base_url = (
            "https://api.example.com"  # pragma: allowlist secret
        )
        mock_client.backend_config.api_base_url = "https://api.example.com/api/v1"
        mock_client.auth_manager = AsyncMock()
        mock_client.auth_manager.augment_headers = AsyncMock(return_value={})
        mock_client._map_to_backend_status = Mock(return_value="FAILED")
        mock_client._normalize_execution_mode = Mock(return_value="local")
        mock_client._sanitize_error_message = Mock(
            return_value="evaluation raised ValueError"
        )

        ops = TrialOperations(mock_client)
        ops._handle_trial_success_response = AsyncMock(return_value=True)

        mock_response = Mock()
        mock_response.status = 201

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = Mock(return_value=mock_post_ctx)

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "traigent.cloud.trial_operations.is_backend_offline",
                return_value=False,
            ),
            patch("traigent.cloud.trial_operations.AIOHTTP_AVAILABLE", True),
            patch(
                "traigent.cloud.trial_operations.validate_configuration_run_submission",
            ),
            patch("traigent.cloud.trial_operations.aiohttp") as mock_aiohttp,
        ):
            mock_aiohttp.ClientSession = Mock(return_value=mock_session_ctx)
            mock_aiohttp.ClientTimeout = Mock()

            result = await ops.submit_trial_result_via_session(
                session_id="session_123",
                trial_id="trial_001",
                config={"temperature": 0.2},
                metrics={},
                status="failed",
                error_message="evaluation raised ValueError",
            )

        assert result is True
        posted_payload = mock_session.post.call_args.kwargs["json"]
        assert posted_payload["error_message"] == "evaluation raised ValueError"
        assert "error" not in posted_payload

    @pytest.mark.asyncio
    async def test_invalid_configuration_run_submission_does_not_post(self) -> None:
        """Schema validation failure must stop trial result network submission."""
        mock_client = Mock()
        mock_client.backend_config = Mock()
        mock_client.backend_config.backend_base_url = "https://api.example.com"
        mock_client.backend_config.api_base_url = "https://api.example.com"
        mock_client.auth_manager = AsyncMock()
        mock_client.auth_manager.augment_headers = AsyncMock(return_value={})
        mock_client._map_to_backend_status = Mock(return_value="completed")
        mock_client._normalize_execution_mode = Mock(return_value="local")
        mock_client._sanitize_error_message = Mock(return_value="")

        ops = TrialOperations(mock_client)

        with (
            patch(
                "traigent.cloud.trial_operations.is_backend_offline",
                return_value=False,
            ),
            patch("traigent.cloud.trial_operations.AIOHTTP_AVAILABLE", True),
            patch(
                "traigent.cloud.trial_operations.validate_configuration_run_submission",
                side_effect=ValueError("invalid payload"),
            ),
            patch("traigent.cloud.trial_operations.aiohttp") as mock_aiohttp,
        ):
            result = await ops.submit_trial_result_via_session(
                session_id="test-session",
                trial_id="test-trial",
                config={"model": "gpt-4"},
                metrics={"accuracy": 0.95},
                status="completed",
            )

        assert result is False
        mock_client.auth_manager.augment_headers.assert_not_awaited()
        mock_aiohttp.ClientSession.assert_not_called()

    @pytest.mark.asyncio
    async def test_invalid_metric_key_rejects_submission(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Traigent#1724: invalid metric keys hard-fail the submission.

        MeasuresDict is a hard contract mirroring the schema validation a few
        lines below it — a trial whose measures violate the contract must be
        rejected, not silently submitted with unvalidated numeric metrics
        (the previous fail-open behavior).
        """
        mock_client = Mock()
        mock_client.backend_config = Mock()
        mock_client.backend_config.backend_base_url = (
            "https://api.example.com"  # pragma: allowlist secret
        )
        mock_client.backend_config.api_base_url = "https://api.example.com"
        mock_client.auth_manager = AsyncMock()
        mock_client.auth_manager.augment_headers = AsyncMock(return_value={})
        mock_client._map_to_backend_status = Mock(return_value="completed")
        mock_client._normalize_execution_mode = Mock(return_value="local")
        mock_client._sanitize_error_message = Mock(return_value="")

        ops = TrialOperations(mock_client)
        ops._handle_trial_success_response = AsyncMock(return_value=True)

        # Use a metric key with a hyphen — MeasuresDict rejects non-identifier keys
        invalid_metrics = {"invalid-key": 0.95}

        with (
            patch(
                "traigent.cloud.trial_operations.is_backend_offline",
                return_value=False,
            ),
            patch(
                "traigent.cloud.trial_operations.AIOHTTP_AVAILABLE",
                True,
            ),
            patch("traigent.cloud.trial_operations.aiohttp") as mock_aiohttp,
        ):
            with caplog.at_level(
                logging.ERROR, logger="traigent.cloud.trial_operations"
            ):
                result = await ops.submit_trial_result_via_session(
                    session_id="test-session",
                    trial_id="test-trial",
                    config={"model": "gpt-4"},
                    metrics=invalid_metrics,
                    status="completed",
                )

            # Rejected outright — no silent fallback to unvalidated metrics.
            assert result is False
            assert any("Invalid trial metrics" in msg for msg in caplog.messages)
            # Validation failure happens before any network call is attempted.
            mock_aiohttp.ClientSession.assert_not_called()

    @pytest.mark.asyncio
    async def test_transport_fields_do_not_trigger_measuresdict_warnings(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """summary_stats/measures should be extracted before MeasuresDict validation."""
        mock_client = Mock()
        mock_client.backend_config = Mock()
        mock_client.backend_config.backend_base_url = (
            "https://api.example.com"  # pragma: allowlist secret
        )
        mock_client.backend_config.api_base_url = "https://api.example.com"
        mock_client.auth_manager = AsyncMock()
        mock_client.auth_manager.augment_headers = AsyncMock(return_value={})
        mock_client._map_to_backend_status = Mock(return_value="completed")
        mock_client._normalize_execution_mode = Mock(return_value="local")
        mock_client._sanitize_error_message = Mock(return_value="")

        ops = TrialOperations(mock_client)
        ops._handle_trial_success_response = AsyncMock(return_value=True)

        metrics = {
            "accuracy": 0.95,
            "summary_stats": {"metrics": {"accuracy": 0.95}},
            "measures": [
                {
                    "example_id": "ex_123",
                    "metrics": {"accuracy": 0.95},
                }
            ],
        }

        mock_response = Mock()
        mock_response.status = 200

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = Mock(return_value=mock_post_ctx)

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "traigent.cloud.trial_operations.is_backend_offline",
                return_value=False,
            ),
            patch(
                "traigent.cloud.trial_operations.AIOHTTP_AVAILABLE",
                True,
            ),
            patch(
                "traigent.cloud.trial_operations.validate_configuration_run_submission",
            ),
            patch("traigent.cloud.trial_operations.aiohttp") as mock_aiohttp,
        ):
            mock_aiohttp.ClientSession = Mock(return_value=mock_session_ctx)
            mock_aiohttp.ClientTimeout = Mock()

            with caplog.at_level(logging.WARNING):
                result = await ops.submit_trial_result_via_session(
                    session_id="test-session",
                    trial_id="test-trial",
                    config={"model": "gpt-4"},
                    metrics=metrics,
                    status="completed",
                )

            assert result is True
            assert not any(
                "Measure 'summary_stats' has non-numeric value type dict" in msg
                for msg in caplog.messages
            )
            assert not any(
                "Measure 'measures' has non-numeric value type list" in msg
                for msg in caplog.messages
            )

            call_args = ops._handle_trial_success_response.call_args
            assert call_args is not None
            assert call_args.args[5] == {"accuracy": 0.95}

    @pytest.mark.asyncio
    async def test_summary_stats_metadata_uses_sdk_version_without_execution_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Summary stats submission metadata uses package version and omits execution_mode."""
        monkeypatch.setenv("TRAIGENT_FORCE_VERSION", "8.7.6")
        mock_client = Mock()
        mock_client.backend_config = Mock()
        mock_client.backend_config.backend_base_url = "https://api.example.com"
        mock_client.backend_config.api_base_url = "https://api.example.com"
        mock_client.auth_manager = AsyncMock()
        mock_client.auth_manager.augment_headers = AsyncMock(return_value={})
        mock_client._map_to_backend_status = Mock(return_value="completed")

        ops = TrialOperations(mock_client)

        mock_response = Mock()
        mock_response.status = 201

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = Mock(return_value=mock_post_ctx)

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "traigent.cloud.trial_operations.is_backend_offline",
                return_value=False,
            ),
            patch("traigent.cloud.trial_operations.AIOHTTP_AVAILABLE", True),
            patch(
                "traigent.cloud.trial_operations.validate_configuration_run_submission",
            ),
            patch("traigent.cloud.trial_operations.aiohttp") as mock_aiohttp,
        ):
            mock_aiohttp.ClientSession = Mock(return_value=mock_session_ctx)
            mock_aiohttp.ClientTimeout = Mock()

            result = await ops.submit_summary_stats(
                session_id="test-session",
                trial_id="test-trial",
                config={"model": "gpt-4"},
                summary_stats={"metrics": {"accuracy": 0.95}},
                status="completed",
            )

        assert result is True
        payload = mock_session.post.call_args.kwargs["json"]
        metadata = payload["metadata"]
        assert metadata["mode"] == "privacy"
        assert metadata["sdk_version"] == "8.7.6"
        assert "execution_mode" not in metadata


class TestSummaryStatsValidation:
    """Tests for fail-closed summary_stats submission validation."""

    @pytest.mark.asyncio
    async def test_invalid_summary_stats_submission_does_not_post(self) -> None:
        """Schema validation failure must stop summary stats network submission."""
        mock_client = Mock()
        mock_client.backend_config = Mock()
        mock_client.backend_config.backend_base_url = "https://api.example.com"
        mock_client.backend_config.api_base_url = "https://api.example.com"
        mock_client.auth_manager = AsyncMock()
        mock_client.auth_manager.augment_headers = AsyncMock(return_value={})
        mock_client._map_to_backend_status = Mock(return_value="completed")

        ops = TrialOperations(mock_client)

        with (
            patch(
                "traigent.cloud.trial_operations.is_backend_offline",
                return_value=False,
            ),
            patch("traigent.cloud.trial_operations.AIOHTTP_AVAILABLE", True),
            patch(
                "traigent.cloud.trial_operations.validate_configuration_run_submission",
                side_effect=ValueError("invalid summary"),
            ),
            patch("traigent.cloud.trial_operations.aiohttp") as mock_aiohttp,
        ):
            result = await ops.submit_summary_stats(
                session_id="test-session",
                trial_id="test-trial",
                config={"model": "gpt-4"},
                summary_stats={"metrics": {"accuracy": 0.95}},
                status="completed",
            )

        assert result is False
        mock_client.auth_manager.augment_headers.assert_not_awaited()
        mock_aiohttp.ClientSession.assert_not_called()


class TestPrivacyConfigRedactionSubmission:
    """Privacy-mode config redaction must apply to the actual POST payload."""

    @staticmethod
    def _make_ops(privacy_enabled: bool = False) -> TrialOperations:
        mock_client = Mock()
        mock_client.backend_config = Mock()
        mock_client.backend_config.backend_base_url = "https://api.example.com"
        mock_client.backend_config.api_base_url = "https://api.example.com"
        mock_client.traigent_config = SimpleNamespace(privacy_enabled=privacy_enabled)
        mock_client.auth_manager = AsyncMock()
        mock_client.auth_manager.augment_headers = AsyncMock(return_value={})
        mock_client._map_to_backend_status = Mock(return_value="completed")
        mock_client._normalize_execution_mode = Mock(return_value="hybrid")
        mock_client._sanitize_error_message = Mock(return_value="")
        return TrialOperations(mock_client)

    @staticmethod
    def _install_post_mock(mock_aiohttp: Mock, status: int = 201) -> Mock:
        mock_response = Mock()
        mock_response.status = status
        mock_response.text = AsyncMock(return_value="")

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = Mock(return_value=mock_post_ctx)

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_aiohttp.ClientSession = Mock(return_value=mock_session_ctx)
        mock_aiohttp.ClientTimeout = Mock()
        return mock_session

    @pytest.mark.asyncio
    async def test_privacy_summary_stats_redacts_sensitive_config_values(
        self,
    ) -> None:
        sentinel = "SENTINEL-PII-8842"
        ops = self._make_ops()

        with (
            patch(
                "traigent.cloud.trial_operations.is_backend_offline",
                return_value=False,
            ),
            patch("traigent.cloud.trial_operations.AIOHTTP_AVAILABLE", True),
            patch(
                "traigent.cloud.trial_operations.validate_configuration_run_submission",
            ),
            patch("traigent.cloud.trial_operations.aiohttp") as mock_aiohttp,
        ):
            mock_session = self._install_post_mock(mock_aiohttp)

            result = await ops.submit_summary_stats(
                session_id="test-session",
                trial_id="test-trial",
                config={"system_prompt": sentinel, "temperature": 0.7},
                summary_stats={"metrics": {"accuracy": 0.95}},
                status="completed",
            )

        assert result is True
        payload = mock_session.post.call_args.kwargs["json"]
        assert "system_prompt" in payload["config"]
        assert payload["config"]["system_prompt"] == "[REDACTED:17 chars]"
        assert payload["config"]["temperature"] == 0.7
        assert sentinel not in str(payload)

    @pytest.mark.asyncio
    async def test_privacy_summary_stats_redacts_unknown_string_config_keys(
        self,
    ) -> None:
        sentinel = "SENTINEL-PII-8842"
        ops = self._make_ops()

        with (
            patch(
                "traigent.cloud.trial_operations.is_backend_offline",
                return_value=False,
            ),
            patch("traigent.cloud.trial_operations.AIOHTTP_AVAILABLE", True),
            patch(
                "traigent.cloud.trial_operations.validate_configuration_run_submission",
            ),
            patch("traigent.cloud.trial_operations.aiohttp") as mock_aiohttp,
        ):
            mock_session = self._install_post_mock(mock_aiohttp)

            result = await ops.submit_summary_stats(
                session_id="test-session",
                trial_id="test-trial",
                config={
                    "persona_instructions": sentinel,
                    "temperature": 0.2,
                    "enabled": True,
                    "optional": None,
                },
                summary_stats={"metrics": {"accuracy": 0.95}},
                status="completed",
            )

        assert result is True
        payload = mock_session.post.call_args.kwargs["json"]
        assert payload["config"]["persona_instructions"] == "[REDACTED:17 chars]"
        assert payload["config"]["temperature"] == 0.2
        assert payload["config"]["enabled"] is True
        assert payload["config"]["optional"] is None
        assert sentinel not in str(payload)

    @pytest.mark.asyncio
    async def test_privacy_trial_start_redacts_config_payload(self) -> None:
        sentinel = "SENTINEL-PII-8842"
        ops = self._make_ops(privacy_enabled=True)

        with (
            patch(
                "traigent.cloud.trial_operations.is_backend_offline",
                return_value=False,
            ),
            patch("traigent.cloud.trial_operations.AIOHTTP_AVAILABLE", True),
            patch("traigent.cloud.trial_operations.aiohttp") as mock_aiohttp,
        ):
            mock_session = self._install_post_mock(mock_aiohttp)

            result = await ops.register_trial_start(
                session_id="test-session",
                trial_id="test-trial",
                config={"system_prompt": sentinel, "temperature": 0.7},
            )

        assert result is True
        payload = mock_session.post.call_args.kwargs["json"]
        assert payload["config"]["system_prompt"] == "[REDACTED:17 chars]"
        assert payload["config"]["temperature"] == 0.7
        assert sentinel not in str(payload)

    @pytest.mark.asyncio
    async def test_default_trial_submission_keeps_tuned_config_values(
        self,
    ) -> None:
        sentinel = "SENTINEL-PII-8842"
        ops = self._make_ops()
        ops._handle_trial_success_response = AsyncMock(return_value=True)

        with (
            patch(
                "traigent.cloud.trial_operations.is_backend_offline",
                return_value=False,
            ),
            patch("traigent.cloud.trial_operations.AIOHTTP_AVAILABLE", True),
            patch(
                "traigent.cloud.trial_operations.validate_configuration_run_submission",
            ),
            patch("traigent.cloud.trial_operations.aiohttp") as mock_aiohttp,
        ):
            mock_session = self._install_post_mock(mock_aiohttp)

            result = await ops.submit_trial_result_via_session(
                session_id="test-session",
                trial_id="test-trial",
                config={"system_prompt": sentinel, "temperature": 0.7},
                metrics={"accuracy": 0.95},
                status="completed",
                execution_mode="hybrid",
            )

        assert result is True
        payload = mock_session.post.call_args.kwargs["json"]
        assert payload["config"]["system_prompt"] == sentinel
        assert payload["config"]["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_privacy_summary_stats_fails_closed_when_redaction_fails(
        self,
    ) -> None:
        ops = self._make_ops()

        with (
            patch(
                "traigent.cloud.trial_operations.is_backend_offline",
                return_value=False,
            ),
            patch("traigent.cloud.trial_operations.AIOHTTP_AVAILABLE", True),
            patch.object(
                TrialOperations,
                "_redact_privacy_config",
                side_effect=RuntimeError("redaction failed"),
            ),
            patch("traigent.cloud.trial_operations.aiohttp") as mock_aiohttp,
        ):
            result = await ops.submit_summary_stats(
                session_id="test-session",
                trial_id="test-trial",
                config={"system_prompt": "SENTINEL-PII-8842"},
                summary_stats={"metrics": {"accuracy": 0.95}},
                status="completed",
            )

        assert result is False
        mock_aiohttp.ClientSession.assert_not_called()


class TestWeightedScoreUpdates:
    """Tests for weighted-score backend update accounting."""

    @pytest.mark.asyncio
    async def test_missing_configuration_run_is_not_counted_as_update(self) -> None:
        """Backend not-found responses should not report an applied update."""
        mock_client = Mock()
        mock_client.backend_config = Mock()
        mock_client.backend_config.backend_base_url = "https://api.example.com"
        mock_client.backend_config.api_base_url = "https://api.example.com"
        mock_client.auth_manager = AsyncMock()
        mock_client.auth_manager.augment_headers = AsyncMock(return_value={})

        ops = TrialOperations(mock_client)

        mock_response = Mock()
        mock_response.status = 400
        mock_response.text = AsyncMock(return_value="Configuration run not found")

        mock_put_ctx = AsyncMock()
        mock_put_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_put_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.put = Mock(return_value=mock_put_ctx)

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "traigent.cloud.trial_operations.is_backend_offline",
                return_value=False,
            ),
            patch("traigent.cloud.trial_operations.AIOHTTP_AVAILABLE", True),
            patch("traigent.cloud.trial_operations.aiohttp") as mock_aiohttp,
        ):
            mock_aiohttp.ClientSession = Mock(return_value=mock_session_ctx)
            mock_aiohttp.ClientTimeout = Mock()

            result = await ops.update_trial_weighted_scores(
                trial_id="trial-missing",
                weighted_score=0.73,
            )

        assert result is False
        mock_session.put.assert_called_once()


class TestSummaryStatsLogging:
    """Test logging behavior for optional summary_stats metadata."""

    def test_missing_summary_stats_is_debug_only(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Missing summary_stats should not emit a warning-level user log."""
        mock_client = Mock()
        mock_client.backend_config = Mock()
        mock_client.backend_config.backend_base_url = "https://api.example.com"
        mock_client.auth_manager = Mock()
        ops = TrialOperations(mock_client)

        with caplog.at_level(logging.DEBUG, logger="traigent.cloud.trial_operations"):
            ops._log_summary_stats_debug("trial_abc", None)

        assert any(
            "No summary_stats provided for trial trial_abc (optional)" in msg
            for msg in caplog.messages
        )
        assert not any(
            "No summary_stats found for trial trial_abc" in msg
            for msg in caplog.messages
        )


class TestOfflineModeReturnsNone:
    """Offline mode must return None (not True) so callers don't treat skip as success.

    Per workspace Rule 2 (no fake completion), returning True when no backend
    call was made would conflate 'skipped' with 'succeeded'.
    """

    def _make_ops(self) -> TrialOperations:
        mock_client = Mock()
        mock_client.backend_config = Mock()
        mock_client.backend_config.backend_base_url = "https://api.example.com"
        mock_client.backend_config.api_base_url = "https://api.example.com"
        mock_client.auth_manager = AsyncMock()
        mock_client.auth_manager.augment_headers = AsyncMock(return_value={})
        mock_client._map_to_backend_status = Mock(return_value="ACTIVE")
        mock_client._normalize_execution_mode = Mock(return_value="local")
        return TrialOperations(mock_client)

    @pytest.mark.asyncio
    async def test_register_trial_start_offline_returns_none(self) -> None:
        """register_trial_start returns None when backend is offline."""
        ops = self._make_ops()
        with patch(
            "traigent.cloud.trial_operations.is_backend_offline", return_value=True
        ):
            result = await ops.register_trial_start("sess-1", "trial-1", {"k": "v"})
        assert result is None

    @pytest.mark.asyncio
    async def test_submit_trial_result_offline_returns_none(self) -> None:
        """submit_trial_result_via_session returns None when backend is offline."""
        ops = self._make_ops()
        with patch(
            "traigent.cloud.trial_operations.is_backend_offline", return_value=True
        ):
            result = await ops.submit_trial_result_via_session(
                session_id="sess-1",
                trial_id="trial-1",
                config={"k": "v"},
                metrics={"accuracy": 0.9},
                status="completed",
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_submit_summary_stats_offline_returns_none(self) -> None:
        """submit_summary_stats returns None when backend is offline."""
        ops = self._make_ops()
        with patch(
            "traigent.cloud.trial_operations.is_backend_offline", return_value=True
        ):
            result = await ops.submit_summary_stats(
                session_id="sess-1",
                trial_id="trial-1",
                config={"k": "v"},
                summary_stats={"metrics": {"accuracy": 0.9}},
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_update_trial_weighted_scores_offline_returns_none(self) -> None:
        """update_trial_weighted_scores returns None when backend is offline."""
        ops = self._make_ops()
        with patch(
            "traigent.cloud.trial_operations.is_backend_offline", return_value=True
        ):
            result = await ops.update_trial_weighted_scores(
                trial_id="trial-1",
                weighted_score=0.85,
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_register_trial_start_exception_offline_returns_none(self) -> None:
        """Exception path in register_trial_start also returns None when offline."""
        ops = self._make_ops()

        # First call to is_backend_offline returns False (enter try block),
        # second call returns True (in exception handler).
        with (
            patch(
                "traigent.cloud.trial_operations.is_backend_offline",
                side_effect=[False, True],
            ),
            patch("traigent.cloud.trial_operations.AIOHTTP_AVAILABLE", True),
            patch("traigent.cloud.trial_operations.aiohttp") as mock_aiohttp,
        ):
            mock_aiohttp.ClientSession = Mock(
                side_effect=ConnectionError("simulated offline")
            )
            result = await ops.register_trial_start("sess-1", "trial-1", {"k": "v"})
        assert result is None

    def test_register_trial_start_sync_offline_returns_none(self) -> None:
        """Sync wrapper also returns None when offline."""
        ops = self._make_ops()
        with patch(
            "traigent.cloud.trial_operations.is_backend_offline", return_value=True
        ):
            result = ops.register_trial_start_sync("sess-1", "trial-1", {"k": "v"})
        assert result is None

    @pytest.mark.asyncio
    async def test_register_trial_start_sync_running_loop_offline_returns_none(
        self,
    ) -> None:
        """Sync wrapper preserves None even when called inside a running loop."""
        ops = self._make_ops()
        with patch(
            "traigent.cloud.trial_operations.is_backend_offline", return_value=True
        ):
            result = ops.register_trial_start_sync("sess-1", "trial-1", {"k": "v"})
        assert result is None

    @pytest.mark.asyncio
    async def test_submit_trial_result_exception_offline_returns_none(self) -> None:
        """Exception path in submit_trial_result_via_session returns None offline."""
        ops = self._make_ops()

        with (
            patch(
                "traigent.cloud.trial_operations.is_backend_offline",
                side_effect=[False, True],
            ),
            patch("traigent.cloud.trial_operations.AIOHTTP_AVAILABLE", True),
            patch("traigent.cloud.trial_operations.aiohttp") as mock_aiohttp,
        ):
            mock_aiohttp.ClientSession = Mock(
                side_effect=ConnectionError("simulated offline")
            )
            result = await ops.submit_trial_result_via_session(
                session_id="sess-1",
                trial_id="trial-1",
                config={"k": "v"},
                metrics={"accuracy": 0.9},
                status="completed",
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_submit_summary_stats_exception_offline_returns_none(self) -> None:
        """Exception path in submit_summary_stats returns None offline."""
        ops = self._make_ops()

        with (
            patch(
                "traigent.cloud.trial_operations.is_backend_offline",
                side_effect=[False, True],
            ),
            patch("traigent.cloud.trial_operations.AIOHTTP_AVAILABLE", True),
            patch("traigent.cloud.trial_operations.aiohttp") as mock_aiohttp,
        ):
            mock_aiohttp.ClientSession = Mock(
                side_effect=ConnectionError("simulated offline")
            )
            result = await ops.submit_summary_stats(
                session_id="sess-1",
                trial_id="trial-1",
                config={"k": "v"},
                summary_stats={"metrics": {"accuracy": 0.9}},
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_offline_result_is_falsy(self) -> None:
        """None is falsy so callers using `if result:` treat it as non-success."""
        ops = self._make_ops()
        with patch(
            "traigent.cloud.trial_operations.is_backend_offline", return_value=True
        ):
            result = await ops.register_trial_start("sess-1", "trial-1", {"k": "v"})
        # The critical invariant: callers must NOT interpret this as success.
        assert not result


class TestMeasuresLogging:
    """Test logging behavior for optional measures metadata."""

    def test_missing_measures_is_debug_only(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Missing measures should not emit a warning-level user log."""
        mock_client = Mock()
        mock_client.backend_config = Mock()
        mock_client.backend_config.backend_base_url = "https://api.example.com"
        mock_client.auth_manager = Mock()
        ops = TrialOperations(mock_client)

        with caplog.at_level(logging.DEBUG, logger="traigent.cloud.trial_operations"):
            ops._log_measures_debug("trial_xyz", None)

        assert any(
            "No measures provided for trial trial_xyz (optional)" in msg
            for msg in caplog.messages
        )
        assert not any(
            "No measures found for trial trial_xyz" in msg for msg in caplog.messages
        )


class TestMeasuresUpdateFailureVisibility:
    """Traigent#1724: a failed measures backfill must not look like ordinary success."""

    @pytest.mark.asyncio
    async def test_measures_update_failure_is_surfaced_not_swallowed(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """_update_config_run_measures returning False is logged and recorded.

        The overall trial submission still reports success (the POST to
        /sessions/{id}/results already succeeded with 2xx), but the failed
        measures backfill must be visible: a WARNING log distinguishable
        from the ordinary "Submitted trial result" debug line, and a marker
        on the result metadata.
        """
        mock_client = Mock()
        mock_client._update_config_run_status = AsyncMock(return_value=True)
        mock_client._update_config_run_measures = AsyncMock(return_value=False)

        ops = TrialOperations(mock_client)

        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={"continue_optimization": True})

        result_data: dict = {"metadata": {}}
        clean_metrics = {"accuracy": 0.9}

        with caplog.at_level(logging.WARNING, logger="traigent.cloud.trial_operations"):
            result = await ops._handle_trial_success_response(
                response=mock_response,
                session_id="session_123",
                trial_id="trial_001",
                backend_status="COMPLETED",
                result_data=result_data,
                clean_metrics=clean_metrics,
            )

        # Trial submission itself still succeeded.
        assert result is True
        mock_client._update_config_run_measures.assert_awaited_once_with(
            "trial_001", clean_metrics, None
        )
        # But the degraded measures backfill is not swallowed silently.
        assert result_data["metadata"]["measures_update_degraded"] is True
        assert any("Measures backfill failed" in msg for msg in caplog.messages)

    @pytest.mark.asyncio
    async def test_measures_update_success_leaves_no_degraded_marker(self) -> None:
        """Happy path stays byte-identical: no marker when the update succeeds."""
        mock_client = Mock()
        mock_client._update_config_run_status = AsyncMock(return_value=True)
        mock_client._update_config_run_measures = AsyncMock(return_value=True)

        ops = TrialOperations(mock_client)

        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={"continue_optimization": True})

        result_data: dict = {"metadata": {}}

        result = await ops._handle_trial_success_response(
            response=mock_response,
            session_id="session_123",
            trial_id="trial_001",
            backend_status="COMPLETED",
            result_data=result_data,
            clean_metrics={"accuracy": 0.9},
        )

        assert result is True
        assert "measures_update_degraded" not in result_data["metadata"]


class TestConfigRunStatusErrorMessageForwarding:
    """Traigent#1885 (companion to TraigentBackend#2002): a failed/pruned
    trial's error_message must reach _update_config_run_status instead of
    being dropped on the status-only PUT."""

    @pytest.mark.asyncio
    async def test_failure_path_forwards_error_message(self) -> None:
        """result_data carrying a sanitized error_message (set upstream in
        submit_trial_result_via_session when the caller supplied one) must
        be passed through to _update_config_run_status."""
        mock_client = Mock()
        mock_client._update_config_run_status = AsyncMock(return_value=True)
        mock_client._update_config_run_measures = AsyncMock(return_value=True)

        ops = TrialOperations(mock_client)

        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={"continue_optimization": True})

        result_data: dict = {
            "metadata": {},
            "error_message": "evaluator raised ValueError",
        }

        result = await ops._handle_trial_success_response(
            response=mock_response,
            session_id="session_123",
            trial_id="trial_001",
            backend_status="FAILED",
            result_data=result_data,
            clean_metrics={},
        )

        assert result is True
        mock_client._update_config_run_status.assert_awaited_once_with(
            "trial_001", "FAILED", error_message="evaluator raised ValueError"
        )

    @pytest.mark.asyncio
    async def test_success_path_forwards_no_error_message(self) -> None:
        """A successful trial has no error_message in result_data, so
        _update_config_run_status must be called with None — no wire
        regression for the happy path."""
        mock_client = Mock()
        mock_client._update_config_run_status = AsyncMock(return_value=True)
        mock_client._update_config_run_measures = AsyncMock(return_value=True)

        ops = TrialOperations(mock_client)

        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={"continue_optimization": True})

        result_data: dict = {"metadata": {}}

        result = await ops._handle_trial_success_response(
            response=mock_response,
            session_id="session_123",
            trial_id="trial_001",
            backend_status="COMPLETED",
            result_data=result_data,
            clean_metrics={"accuracy": 0.9},
        )

        assert result is True
        mock_client._update_config_run_status.assert_awaited_once_with(
            "trial_001", "COMPLETED", error_message=None
        )


def _mk_slot_client():
    """Backend client stub configured for request_trial_slot HTTP mocking."""
    mock_client = Mock()
    mock_client.backend_config = Mock()
    mock_client.backend_config.backend_base_url = "http://localhost:5000"
    mock_client.backend_config.api_base_url = "http://localhost:5000/api/v1"
    mock_client.auth_manager = AsyncMock()
    mock_client.auth_manager.augment_headers = AsyncMock(return_value={})
    return mock_client


def _aiohttp_post_returning(mock_response):
    """Build (mock_aiohttp ClientSession factory, mock_session) for a POST."""
    mock_post_ctx = AsyncMock()
    mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post_ctx.__aexit__ = AsyncMock(return_value=False)
    mock_session = AsyncMock()
    mock_session.post = Mock(return_value=mock_post_ctx)
    mock_session_ctx = AsyncMock()
    mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_ctx.__aexit__ = AsyncMock(return_value=False)
    return mock_session_ctx, mock_session


def _mk_submission_client(*, armed_sessions: set[str] | None = None):
    mock_client = Mock()
    mock_client.backend_config = Mock()
    mock_client.backend_config.backend_base_url = "https://api.example.com"
    mock_client.backend_config.api_base_url = "https://api.example.com/api/v1"
    mock_client.auth_manager = AsyncMock()
    mock_client.auth_manager.augment_headers = AsyncMock(return_value={})
    mock_client._cost_budget_armed_sessions = set(armed_sessions or set())
    mock_client._map_to_backend_status = Mock(
        side_effect=lambda status, endpoint="config_run": {
            "completed": "COMPLETED",
            "running": "RUNNING",
            "in_progress": "RUNNING",
            "failed": "FAILED",
        }.get(str(status).lower(), str(status).upper())
    )
    mock_client._normalize_execution_mode = Mock(return_value="local")
    mock_client._sanitize_error_message = Mock(return_value="")
    return mock_client


class TestBudgetedSessionResultCostGuarantee:
    @pytest.mark.asyncio
    async def test_budgeted_completed_submission_backfills_zero_cost(self) -> None:
        mock_client = _mk_submission_client(armed_sessions={"session_123"})
        ops = TrialOperations(mock_client)
        ops._handle_trial_success_response = AsyncMock(return_value=True)
        mock_response = Mock()
        mock_response.status = 201
        mock_session_ctx, mock_session = _aiohttp_post_returning(mock_response)

        with (
            patch(
                "traigent.cloud.trial_operations.is_backend_offline",
                return_value=False,
            ),
            patch("traigent.cloud.trial_operations.AIOHTTP_AVAILABLE", True),
            patch(
                "traigent.cloud.trial_operations.validate_configuration_run_submission",
            ),
            patch("traigent.cloud.trial_operations.aiohttp") as mock_aiohttp,
        ):
            mock_aiohttp.ClientSession = Mock(return_value=mock_session_ctx)
            mock_aiohttp.ClientTimeout = Mock()

            result = await ops.submit_trial_result_via_session(
                session_id="session_123",
                trial_id="trial_001",
                config={"temperature": 0.2},
                metrics={"accuracy": 0.95},
                status="completed",
            )

        assert result is True
        payload = mock_session.post.call_args.kwargs["json"]
        assert payload["metrics"]["cost"] == 0.0

    @pytest.mark.asyncio
    async def test_budgeted_completed_submission_promotes_metadata_cost(self) -> None:
        mock_client = _mk_submission_client(armed_sessions={"session_123"})
        ops = TrialOperations(mock_client)
        ops._handle_trial_success_response = AsyncMock(return_value=True)
        mock_response = Mock()
        mock_response.status = 201
        mock_session_ctx, mock_session = _aiohttp_post_returning(mock_response)

        with (
            patch(
                "traigent.cloud.trial_operations.is_backend_offline",
                return_value=False,
            ),
            patch("traigent.cloud.trial_operations.AIOHTTP_AVAILABLE", True),
            patch(
                "traigent.cloud.trial_operations.validate_configuration_run_submission",
            ),
            patch("traigent.cloud.trial_operations.aiohttp") as mock_aiohttp,
        ):
            mock_aiohttp.ClientSession = Mock(return_value=mock_session_ctx)
            mock_aiohttp.ClientTimeout = Mock()

            result = await ops.submit_trial_result_via_session(
                session_id="session_123",
                trial_id="trial_001",
                config={"temperature": 0.2},
                metrics={"accuracy": 0.95},
                status="completed",
                metadata={"total_cost": 0.42},
            )

        assert result is True
        payload = mock_session.post.call_args.kwargs["json"]
        assert payload["metrics"]["cost"] == pytest.approx(0.42)

    @pytest.mark.asyncio
    async def test_non_budget_completed_submission_does_not_add_cost(self) -> None:
        mock_client = _mk_submission_client()
        ops = TrialOperations(mock_client)
        ops._handle_trial_success_response = AsyncMock(return_value=True)
        mock_response = Mock()
        mock_response.status = 201
        mock_session_ctx, mock_session = _aiohttp_post_returning(mock_response)

        with (
            patch(
                "traigent.cloud.trial_operations.is_backend_offline",
                return_value=False,
            ),
            patch("traigent.cloud.trial_operations.AIOHTTP_AVAILABLE", True),
            patch(
                "traigent.cloud.trial_operations.validate_configuration_run_submission",
            ),
            patch("traigent.cloud.trial_operations.aiohttp") as mock_aiohttp,
        ):
            mock_aiohttp.ClientSession = Mock(return_value=mock_session_ctx)
            mock_aiohttp.ClientTimeout = Mock()

            result = await ops.submit_trial_result_via_session(
                session_id="session_123",
                trial_id="trial_001",
                config={"temperature": 0.2},
                metrics={"accuracy": 0.95},
                status="completed",
            )

        assert result is True
        payload = mock_session.post.call_args.kwargs["json"]
        assert "cost" not in payload["metrics"]

    @pytest.mark.asyncio
    async def test_budgeted_non_completed_submission_does_not_add_cost(self) -> None:
        mock_client = _mk_submission_client(armed_sessions={"session_123"})
        ops = TrialOperations(mock_client)
        ops._handle_trial_success_response = AsyncMock(return_value=True)
        mock_response = Mock()
        mock_response.status = 201
        mock_session_ctx, mock_session = _aiohttp_post_returning(mock_response)

        with (
            patch(
                "traigent.cloud.trial_operations.is_backend_offline",
                return_value=False,
            ),
            patch("traigent.cloud.trial_operations.AIOHTTP_AVAILABLE", True),
            patch(
                "traigent.cloud.trial_operations.validate_configuration_run_submission",
            ),
            patch("traigent.cloud.trial_operations.aiohttp") as mock_aiohttp,
        ):
            mock_aiohttp.ClientSession = Mock(return_value=mock_session_ctx)
            mock_aiohttp.ClientTimeout = Mock()

            result = await ops.submit_trial_result_via_session(
                session_id="session_123",
                trial_id="trial_001",
                config={"temperature": 0.2},
                metrics={"accuracy": 0.95},
                status="running",
            )

        assert result is True
        payload = mock_session.post.call_args.kwargs["json"]
        assert payload["status"] == "RUNNING"
        assert "cost" not in payload["metrics"]

    @pytest.mark.asyncio
    async def test_budgeted_running_trial_start_does_not_add_cost(self) -> None:
        mock_client = _mk_submission_client(armed_sessions={"session_123"})
        ops = TrialOperations(mock_client)
        mock_response = Mock()
        mock_response.status = 201
        mock_session_ctx, mock_session = _aiohttp_post_returning(mock_response)

        with (
            patch(
                "traigent.cloud.trial_operations.is_backend_offline",
                return_value=False,
            ),
            patch("traigent.cloud.trial_operations.AIOHTTP_AVAILABLE", True),
            patch("traigent.cloud.trial_operations.aiohttp") as mock_aiohttp,
        ):
            mock_aiohttp.ClientSession = Mock(return_value=mock_session_ctx)
            mock_aiohttp.ClientTimeout = Mock()

            result = await ops.register_trial_start(
                session_id="session_123",
                trial_id="trial_001",
                config={"temperature": 0.2},
            )

        assert result is True
        payload = mock_session.post.call_args.kwargs["json"]
        assert payload["status"] == "RUNNING"
        assert payload["metrics"] == {}

    @pytest.mark.asyncio
    async def test_budgeted_summary_stats_completed_backfills_cost(self) -> None:
        mock_client = _mk_submission_client(armed_sessions={"session_123"})
        ops = TrialOperations(mock_client)
        mock_response = Mock()
        mock_response.status = 201
        mock_session_ctx, mock_session = _aiohttp_post_returning(mock_response)

        with (
            patch(
                "traigent.cloud.trial_operations.is_backend_offline",
                return_value=False,
            ),
            patch("traigent.cloud.trial_operations.AIOHTTP_AVAILABLE", True),
            patch(
                "traigent.cloud.trial_operations.validate_configuration_run_submission",
            ),
            patch("traigent.cloud.trial_operations.aiohttp") as mock_aiohttp,
        ):
            mock_aiohttp.ClientSession = Mock(return_value=mock_session_ctx)
            mock_aiohttp.ClientTimeout = Mock()

            result = await ops.submit_summary_stats(
                session_id="session_123",
                trial_id="trial_001",
                config={"temperature": 0.2},
                summary_stats={"metrics": {"accuracy": 0.95}},
                status="completed",
            )

        assert result is True
        payload = mock_session.post.call_args.kwargs["json"]
        assert payload["metrics"]["cost"] == 0.0


class TestRequestTrialSlot:
    """The backend mints configuration_run ids only via next-trial; the SDK
    must obtain that id before submitting results (a client hash 404s)."""

    @pytest.mark.asyncio
    async def test_returns_backend_minted_trial_id(self) -> None:
        ops = TrialOperations(_mk_slot_client())

        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "should_continue": True,
                "suggestion": {
                    "trial_id": "trial_be_minted_abc",
                    "session_id": "sess-1",
                    "config": {"variant": "cheap"},
                },
            }
        )
        mock_session_ctx, mock_session = _aiohttp_post_returning(mock_response)

        with (
            patch(
                "traigent.cloud.trial_operations.is_backend_offline",
                return_value=False,
            ),
            patch("traigent.cloud.trial_operations.AIOHTTP_AVAILABLE", True),
            patch("traigent.cloud.trial_operations.aiohttp") as mock_aiohttp,
        ):
            mock_aiohttp.ClientSession = Mock(return_value=mock_session_ctx)
            mock_aiohttp.ClientTimeout = Mock()
            slot = await ops.request_trial_slot("sess-1")

        assert slot == TrialSlotResult.acquired("trial_be_minted_abc")
        assert slot
        # It hit the next-trial endpoint (where the backend mints the slot).
        url = mock_session.post.call_args.args[0]
        assert url.endswith("/sessions/sess-1/next-trial")

    @pytest.mark.asyncio
    async def test_should_continue_false_returns_completion_result(self) -> None:
        """Graceful backend completion is distinct from genuine no-slot/error."""
        ops = TrialOperations(_mk_slot_client())

        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"should_continue": False, "suggestion": None}
        )
        mock_session_ctx, _ = _aiohttp_post_returning(mock_response)

        with (
            patch(
                "traigent.cloud.trial_operations.is_backend_offline",
                return_value=False,
            ),
            patch("traigent.cloud.trial_operations.AIOHTTP_AVAILABLE", True),
            patch("traigent.cloud.trial_operations.aiohttp") as mock_aiohttp,
        ):
            mock_aiohttp.ClientSession = Mock(return_value=mock_session_ctx)
            mock_aiohttp.ClientTimeout = Mock()
            slot = await ops.request_trial_slot("sess-1")

        assert slot == TrialSlotResult.complete()
        assert not slot

    @pytest.mark.asyncio
    async def test_should_continue_true_without_id_returns_unavailable(self) -> None:
        """A malformed/empty continuation response is not terminal completion."""
        ops = TrialOperations(_mk_slot_client())

        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"should_continue": True, "suggestion": {}}
        )
        mock_session_ctx, _ = _aiohttp_post_returning(mock_response)

        with (
            patch(
                "traigent.cloud.trial_operations.is_backend_offline",
                return_value=False,
            ),
            patch("traigent.cloud.trial_operations.AIOHTTP_AVAILABLE", True),
            patch("traigent.cloud.trial_operations.aiohttp") as mock_aiohttp,
        ):
            mock_aiohttp.ClientSession = Mock(return_value=mock_session_ctx)
            mock_aiohttp.ClientTimeout = Mock()
            slot = await ops.request_trial_slot("sess-1")

        assert slot == TrialSlotResult.unavailable()
        assert not slot.optimization_complete

    @pytest.mark.asyncio
    async def test_non_2xx_returns_unavailable(self) -> None:
        ops = TrialOperations(_mk_slot_client())

        mock_response = Mock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="server error")
        mock_session_ctx, _ = _aiohttp_post_returning(mock_response)

        with (
            patch(
                "traigent.cloud.trial_operations.is_backend_offline",
                return_value=False,
            ),
            patch("traigent.cloud.trial_operations.AIOHTTP_AVAILABLE", True),
            patch("traigent.cloud.trial_operations.aiohttp") as mock_aiohttp,
        ):
            mock_aiohttp.ClientSession = Mock(return_value=mock_session_ctx)
            mock_aiohttp.ClientTimeout = Mock()
            slot = await ops.request_trial_slot("sess-1")

        assert slot == TrialSlotResult.unavailable()
        assert not slot.optimization_complete

    @pytest.mark.asyncio
    async def test_transport_error_returns_unavailable(self) -> None:
        ops = TrialOperations(_mk_slot_client())

        with (
            patch(
                "traigent.cloud.trial_operations.is_backend_offline",
                return_value=False,
            ),
            patch("traigent.cloud.trial_operations.AIOHTTP_AVAILABLE", True),
            patch("traigent.cloud.trial_operations.aiohttp") as mock_aiohttp,
        ):
            mock_aiohttp.ClientSession = Mock(side_effect=ConnectionError("boom"))
            mock_aiohttp.ClientTimeout = Mock()
            slot = await ops.request_trial_slot("sess-1")

        assert slot == TrialSlotResult.unavailable()
        assert not slot.optimization_complete

    @pytest.mark.asyncio
    async def test_offline_mode_returns_unavailable(self) -> None:
        """Offline mode skips the backend entirely (no fake slot)."""
        mock_client = _mk_slot_client()
        ops = TrialOperations(mock_client)
        with patch(
            "traigent.cloud.trial_operations.is_backend_offline",
            return_value=True,
        ):
            slot = await ops.request_trial_slot("sess-1")
        assert slot == TrialSlotResult.unavailable()
        mock_client.auth_manager.augment_headers.assert_not_awaited()


class TestHandle400NotFound:
    """HTTP 400 'not found' from the results endpoint is a transient per-worker
    session-storage condition (BE #1194) and must be handled gracefully:
    - logged at INFO (not WARNING/ERROR) with the transient sync guidance
    - submit_trial_result_via_session returns None (skipped) not False (hard failure)
    - non-session "not found" 400s (e.g. "Trial not found") stay at WARNING/False
    """

    def _make_ops(self) -> "TrialOperations":
        mock_client = Mock()
        mock_client.backend_config = Mock()
        mock_client.backend_config.backend_base_url = "https://portal.traigent.ai"
        mock_client.backend_config.api_base_url = "https://portal.traigent.ai/api/v1"
        mock_client.auth_manager = AsyncMock()
        mock_client.auth_manager.augment_headers = AsyncMock(return_value={})
        mock_client._map_to_backend_status = Mock(return_value="COMPLETED")
        mock_client._normalize_execution_mode = Mock(return_value="local")
        mock_client._sanitize_error_message = Mock(return_value="")
        return TrialOperations(mock_client)

    def test_handle_trial_error_response_400_session_not_found_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """400 + session not found body → INFO log with transient guidance and
        the backend detail in the log, not WARNING."""
        ops = self._make_ops()
        error_body = '{"error": "Session abc123 not found"}'

        with caplog.at_level(logging.DEBUG, logger="traigent.cloud.trial_operations"):
            is_transient = ops._handle_trial_error_response(
                status=400,
                trial_id="trial_xyz",
                session_id="abc123",
                url="https://portal.traigent.ai/api/v1/sessions/abc123/results",
                error_text=error_body,
            )

        assert is_transient is True, (
            "_handle_trial_error_response must return True for session not-found"
        )

        # Must log at INFO, not WARNING or ERROR
        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        warn_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert info_records, (
            "Expected at least one INFO log record for transient not-found"
        )
        assert not warn_records, (
            "No WARNING or ERROR log records should be emitted for a transient "
            f"session-not-found 400; got: {[r.getMessage() for r in warn_records]}"
        )

        combined = " ".join(r.getMessage() for r in info_records)
        # Sync guidance must appear so users know how to recover
        assert "sync" in combined, (
            "INFO log should include sync guidance for transient not-found"
        )
        # The parsed backend detail must appear in the log so the body is not
        # silently swallowed — this is the text extracted from the JSON "error" key.
        assert "Session abc123 not found" in combined, (
            "INFO log must include the backend detail from the response body; "
            f"got: {combined!r}"
        )

    def test_handle_trial_error_response_400_validation_logs_backend_reason(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Permanent validation 400s must surface details.reason, not a guess."""

        ops = self._make_ops()
        error_body = json.dumps(
            {
                "details": {
                    "reason": [
                        'submitted config["model"] is outside the declared categorical domain'
                    ]
                },
                "error": "Invalid request data...",
                "error_code": "VALIDATION_ERROR",
                "message": "Invalid request data...",
                "success": False,
            }
        )

        with caplog.at_level(logging.ERROR, logger="traigent.cloud.trial_operations"):
            result = ops._handle_trial_error_response(
                status=400,
                trial_id="trial_xyz",
                session_id="sess_abc",
                url="https://portal.traigent.ai/api/v1/sessions/sess_abc/results",
                error_text=error_body,
            )

        assert isinstance(result, TrialSubmissionResult)
        assert result.permanent_rejection is True
        assert result.reason is not None
        assert "outside the declared categorical domain" in result.reason

        combined = " ".join(r.getMessage() for r in caplog.records)
        assert "outside the declared categorical domain" in combined
        assert "cost_usd" not in combined

    def test_handle_trial_error_response_400_trial_not_found_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """400 + 'Trial not found' body must NOT be treated as transient session
        storage — it should stay at WARNING/False to avoid masking real errors."""
        ops = self._make_ops()
        # "Trial not found in session" is a different error class — not a transient
        # per-worker session storage miss.
        error_body = '{"error": "Trial trial_xyz not found in session sess_abc"}'

        with caplog.at_level(logging.DEBUG, logger="traigent.cloud.trial_operations"):
            is_transient = ops._handle_trial_error_response(
                status=400,
                trial_id="trial_xyz",
                session_id="sess_abc",
                url="https://portal.traigent.ai/api/v1/sessions/sess_abc/results",
                error_text=error_body,
            )

        assert not is_transient, (
            "'Trial not found in session' is not a transient session storage miss"
        )

        warn_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warn_records, "Expected WARNING log for non-session not-found 400"

    @pytest.mark.asyncio
    async def test_submit_trial_result_400_session_not_found_returns_none(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """submit_trial_result_via_session must return None (not False) for a
        400 session-not-found so _log_trial_to_backend does not flag backend
        degraded.  The backend detail must appear in the log."""
        ops = self._make_ops()

        mock_response = Mock()
        mock_response.status = 400
        error_json = '{"error": "Session sess_edge not found"}'
        mock_response.text = AsyncMock(return_value=error_json)
        mock_session_ctx, _ = _aiohttp_post_returning(mock_response)

        with (
            patch(
                "traigent.cloud.trial_operations.is_backend_offline",
                return_value=False,
            ),
            patch("traigent.cloud.trial_operations.AIOHTTP_AVAILABLE", True),
            patch(
                "traigent.cloud.trial_operations.validate_configuration_run_submission",
            ),
            patch("traigent.cloud.trial_operations.aiohttp") as mock_aiohttp,
            caplog.at_level(logging.DEBUG, logger="traigent.cloud.trial_operations"),
        ):
            mock_aiohttp.ClientSession = Mock(return_value=mock_session_ctx)
            mock_aiohttp.ClientTimeout = Mock()

            result = await ops.submit_trial_result_via_session(
                session_id="sess_edge",
                trial_id="trial_001",
                config={"temperature": 0.5},
                metrics={"score": 0.8},
                status="completed",
                execution_mode="local",
            )

        # Must return None (transient/skipped), not False (hard failure)
        assert result is None, (
            f"Expected None for 400 session-not-found (transient skip), got {result!r}"
        )

        # The backend detail from the response body must be visible in the log.
        # This is the key fix: the error is not silently swallowed.
        all_log_text = " ".join(r.getMessage() for r in caplog.records)
        assert "Session sess_edge not found" in all_log_text, (
            "Backend response body detail must appear in logs. "
            f"All log text: {all_log_text!r}"
        )
