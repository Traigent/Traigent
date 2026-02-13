"""Tests for trial_operations.py - particularly new code paths."""

import logging
from unittest.mock import AsyncMock, Mock, patch

import pytest

from traigent.cloud.trial_operations import TrialOperations


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


class TestMeasuresDictValidationInSubmission:
    """Test MeasuresDict validation warning path in submit_trial_result_via_session."""

    @pytest.mark.asyncio
    async def test_invalid_metric_key_logs_warning_and_submits_unvalidated(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Invalid metric keys trigger MeasuresDict warning but submission continues."""
        mock_client = Mock()
        mock_client.backend_config = Mock()
        mock_client.backend_config.backend_base_url = (
            "https://api.example.com"  # pragma: allowlist secret
        )
        mock_client.backend_config.api_base_url = "https://api.example.com"
        mock_client.auth_manager = AsyncMock()
        mock_client.auth_manager.augment_headers = AsyncMock(return_value={})
        mock_client._map_to_backend_status = Mock(return_value="completed")
        mock_client._normalize_execution_mode = Mock(return_value="edge_analytics")
        mock_client._sanitize_error_message = Mock(return_value="")

        ops = TrialOperations(mock_client)
        ops._handle_trial_success_response = AsyncMock(return_value=True)

        # Use a metric key with a hyphen — MeasuresDict rejects non-identifier keys
        invalid_metrics = {"invalid-key": 0.95}

        # Build nested async context manager mocks for aiohttp
        mock_response = Mock()
        mock_response.status = 200

        # post() returns an async context manager yielding mock_response
        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = Mock(return_value=mock_post_ctx)

        # ClientSession() returns an async context manager yielding mock_session
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

            with caplog.at_level(
                logging.WARNING, logger="traigent.cloud.trial_operations"
            ):
                result = await ops.submit_trial_result_via_session(
                    session_id="test-session",
                    trial_id="test-trial",
                    config={"model": "gpt-4"},
                    metrics=invalid_metrics,
                    status="completed",
                )

            # The warning should have been logged
            assert any("Metrics validation warning" in msg for msg in caplog.messages)
            # Submission should still proceed (True = success)
            assert result is True
