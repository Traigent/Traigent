"""Tests for trial_operations.py - particularly new code paths."""

from unittest.mock import Mock

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
        data = {"api_key": "sk-1234567890abcdef"}
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
                "api_key": "secret123",
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
