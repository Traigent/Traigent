"""Unit tests for APIKeyManager class.

Tests for API key lifecycle management, validation, and rotation.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Security FUNC-SECURITY REQ-SEC-010

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from traigent.cloud.api_key_manager import API_KEY_TOKEN_TTL, APIKeyManager
from traigent.cloud.auth import APIKey, AuthCredentials, SecureToken, UnifiedAuthConfig


@pytest.fixture
def config() -> UnifiedAuthConfig:
    """Create test configuration."""
    return UnifiedAuthConfig(
        api_key_default_ttl_days=30,
        api_key_warning_days=14,
        api_key_critical_days=7,
    )


@pytest.fixture
def manager(config: UnifiedAuthConfig) -> APIKeyManager:
    """Create APIKeyManager instance."""
    return APIKeyManager(config)


@pytest.fixture
def valid_api_key() -> str:
    """Create a valid API key for testing."""
    # tg_ prefix + 61 alphanumeric chars = 64 total
    return "tg_" + "a" * 61


class TestMaskKey:
    """Tests for mask_key static method."""

    def test_mask_key_normal(self) -> None:
        """Test masking a normal API key."""
        key = "tg_abcdefghijklmnopqrstuvwxyz1234567890ABCDEFGHIJKLMNOPQRSTUVWXY"
        masked = APIKeyManager.mask_key(key)
        assert masked.startswith("tg_a")
        assert masked.endswith("VWXY")
        assert "*" in masked

    def test_mask_key_empty(self) -> None:
        """Test masking an empty key."""
        assert APIKeyManager.mask_key("") == ""

    def test_mask_key_short(self) -> None:
        """Test masking a key shorter than 8 chars."""
        assert APIKeyManager.mask_key("abc") == "***"
        assert APIKeyManager.mask_key("12345678") == "********"

    def test_mask_key_exactly_9_chars(self) -> None:
        """Test masking a key with exactly 9 chars."""
        masked = APIKeyManager.mask_key("123456789")
        assert masked == "1234*6789"


class TestSetToken:
    """Tests for set_token method."""

    def test_set_token_valid(self, manager: APIKeyManager, valid_api_key: str) -> None:
        """Test setting a valid API key token."""
        manager.set_token(valid_api_key, source="test")

        assert manager.api_key_token is not None
        assert manager.api_key_preview is not None
        assert manager.api_key_source == "test"
        assert manager.api_key_last_rotated is not None

    def test_set_token_clears_on_empty(self, manager: APIKeyManager) -> None:
        """Test that empty key clears token."""
        # First set a token
        manager._api_key_token = MagicMock()
        manager._api_key_preview = "preview"

        # Then clear it with empty key
        manager.set_token("", source="test")

        assert manager.api_key_token is None
        assert manager.api_key_preview is None

    def test_set_token_with_expiry(
        self, manager: APIKeyManager, valid_api_key: str
    ) -> None:
        """Test setting token with custom expiry."""
        expires_at = datetime.now(timezone.utc) + timedelta(days=60)
        manager.set_token(valid_api_key, source="test", expires_at=expires_at)

        assert manager.api_key_expiry is not None
        assert manager.api_key_expiry == expires_at

    def test_set_token_adds_timezone_to_naive_expiry(
        self, manager: APIKeyManager, valid_api_key: str
    ) -> None:
        """Test that naive datetime expiry gets UTC timezone."""
        naive_expires = datetime.now() + timedelta(days=60)
        manager.set_token(valid_api_key, source="test", expires_at=naive_expires)

        assert manager.api_key_expiry is not None
        assert manager.api_key_expiry.tzinfo is not None

    def test_set_token_invalid_key_continues_without_token(
        self, manager: APIKeyManager
    ) -> None:
        """Test that invalid key logs warning but continues."""
        # Very short key will fail SecureToken validation
        manager.set_token("ab", source="test")

        # Token should be None, but source should still be set
        assert manager.api_key_source == "test"


class TestClearToken:
    """Tests for clear_token method."""

    def test_clear_token(self, manager: APIKeyManager, valid_api_key: str) -> None:
        """Test clearing all API key state."""
        manager.set_token(valid_api_key, source="test")
        manager.clear_token()

        assert manager.api_key_token is None
        assert manager.api_key_preview is None
        assert manager.api_key_source is None
        assert manager.api_key_expiry is None
        assert manager.api_key_last_rotated is None

    def test_clear_token_when_empty(self, manager: APIKeyManager) -> None:
        """Test clearing when already empty."""
        manager.clear_token()  # Should not raise
        assert manager.api_key_token is None


class TestHasKey:
    """Tests for has_key method."""

    def test_has_key_with_token(
        self, manager: APIKeyManager, valid_api_key: str
    ) -> None:
        """Test has_key returns True when token is set."""
        manager.set_token(valid_api_key, source="test")
        assert manager.has_key() is True

    def test_has_key_without_token(self, manager: APIKeyManager) -> None:
        """Test has_key returns False when no token."""
        assert manager.has_key() is False

    def test_has_key_with_credentials_callback(self, manager: APIKeyManager) -> None:
        """Test has_key checks credentials callback."""
        mock_credentials = MagicMock()
        mock_credentials.api_key = "some-key"
        manager.set_callbacks(get_credentials=lambda: mock_credentials)

        assert manager.has_key() is True


class TestGetKeyForInternalUse:
    """Tests for get_key_for_internal_use method."""

    def test_get_key_from_token(
        self, manager: APIKeyManager, valid_api_key: str
    ) -> None:
        """Test getting key from secure token."""
        manager.set_token(valid_api_key, source="test")

        key = manager.get_key_for_internal_use()
        assert key == valid_api_key

    def test_get_key_from_credentials(self, manager: APIKeyManager) -> None:
        """Test getting key from credentials callback."""
        mock_credentials = MagicMock()
        mock_credentials.api_key = "credentials-key"
        manager.set_callbacks(get_credentials=lambda: mock_credentials)

        key = manager.get_key_for_internal_use()
        assert key == "credentials-key"

    def test_get_key_returns_none_when_nothing_available(
        self, manager: APIKeyManager
    ) -> None:
        """Test returns None when no key available."""
        assert manager.get_key_for_internal_use() is None


class TestValidateFormat:
    """Tests for validate_format method."""

    def test_validate_format_valid_key(
        self, manager: APIKeyManager, valid_api_key: str
    ) -> None:
        """Test validation of valid API key."""
        assert manager.validate_format(valid_api_key) is True

    def test_validate_format_wrong_prefix(self, manager: APIKeyManager) -> None:
        """Test validation rejects wrong prefix."""
        key = "ak_" + "a" * 61
        assert manager.validate_format(key) is False

    def test_validate_format_wrong_length(self, manager: APIKeyManager) -> None:
        """Test validation rejects wrong length."""
        key = "tg_" + "a" * 50  # Too short
        assert manager.validate_format(key) is False

    def test_validate_format_invalid_chars(self, manager: APIKeyManager) -> None:
        """Test validation rejects invalid characters."""
        key = "tg_" + "a" * 50 + "!@#$%^&*()+"
        assert manager.validate_format(key) is False

    def test_validate_format_none(self, manager: APIKeyManager) -> None:
        """Test validation rejects None."""
        assert manager.validate_format(None) is False

    def test_validate_format_empty(self, manager: APIKeyManager) -> None:
        """Test validation rejects empty string."""
        assert manager.validate_format("") is False

    def test_validate_format_not_string(self, manager: APIKeyManager) -> None:
        """Test validation rejects non-string."""
        assert manager.validate_format(12345) is False  # type: ignore


class TestGetStatus:
    """Tests for get_status method."""

    def test_get_status_missing(self, manager: APIKeyManager) -> None:
        """Test status when no key is set."""
        status = manager.get_status()
        assert status["state"] == "missing"
        assert status["preview"] is None

    def test_get_status_ok(self, manager: APIKeyManager, valid_api_key: str) -> None:
        """Test status when key is healthy."""
        expires = datetime.now(timezone.utc) + timedelta(days=30)
        manager.set_token(valid_api_key, source="test", expires_at=expires)

        status = manager.get_status()
        assert status["state"] == "ok"
        assert status["preview"] is not None
        assert status["days_remaining"] > 14

    def test_get_status_warning(
        self, manager: APIKeyManager, valid_api_key: str
    ) -> None:
        """Test status when key is approaching expiry."""
        expires = datetime.now(timezone.utc) + timedelta(days=10)
        manager.set_token(valid_api_key, source="test", expires_at=expires)

        status = manager.get_status()
        assert status["state"] == "warning"
        assert 7 < status["days_remaining"] <= 14

    def test_get_status_critical(
        self, manager: APIKeyManager, valid_api_key: str
    ) -> None:
        """Test status when key is critically close to expiry."""
        expires = datetime.now(timezone.utc) + timedelta(days=3)
        manager.set_token(valid_api_key, source="test", expires_at=expires)

        status = manager.get_status()
        assert status["state"] == "critical"
        assert status["days_remaining"] <= 7

    def test_get_status_expired(
        self, manager: APIKeyManager, valid_api_key: str
    ) -> None:
        """Test status when key has expired."""
        expires = datetime.now(timezone.utc) - timedelta(days=1)
        manager.set_token(valid_api_key, source="test", expires_at=expires)

        status = manager.get_status()
        assert status["state"] == "expired"
        assert status["days_remaining"] < 0


class TestCheckRotation:
    """Tests for check_rotation method."""

    def test_check_rotation_healthy(
        self, manager: APIKeyManager, valid_api_key: str
    ) -> None:
        """Test rotation check when key is healthy."""
        expires = datetime.now(timezone.utc) + timedelta(days=30)
        manager.set_token(valid_api_key, source="test", expires_at=expires)

        assert manager.check_rotation() is True

    def test_check_rotation_warning(
        self, manager: APIKeyManager, valid_api_key: str
    ) -> None:
        """Test rotation check when key needs attention."""
        expires = datetime.now(timezone.utc) + timedelta(days=10)
        manager.set_token(valid_api_key, source="test", expires_at=expires)

        assert manager.check_rotation() is False

    def test_check_rotation_critical(
        self, manager: APIKeyManager, valid_api_key: str
    ) -> None:
        """Test rotation check when key is critical."""
        expires = datetime.now(timezone.utc) + timedelta(days=3)
        manager.set_token(valid_api_key, source="test", expires_at=expires)

        assert manager.check_rotation() is False


class TestPersistApiKey:
    """Tests for persist_api_key method."""

    def test_persist_api_key_from_credentials(
        self, manager: APIKeyManager, valid_api_key: str
    ) -> None:
        """Test persisting API key from credentials."""
        credentials = AuthCredentials(api_key=valid_api_key)
        manager.persist_api_key(credentials)

        assert manager.api_key_token is not None
        assert manager.api_key is not None
        assert manager.api_key.key == valid_api_key

    def test_persist_api_key_with_source_metadata(
        self, manager: APIKeyManager, valid_api_key: str
    ) -> None:
        """Test persisting API key preserves source metadata."""
        credentials = AuthCredentials(api_key=valid_api_key, metadata={"source": "cli"})
        manager.persist_api_key(credentials)

        assert manager.api_key_source == "cli"

    def test_persist_api_key_no_key(self, manager: APIKeyManager) -> None:
        """Test persist does nothing without key."""
        credentials = AuthCredentials()
        manager.persist_api_key(credentials)

        assert manager.api_key is None


class TestGetInfo:
    """Tests for get_info method."""

    def test_get_info_with_api_key_object(
        self, manager: APIKeyManager, valid_api_key: str
    ) -> None:
        """Test get_info returns info from APIKey object."""
        credentials = AuthCredentials(api_key=valid_api_key)
        manager.persist_api_key(credentials)

        info = manager.get_info()
        assert info is not None
        assert info["name"] == "default"
        assert info["is_valid"] is True
        assert "permissions" in info

    def test_get_info_with_env_key(
        self, manager: APIKeyManager, valid_api_key: str
    ) -> None:
        """Test get_info returns info for environment key."""
        manager.set_token(valid_api_key, source="env")

        info = manager.get_info()
        assert info is not None
        assert info["name"] == "environment"
        assert info["is_valid"] is True

    def test_get_info_no_key(self, manager: APIKeyManager) -> None:
        """Test get_info returns None when no key."""
        info = manager.get_info()
        assert info is None


class TestCallbacks:
    """Tests for callback functionality."""

    def test_set_callbacks(self, manager: APIKeyManager) -> None:
        """Test setting callback functions."""
        get_creds = MagicMock(return_value=None)
        get_provided = MagicMock(return_value=None)
        get_last_result = MagicMock(return_value=None)

        manager.set_callbacks(
            get_credentials=get_creds,
            get_provided_credentials=get_provided,
            get_last_auth_result=get_last_result,
        )

        assert manager._get_credentials_fn is get_creds
        assert manager._get_provided_credentials_fn is get_provided
        assert manager._get_last_auth_result_fn is get_last_result

    def test_callbacks_used_in_has_key(self, manager: APIKeyManager) -> None:
        """Test that callbacks are used in has_key."""
        mock_creds = MagicMock()
        mock_creds.api_key = "test-key"
        get_creds = MagicMock(return_value=mock_creds)

        manager.set_callbacks(get_credentials=get_creds)

        result = manager.has_key()
        assert result is True
        get_creds.assert_called_once()


class TestApiKeyObject:
    """Tests for APIKey object management."""

    def test_set_api_key_object(
        self, manager: APIKeyManager, valid_api_key: str
    ) -> None:
        """Test setting APIKey object directly."""
        api_key = APIKey(
            key=valid_api_key,
            name="test-key",
            created_at=datetime.now(timezone.utc),
            permissions={"read": True},
        )

        manager.set_api_key_object(api_key)

        assert manager.api_key is api_key
        assert manager.api_key.name == "test-key"

    def test_clear_api_key_object(
        self, manager: APIKeyManager, valid_api_key: str
    ) -> None:
        """Test clearing APIKey object."""
        api_key = APIKey(
            key=valid_api_key,
            name="test-key",
            created_at=datetime.now(timezone.utc),
        )
        manager.set_api_key_object(api_key)

        manager.clear_api_key_object()

        assert manager.api_key is None
