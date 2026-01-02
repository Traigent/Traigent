#!/usr/bin/env python3
"""Test secure authentication manager and resilient client."""

import os
import sys
import time
from unittest.mock import AsyncMock

import pytest

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from traigent.cloud.auth import AuthManager as SecureAuthManager
from traigent.cloud.auth import (
    SecureToken,
    constant_time_compare,
    log_auth_event,
)
from traigent.cloud.resilient_client import (
    ErrorType,
    ResilientClient,
)


class TestSecureToken:
    """Test SecureToken class."""

    def test_token_creation(self):
        """Test secure token creation and validation."""
        token = SecureToken(
            _value="x" * 100, _expires_at=time.time() + 3600  # Long enough token
        )
        assert not token.is_expired
        assert token.time_until_expiry > 0

    def test_token_expiry(self):
        """Test token expiry detection."""
        # Create expired token
        token = SecureToken(
            _value="x" * 100, _expires_at=time.time() - 1  # Already expired
        )
        assert token.is_expired
        assert token.time_until_expiry == 0

    def test_token_never_exposed(self):
        """Test that token value is never exposed in string representations."""
        token_value = "super_secret_token_value_12345"
        token = SecureToken(_value=token_value, _expires_at=time.time() + 3600)

        # Check string representations don't contain token
        assert token_value not in str(token)
        assert token_value not in repr(token)

    def test_token_header_generation(self):
        """Test authorization header generation."""
        token = SecureToken(
            _value="test_token_value_with_minimum_length_requirement",
            _expires_at=time.time() + 3600,
        )

        header = token.get_header()
        assert "Authorization" in header
        assert header["Authorization"].startswith("Bearer ")

    def test_expired_token_raises_error(self):
        """Test that expired token raises error when getting header."""
        token = SecureToken(_value="x" * 100, _expires_at=time.time() - 1)

        with pytest.raises(ValueError, match="expired"):
            token.get_header()

    def test_token_clearing(self):
        """Test secure token clearing."""
        token = SecureToken(
            _value="secret_value_with_minimum_length_requirement",
            _expires_at=time.time() + 3600,
        )

        token.clear()
        # After clearing, the token should not have the value attribute
        assert not hasattr(token, "_value")


class TestSecureAuthManager:
    """Test SecureAuthManager class."""

    @pytest.mark.asyncio
    async def test_credential_validation(self):
        """Test credential validation."""
        from unittest.mock import patch

        manager = SecureAuthManager()

        # Disable dev mode to test strict validation
        # Note: _validate_credentials delegates to _password_auth_handler,
        # so we need to patch the handler's _is_dev_mode_enabled, not the manager's
        with patch.object(
            manager._password_auth_handler, "_is_dev_mode_enabled", return_value=False
        ):
            # Invalid credentials - missing fields
            assert not manager._validate_credentials({})
            assert not manager._validate_credentials({"email": "test@example.com"})
            assert not manager._validate_credentials({"password": "password123"})

            # Invalid email format
            assert not manager._validate_credentials(
                {"email": "notanemail", "password": "password123"}
            )

            # Password too short
            assert not manager._validate_credentials(
                {"email": "test@example.com", "password": "short"}
            )

            # Valid credentials
            assert manager._validate_credentials(
                {"email": "test@example.com", "password": "validpassword123"}
            )

    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test rate limiting after failed attempts."""
        manager = SecureAuthManager()

        # Record multiple failures
        for _ in range(3):
            manager._record_failure()

        # Should be rate limited now
        assert manager._should_rate_limit()

        # Wait time should increase exponentially
        wait_time = manager._get_rate_limit_wait()
        assert wait_time > 0

    @pytest.mark.asyncio
    async def test_authentication_without_implementation(self):
        """Test authentication fails gracefully when not implemented."""
        from unittest.mock import patch

        manager = SecureAuthManager()

        # Disable dev mode to test actual backend authentication behavior
        # Note: authenticate delegates to _password_auth_handler which has its own
        # _is_dev_mode_enabled check, so we need to patch it there
        with patch.object(
            manager._password_auth_handler, "_is_dev_mode_enabled", return_value=False
        ):
            result = await manager.authenticate(
                {"email": "test@example.com", "password": "validpassword123"}
            )

            # Should indicate failure since backend not implemented
            assert result.success is False

    def test_is_authenticated(self):
        """Test authentication status check."""
        manager = SecureAuthManager()

        # Not authenticated initially
        assert not manager.is_authenticated()

        # Store a token
        manager._current_token = SecureToken(
            _value="x" * 100, _expires_at=time.time() + 3600
        )

        # Now authenticated
        assert manager.is_authenticated()

        # With expired token
        manager._current_token = SecureToken(
            _value="x" * 100, _expires_at=time.time() - 1
        )

        # Not authenticated
        assert not manager.is_authenticated()

    def test_constant_time_compare(self):
        """Test constant time string comparison."""
        # Equal strings
        assert constant_time_compare("test", "test")
        assert constant_time_compare("", "")

        # Different strings
        assert not constant_time_compare("test", "different")
        assert not constant_time_compare("test", "")

    def test_log_auth_event(self):
        """Test security audit logging."""
        # Should not raise any errors - verify completion
        result1 = log_auth_event("login", True, {"user": "test@example.com"})
        result2 = log_auth_event("refresh", False, {"reason": "expired"})

        # Should sanitize sensitive data
        result3 = log_auth_event(
            "login",
            True,
            {
                "user": "test@example.com",
                "password": "should_not_appear",
                "token": "should_not_appear",
            },
        )
        # Function completed successfully
        assert result1 is None  # Function returns None
        assert result2 is None
        assert result3 is None


class TestResilientClient:
    """Test ResilientClient class."""

    def test_error_classification(self):
        """Test error type classification."""
        client = ResilientClient()

        # Network errors
        assert (
            client.classify_error(Exception("Connection refused")) == ErrorType.NETWORK
        )
        assert client.classify_error(Exception("timeout")) == ErrorType.NETWORK

        # Auth errors
        assert client.classify_error(Exception("401 Unauthorized")) == ErrorType.AUTH
        assert client.classify_error(Exception("403 Forbidden")) == ErrorType.AUTH

        # Rate limit errors
        assert (
            client.classify_error(Exception("429 Too Many Requests"))
            == ErrorType.RATE_LIMIT
        )

        # Server errors
        assert (
            client.classify_error(Exception("500 Internal Server Error"))
            == ErrorType.SERVER
        )
        assert client.classify_error(Exception("502 Bad Gateway")) == ErrorType.SERVER

        # Client errors
        assert client.classify_error(Exception("400 Bad Request")) == ErrorType.CLIENT
        assert client.classify_error(Exception("404 Not Found")) == ErrorType.CLIENT

    def test_should_retry(self):
        """Test retry decision logic."""
        client = ResilientClient(max_retries=3)

        # Should retry network errors
        assert client.should_retry(Exception("Connection timeout"), 0)
        assert client.should_retry(Exception("Connection timeout"), 1)

        # Should not retry auth errors
        assert not client.should_retry(Exception("401 Unauthorized"), 0)

        # Should not retry client errors
        assert not client.should_retry(Exception("400 Bad Request"), 0)

        # Should retry server errors
        assert client.should_retry(Exception("500 Server Error"), 0)

        # Should not retry after max attempts
        assert not client.should_retry(Exception("Connection timeout"), 3)

    def test_calculate_delay(self):
        """Test exponential backoff calculation."""
        client = ResilientClient(base_delay=1.0, jitter_factor=0)

        # Exponential backoff: 1s, 2s, 4s, 8s...
        assert client.calculate_delay(0, Exception("error")) == 1.0
        assert client.calculate_delay(1, Exception("error")) == 2.0
        assert client.calculate_delay(2, Exception("error")) == 4.0
        assert client.calculate_delay(3, Exception("error")) == 8.0

        # Rate limit errors get longer delays
        rate_limit_error = Exception("429 Rate Limited")
        delay = client.calculate_delay(0, rate_limit_error)
        assert delay >= 2.0  # At least 2x base delay

    @pytest.mark.asyncio
    async def test_execute_with_retry_success(self):
        """Test successful operation execution."""
        client = ResilientClient()

        # Mock operation that succeeds
        operation = AsyncMock(return_value="success")

        result = await client.execute_with_retry(operation)
        assert result == "success"
        operation.assert_called_once()

        # Check statistics
        stats = client.get_statistics()
        assert stats["total_requests"] == 1
        assert stats["successful_requests"] == 1
        assert stats["failed_requests"] == 0

    @pytest.mark.asyncio
    async def test_execute_with_retry_eventual_success(self):
        """Test operation that fails then succeeds."""
        client = ResilientClient(base_delay=0.01)  # Short delay for testing

        # Mock operation that fails twice then succeeds
        operation = AsyncMock(
            side_effect=[
                Exception("Connection timeout"),
                Exception("Connection timeout"),
                "success",
            ]
        )

        result = await client.execute_with_retry(operation)
        assert result == "success"
        assert operation.call_count == 3

        # Check statistics
        stats = client.get_statistics()
        assert stats["total_requests"] == 1
        assert stats["successful_requests"] == 1
        assert stats["total_retries"] == 2
        assert stats["retry_successes"] == 1

    @pytest.mark.asyncio
    async def test_execute_with_retry_non_retryable(self):
        """Test non-retryable error stops immediately."""
        client = ResilientClient()

        # Mock operation that returns auth error
        operation = AsyncMock(side_effect=Exception("401 Unauthorized"))

        with pytest.raises(Exception, match="401"):
            await client.execute_with_retry(operation)

        # Should only call once (no retries for auth errors)
        operation.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_with_retry_exhausted(self):
        """Test all retries exhausted."""
        client = ResilientClient(max_retries=2, base_delay=0.01)

        # Mock operation that always fails
        operation = AsyncMock(side_effect=Exception("Connection timeout"))

        with pytest.raises(Exception, match="Connection timeout"):
            await client.execute_with_retry(operation)

        # Should call initial + 2 retries = 3 times
        assert operation.call_count == 3

        # Check statistics
        stats = client.get_statistics()
        assert stats["failed_requests"] == 1
        assert stats["total_retries"] == 2

    def test_error_sanitization(self):
        """Test error message sanitization."""
        client = ResilientClient()

        # Should sanitize bearer tokens
        error = Exception("Authorization: Bearer super_secret_token failed")
        sanitized = client._sanitize_error(error)
        assert "super_secret_token" not in sanitized
        assert "[REDACTED]" in sanitized

        # Should sanitize passwords
        error = Exception("Password: mypassword123 is invalid")
        sanitized = client._sanitize_error(error)
        assert "mypassword123" not in sanitized

        # Should sanitize emails
        error = Exception("User test@example.com not found")
        sanitized = client._sanitize_error(error)
        assert "test@example.com" not in sanitized

    def test_statistics(self):
        """Test statistics tracking and calculation."""
        client = ResilientClient()

        # Initial stats
        stats = client.get_statistics()
        assert stats["success_rate"] == 0
        assert stats["retry_rate"] == 0

        # Reset statistics
        client.reset_statistics()
        stats = client.get_statistics()
        assert stats["total_requests"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
