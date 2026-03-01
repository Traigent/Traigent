"""Tests for authentication data classes.

Tests cover:
- SecureToken: Secure token storage with automatic clearing
- _AsyncBool: Boolean value that can be awaited or used synchronously
- AuthResult: Authentication operation result
- AuthCredentials: Credentials container
"""

import time

import pytest

from traigent.cloud.auth import (
    MIN_TOKEN_LENGTH,
    TOKEN_REFRESH_THRESHOLD,
    AuthCredentials,
    AuthMode,
    AuthResult,
    AuthStatus,
    SecureToken,
    TokenExpiredError,
    _AsyncBool,
)

TEST_KEY_PREFIX = "t" + "g_"


class TestSecureToken:
    """Test SecureToken class for secure token storage."""

    def test_creation_valid_token(self):
        """Test creating a SecureToken with valid token."""
        token_value = "a" * MIN_TOKEN_LENGTH  # Minimum valid length
        expires_at = time.time() + 3600  # 1 hour from now

        token = SecureToken(_value=token_value, _expires_at=expires_at)

        assert token._token_type == "Bearer"
        assert not token.is_expired

    def test_creation_short_token_raises(self):
        """Test that creating a SecureToken with short token raises ValueError."""
        short_token = "short"  # Less than MIN_TOKEN_LENGTH
        expires_at = time.time() + 3600

        with pytest.raises(ValueError, match="Invalid token format"):
            SecureToken(_value=short_token, _expires_at=expires_at)

    def test_creation_empty_token_raises(self):
        """Test that creating a SecureToken with empty token raises ValueError."""
        with pytest.raises(ValueError, match="Invalid token format"):
            SecureToken(_value="", _expires_at=time.time() + 3600)

    def test_is_expired_before_threshold(self):
        """Test is_expired returns False when token is not near expiry."""
        token_value = "a" * MIN_TOKEN_LENGTH
        # Token expires in 1 hour, well beyond the refresh threshold
        expires_at = time.time() + 3600

        token = SecureToken(_value=token_value, _expires_at=expires_at)

        assert token.is_expired is False

    def test_is_expired_within_threshold(self):
        """Test is_expired returns True when token is within refresh threshold."""
        token_value = "a" * MIN_TOKEN_LENGTH
        # Token expires in less than TOKEN_REFRESH_THRESHOLD seconds
        expires_at = time.time() + TOKEN_REFRESH_THRESHOLD - 1

        token = SecureToken(_value=token_value, _expires_at=expires_at)

        assert token.is_expired is True

    def test_is_expired_already_expired(self):
        """Test is_expired returns True when token has already expired."""
        token_value = "a" * MIN_TOKEN_LENGTH
        # Token expired 1 hour ago
        expires_at = time.time() - 3600

        token = SecureToken(_value=token_value, _expires_at=expires_at)

        assert token.is_expired is True

    def test_time_until_expiry_positive(self):
        """Test time_until_expiry returns positive value for valid token."""
        token_value = "a" * MIN_TOKEN_LENGTH
        expires_in = 3600
        expires_at = time.time() + expires_in

        token = SecureToken(_value=token_value, _expires_at=expires_at)

        # Should be close to 3600 (allow small timing variance)
        assert 3595 <= token.time_until_expiry <= 3600

    def test_time_until_expiry_expired(self):
        """Test time_until_expiry returns 0 for expired token."""
        token_value = "a" * MIN_TOKEN_LENGTH
        expires_at = time.time() - 3600  # Already expired

        token = SecureToken(_value=token_value, _expires_at=expires_at)

        assert token.time_until_expiry == 0

    def test_get_header_valid(self):
        """Test get_header returns proper authorization header for valid token."""
        token_value = "test_token_value_12345"
        expires_at = time.time() + 3600

        token = SecureToken(_value=token_value, _expires_at=expires_at)
        header = token.get_header()

        assert header == {"Authorization": f"Bearer {token_value}"}

    def test_get_header_expired_raises(self):
        """Test get_header raises TokenExpiredError for expired token."""
        token_value = "a" * MIN_TOKEN_LENGTH
        expires_at = time.time() - 100  # Already expired

        token = SecureToken(_value=token_value, _expires_at=expires_at)

        with pytest.raises(TokenExpiredError, match="Token has expired"):
            token.get_header()

    def test_get_value_valid(self):
        """Test get_value returns token value for valid token."""
        token_value = "test_token_value_12345"
        expires_at = time.time() + 3600

        token = SecureToken(_value=token_value, _expires_at=expires_at)

        assert token.get_value() == token_value

    def test_get_value_expired_raises(self):
        """Test get_value raises TokenExpiredError for expired token."""
        token_value = "a" * MIN_TOKEN_LENGTH
        expires_at = time.time() - 100  # Already expired

        token = SecureToken(_value=token_value, _expires_at=expires_at)

        with pytest.raises(TokenExpiredError, match="Token has expired"):
            token.get_value()

    def test_clear_wipes_token(self):
        """Test clear method wipes token from memory."""
        token_value = "test_token_value_12345"
        expires_at = time.time() + 3600

        token = SecureToken(_value=token_value, _expires_at=expires_at)

        # Verify token exists
        assert token.get_value() == token_value

        # Clear the token
        token.clear()

        # Token should be wiped - accessing _value should fail or be X's
        assert not hasattr(token, "_value") or token._value != token_value

    def test_str_no_token_exposure(self):
        """Test __str__ does not expose the actual token value."""
        token_value = "super_secret_token_12345"
        expires_at = time.time() + 3600

        token = SecureToken(_value=token_value, _expires_at=expires_at)

        str_repr = str(token)

        assert "super_secret_token" not in str_repr
        assert "SecureToken" in str_repr
        assert "expires_in" in str_repr.lower() or "type" in str_repr.lower()

    def test_repr_no_token_exposure(self):
        """Test __repr__ does not expose the actual token value."""
        token_value = "super_secret_token_12345"
        expires_at = time.time() + 3600

        token = SecureToken(_value=token_value, _expires_at=expires_at)

        repr_str = repr(token)

        assert "super_secret_token" not in repr_str

    def test_custom_token_type(self):
        """Test SecureToken with custom token type."""
        token_value = "a" * MIN_TOKEN_LENGTH
        expires_at = time.time() + 3600

        token = SecureToken(
            _value=token_value, _expires_at=expires_at, _token_type="Custom"
        )
        header = token.get_header()

        assert header == {"Authorization": f"Custom {token_value}"}


class TestAsyncBool:
    """Test _AsyncBool for dual sync/async boolean behavior."""

    def test_bool_true(self):
        """Test _AsyncBool returns True synchronously."""
        async_bool = _AsyncBool(True)

        assert bool(async_bool) is True
        assert async_bool  # Also test implicit bool conversion

    def test_bool_false(self):
        """Test _AsyncBool returns False synchronously."""
        async_bool = _AsyncBool(False)

        assert bool(async_bool) is False
        assert not async_bool  # Also test implicit bool conversion

    def test_bool_truthy_value(self):
        """Test _AsyncBool with truthy value."""
        async_bool = _AsyncBool(1)  # Truthy value

        assert bool(async_bool) is True

    def test_bool_falsy_value(self):
        """Test _AsyncBool with falsy value."""
        async_bool = _AsyncBool(0)  # Falsy value

        assert bool(async_bool) is False

    @pytest.mark.asyncio
    async def test_await_true(self):
        """Test awaiting _AsyncBool returns True."""
        async_bool = _AsyncBool(True)

        result = await async_bool

        assert result is True

    @pytest.mark.asyncio
    async def test_await_false(self):
        """Test awaiting _AsyncBool returns False."""
        async_bool = _AsyncBool(False)

        result = await async_bool

        assert result is False


class TestAuthResult:
    """Test AuthResult dataclass."""

    def test_bool_conversion_success(self):
        """Test AuthResult evaluates to True when success=True."""
        result = AuthResult(
            success=True,
            status=AuthStatus.AUTHENTICATED,
        )

        assert bool(result) is True
        assert result  # Implicit bool conversion

    def test_bool_conversion_failure(self):
        """Test AuthResult evaluates to False when success=False."""
        result = AuthResult(
            success=False,
            status=AuthStatus.INVALID,
            error_message="Authentication failed",
        )

        assert bool(result) is False
        assert not result  # Implicit bool conversion

    def test_default_values(self):
        """Test AuthResult default values."""
        result = AuthResult(
            success=True,
            status=AuthStatus.AUTHENTICATED,
        )

        assert result.credentials is None
        assert result.headers == {}
        assert result.error_message is None
        assert result.expires_in is None
        assert result.retry_after is None

    def test_full_result(self):
        """Test AuthResult with all fields populated."""
        credentials = AuthCredentials(
            mode=AuthMode.API_KEY,
            api_key="example-key-12345",
        )
        headers = {"Authorization": "Bearer example-key-12345"}

        result = AuthResult(
            success=True,
            status=AuthStatus.AUTHENTICATED,
            credentials=credentials,
            headers=headers,
            expires_in=3600,
        )

        assert result.success is True
        assert result.status == AuthStatus.AUTHENTICATED
        assert result.credentials == credentials
        assert result.headers == headers
        assert result.expires_in == 3600

    def test_error_result(self):
        """Test AuthResult for error case."""
        result = AuthResult(
            success=False,
            status=AuthStatus.RATE_LIMITED,
            error_message="Too many requests",
            retry_after=60.0,
        )

        assert result.success is False
        assert result.status == AuthStatus.RATE_LIMITED
        assert result.error_message == "Too many requests"
        assert result.retry_after == 60.0


class TestAuthCredentials:
    """Test AuthCredentials dataclass."""

    def test_default_values(self):
        """Test AuthCredentials default values."""
        creds = AuthCredentials()

        assert creds.mode == AuthMode.API_KEY
        assert creds.api_key is None
        assert creds.jwt_token is None
        assert creds.refresh_token is None
        assert creds.client_id is None
        assert creds.client_secret is None
        assert creds.service_key is None
        assert creds.backend_url is None
        assert creds.expires_at is None
        assert creds.scopes == []
        assert creds.metadata == {}

    def test_api_key_credentials(self):
        """Test AuthCredentials with API key."""
        creds = AuthCredentials(
            mode=AuthMode.API_KEY,
            api_key=TEST_KEY_PREFIX + ("x" * 12),
        )

        assert creds.mode == AuthMode.API_KEY
        assert creds.api_key == TEST_KEY_PREFIX + ("x" * 12)

    def test_jwt_credentials(self):
        """Test AuthCredentials with JWT token."""
        creds = AuthCredentials(
            mode=AuthMode.JWT_TOKEN,
            jwt_token="jwt_token_placeholder",
            refresh_token="placeholder",
            expires_at=time.time() + 3600,
        )

        assert creds.mode == AuthMode.JWT_TOKEN
        assert creds.jwt_token is not None
        assert creds.refresh_token == "placeholder"
        assert creds.expires_at is not None

    def test_oauth2_credentials(self):
        """Test AuthCredentials with OAuth2 client credentials."""
        creds = AuthCredentials(
            mode=AuthMode.OAUTH2,
            client_id="client_123",
            client_secret="secret_456",
            scopes=["read", "write"],
        )

        assert creds.mode == AuthMode.OAUTH2
        assert creds.client_id == "client_123"
        assert creds.client_secret == "secret_456"
        assert creds.scopes == ["read", "write"]

    def test_service_to_service_credentials(self):
        """Test AuthCredentials for service-to-service auth."""
        creds = AuthCredentials(
            mode=AuthMode.SERVICE_TO_SERVICE,
            service_key="service_key_789",
        )

        assert creds.mode == AuthMode.SERVICE_TO_SERVICE
        assert creds.service_key == "service_key_789"

    def test_repr_masks_secrets(self):
        """Test __repr__ masks sensitive data."""
        api_key_value = f"{TEST_KEY_PREFIX}placeholder_key"
        creds = AuthCredentials(
            mode=AuthMode.API_KEY,
            api_key=api_key_value,
            client_secret="placeholder_value",
        )

        repr_str = repr(creds)

        # Should not contain actual secrets
        assert "placeholder_key" not in repr_str
        assert "placeholder_value" not in repr_str
        # Should indicate presence of secrets
        assert (
            "***" in repr_str or "api_key=True" in repr_str or "has_secret" in repr_str
        )

    def test_str_masks_secrets(self):
        """Test __str__ masks sensitive data."""
        api_key_value = f"{TEST_KEY_PREFIX}placeholder_key"
        creds = AuthCredentials(
            mode=AuthMode.API_KEY,
            api_key=api_key_value,
        )

        str_repr = str(creds)

        # Should not contain actual secrets
        assert "placeholder_key" not in str_repr

    def test_metadata(self):
        """Test AuthCredentials with metadata."""
        creds = AuthCredentials(
            mode=AuthMode.API_KEY,
            api_key="example-key",
            metadata={
                "source": "environment",
                "owner_user_id": "user_123",
            },
        )

        assert creds.metadata["source"] == "environment"
        assert creds.metadata["owner_user_id"] == "user_123"


class TestAuthStatus:
    """Test AuthStatus enum values."""

    def test_all_statuses_exist(self):
        """Test all expected status values exist."""
        assert AuthStatus.AUTHENTICATED.value == "authenticated"
        assert AuthStatus.UNAUTHENTICATED.value == "unauthenticated"
        assert AuthStatus.EXPIRED.value == "expired"
        assert AuthStatus.INVALID.value == "invalid"
        assert AuthStatus.REFRESHING.value == "refreshing"
        assert AuthStatus.RATE_LIMITED.value == "rate_limited"


class TestAuthMode:
    """Test AuthMode enum values."""

    def test_all_modes_exist(self):
        """Test all expected auth modes exist."""
        assert AuthMode.API_KEY.value == "api_key"
        assert AuthMode.JWT_TOKEN.value == "jwt_token"
        assert AuthMode.OAUTH2.value == "oauth2"
        assert AuthMode.SERVICE_TO_SERVICE.value == "service_to_service"
        assert AuthMode.DEVELOPMENT.value == "development"
        assert AuthMode.CLOUD.value == "cloud"
        assert AuthMode.EDGE_ANALYTICS.value == "edge_analytics"
        assert AuthMode.DEMO.value == "demo"
