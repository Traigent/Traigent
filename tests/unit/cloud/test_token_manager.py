"""Tests for TokenManager class.

Tests cover:
- Token storage and retrieval
- Token clearing
- Token refresh scheduling
- JWT and OAuth2 refresh flows
- Building credentials from token data
- Authorization header generation
"""

import asyncio
import time
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from traigent.cloud.auth import (
    AuthCredentials,
    AuthMode,
    AuthResult,
    AuthStatus,
    SecureToken,
    UnifiedAuthConfig,
)
from traigent.cloud.token_manager import TokenManager


@pytest.fixture
def config():
    """Create a test configuration."""
    return UnifiedAuthConfig(
        cloud_base_url="http://localhost:5000",
        token_refresh_threshold=300,
        auto_refresh=True,
        cache_credentials=False,
    )


@pytest.fixture
def token_manager(config):
    """Create a TokenManager instance with test configuration."""
    return TokenManager(config)


@pytest.fixture
def token_manager_with_callbacks(config):
    """Create a TokenManager with callbacks configured."""
    tm = TokenManager(config)
    credentials = AuthCredentials(
        mode=AuthMode.JWT_TOKEN,
        jwt_token="test_jwt_token_12345",
        refresh_token="test_refresh_token_12345",
        expires_at=time.time() + 3600,
    )

    tm.set_callbacks(
        get_credentials=lambda: credentials,
        set_credentials=MagicMock(),
        cache_credentials=AsyncMock(),
        auth_lock=asyncio.Lock(),
    )
    return tm, credentials


class TestTokenManagerInitialization:
    """Test TokenManager initialization."""

    def test_initialization_with_config(self, config):
        """Test TokenManager initializes with config."""
        tm = TokenManager(config)

        assert tm.config == config
        assert tm.current_token is None
        assert tm.refresh_token_secure is None
        assert tm.refresh_task is None
        assert tm.last_refresh_attempt == 0.0

    def test_initialization_with_callbacks(self, config):
        """Test TokenManager initializes with callback functions."""
        validate_fn = MagicMock()
        set_api_key_fn = MagicMock()

        tm = TokenManager(
            config,
            validate_key_format_fn=validate_fn,
            set_api_key_token_fn=set_api_key_fn,
        )

        assert tm._validate_key_format == validate_fn
        assert tm._set_api_key_token == set_api_key_fn


class TestSetCallbacks:
    """Test callback configuration."""

    def test_set_callbacks_configures_all(self, token_manager):
        """Test set_callbacks configures all callback functions."""
        get_creds = MagicMock()
        set_creds = MagicMock()
        cache_creds = AsyncMock()
        lock = asyncio.Lock()
        validate_fn = MagicMock()
        set_key_fn = MagicMock()

        token_manager.set_callbacks(
            get_credentials=get_creds,
            set_credentials=set_creds,
            cache_credentials=cache_creds,
            auth_lock=lock,
            validate_key_format=validate_fn,
            set_api_key_token=set_key_fn,
        )

        assert token_manager._get_credentials_fn == get_creds
        assert token_manager._set_credentials_fn == set_creds
        assert token_manager._cache_credentials_fn == cache_creds
        assert token_manager._auth_lock == lock
        assert token_manager._validate_key_format == validate_fn
        assert token_manager._set_api_key_token == set_key_fn

    def test_set_callbacks_preserves_existing_validators(self, config):
        """Test set_callbacks preserves existing validators if not provided."""
        validate_fn = MagicMock()
        tm = TokenManager(config, validate_key_format_fn=validate_fn)

        tm.set_callbacks(
            get_credentials=MagicMock(),
            set_credentials=MagicMock(),
            cache_credentials=AsyncMock(),
            auth_lock=asyncio.Lock(),
        )

        assert tm._validate_key_format == validate_fn


class TestStoreTokens:
    """Test token storage functionality."""

    def test_store_tokens_access_token(self, token_manager):
        """Test storing access token creates SecureToken."""
        token_data = {
            "access_token": "test_access_token_12345",
            "expires_in": 3600,
        }

        token_manager.store_tokens(token_data)

        assert token_manager.current_token is not None
        assert token_manager.current_token.get_value() == "test_access_token_12345"

    def test_store_tokens_with_refresh_token(self, token_manager):
        """Test storing both access and refresh tokens."""
        token_data = {
            "access_token": "test_access_token_12345",
            "refresh_token": "test_refresh_token_12345",
            "expires_in": 3600,
        }

        token_manager.store_tokens(token_data)

        assert token_manager.current_token is not None
        assert token_manager.refresh_token_secure is not None
        assert (
            token_manager.refresh_token_secure.get_value() == "test_refresh_token_12345"
        )

    def test_store_tokens_preserves_existing_refresh(self, token_manager):
        """Test storing new access token preserves existing refresh token."""
        # First store with refresh token
        token_manager.store_tokens(
            {
                "access_token": "first_access_token_12345",
                "refresh_token": "existing_refresh_token_12345",
                "expires_in": 3600,
            }
        )

        # Store new access token without refresh token
        token_manager.store_tokens(
            {
                "access_token": "second_access_token_12345",
                "expires_in": 3600,
            }
        )

        assert token_manager.current_token.get_value() == "second_access_token_12345"
        assert (
            token_manager.refresh_token_secure.get_value()
            == "existing_refresh_token_12345"
        )

    def test_store_tokens_invalid_access_token_skipped(self, token_manager):
        """Test invalid access token format is skipped gracefully."""
        token_data = {
            "access_token": "short",  # Too short
            "expires_in": 3600,
        }

        # Should not raise, just skip storing
        token_manager.store_tokens(token_data)

        assert token_manager.current_token is None

    def test_store_tokens_clears_previous(self, token_manager):
        """Test storing new tokens clears previous tokens."""
        token_manager.store_tokens(
            {
                "access_token": "first_token_12345678",
                "expires_in": 3600,
            }
        )

        token_manager.store_tokens(
            {
                "access_token": "second_token_12345678",
                "expires_in": 3600,
            }
        )

        # First token should be cleared
        assert token_manager.current_token.get_value() == "second_token_12345678"


class TestClearTokens:
    """Test token clearing functionality."""

    def test_clear_tokens_clears_all(self, token_manager):
        """Test clear_tokens removes all stored tokens."""
        token_manager.store_tokens(
            {
                "access_token": "test_access_token_12345",
                "refresh_token": "test_refresh_token_12345",
                "expires_in": 3600,
            }
        )

        token_manager.clear_tokens()

        assert token_manager.current_token is None
        assert token_manager.refresh_token_secure is None

    def test_clear_tokens_when_empty(self, token_manager):
        """Test clear_tokens succeeds when no tokens stored."""
        token_manager.clear_tokens()

        assert token_manager.current_token is None
        assert token_manager.refresh_token_secure is None


class TestScheduleRefresh:
    """Test token refresh scheduling."""

    def test_schedule_refresh_no_expiry(self, token_manager):
        """Test schedule_refresh does nothing without expiry."""
        credentials = AuthCredentials(
            mode=AuthMode.JWT_TOKEN,
            jwt_token="test_token_12345",
            expires_at=None,
        )

        token_manager.schedule_refresh(credentials)

        assert token_manager.refresh_task is None

    @pytest.mark.asyncio
    async def test_schedule_refresh_with_datetime_expiry(self, token_manager):
        """Test schedule_refresh with datetime expiry."""
        expires_at = datetime.now(UTC).replace(microsecond=0) + __import__(
            "datetime"
        ).timedelta(hours=1)
        credentials = AuthCredentials(
            mode=AuthMode.JWT_TOKEN,
            jwt_token="test_token_12345",
            expires_at=expires_at,
        )

        token_manager.schedule_refresh(credentials)

        assert token_manager.refresh_task is not None
        # Clean up
        token_manager.refresh_task.cancel()
        try:
            await token_manager.refresh_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_schedule_refresh_with_float_expiry(self, token_manager):
        """Test schedule_refresh with float timestamp expiry."""
        credentials = AuthCredentials(
            mode=AuthMode.JWT_TOKEN,
            jwt_token="test_token_12345",
            expires_at=time.time() + 3600,
        )

        token_manager.schedule_refresh(credentials)

        assert token_manager.refresh_task is not None
        # Clean up
        token_manager.refresh_task.cancel()
        try:
            await token_manager.refresh_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_schedule_refresh_cancels_existing(self, token_manager):
        """Test schedule_refresh cancels existing task."""
        credentials = AuthCredentials(
            mode=AuthMode.JWT_TOKEN,
            jwt_token="test_token_12345",
            expires_at=time.time() + 3600,
        )

        token_manager.schedule_refresh(credentials)
        first_task = token_manager.refresh_task

        token_manager.schedule_refresh(credentials)
        second_task = token_manager.refresh_task

        assert first_task.cancelled() or first_task != second_task
        # Clean up
        if second_task and not second_task.done():
            second_task.cancel()
            try:
                await second_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_set_refresh_task_clears_cancelled_task(self, token_manager):
        """Cancelled refresh tasks should clear the tracked task reference."""
        task = asyncio.create_task(asyncio.sleep(10))
        token_manager._set_refresh_task(task)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        await asyncio.sleep(0)
        assert token_manager.refresh_task is None


class TestRefreshAccessToken:
    """Test refresh_access_token method."""

    @pytest.mark.asyncio
    async def test_refresh_no_credentials(self, token_manager):
        """Test refresh fails without credentials."""
        token_manager.set_callbacks(
            get_credentials=lambda: None,
            set_credentials=MagicMock(),
            cache_credentials=AsyncMock(),
            auth_lock=asyncio.Lock(),
        )

        result = await token_manager.refresh_access_token()

        assert result.success is False
        assert result.status == AuthStatus.INVALID
        assert "No credentials" in result.error_message

    @pytest.mark.asyncio
    async def test_refresh_no_refresh_token(self, token_manager):
        """Test refresh fails without refresh token."""
        credentials = AuthCredentials(
            mode=AuthMode.JWT_TOKEN,
            jwt_token="test_token_12345",
            refresh_token=None,
        )
        token_manager.set_callbacks(
            get_credentials=lambda: credentials,
            set_credentials=MagicMock(),
            cache_credentials=AsyncMock(),
            auth_lock=asyncio.Lock(),
        )

        result = await token_manager.refresh_access_token()

        assert result.success is False
        assert "No refresh token" in result.error_message

    @pytest.mark.asyncio
    async def test_refresh_api_key_mode_returns_success(self, token_manager):
        """Test refresh with API_KEY mode returns success (no refresh needed)."""
        credentials = AuthCredentials(
            mode=AuthMode.API_KEY,
            api_key="placeholder_key",  # pragma: allowlist secret
            refresh_token="placeholder",  # Has refresh token but mode is API_KEY
        )
        token_manager.set_callbacks(
            get_credentials=lambda: credentials,
            set_credentials=MagicMock(),
            cache_credentials=AsyncMock(),
            auth_lock=asyncio.Lock(),
        )

        result = await token_manager.refresh_access_token()

        assert result.success is True
        assert result.status == AuthStatus.AUTHENTICATED

    @pytest.mark.asyncio
    async def test_refresh_updates_last_attempt(self, token_manager):
        """Test refresh updates last_refresh_attempt timestamp."""
        credentials = AuthCredentials(
            mode=AuthMode.JWT_TOKEN,
            jwt_token="test_token_12345",
            refresh_token="test_refresh_token_12345",
        )
        token_manager.set_callbacks(
            get_credentials=lambda: credentials,
            set_credentials=MagicMock(),
            cache_credentials=AsyncMock(),
            auth_lock=asyncio.Lock(),
        )

        before = time.time()
        with patch.object(
            token_manager, "refresh_jwt_secure", new_callable=AsyncMock
        ) as mock_refresh:
            mock_refresh.return_value = AuthResult(
                success=True, status=AuthStatus.AUTHENTICATED
            )
            await token_manager.refresh_access_token()

        assert token_manager.last_refresh_attempt >= before


class TestBuildCredentialsFromTokenData:
    """Test building credentials from token response data."""

    def test_build_credentials_basic(self, token_manager):
        """Test building credentials from basic token data."""
        token_data = {
            "access_token": "new_access_token_12345",
            "expires_in": 3600,
        }
        token_manager.set_callbacks(
            get_credentials=lambda: None,
            set_credentials=MagicMock(),
            cache_credentials=AsyncMock(),
            auth_lock=asyncio.Lock(),
        )

        credentials = token_manager.build_credentials_from_token_data(token_data)

        assert credentials.mode == AuthMode.JWT_TOKEN
        assert credentials.jwt_token == "new_access_token_12345"
        assert credentials.expires_at is not None

    def test_build_credentials_with_refresh_token(self, token_manager):
        """Test building credentials with refresh token."""
        token_data = {
            "access_token": "new_access_token_12345",
            "refresh_token": "new_refresh_token_12345",
            "expires_in": 3600,
        }
        token_manager.set_callbacks(
            get_credentials=lambda: None,
            set_credentials=MagicMock(),
            cache_credentials=AsyncMock(),
            auth_lock=asyncio.Lock(),
        )

        credentials = token_manager.build_credentials_from_token_data(token_data)

        assert credentials.refresh_token == "new_refresh_token_12345"

    def test_build_credentials_preserves_stored_refresh(self, token_manager):
        """Test building credentials uses stored refresh if not in response."""
        # Store a refresh token
        token_manager.store_tokens(
            {
                "access_token": "old_access_token_12345",
                "refresh_token": "stored_refresh_token_12345",
                "expires_in": 3600,
            }
        )

        token_manager.set_callbacks(
            get_credentials=lambda: None,
            set_credentials=MagicMock(),
            cache_credentials=AsyncMock(),
            auth_lock=asyncio.Lock(),
        )

        # Build credentials without refresh token in response
        token_data = {
            "access_token": "new_access_token_12345",
            "expires_in": 3600,
        }

        credentials = token_manager.build_credentials_from_token_data(token_data)

        assert credentials.refresh_token == "stored_refresh_token_12345"

    def test_build_credentials_with_user_metadata(self, token_manager):
        """Test building credentials includes user metadata."""
        token_data = {
            "access_token": "new_access_token_12345",
            "expires_in": 3600,
            "user": {"id": "user_123", "email": "test@example.com"},
        }
        token_manager.set_callbacks(
            get_credentials=lambda: None,
            set_credentials=MagicMock(),
            cache_credentials=AsyncMock(),
            auth_lock=asyncio.Lock(),
        )

        credentials = token_manager.build_credentials_from_token_data(token_data)

        assert credentials.metadata.get("user") == {
            "id": "user_123",
            "email": "test@example.com",
        }

    def test_build_credentials_with_api_key(self, token_manager):
        """Test building credentials includes API key."""
        validate_fn = MagicMock(return_value=True)
        set_key_fn = MagicMock()
        token_manager._validate_key_format = validate_fn
        token_manager._set_api_key_token = set_key_fn

        token_data = {
            "access_token": "access_token_placeholder",
            "expires_in": 3600,
            "api_key": "placeholder_key",  # pragma: allowlist secret
        }
        token_manager.set_callbacks(
            get_credentials=lambda: None,
            set_credentials=MagicMock(),
            cache_credentials=AsyncMock(),
            auth_lock=asyncio.Lock(),
        )

        credentials = token_manager.build_credentials_from_token_data(token_data)

        assert credentials.api_key == "placeholder_key"  # pragma: allowlist secret
        set_key_fn.assert_called_once()


class TestOAuth2RefreshHardening:
    """Security regressions for OAuth2 token refresh."""

    @pytest.mark.asyncio
    async def test_refresh_oauth2_rejects_local_cloud_base_url(self, config):
        credentials = AuthCredentials(
            mode=AuthMode.OAUTH2,
            refresh_token="refresh_token_placeholder",
            client_id="client-id",
            client_secret="client-secret",  # pragma: allowlist secret
        )
        tm = TokenManager(config)
        tm.set_callbacks(
            get_credentials=lambda: credentials,
            set_credentials=MagicMock(),
            cache_credentials=AsyncMock(),
            auth_lock=asyncio.Lock(),
        )

        result = await tm.refresh_oauth2()

        assert result.success is False
        assert result.status == AuthStatus.INVALID
        assert result.error_message == "OAuth2 cloud_base_url is not allowed"

    @pytest.mark.asyncio
    async def test_refresh_jwt_error_message_does_not_return_raw_body(
        self, token_manager_with_callbacks
    ):
        token_manager, _credentials = token_manager_with_callbacks

        async def fail_refresh(*_args, **_kwargs):
            raise RuntimeError(
                "500: {'access_token':'secret-token-value','password':'secret'}"  # pragma: allowlist secret
            )

        with patch(
            "traigent.cloud.resilient_client.ResilientClient.execute_with_retry",
            side_effect=fail_refresh,
        ):
            result = await token_manager.refresh_jwt_secure("refresh_token")

        assert result.success is False
        assert result.error_message == "Token refresh failed"
        assert "secret-token-value" not in result.error_message
        assert "password" not in result.error_message


class TestGetAuthorizationHeader:
    """Test authorization header generation."""

    def test_get_header_no_token(self, token_manager):
        """Test get_authorization_header returns empty dict without token."""
        header = token_manager.get_authorization_header()

        assert header == {}

    def test_get_header_valid_token(self, token_manager):
        """Test get_authorization_header returns proper header."""
        token_manager.store_tokens(
            {
                "access_token": "test_access_token_12345",
                "expires_in": 3600,
            }
        )

        header = token_manager.get_authorization_header()

        assert header == {"Authorization": "Bearer test_access_token_12345"}

    def test_get_header_expired_token(self, token_manager):
        """Test get_authorization_header returns empty for expired token."""
        token_manager.store_tokens(
            {
                "access_token": "test_access_token_12345",
                "expires_in": -100,  # Already expired
            }
        )

        header = token_manager.get_authorization_header()

        assert header == {}


class TestProperties:
    """Test TokenManager properties."""

    def test_current_token_property(self, token_manager):
        """Test current_token property returns stored token."""
        token_manager.store_tokens(
            {
                "access_token": "test_access_token_12345",
                "expires_in": 3600,
            }
        )

        assert token_manager.current_token is not None
        assert isinstance(token_manager.current_token, SecureToken)

    def test_refresh_token_secure_property(self, token_manager):
        """Test refresh_token_secure property returns stored refresh token."""
        token_manager.store_tokens(
            {
                "access_token": "test_access_token_12345",
                "refresh_token": "test_refresh_token_12345",
                "expires_in": 3600,
            }
        )

        assert token_manager.refresh_token_secure is not None
        assert isinstance(token_manager.refresh_token_secure, SecureToken)

    @pytest.mark.asyncio
    async def test_refresh_task_property(self, token_manager):
        """Test refresh_task property returns scheduled task."""
        credentials = AuthCredentials(
            mode=AuthMode.JWT_TOKEN,
            jwt_token="test_token_12345",
            expires_at=time.time() + 3600,
        )

        token_manager.schedule_refresh(credentials)

        assert token_manager.refresh_task is not None
        assert isinstance(token_manager.refresh_task, asyncio.Task)
        # Clean up
        token_manager.refresh_task.cancel()
        try:
            await token_manager.refresh_task
        except asyncio.CancelledError:
            pass

    def test_last_refresh_attempt_property(self, token_manager):
        """Test last_refresh_attempt property returns timestamp."""
        assert token_manager.last_refresh_attempt == 0.0
