"""Tests for Traigent Cloud Service authentication."""

import asyncio
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from traigent.cloud.auth import (
    APIKey,
    AuthCredentials,
    AuthenticationError,
    AuthManager,
    AuthMode,
    AuthStatus,
    UnifiedAuthConfig,
)


def _mock_backend_validate(success: bool = True):
    """Patch ``AuthManager._validate_api_key_with_backend`` for offline tests.

    Returns a context manager. Use ``with _mock_backend_validate():`` around
    code that exercises ``_authenticate_api_key`` so tests do not require a
    live backend. ``success=False`` simulates a backend rejection.
    """

    async def _ok(self, api_key):  # noqa: ARG001
        return None

    async def _fail(self, api_key):  # noqa: ARG001
        return "mocked-failure"

    return patch.object(
        AuthManager,
        "_validate_api_key_with_backend",
        new=_ok if success else _fail,
    )


# Shared dev-token used by tests exercising AuthMode.DEVELOPMENT.
_DEV_TOKEN = "test-dev-token-shared-secret"


def _dev_env():
    """Set TRAIGENT_DEV_AUTH_TOKEN for the duration of a test."""
    return patch.dict(
        "os.environ", {"TRAIGENT_DEV_AUTH_TOKEN": _DEV_TOKEN}, clear=False
    )


class TestAPIKey:
    """Test cases for APIKey dataclass."""

    def test_api_key_creation(self):
        """Test APIKey creation with all parameters."""
        created_at = datetime.now(UTC)
        expires_at = created_at + timedelta(days=365)

        api_key = APIKey(
            key="tg_test_key_12345",
            name="test_key",
            created_at=created_at,
            expires_at=expires_at,
            permissions={"optimize": True, "analytics": False},
            usage_limit=1000,
        )

        assert api_key.key == "tg_test_key_12345"
        assert api_key.name == "test_key"
        assert api_key.created_at == created_at
        assert api_key.expires_at == expires_at
        assert api_key.permissions == {"optimize": True, "analytics": False}
        assert api_key.usage_limit == 1000

    def test_api_key_default_permissions(self):
        """Test APIKey with default permissions."""
        api_key = APIKey(key="test_key", name="test", created_at=datetime.now(UTC))

        expected_permissions = {"optimize": True, "analytics": True, "billing": False}
        assert api_key.permissions == expected_permissions

    def test_api_key_is_valid_not_expired(self):
        """Test API key validity when not expired."""
        api_key = APIKey(
            key="valid_key",
            name="test",
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(days=1),
        )

        assert api_key.is_valid() is True

    def test_api_key_is_valid_expired(self):
        """Test API key validity when expired."""
        api_key = APIKey(
            key="expired_key",
            name="test",
            created_at=datetime.now(UTC) - timedelta(days=2),
            expires_at=datetime.now(UTC) - timedelta(days=1),
        )

        assert api_key.is_valid() is False

    def test_api_key_is_valid_no_expiry(self):
        """Test API key validity with no expiry date."""
        api_key = APIKey(
            key="no_expiry_key",
            name="test",
            created_at=datetime.now(UTC),
            expires_at=None,
        )

        assert api_key.is_valid() is True

    def test_api_key_is_valid_empty_key(self):
        """Test API key validity with empty key."""
        api_key = APIKey(key="", name="test", created_at=datetime.now(UTC))

        assert api_key.is_valid() is False

    def test_api_key_has_permission(self):
        """Test permission checking."""
        api_key = APIKey(
            key="test_key",
            name="test",
            created_at=datetime.now(UTC),
            permissions={"optimize": True, "analytics": False},
        )

        assert api_key.has_permission("optimize") is True
        assert api_key.has_permission("analytics") is False
        assert api_key.has_permission("nonexistent") is False


class TestAuthManager:
    """Regression coverage for AuthManager."""

    def test_initialization_with_key(self):
        sample_key = "tg_" + "a" * 61
        manager = AuthManager(api_key=sample_key)
        assert manager.has_api_key()
        preview = manager.get_api_key_preview()
        assert preview is not None and preview.startswith("tg_")

    def test_initialization_from_env(self):
        env_key = "tg_" + "b" * 61
        with patch.dict("os.environ", {"TRAIGENT_API_KEY": env_key}):
            manager = AuthManager()
            assert manager.has_api_key()
            preview = manager.get_api_key_preview()
            assert preview is not None and preview.startswith("tg_")

    def test_initialization_without_key(self):
        with patch.dict("os.environ", {}, clear=True):
            manager = AuthManager()
            assert manager.has_api_key() is False

    @pytest.mark.asyncio
    async def test_authenticate_success(self):
        manager = AuthManager(api_key="tg_" + "x" * 61)
        with _mock_backend_validate():
            result = await manager.authenticate()

        assert result.success is True
        assert manager._authenticated is True
        assert manager._api_key is not None

    @pytest.mark.asyncio
    async def test_authenticate_without_key(self):
        with patch.dict("os.environ", {}, clear=True):
            manager = AuthManager()
            result = await manager.authenticate()

            assert result.success is False
            assert manager._authenticated is False

    @pytest.mark.asyncio
    async def test_authenticate_invalid_key(self):
        manager = AuthManager(api_key="short")
        result = await manager.authenticate()

        assert result.success is False
        assert manager._authenticated is False

    @pytest.mark.asyncio
    async def test_is_authenticated(self):
        manager = AuthManager(api_key="tg_" + "x" * 61)
        with _mock_backend_validate():
            await manager.authenticate()

        assert await manager.is_authenticated() is True

    @pytest.mark.asyncio
    async def test_get_headers_success(self):
        manager = AuthManager(api_key="tg_" + "x" * 61)
        with _mock_backend_validate():
            auth_result = await manager.authenticate()
            # Explicitly assert authentication succeeded BEFORE asking for headers.
            assert auth_result.success is True
            assert await manager.is_authenticated() is True

            headers = await manager.get_auth_headers()
        # API key may be in X-API-Key or Authorization header
        assert "X-API-Key" in headers or "Authorization" in headers

    @pytest.mark.asyncio
    async def test_get_headers_raises_on_auth_failure(self):
        """B4: get_auth_headers must fail closed when authenticate() fails.

        Previously, a backend-rejected key still produced X-API-Key /
        Authorization headers via a fallback path. The fallback was removed,
        so an AuthenticationError must be raised instead.
        """
        manager = AuthManager(api_key="tg_" + "x" * 61)
        with _mock_backend_validate(success=False):
            with pytest.raises(AuthenticationError):
                await manager.get_auth_headers()

    @pytest.mark.asyncio
    async def test_get_headers_without_credentials(self):
        with patch.dict("os.environ", {}, clear=True):
            manager = AuthManager()
            with pytest.raises(AuthenticationError):
                await manager.get_auth_headers()

    @pytest.mark.asyncio
    async def test_refresh_authentication_without_token(self):
        manager = AuthManager(api_key="tg_" + "x" * 61)
        await manager.authenticate()
        result = await manager.refresh_authentication()
        assert result.success is False

    @pytest.mark.asyncio
    async def test_logout_clears_state(self):
        manager = AuthManager(api_key="tg_" + "x" * 61)
        await manager.authenticate()

        success = await manager.logout()
        assert success is True
        assert manager._credentials is None
        assert manager._auth_status == AuthStatus.UNAUTHENTICATED
        assert manager._authenticated is False

    def test_set_api_key_updates_credentials(self):
        manager = AuthManager()
        manager.set_api_key("tg_" + "x" * 61)
        assert manager.has_api_key()
        preview = manager.get_api_key_preview()
        assert preview is not None and preview.startswith("tg_")

    def test_get_api_key_info_authenticated(self):
        manager = AuthManager(api_key="tg_" + "x" * 61)
        manager._api_key = APIKey(
            key="tg_" + "x" * 61,
            name="test_key",
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(days=30),
        )

        info = manager.get_api_key_info()
        assert info is not None
        assert info["name"] == "test_key"
        assert info["is_valid"] is True

    def test_get_api_key_info_without_authentication(self):
        with patch.dict("os.environ", {}, clear=True):
            manager = AuthManager()
            assert manager.get_api_key_info() is None

    def test_rotate_api_key_updates_status(self):
        original_key = "tg_" + "x" * 61
        new_key = "tg_" + "y" * 61
        manager = AuthManager(api_key=original_key)
        initial_status = manager.get_api_key_status()
        assert initial_status["preview"].startswith("tg_")

        manager.rotate_api_key(new_key)

        rotated_status = manager.get_api_key_status()
        assert rotated_status["preview"].startswith("tg_")
        assert "y" in rotated_status["preview"]
        assert manager._api_key_last_rotated is not None

    def test_get_owner_fingerprint_includes_metadata(self):
        manager = AuthManager()
        api_key_value = "tg_" + "o" * 61
        metadata = {
            "owner_user_id": "owner-123",
            "owner_api_key_id": "key-789",
            "created_by": "creator-456",
            "owner_scope": ["optimize"],
            "source": "unit-test",
        }

        manager._credentials = AuthCredentials(
            mode=AuthMode.API_KEY,
            api_key=api_key_value,
            metadata=metadata,
        )
        manager._api_key_preview = "tg_prev_1234"
        manager._api_key_source = "unit-test"

        fingerprint = manager.get_owner_fingerprint()

        assert fingerprint["owner_user_id"] == "owner-123"
        assert fingerprint["owner_api_key_id"] == "key-789"
        assert fingerprint["created_by"] == "creator-456"
        assert fingerprint["owner_scope"] == ["optimize"]
        assert fingerprint["credential_source"] == "unit-test"
        assert fingerprint["owner_api_key_preview"] == "tg_prev_1234"
        assert fingerprint["auth_mode"] == AuthMode.API_KEY.value
        assert fingerprint["metadata_present"] is True

    def test_get_owner_fingerprint_fallbacks(self):
        manager = AuthManager()
        api_key_value = "tg_" + "p" * 61
        manager._credentials = AuthCredentials(
            mode=AuthMode.API_KEY,
            api_key=api_key_value,
            metadata={"user_id": "legacy-user", "scopes": ["analytics"]},
        )
        manager._api_key_source = "environment"

        fingerprint = manager.get_owner_fingerprint()

        assert fingerprint["owner_user_id"] == "legacy-user"
        assert fingerprint["created_by"] == "legacy-user"
        assert fingerprint["owner_api_key_id"] is None
        assert fingerprint["owner_scope"] == ["analytics"]
        assert fingerprint["credential_source"] == "environment"

    def test_api_key_rotation_health_checks(self):
        config = UnifiedAuthConfig(
            api_key_default_ttl_days=90,
            api_key_warning_days=10,
            api_key_critical_days=3,
        )
        manager = AuthManager(config)

        # Set key expiring in five days -> warning
        warning_expiry = datetime.now(UTC) + timedelta(days=5)
        manager.set_api_key("tg_" + "w" * 61, expires_at=warning_expiry)

        status = manager.get_api_key_status()
        assert status["state"] == "warning"
        assert manager.check_api_key_rotation() is False

        # Rotate with key expiring in one day -> critical
        critical_expiry = datetime.now(UTC) + timedelta(days=1)
        manager.rotate_api_key("tg_" + "c" * 61, expires_at=critical_expiry)

        status = manager.get_api_key_status()
        assert status["state"] == "critical"
        assert manager.check_api_key_rotation() is False

        # Rotate with already expired key -> expired
        expired_expiry = datetime.now(UTC) - timedelta(days=1)
        manager.rotate_api_key("tg_" + "e" * 61, expires_at=expired_expiry)

        status = manager.get_api_key_status()
        assert status["state"] == "expired"
        assert manager.check_api_key_rotation() is False


class TestDemoAuthManager:
    """Maintain historical test names for demo-mode coverage."""

    DEMO_KEY = "tg_demo_" + "x" * 56

    @pytest.mark.asyncio
    async def test_demo_auth_manager_initialization(self):
        manager = AuthManager(api_key=self.DEMO_KEY)
        with _mock_backend_validate():
            result = await manager.authenticate()

        assert result.success is True
        assert manager.has_api_key()
        assert manager._authenticated is True

    @pytest.mark.asyncio
    async def test_demo_authenticate_always_succeeds(self):
        manager = AuthManager(api_key=self.DEMO_KEY)
        with _mock_backend_validate():
            result = await manager.authenticate()
        assert result.success is True

    @pytest.mark.asyncio
    async def test_demo_auth_permissions(self):
        manager = AuthManager(api_key=self.DEMO_KEY)
        with _mock_backend_validate():
            await manager.authenticate()
        assert manager._api_key is not None
        assert manager._api_key.has_permission("optimize") is True

    @pytest.mark.asyncio
    async def test_demo_auth_headers(self):
        manager = AuthManager(api_key=self.DEMO_KEY)
        with _mock_backend_validate():
            await manager.authenticate()
        headers = await manager.get_auth_headers()
        # API key may be in X-API-Key or Authorization header
        assert "X-API-Key" in headers or "Authorization" in headers


class TestAuthenticationError:
    """Test cases for AuthenticationError exception."""

    def test_authentication_error_creation(self):
        error = AuthenticationError("Authentication failed")
        assert str(error) == "Authentication failed"

    def test_authentication_error_inheritance(self):
        error = AuthenticationError("Test error")
        assert isinstance(error, Exception)


class TestAuthenticationModes:
    """Test cases for different authentication modes (_authenticate_by_mode dispatch)."""

    @pytest.mark.asyncio
    async def test_authenticate_by_mode_api_key(self):
        """Test _authenticate_by_mode dispatches to API key authentication."""
        manager = AuthManager(api_key="tg_" + "x" * 61)
        credentials = AuthCredentials(
            mode=AuthMode.API_KEY,
            api_key="tg_" + "x" * 61,
        )
        with _mock_backend_validate():
            result = await manager._authenticate_by_mode(credentials)

        assert result.success is True
        assert result.status == AuthStatus.AUTHENTICATED

    @pytest.mark.asyncio
    async def test_authenticate_by_mode_unsupported(self):
        """Test _authenticate_by_mode returns error for unsupported mode."""
        manager = AuthManager()
        # Create credentials with a mode that has no handler (CLOUD mode)
        credentials = AuthCredentials(
            mode=AuthMode.CLOUD,
        )
        result = await manager._authenticate_by_mode(credentials)

        assert result.success is False
        assert result.status == AuthStatus.INVALID
        assert "Unsupported authentication mode" in result.error_message

    @pytest.mark.asyncio
    async def test_authenticate_jwt_success(self):
        """Test JWT authentication with valid token."""
        manager = AuthManager()

        # Create a valid-looking JWT token (mock validation)
        jwt_token = (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0Iiwibm" + "a" * 50
        )

        credentials = AuthCredentials(
            mode=AuthMode.JWT_TOKEN,
            jwt_token=jwt_token,
        )

        # Mock the JWT validator to return success (patched at import location)
        with patch(
            "traigent.security.jwt_validator.get_secure_jwt_validator"
        ) as mock_validator:
            mock_result = type(
                "ValidationResult",
                (),
                {
                    "valid": True,
                    "claims": {"sub": "test"},
                    "warnings": [],
                    "expires_at": None,
                    "error": None,
                },
            )()
            mock_validator.return_value.validate_token.return_value = mock_result

            result = await manager._authenticate_jwt(credentials)

            assert result.success is True
            assert result.status == AuthStatus.AUTHENTICATED
            assert "Authorization" in result.headers

    @pytest.mark.asyncio
    async def test_authenticate_jwt_missing_token(self):
        """Test JWT authentication fails when token is not provided."""
        manager = AuthManager()
        credentials = AuthCredentials(
            mode=AuthMode.JWT_TOKEN,
            jwt_token=None,
        )

        result = await manager._authenticate_jwt(credentials)

        assert result.success is False
        assert result.status == AuthStatus.INVALID
        assert "JWT token not provided" in result.error_message

    @pytest.mark.asyncio
    async def test_authenticate_jwt_invalid_token(self):
        """Test JWT authentication fails with invalid token."""
        manager = AuthManager()
        credentials = AuthCredentials(
            mode=AuthMode.JWT_TOKEN,
            jwt_token="invalid_jwt_token",
        )

        # Mock the JWT validator to return failure
        with patch(
            "traigent.security.jwt_validator.get_secure_jwt_validator"
        ) as mock_validator:
            mock_result = type(
                "ValidationResult",
                (),
                {
                    "valid": False,
                    "error": "Invalid token format",
                    "warnings": [],
                    "expires_at": None,
                },
            )()
            mock_validator.return_value.validate_token.return_value = mock_result

            result = await manager._authenticate_jwt(credentials)

            assert result.success is False
            assert result.status == AuthStatus.INVALID

    @pytest.mark.asyncio
    async def test_authenticate_oauth2_success(self):
        """Test OAuth2 authentication with valid credentials."""
        manager = AuthManager()
        credentials = AuthCredentials(
            mode=AuthMode.OAUTH2,
            client_id="test_client_id",
            client_secret="test_client_secret",
            scopes=["read", "write"],
        )

        # Mock the OAuth2 client credentials flow
        mock_token_data = {
            "access_token": "mock_access_token_" + "x" * 50,
            "refresh_token": "mock_refresh_token",
            "expires_in": 3600,
        }

        with patch.object(
            manager, "_oauth2_client_credentials_flow", return_value=mock_token_data
        ):
            result = await manager._authenticate_oauth2(credentials)

            assert result.success is True
            assert result.status == AuthStatus.AUTHENTICATED
            assert result.expires_in == 3600
            assert "Authorization" in result.headers

    @pytest.mark.asyncio
    async def test_authenticate_oauth2_missing_credentials(self):
        """Test OAuth2 authentication fails when credentials missing."""
        manager = AuthManager()

        # Missing client_id
        credentials = AuthCredentials(
            mode=AuthMode.OAUTH2,
            client_id=None,
            client_secret="test_secret",
        )
        result = await manager._authenticate_oauth2(credentials)

        assert result.success is False
        assert result.status == AuthStatus.INVALID
        assert "OAuth2 client credentials not provided" in result.error_message

        # Missing client_secret
        credentials = AuthCredentials(
            mode=AuthMode.OAUTH2,
            client_id="test_id",
            client_secret=None,
        )
        result = await manager._authenticate_oauth2(credentials)

        assert result.success is False
        assert result.status == AuthStatus.INVALID

    @pytest.mark.asyncio
    async def test_authenticate_oauth2_with_existing_token(self):
        """Test OAuth2 uses existing JWT token if available."""
        manager = AuthManager()
        jwt_token = "existing_token_" + "x" * 50

        credentials = AuthCredentials(
            mode=AuthMode.OAUTH2,
            client_id="test_client_id",
            client_secret="test_client_secret",
            jwt_token=jwt_token,  # Existing token
        )

        # Mock the JWT authentication since OAuth2 delegates to it
        with patch(
            "traigent.security.jwt_validator.get_secure_jwt_validator"
        ) as mock_validator:
            mock_result = type(
                "ValidationResult",
                (),
                {
                    "valid": True,
                    "claims": {"sub": "test"},
                    "warnings": [],
                    "expires_at": None,
                    "error": None,
                },
            )()
            mock_validator.return_value.validate_token.return_value = mock_result

            result = await manager._authenticate_oauth2(credentials)

            # Should delegate to JWT auth
            assert result.success is True
            assert result.status == AuthStatus.AUTHENTICATED

    @pytest.mark.asyncio
    async def test_authenticate_service_to_service_success(self):
        """Test service-to-service authentication with valid service key."""
        manager = AuthManager()
        credentials = AuthCredentials(
            mode=AuthMode.SERVICE_TO_SERVICE,
            service_key="test_service_key_12345",
        )
        manager._credentials = credentials

        result = await manager._authenticate_service_to_service(credentials)

        assert result.success is True
        assert result.status == AuthStatus.AUTHENTICATED
        assert "Authorization" in result.headers
        assert result.headers["Authorization"].startswith("Service ")
        assert result.headers["X-Service-Key"] == "test_service_key_12345"

    @pytest.mark.asyncio
    async def test_authenticate_service_to_service_missing_key(self):
        """Test service-to-service authentication fails when key missing."""
        manager = AuthManager()
        credentials = AuthCredentials(
            mode=AuthMode.SERVICE_TO_SERVICE,
            service_key=None,
        )

        result = await manager._authenticate_service_to_service(credentials)

        assert result.success is False
        assert result.status == AuthStatus.INVALID
        assert "Service key not provided" in result.error_message

    @pytest.mark.asyncio
    async def test_authenticate_development_success(self):
        """Test development mode auth succeeds when env token + creds match."""
        manager = AuthManager()
        credentials = AuthCredentials(
            mode=AuthMode.DEVELOPMENT,
            metadata={"dev_user": "test_developer", "dev_token": _DEV_TOKEN},
        )

        with _dev_env():
            result = await manager._authenticate_development(credentials)

        assert result.success is True
        assert result.status == AuthStatus.AUTHENTICATED
        assert result.headers["X-Development-Mode"] == "true"
        assert result.headers["X-Dev-User"] == "test_developer"

    @pytest.mark.asyncio
    async def test_authenticate_development_default_user(self):
        """Test development mode uses default user if not specified."""
        manager = AuthManager()
        credentials = AuthCredentials(
            mode=AuthMode.DEVELOPMENT,
            metadata={"dev_token": _DEV_TOKEN},  # No dev_user specified
        )

        with _dev_env():
            result = await manager._authenticate_development(credentials)

        assert result.success is True
        assert result.headers["X-Dev-User"] == "developer"

    @pytest.mark.asyncio
    async def test_authenticate_development_rejects_without_env(self):
        """Development mode must reject when TRAIGENT_DEV_AUTH_TOKEN is unset."""
        manager = AuthManager()
        credentials = AuthCredentials(
            mode=AuthMode.DEVELOPMENT,
            metadata={"dev_token": "anything"},
        )

        with patch.dict("os.environ", {}, clear=False) as _:
            import os as _os

            _os.environ.pop("TRAIGENT_DEV_AUTH_TOKEN", None)
            result = await manager._authenticate_development(credentials)

        assert result.success is False
        assert result.status == AuthStatus.INVALID
        assert "TRAIGENT_DEV_AUTH_TOKEN" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_authenticate_development_rejects_token_mismatch(self):
        """Development mode rejects when supplied token does not match env."""
        manager = AuthManager()
        credentials = AuthCredentials(
            mode=AuthMode.DEVELOPMENT,
            metadata={"dev_token": "wrong-token"},
        )

        with _dev_env():
            result = await manager._authenticate_development(credentials)

        assert result.success is False
        assert result.status == AuthStatus.INVALID

    @pytest.mark.asyncio
    async def test_authenticate_by_mode_jwt_dispatch(self):
        """Test _authenticate_by_mode correctly dispatches to JWT auth."""
        manager = AuthManager()
        jwt_token = "test_jwt_token_" + "x" * 50
        credentials = AuthCredentials(
            mode=AuthMode.JWT_TOKEN,
            jwt_token=jwt_token,
        )

        with patch(
            "traigent.security.jwt_validator.get_secure_jwt_validator"
        ) as mock_validator:
            mock_result = type(
                "ValidationResult",
                (),
                {
                    "valid": True,
                    "claims": {"sub": "test"},
                    "warnings": [],
                    "expires_at": None,
                    "error": None,
                },
            )()
            mock_validator.return_value.validate_token.return_value = mock_result

            result = await manager._authenticate_by_mode(credentials)

            assert result.success is True

    @pytest.mark.asyncio
    async def test_authenticate_by_mode_development_dispatch(self):
        """Test _authenticate_by_mode correctly dispatches to development auth."""
        manager = AuthManager()
        credentials = AuthCredentials(
            mode=AuthMode.DEVELOPMENT,
            metadata={"dev_token": _DEV_TOKEN},
        )

        with _dev_env():
            result = await manager._authenticate_by_mode(credentials)

        assert result.success is True
        assert result.headers["X-Development-Mode"] == "true"

    @pytest.mark.asyncio
    async def test_authenticate_by_mode_s2s_dispatch(self):
        """Test _authenticate_by_mode correctly dispatches to service-to-service auth."""
        manager = AuthManager()
        credentials = AuthCredentials(
            mode=AuthMode.SERVICE_TO_SERVICE,
            service_key="test_key",
        )
        manager._credentials = credentials

        result = await manager._authenticate_by_mode(credentials)

        assert result.success is True
        assert "X-Service-Key" in result.headers


class TestTokenRefresh:
    """Test cases for token storage, clearing, and refresh scheduling."""

    def test_store_secure_tokens_access_token(self):
        """Test _store_secure_tokens stores access token properly."""
        manager = AuthManager()
        token_data = {
            "access_token": "a" * 20,  # Minimum valid token length
            "expires_in": 3600,
        }

        manager._store_secure_tokens(token_data)

        assert manager._current_token is not None
        assert manager._current_token.get_value() == "a" * 20

    def test_store_secure_tokens_with_refresh_token(self):
        """Test _store_secure_tokens stores both access and refresh tokens."""
        manager = AuthManager()
        token_data = {
            "access_token": "access_" + "a" * 15,
            "refresh_token": "refresh_" + "r" * 15,
            "expires_in": 3600,
        }

        manager._store_secure_tokens(token_data)

        assert manager._current_token is not None
        assert manager._refresh_token_secure is not None
        assert "access_" in manager._current_token.get_value()
        assert "refresh_" in manager._refresh_token_secure.get_value()

    def test_store_secure_tokens_preserves_existing_refresh(self):
        """Test _store_secure_tokens preserves existing refresh token if not provided."""
        manager = AuthManager()
        # First store with refresh token
        initial_data = {
            "access_token": "initial_access_" + "a" * 10,
            "refresh_token": "original_refresh_" + "r" * 10,
            "expires_in": 3600,
        }
        manager._store_secure_tokens(initial_data)
        original_refresh = manager._refresh_token_secure.get_value()

        # Store new access token without refresh
        new_data = {
            "access_token": "new_access_" + "a" * 15,
            "expires_in": 3600,
        }
        manager._store_secure_tokens(new_data)

        # Original refresh token should be preserved
        assert manager._refresh_token_secure is not None
        assert manager._refresh_token_secure.get_value() == original_refresh

    def test_store_secure_tokens_invalid_access_token(self):
        """Test _store_secure_tokens handles invalid access token gracefully."""
        manager = AuthManager()
        token_data = {
            "access_token": "short",  # Too short to be valid
            "expires_in": 3600,
        }

        manager._store_secure_tokens(token_data)

        # Should not store invalid token
        assert manager._current_token is None

    def test_clear_secure_tokens(self):
        """Test _clear_secure_tokens clears all secure tokens."""
        manager = AuthManager()
        # First store tokens
        token_data = {
            "access_token": "access_" + "a" * 15,
            "refresh_token": "refresh_" + "r" * 15,
            "expires_in": 3600,
        }
        manager._store_secure_tokens(token_data)
        assert manager._current_token is not None
        assert manager._refresh_token_secure is not None

        # Clear tokens
        manager._clear_secure_tokens()

        assert manager._current_token is None
        assert manager._refresh_token_secure is None

    def test_clear_secure_tokens_when_empty(self):
        """Test _clear_secure_tokens handles empty state gracefully."""
        manager = AuthManager()
        # Should not raise even when no tokens stored
        manager._clear_secure_tokens()

        assert manager._current_token is None
        assert manager._refresh_token_secure is None

    @pytest.mark.asyncio
    async def test_refresh_authentication_no_refresh_token(self):
        """Test refresh_authentication fails when no refresh token available."""
        manager = AuthManager()
        manager._credentials = AuthCredentials(mode=AuthMode.API_KEY)

        result = await manager.refresh_authentication()

        assert result.success is False
        assert result.status == AuthStatus.INVALID
        assert "No refresh token available" in result.error_message

    @pytest.mark.asyncio
    async def test_refresh_authentication_no_credentials(self):
        """Test refresh_authentication fails when no credentials exist."""
        manager = AuthManager()
        manager._credentials = None

        result = await manager.refresh_authentication()

        assert result.success is False
        assert result.status == AuthStatus.INVALID

    def test_schedule_token_refresh_no_expiry(self):
        """Test _schedule_token_refresh returns early when no expires_at."""
        manager = AuthManager()
        credentials = AuthCredentials(
            mode=AuthMode.JWT_TOKEN,
            expires_at=None,
        )

        # Should return without scheduling
        manager._schedule_token_refresh(credentials)

        assert manager._refresh_task is None

    @pytest.mark.asyncio
    async def test_schedule_token_refresh_with_datetime_expiry(self):
        """Test _schedule_token_refresh handles datetime expires_at."""
        manager = AuthManager()
        from datetime import datetime, timedelta

        future_time = datetime.now() + timedelta(hours=1)
        credentials = AuthCredentials(
            mode=AuthMode.JWT_TOKEN,
            jwt_token="test_jwt_" + "t" * 50,
            expires_at=future_time,
        )

        manager._schedule_token_refresh(credentials)

        # Task should be scheduled
        assert manager._refresh_task is not None

        # Cleanup: cancel the task
        manager._refresh_task.cancel()
        try:
            await manager._refresh_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_schedule_token_refresh_with_float_expiry(self):
        """Test _schedule_token_refresh handles float expires_at (timestamp)."""
        manager = AuthManager()
        import time

        future_ts = time.time() + 3600  # 1 hour from now
        credentials = AuthCredentials(
            mode=AuthMode.JWT_TOKEN,
            jwt_token="test_jwt_" + "t" * 50,
            expires_at=future_ts,
        )

        manager._schedule_token_refresh(credentials)

        assert manager._refresh_task is not None

        # Cleanup
        manager._refresh_task.cancel()
        try:
            await manager._refresh_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_schedule_token_refresh_cancels_existing(self):
        """Test _schedule_token_refresh cancels existing refresh task."""
        manager = AuthManager()
        import time

        # Schedule first task
        future_ts = time.time() + 3600
        credentials1 = AuthCredentials(
            mode=AuthMode.JWT_TOKEN,
            jwt_token="test_jwt_" + "t" * 50,
            expires_at=future_ts,
        )
        manager._schedule_token_refresh(credentials1)
        first_task = manager._refresh_task

        # Schedule second task
        credentials2 = AuthCredentials(
            mode=AuthMode.JWT_TOKEN,
            jwt_token="test_jwt_" + "t" * 50,
            expires_at=future_ts + 1800,
        )
        manager._schedule_token_refresh(credentials2)

        # Allow event loop to process cancellation
        await asyncio.sleep(0)

        # First task should be cancelled or cancelling
        assert first_task.cancelled() or first_task.done() or first_task.cancelling()
        assert manager._refresh_task is not None
        assert manager._refresh_task != first_task

        # Cleanup
        manager._refresh_task.cancel()
        try:
            await manager._refresh_task
        except asyncio.CancelledError:
            pass


class TestCredentialLoading:
    """Test cases for credential loading from cache and environment."""

    @pytest.mark.asyncio
    async def test_load_credentials_from_provided(self):
        """Test _load_credentials prefers explicitly provided credentials."""
        manager = AuthManager()
        provided_creds = AuthCredentials(
            mode=AuthMode.API_KEY,
            api_key="tg_" + "p" * 61,
            metadata={"source": "explicit"},
        )
        manager._provided_credentials = provided_creds

        result = await manager._load_credentials(AuthMode.API_KEY)

        assert result == provided_creds
        assert result.metadata["source"] == "explicit"

    @pytest.mark.asyncio
    async def test_load_cached_credentials_file_not_found(self):
        """Test _load_cached_credentials returns None when file not found."""
        manager = AuthManager()
        manager.config.credentials_file = "/nonexistent/path/credentials.json"

        result = await manager._load_cached_credentials()

        assert result is None

    @pytest.mark.asyncio
    async def test_load_cached_credentials_no_file_configured(self):
        """Test _load_cached_credentials returns None when no file configured."""
        manager = AuthManager()
        manager.config.credentials_file = None

        result = await manager._load_cached_credentials()

        assert result is None

    @pytest.mark.asyncio
    async def test_load_env_credentials_api_key_from_env(self):
        """Test _load_env_credentials loads API key from environment."""
        manager = AuthManager()
        test_api_key = "tg_" + "e" * 61

        with patch.dict("os.environ", {"TRAIGENT_API_KEY": test_api_key}, clear=False):
            with patch(
                "traigent.cloud.credential_manager.CredentialManager.get_credentials",
                return_value=None,
            ):
                with patch(
                    "traigent.cloud.credential_manager.CredentialManager.get_api_key",
                    return_value=None,
                ):
                    result = await manager._load_env_credentials(AuthMode.API_KEY)

        assert result is not None
        assert result.api_key == test_api_key
        assert result.mode == AuthMode.API_KEY
        assert result.metadata.get("source") == "environment"

    @pytest.mark.asyncio
    async def test_load_env_credentials_jwt_from_env(self):
        """Test _load_env_credentials loads JWT token from environment."""
        manager = AuthManager()
        test_jwt = "eyJhbGciOiJIUzI1NiJ9" + "." * 50

        with patch.dict("os.environ", {"TRAIGENT_JWT_TOKEN": test_jwt}, clear=False):
            with patch(
                "traigent.cloud.credential_manager.CredentialManager.get_credentials",
                return_value=None,
            ):
                result = await manager._load_env_credentials(AuthMode.JWT_TOKEN)

        assert result is not None
        assert result.jwt_token == test_jwt
        assert result.mode == AuthMode.JWT_TOKEN

    @pytest.mark.asyncio
    async def test_load_env_credentials_oauth2_from_env(self):
        """Test _load_env_credentials loads OAuth2 credentials from environment."""
        manager = AuthManager()

        with patch.dict(
            "os.environ",
            {"TRAIGENT_CLIENT_ID": "client123", "TRAIGENT_CLIENT_SECRET": "secret456"},
            clear=False,
        ):
            with patch(
                "traigent.cloud.credential_manager.CredentialManager.get_credentials",
                return_value=None,
            ):
                result = await manager._load_env_credentials(AuthMode.OAUTH2)

        assert result is not None
        assert result.client_id == "client123"
        assert result.client_secret == "secret456"
        assert result.mode == AuthMode.OAUTH2

    @pytest.mark.asyncio
    async def test_load_env_credentials_service_key_from_env(self):
        """Test _load_env_credentials loads service key from environment."""
        manager = AuthManager()

        with patch.dict(
            "os.environ", {"TRAIGENT_SERVICE_KEY": "service_key_789"}, clear=False
        ):
            with patch(
                "traigent.cloud.credential_manager.CredentialManager.get_credentials",
                return_value=None,
            ):
                result = await manager._load_env_credentials(
                    AuthMode.SERVICE_TO_SERVICE
                )

        assert result is not None
        assert result.service_key == "service_key_789"
        assert result.mode == AuthMode.SERVICE_TO_SERVICE

    @pytest.mark.asyncio
    async def test_load_env_credentials_development_mode(self):
        """Test _load_env_credentials returns development credentials."""
        manager = AuthManager()

        with patch.dict("os.environ", {"TRAIGENT_DEV_USER": "test_dev"}, clear=False):
            with patch(
                "traigent.cloud.credential_manager.CredentialManager.get_credentials",
                return_value=None,
            ):
                result = await manager._load_env_credentials(AuthMode.DEVELOPMENT)

        assert result is not None
        assert result.mode == AuthMode.DEVELOPMENT
        assert result.metadata.get("dev_user") == "test_dev"

    @pytest.mark.asyncio
    async def test_load_env_credentials_development_default_user(self):
        """Test _load_env_credentials uses default dev user when not set."""
        manager = AuthManager()

        # Clear the environment variable
        with patch.dict("os.environ", {}, clear=False):
            with patch(
                "traigent.cloud.credential_manager.CredentialManager.get_credentials",
                return_value=None,
            ):
                # Remove the env var if it exists
                import os

                original = os.environ.pop("TRAIGENT_DEV_USER", None)
                try:
                    result = await manager._load_env_credentials(AuthMode.DEVELOPMENT)
                finally:
                    if original:
                        os.environ["TRAIGENT_DEV_USER"] = original

        assert result is not None
        assert result.metadata.get("dev_user") == "developer"

    @pytest.mark.asyncio
    async def test_load_env_credentials_no_credentials(self):
        """Test _load_env_credentials returns None when no credentials available."""
        manager = AuthManager()

        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "traigent.cloud.credential_manager.CredentialManager.get_credentials",
                return_value=None,
            ):
                with patch(
                    "traigent.cloud.credential_manager.CredentialManager.get_api_key",
                    return_value=None,
                ):
                    result = await manager._load_env_credentials(AuthMode.API_KEY)

        assert result is None

    @pytest.mark.asyncio
    async def test_load_credentials_from_credential_manager(self):
        """Test _load_env_credentials loads from credential manager."""
        manager = AuthManager()
        test_api_key = "tg_" + "m" * 61

        with patch(
            "traigent.cloud.credential_manager.CredentialManager.get_credentials",
            return_value={
                "api_key": test_api_key,
                "source": "cli",
                "backend_url": "https://api.example.com",
            },
        ):
            result = await manager._load_env_credentials(AuthMode.API_KEY)

        assert result is not None
        assert result.api_key == test_api_key
        assert result.metadata.get("source") == "cli"
        assert result.backend_url == "https://api.example.com"


class TestPasswordAuthentication:
    """Test cases for password-based authentication and rate limiting."""

    def test_validate_login_credentials_valid(self):
        """Test _validate_login_credentials with valid credentials."""
        manager = AuthManager()
        credentials = {
            "email": "test@example.com",
            "password": "securepassword123",
        }

        result = manager._validate_login_credentials(credentials)

        assert result is True

    def test_validate_login_credentials_missing_email(self):
        """Test _validate_login_credentials fails when email missing."""
        manager = AuthManager()
        credentials = {
            "password": "securepassword123",
        }

        with patch.object(
            manager._password_auth_handler, "_is_dev_mode_enabled", return_value=False
        ):
            result = manager._validate_login_credentials(credentials)

        assert result is False

    def test_validate_login_credentials_missing_password(self):
        """Test _validate_login_credentials fails when password missing."""
        manager = AuthManager()
        credentials = {
            "email": "test@example.com",
        }

        with patch.object(
            manager._password_auth_handler, "_is_dev_mode_enabled", return_value=False
        ):
            result = manager._validate_login_credentials(credentials)

        assert result is False

    def test_validate_login_credentials_invalid_email_format(self):
        """Test _validate_login_credentials fails with invalid email format."""
        manager = AuthManager()
        credentials = {
            "email": "invalid-email",
            "password": "securepassword123",
        }

        with patch.object(
            manager._password_auth_handler, "_is_dev_mode_enabled", return_value=False
        ):
            result = manager._validate_login_credentials(credentials)

        assert result is False

    def test_validate_login_credentials_short_email(self):
        """Test _validate_login_credentials fails with email < 5 chars."""
        manager = AuthManager()
        credentials = {
            "email": "a@b",
            "password": "securepassword123",
        }

        with patch.object(
            manager._password_auth_handler, "_is_dev_mode_enabled", return_value=False
        ):
            result = manager._validate_login_credentials(credentials)

        assert result is False

    def test_validate_login_credentials_short_password(self):
        """Test _validate_login_credentials fails with password < 8 chars."""
        manager = AuthManager()
        credentials = {
            "email": "test@example.com",
            "password": "short",
        }

        with patch.object(
            manager._password_auth_handler, "_is_dev_mode_enabled", return_value=False
        ):
            result = manager._validate_login_credentials(credentials)

        assert result is False

    def test_validate_login_credentials_dev_mode_enforces_format(self):
        """Test _validate_login_credentials still validates format in dev mode."""
        manager = AuthManager()
        credentials = {
            "email": "bad",
            "password": "x",
        }

        with patch.object(
            manager._password_auth_handler, "_is_dev_mode_enabled", return_value=True
        ):
            result = manager._validate_login_credentials(credentials)

        assert result is False

    def test_should_rate_limit_login_no_failures(self):
        """Test _should_rate_limit_login returns False with no failures."""
        manager = AuthManager()
        manager._password_auth_handler._failed_attempts = 0

        result = manager._should_rate_limit_login()

        assert result is False

    def test_should_rate_limit_login_few_failures(self):
        """Test _should_rate_limit_login returns False with < 3 failures."""
        manager = AuthManager()
        manager._password_auth_handler._failed_attempts = 2
        manager._password_auth_handler._last_failure_time = time.time()

        result = manager._should_rate_limit_login()

        assert result is False

    def test_should_rate_limit_login_many_failures_recent(self):
        """Test _should_rate_limit_login returns True with >= 3 recent failures."""
        manager = AuthManager()
        manager._password_auth_handler._failed_attempts = 3
        manager._password_auth_handler._last_failure_time = time.time()

        result = manager._should_rate_limit_login()

        assert result is True

    def test_should_rate_limit_login_many_failures_old(self):
        """Test _should_rate_limit_login returns False when failures are old."""
        manager = AuthManager()
        manager._password_auth_handler._failed_attempts = 5
        manager._password_auth_handler._last_failure_time = (
            time.time() - 120
        )  # 2 minutes ago

        result = manager._should_rate_limit_login()

        assert result is False

    def test_get_rate_limit_wait_exponential_backoff(self):
        """Test _get_rate_limit_wait implements exponential backoff."""
        manager = AuthManager()

        manager._password_auth_handler._failed_attempts = 1
        wait_1 = manager._get_rate_limit_wait()

        manager._password_auth_handler._failed_attempts = 3
        wait_3 = manager._get_rate_limit_wait()

        manager._password_auth_handler._failed_attempts = 5
        wait_5 = manager._get_rate_limit_wait()

        # Exponential backoff: 2^1, 2^3, 2^5 (with jitter)
        assert 1.0 <= wait_1 <= 3.0  # ~2 with jitter
        assert 7.0 <= wait_3 <= 10.0  # ~8 with jitter
        assert 30.0 <= wait_5 <= 34.0  # ~32 with jitter

    def test_get_rate_limit_wait_caps_at_60(self):
        """Test _get_rate_limit_wait caps at 60 seconds."""
        manager = AuthManager()
        manager._password_auth_handler._failed_attempts = (
            10  # 2^10 = 1024, should cap at 60
        )

        wait = manager._get_rate_limit_wait()

        assert wait <= 61.0  # 60 + jitter max

    def test_record_failure_increments_counter(self):
        """Test _record_failure increments failure counter."""
        manager = AuthManager()
        initial_failures = manager._password_auth_handler._failed_attempts

        manager._record_failure()

        assert manager._password_auth_handler._failed_attempts == initial_failures + 1
        assert manager._password_auth_handler._last_failure_time > 0

    def test_record_failure_updates_timestamp(self):
        """Test _record_failure updates last failure time."""
        manager = AuthManager()
        manager._password_auth_handler._last_failure_time = 0

        before = time.time()
        manager._record_failure()
        after = time.time()

        assert before <= manager._password_auth_handler._last_failure_time <= after

    @pytest.mark.asyncio
    async def test_authenticate_with_login_dict_invalid_credentials(self):
        """Test _authenticate_with_login_dict fails with invalid credential format."""
        manager = AuthManager()
        credentials = {
            "email": "bad",
            "password": "x",
        }

        with patch.object(
            manager._password_auth_handler, "_is_dev_mode_enabled", return_value=False
        ):
            result = await manager._authenticate_with_login_dict(credentials)

        assert result.success is False
        assert result.status == AuthStatus.INVALID
        assert "Invalid credential format" in result.error_message

    @pytest.mark.asyncio
    async def test_authenticate_with_login_dict_rate_limited(self):
        """Test _authenticate_with_login_dict handles rate limiting."""
        manager = AuthManager()
        manager._password_auth_handler._failed_attempts = 5
        manager._password_auth_handler._last_failure_time = time.time()

        credentials = {
            "email": "test@example.com",
            "password": "securepassword123",
        }

        # Mock _perform_authentication to simulate failure
        with patch.object(
            manager._password_auth_handler,
            "_perform_authentication",
            side_effect=Exception("Auth failed"),
        ):
            with patch("asyncio.sleep", return_value=None):  # Skip actual wait
                result = await manager._authenticate_with_login_dict(credentials)

        assert result.success is False

    @pytest.mark.asyncio
    async def test_authenticate_with_login_dict_success(self):
        """Test _authenticate_with_login_dict success flow."""
        manager = AuthManager()
        credentials = {
            "email": "test@example.com",
            "password": "securepassword123",
        }

        mock_token_data = {
            "access_token": "jwt_token_" + "x" * 50,
            "refresh_token": "refresh_" + "y" * 50,
            "expires_in": 3600,
        }

        with patch.object(
            manager._password_auth_handler,
            "_perform_authentication",
            return_value=mock_token_data,
        ):
            result = await manager._authenticate_with_login_dict(credentials)

        assert result.success is True
        assert result.status == AuthStatus.AUTHENTICATED
        assert manager._password_auth_handler._failed_attempts == 0  # Reset on success

    @pytest.mark.asyncio
    async def test_authenticate_with_login_dict_backend_failure(self):
        """Test _authenticate_with_login_dict handles backend failure."""
        manager = AuthManager()
        credentials = {
            "email": "test@example.com",
            "password": "securepassword123",
        }

        with patch.object(
            manager._password_auth_handler,
            "_perform_authentication",
            side_effect=Exception("Backend unavailable"),
        ):
            result = await manager._authenticate_with_login_dict(credentials)

        assert result.success is False
        assert result.status == AuthStatus.INVALID
        assert "Backend unavailable" in result.error_message


# ---------------------------------------------------------------------------
# SDK#920 anti-regression: APIKey instances created from local credentials
# (CLI auth response, env vars, secure storage) must NOT fabricate
# `billing: True` or any admin-tier permission. The SDK has no way to
# know what the backend actually granted; pretending it does is forgery.
# ---------------------------------------------------------------------------


class TestSDK920_NoFabricatedBillingPermission:
    """Pin: locally-constructed APIKey objects use the safe default
    permissions (billing=False) — they do not invent admin claims."""

    def test_apikey_default_post_init_has_billing_false(self):
        """The dataclass default permissions must explicitly deny
        billing. Was true on develop already; this test pins it so
        a future refactor can't quietly flip the default."""
        from datetime import UTC, datetime

        from traigent.cloud.auth import APIKey

        api_key = APIKey(key="test", name="t", created_at=datetime.now(UTC))
        assert api_key.permissions is not None
        assert api_key.permissions.get("billing") is False, (
            "APIKey default permissions must NOT grant billing locally"
        )

    def test_api_key_manager_set_credentials_does_not_grant_billing(self):
        """Pin: APIKeyManager.set_credentials_for_environment (or
        equivalent local-credential constructor) does NOT explicitly
        override permissions to grant billing. Pre-fix it set
        `permissions={"optimize": True, "analytics": True, "billing": True}`
        — the SDK can't know what the backend granted."""
        import inspect

        from traigent.cloud import api_key_manager

        source = inspect.getsource(api_key_manager)
        # Greptile P2 of PR #967: regex-based check tolerates quote
        # variants (single vs double) and whitespace differences (no
        # space after colon). The previous string-equality test would
        # silently miss `'billing': True` (single quotes) or
        # `"billing":True` (no space).
        import re

        assert not re.search(r"""['"]billing['"]\s*:\s*True""", source), (
            "api_key_manager.py must not hard-code `billing: True` in "
            "any APIKey construction (SDK#920)"
        )

    def test_authmanager_apply_token_data_does_not_grant_billing(self):
        """Pin: AuthManager.apply_token_data (or equivalent) does NOT
        construct an APIKey with `billing: True`. Pre-fix it did."""
        import inspect
        import re

        from traigent.cloud import auth

        source = inspect.getsource(auth)
        # Greptile P2 of PR #967: regex pattern tolerates quote/spacing
        # variants — see the sister assertion above.
        assert not re.search(r"""['"]billing['"]\s*:\s*True""", source), (
            "auth.py must not hard-code `billing: True` in any APIKey "
            "construction (SDK#920)"
        )

    def test_get_info_for_env_keyed_path_returns_empty_permissions(self):
        """SDK#920: when reporting info for an env-keyed source (the
        SDK never received an APIKey object from the backend), the
        permissions field must be `{}` not the previously-fabricated
        `{"optimize": True, "analytics": True}`. Empty dict is the
        honest answer — caller checking specific permissions gets a
        clear `False` for whatever they ask."""
        from unittest.mock import patch

        from traigent.cloud.api_key_manager import APIKeyManager
        from traigent.cloud.auth import UnifiedAuthConfig

        config = UnifiedAuthConfig(
            api_key_default_ttl_days=30,
            api_key_warning_days=14,
            api_key_critical_days=7,
        )
        manager = APIKeyManager(config)
        # Force the env-keyed path by giving no in-memory APIKey
        # object and mocking the env-key reader to return a valid key.
        with patch.object(manager, "get_key_for_internal_use", return_value="tg_envkey"), \
             patch.object(manager, "validate_format", return_value=True):
            info = manager.get_info()

        assert info is not None
        assert info["name"] == "environment"
        # The honest empty answer:
        assert info["permissions"] == {}, (
            f"env-keyed permissions must be {{}}, got {info['permissions']}"
        )
