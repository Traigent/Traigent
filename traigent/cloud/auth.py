"""Unified authentication module for Traigent SDK.

This module consolidates all authentication functionality with SOC2 compliance,
secure token management, and resilient HTTP client integration.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability CONC-Quality-Security FUNC-CLOUD-HYBRID FUNC-SECURITY REQ-CLOUD-009 REQ-SEC-010 SYNC-CloudHybrid

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, cast

from traigent.cloud._aiohttp_compat import AIOHTTP_AVAILABLE, aiohttp
from traigent.cloud.api_key_manager import APIKeyManager
from traigent.cloud.credential_resolver import CredentialResolver
from traigent.cloud.password_auth_handler import PasswordAuthHandler
from traigent.cloud.token_manager import TokenManager
from traigent.config.backend_config import DEFAULT_LOCAL_URL
from traigent.core.constants import MAX_RETRIES
from traigent.utils.exceptions import AuthenticationError as TraigentAuthenticationError

logger = logging.getLogger(__name__)

# Security constants
TOKEN_REFRESH_THRESHOLD = 300  # Refresh 5 minutes before expiry
MAX_TOKEN_AGE = 86400  # Maximum token age in seconds (24 hours)
MIN_TOKEN_LENGTH = 20  # Minimum acceptable token length
API_KEY_TOKEN_TTL = 31536000  # 365 days


class _AsyncBool:
    """Boolean value that can be awaited or used synchronously."""

    def __init__(self, value: bool) -> None:
        self._value = bool(value)

    def __bool__(self) -> bool:
        return self._value

    def __await__(self):
        async def _coro():
            return self._value

        return _coro().__await__()


# ============================================================================
# Data Models and Enums
# ============================================================================


class AuthMode(Enum):
    """Authentication modes supported by unified auth system."""

    API_KEY = "api_key"
    JWT_TOKEN = "jwt_token"
    OAUTH2 = "oauth2"
    SERVICE_TO_SERVICE = "service_to_service"
    DEVELOPMENT = "development"
    CLOUD = "cloud"
    EDGE_ANALYTICS = "edge_analytics"
    DEMO = "demo"


@dataclass
class AuthCredentials:
    """Authentication credentials container."""

    mode: AuthMode = AuthMode.API_KEY
    api_key: str | None = None
    jwt_token: str | None = None
    refresh_token: str | None = None
    client_id: str | None = None
    client_secret: str | None = None
    service_key: str | None = None
    backend_url: str | None = None  # Resolved from BackendConfig at runtime
    expires_at: float | None = None
    scopes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate credentials on creation."""
        if self.mode in {AuthMode.CLOUD, AuthMode.API_KEY} and not (
            self.api_key or self.jwt_token
        ):
            logger.warning("No credentials provided for cloud authentication mode")

    def __repr__(self) -> str:
        """Secure string representation masking sensitive data."""
        return (
            f"AuthCredentials(mode={self.mode.value}, "
            f"api_key={'***' if self.api_key else None}, "
            f"jwt_token={'***' if self.jwt_token else None}, "
            f"client_id={self.client_id}, "
            f"has_secret={bool(self.client_secret)}, "
            f"has_service_key={bool(self.service_key)}, "
            f"expires_at={self.expires_at})"
        )

    def __str__(self) -> str:
        """Secure string representation for logging."""
        return self.__repr__()


class AuthStatus(Enum):
    """Authentication status."""

    AUTHENTICATED = "authenticated"
    UNAUTHENTICATED = "unauthenticated"
    EXPIRED = "expired"
    INVALID = "invalid"
    REFRESHING = "refreshing"
    RATE_LIMITED = "rate_limited"


@dataclass
class AuthResult:
    """Result of authentication operation."""

    success: bool
    status: AuthStatus
    credentials: AuthCredentials | None = None
    headers: dict[str, str] = field(default_factory=dict)
    error_message: str | None = None
    expires_in: int | None = None
    retry_after: float | None = None

    def __bool__(self) -> bool:  # legacy convenience
        return self.success


@dataclass
class UnifiedAuthConfig:
    """Configuration for unified authentication."""

    default_mode: AuthMode = AuthMode.API_KEY
    backend_base_url: str | None = (
        None  # Will be set from BackendConfig if not provided
    )
    cloud_base_url: str = DEFAULT_LOCAL_URL
    token_refresh_threshold: float = 300.0  # Refresh if expires within 5 minutes
    auto_refresh: bool = True
    cache_credentials: bool = True
    credentials_file: str | None = None
    api_key_default_ttl_days: int = 365
    api_key_warning_days: int = 30
    api_key_critical_days: int = 7

    def __post_init__(self) -> None:
        """Set backend URL from centralized config if not provided."""
        if self.backend_base_url is None:
            from traigent.config.backend_config import BackendConfig

            self.backend_base_url = BackendConfig.get_backend_url()


@dataclass
class APIKey:
    """Traigent Cloud API key configuration."""

    key: str
    name: str
    created_at: datetime | float | int | str
    expires_at: datetime | float | int | str | None = None
    permissions: dict[str, bool] | None = None
    usage_limit: int | None = None

    def __post_init__(self) -> None:
        """Initialise default permissions when not provided."""
        if self.permissions is None:
            self.permissions = {"optimize": True, "analytics": True, "billing": False}

        # Normalise date fields to timezone-aware datetime instances.
        self.created_at = self._coerce_datetime(self.created_at, "created_at")
        if self.expires_at is not None:
            self.expires_at = self._coerce_datetime(self.expires_at, "expires_at")

    def is_valid(self) -> bool:
        """Check if API key is valid and not expired."""
        if self.expires_at:
            # Coerced in __post_init__
            expires_at_dt = cast(datetime, self.expires_at)
            expires_at = (
                expires_at_dt
                if expires_at_dt.tzinfo is not None
                else expires_at_dt.replace(tzinfo=UTC)
            )
            if datetime.now(UTC) > expires_at:
                return False
        return bool(self.key)

    def has_permission(self, permission: str) -> bool:
        """Check if API key grants a specific permission."""
        return self.permissions.get(permission, False) if self.permissions else False

    def __str__(self) -> str:
        """Masked string representation for security."""
        if self.key and len(self.key) > 14:
            return f"APIKey({self.key[:10]}...{self.key[-4:]})"
        return "APIKey(***)"

    @staticmethod
    def _coerce_datetime(
        value: datetime | float | int | str, field_name: str
    ) -> datetime:
        """Coerce various datetime representations to timezone-aware datetime."""

        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, (int, float)):
            dt = datetime.fromtimestamp(float(value), tz=UTC)
        elif isinstance(value, str):
            raw = value.strip()
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            try:
                dt = datetime.fromisoformat(raw)
            except ValueError as exc:
                raise ValueError(
                    f"{field_name} must be datetime, timestamp, or ISO string"
                ) from exc
        else:
            raise TypeError(
                f"{field_name} must be datetime, timestamp, or ISO string, got {type(value).__name__}"
            )

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)

        return dt


# ============================================================================
# Exceptions
# ============================================================================


class AuthenticationError(TraigentAuthenticationError):
    """Base authentication exception."""

    pass


class TokenExpiredError(AuthenticationError, ValueError):
    """Token has expired."""

    pass


class InvalidCredentialsError(AuthenticationError):
    """Invalid credentials provided."""

    pass


# ============================================================================
# Secure Token Management
# ============================================================================


@dataclass
class SecureToken:
    """Secure token storage with automatic clearing."""

    _value: str
    _expires_at: float
    _token_type: str = "Bearer"

    def __post_init__(self) -> None:
        """Validate token on creation."""
        if not self._value or len(self._value) < MIN_TOKEN_LENGTH:
            raise ValueError("Invalid token format") from None

        # Never log the actual token value
        token_hash = hashlib.sha256(self._value.encode()).hexdigest()[:8]
        logger.debug(f"Token created with hash: {token_hash}")

    @property
    def is_expired(self) -> bool:
        """Check if token is expired with safety margin."""
        return time.time() >= (self._expires_at - TOKEN_REFRESH_THRESHOLD)

    @property
    def time_until_expiry(self) -> float:
        """Get seconds until token expires."""
        return max(0, self._expires_at - time.time())

    def get_header(self) -> dict[str, str]:
        """Get authorization header without exposing token in logs."""
        if self.is_expired:
            raise TokenExpiredError("Token has expired")
        return {"Authorization": f"{self._token_type} {self._value}"}

    def get_value(self) -> str:
        """Return raw token value for internal secure use."""
        if self.is_expired:
            raise TokenExpiredError("Token has expired")
        return self._value

    def clear(self) -> None:
        """Securely clear token from memory."""
        if hasattr(self, "_value"):
            self._value = "X" * len(self._value)
            del self._value

    def __del__(self) -> None:
        """Ensure token is cleared when object is destroyed."""
        self.clear()

    def __str__(self):
        """Never expose token in string representation."""
        return f"SecureToken(type={self._token_type}, expires_in={self.time_until_expiry:.0f}s)"

    def __repr__(self):
        """Never expose token in repr."""
        return self.__str__()


# ============================================================================
# Main Authentication Manager
# ============================================================================


class AuthManager:
    """Authentication manager for SDK and backend integration.

    This manager handles authentication for both Traigent Cloud services and
    backend integrations, providing a single entry point for acquiring
    credentials and generating request headers.
    """

    def __init__(
        self,
        api_key: str | None = None,
        config: UnifiedAuthConfig | None = None,
    ) -> None:
        """Initialize authentication manager.

        Args:
            api_key: Optional API key supplied programmatically.
            config: Authentication configuration overrides.
        """
        if isinstance(api_key, UnifiedAuthConfig) and config is None:
            config = api_key
            api_key = None

        self.config = config or UnifiedAuthConfig()

        # Authentication state
        self._credentials: AuthCredentials | None = None
        self._auth_status = AuthStatus.UNAUTHENTICATED
        # Lock ordering: If acquiring both locks, ALWAYS acquire _auth_lock BEFORE
        # _token_op_lock to prevent deadlock. In practice, these locks guard different
        # concerns and should rarely both be needed:
        #   - _auth_lock: Authentication state (login, logout, credential changes)
        #   - _token_op_lock: Token operations (expiration check, refresh, use)
        # Note: asyncio.Lock is NOT reentrant, so nested acquisition would deadlock.
        self._auth_lock = asyncio.Lock()
        self._token_op_lock = asyncio.Lock()

        # API key management (delegated to APIKeyManager)
        self._api_key_manager = APIKeyManager(self.config)
        self._api_key_manager.set_callbacks(
            get_credentials=lambda: self._credentials,
            get_provided_credentials=lambda: self._provided_credentials,
            get_last_auth_result=lambda: self._last_auth_result,
        )
        self._authenticated = False

        # Legacy attribute retained for compatibility with existing tests
        self._unified_auth = self

        self._provided_credentials: AuthCredentials | None = None
        initial_api_key = api_key or os.getenv("TRAIGENT_API_KEY")
        initial_source = (
            "explicit" if api_key else ("environment" if initial_api_key else None)
        )

        if initial_api_key:
            self._api_key_manager.set_token(initial_api_key, source=initial_source)
            self._provided_credentials = AuthCredentials(
                mode=AuthMode.API_KEY,
                api_key=initial_api_key,
                metadata={"source": initial_source} if initial_source else {},
            )

        # Token management (delegated to TokenManager)
        self._token_manager = TokenManager(
            self.config,
            validate_key_format_fn=self._api_key_manager.validate_format,
            set_api_key_token_fn=self._api_key_manager.set_token,
        )

        # Credential resolution (delegated to CredentialResolver)
        self._credential_resolver = CredentialResolver(self.config)
        self._credential_resolver.set_callbacks(
            get_provided_credentials=lambda: self._provided_credentials,
            set_api_key_token=self._api_key_manager.set_token,
            get_api_key_token=lambda: self._api_key_manager.api_key_token,
            increment_cache_hits=lambda: self._stats.__setitem__(
                "cache_hits", self._stats.get("cache_hits", 0) + 1
            ),
        )

        # Password authentication (delegated to PasswordAuthHandler)
        self._password_auth_handler = PasswordAuthHandler()
        self._password_auth_handler.set_callbacks(
            build_credentials=self._build_credentials_from_token_data,
            store_tokens=self._store_secure_tokens,
        )

        self._token_manager.set_callbacks(
            get_credentials=lambda: self._credentials,
            set_credentials=self._set_credentials_and_status,
            cache_credentials=self._cache_credentials,
            auth_lock=self._auth_lock,
        )

        self._last_auth_result: AuthResult | None = None

        # Statistics
        self._stats: dict[str, Any] = {
            "authentication_attempts": 0,
            "successful_authentications": 0,
            "failed_authentications": 0,
            "token_refreshes": 0,
            "cache_hits": 0,
        }

    # ------------------------------------------------------------------
    # Token state delegators (backward compatibility)
    # ------------------------------------------------------------------

    @property
    def _current_token(self) -> SecureToken | None:
        """Delegate to TokenManager for backward compatibility."""
        return self._token_manager.current_token

    @_current_token.setter
    def _current_token(self, value: SecureToken | None) -> None:
        """Delegate to TokenManager for backward compatibility."""
        self._token_manager._current_token = value

    @property
    def _refresh_token_secure(self) -> SecureToken | None:
        """Delegate to TokenManager for backward compatibility."""
        return self._token_manager.refresh_token_secure

    @_refresh_token_secure.setter
    def _refresh_token_secure(self, value: SecureToken | None) -> None:
        """Delegate to TokenManager for backward compatibility."""
        self._token_manager._refresh_token_secure = value

    @property
    def _refresh_task(self) -> asyncio.Task[Any] | None:
        """Delegate to TokenManager for backward compatibility."""
        return self._token_manager.refresh_task

    @_refresh_task.setter
    def _refresh_task(self, value: asyncio.Task[Any] | None) -> None:
        """Delegate to TokenManager for backward compatibility."""
        self._token_manager._refresh_task = value

    @property
    def _last_refresh_attempt(self) -> float:
        """Delegate to TokenManager for backward compatibility."""
        return self._token_manager.last_refresh_attempt

    @_last_refresh_attempt.setter
    def _last_refresh_attempt(self, value: float) -> None:
        """Delegate to TokenManager for backward compatibility."""
        self._token_manager._last_refresh_attempt = value

    def _set_credentials_and_status(
        self, credentials: AuthCredentials | None, status: AuthStatus
    ) -> None:
        """Helper for TokenManager callback to set credentials and status."""
        self._credentials = credentials
        self._auth_status = status

    # ------------------------------------------------------------------
    # API key state delegators (backward compatibility)
    # ------------------------------------------------------------------

    @property
    def _api_key_token(self) -> SecureToken | None:
        """Delegate to APIKeyManager for backward compatibility."""
        return self._api_key_manager.api_key_token

    @_api_key_token.setter
    def _api_key_token(self, value: SecureToken | None) -> None:
        """Delegate to APIKeyManager for backward compatibility."""
        self._api_key_manager._api_key_token = value

    @property
    def _api_key_preview(self) -> str | None:
        """Delegate to APIKeyManager for backward compatibility."""
        return self._api_key_manager.api_key_preview

    @_api_key_preview.setter
    def _api_key_preview(self, value: str | None) -> None:
        """Delegate to APIKeyManager for backward compatibility."""
        self._api_key_manager._api_key_preview = value

    @property
    def _api_key_source(self) -> str | None:
        """Delegate to APIKeyManager for backward compatibility."""
        return self._api_key_manager.api_key_source

    @_api_key_source.setter
    def _api_key_source(self, value: str | None) -> None:
        """Delegate to APIKeyManager for backward compatibility."""
        self._api_key_manager._api_key_source = value

    @property
    def _api_key_expiry(self) -> datetime | None:
        """Delegate to APIKeyManager for backward compatibility."""
        return self._api_key_manager.api_key_expiry

    @_api_key_expiry.setter
    def _api_key_expiry(self, value: datetime | None) -> None:
        """Delegate to APIKeyManager for backward compatibility."""
        self._api_key_manager._api_key_expiry = value

    @property
    def _api_key_last_rotated(self) -> datetime | None:
        """Delegate to APIKeyManager for backward compatibility."""
        return self._api_key_manager.api_key_last_rotated

    @_api_key_last_rotated.setter
    def _api_key_last_rotated(self, value: datetime | None) -> None:
        """Delegate to APIKeyManager for backward compatibility."""
        self._api_key_manager._api_key_last_rotated = value

    @property
    def _api_key(self) -> APIKey | None:
        """Delegate to APIKeyManager for backward compatibility."""
        return self._api_key_manager.api_key

    @_api_key.setter
    def _api_key(self, value: APIKey | None) -> None:
        """Delegate to APIKeyManager for backward compatibility."""
        self._api_key_manager._api_key = value

    # ------------------------------------------------------------------
    # API key management methods (delegated to APIKeyManager)
    # ------------------------------------------------------------------

    @staticmethod
    def _mask_api_key(api_key: str) -> str:
        """Delegate to APIKeyManager.mask_key."""
        return APIKeyManager.mask_key(api_key)

    def _set_api_key_token(
        self,
        api_key: str,
        source: str | None = None,
        expires_at: datetime | None = None,
    ) -> None:
        """Delegate to APIKeyManager.set_token."""
        self._api_key_manager.set_token(api_key, source=source, expires_at=expires_at)

    def _clear_api_key_token(self) -> None:
        """Delegate to APIKeyManager.clear_token."""
        self._api_key_manager.clear_token()

    def get_api_key_preview(self) -> str | None:
        """Return a masked preview of the currently configured API key."""
        return self._api_key_manager.get_preview()

    def has_api_key(self) -> bool:
        """Return True if an API key has been configured."""
        return self._api_key_manager.has_key()

    def _get_api_key_for_internal_use(self) -> str | None:
        """Delegate to APIKeyManager.get_key_for_internal_use."""
        return self._api_key_manager.get_key_for_internal_use()

    # Primary Authentication Interface

    async def authenticate(
        self, credentials: AuthCredentials | None = None, mode: AuthMode | None = None
    ) -> AuthResult:
        """Authenticate with unified auth system returning detailed result.

        The return value matches the legacy implementation so callers can
        inspect ``result.success`` while also allowing access to the raw object
        through ``self._last_auth_result``.
        """

        result = await self.authenticate_with_result(credentials, mode)
        self._last_auth_result = result
        return result

    def set_api_key(
        self,
        api_key: str | None,
        *,
        expires_at: datetime | None = None,
        source: str = "explicit",
    ) -> None:
        """Set or clear an explicit API key for authentication."""

        if api_key:
            self._set_api_key_token(api_key, source=source, expires_at=expires_at)
            self._provided_credentials = AuthCredentials(
                mode=AuthMode.API_KEY,
                api_key=api_key,
                metadata={
                    "source": source,
                    **(
                        {"expires_at": expires_at.isoformat()}
                        if expires_at is not None
                        else {}
                    ),
                },
            )
        else:
            self._provided_credentials = None
            self._clear_api_key_token()

        # Reset state so the next authentication uses updated credentials
        self._credentials = None
        self._current_token = None
        self._auth_status = AuthStatus.UNAUTHENTICATED
        self._api_key = None
        self._authenticated = False

    def rotate_api_key(
        self, api_key: str, *, expires_at: datetime | None = None
    ) -> None:
        """Rotate the API key and refresh internal caches."""

        self.set_api_key(api_key, expires_at=expires_at, source="rotation")
        logger.info("API key rotated successfully")

    def get_api_key_status(self) -> dict[str, Any]:
        """Return structured API key status for monitoring."""
        return self._api_key_manager.get_status()

    def check_api_key_rotation(self) -> bool:
        """Log rotation guidance and return True when the key is healthy."""
        return self._api_key_manager.check_rotation()

    def _validate_key_format(self, key: str | None) -> bool:
        """Validate API key format."""
        return self._api_key_manager.validate_format(key)

    def clear(self) -> None:
        """Clear authentication state synchronously."""

        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()

        self._clear_secure_tokens()
        self._credentials = None
        self._auth_status = AuthStatus.UNAUTHENTICATED
        self._api_key = None
        self._authenticated = False

    async def authenticate_with_result(
        self, credentials: AuthCredentials | None = None, mode: AuthMode | None = None
    ) -> AuthResult:
        """Authenticate with unified auth system returning detailed result.

        Args:
            credentials: Authentication credentials
            mode: Authentication mode override

        Returns:
            AuthResult with authentication status and headers
        """
        async with self._auth_lock:
            self._stats["authentication_attempts"] += 1
            try:
                resolved_creds = await self._resolve_credentials(credentials, mode)

                if isinstance(resolved_creds, dict):
                    return await self._authenticate_with_dict(resolved_creds)

                if resolved_creds is None:
                    return AuthResult(
                        success=False,
                        status=AuthStatus.UNAUTHENTICATED,
                        error_message="No credentials provided",
                    )

                result = await self._authenticate_by_mode(resolved_creds)
                if result.success:
                    await self._handle_successful_auth(resolved_creds, result)
                else:
                    self._record_failed_authentication()

                return result

            except Exception as e:
                logger.error(f"Authentication failed: {e}")
                self._stats["failed_authentications"] += 1
                return AuthResult(
                    success=False, status=AuthStatus.INVALID, error_message=str(e)
                )

    async def _resolve_credentials(
        self,
        credentials: AuthCredentials | dict[str, Any] | None,
        mode: AuthMode | None,
    ) -> AuthCredentials | dict[str, Any] | None:
        """Delegate to CredentialResolver.resolve."""
        return await self._credential_resolver.resolve(credentials, mode)

    async def _authenticate_with_dict(
        self, credentials_dict: dict[str, Any]
    ) -> AuthResult:
        """Authenticate using dictionary credentials."""

        dict_result = await self._authenticate_with_login_dict(credentials_dict)
        if dict_result.success:
            await self._handle_dict_success(dict_result)
        else:
            self._record_failed_authentication()
        return dict_result

    async def _handle_dict_success(self, result: AuthResult) -> None:
        """Apply state updates for successful dict-based authentication."""

        self._credentials = result.credentials
        self._auth_status = result.status
        self._stats["successful_authentications"] += 1
        self._authenticated = True

        if result.credentials and self.config.cache_credentials:
            await self._cache_credentials(result.credentials)

        if (
            result.credentials
            and self.config.auto_refresh
            and result.credentials.refresh_token
        ):
            self._schedule_token_refresh(result.credentials)

    async def _authenticate_by_mode(self, credentials: AuthCredentials) -> AuthResult:
        """Dispatch authentication logic based on credential mode."""

        if credentials.mode == AuthMode.API_KEY:
            return await self._authenticate_api_key(credentials)
        if credentials.mode == AuthMode.JWT_TOKEN:
            return await self._authenticate_jwt(credentials)
        if credentials.mode == AuthMode.OAUTH2:
            return await self._authenticate_oauth2(credentials)
        if credentials.mode == AuthMode.SERVICE_TO_SERVICE:
            return await self._authenticate_service_to_service(credentials)
        if credentials.mode == AuthMode.DEVELOPMENT:
            return await self._authenticate_development(credentials)

        return AuthResult(
            success=False,
            status=AuthStatus.INVALID,
            error_message=f"Unsupported authentication mode: {credentials.mode}",
        )

    async def _handle_successful_auth(
        self, credentials: AuthCredentials, result: AuthResult
    ) -> None:
        """Update state after a successful authentication attempt."""

        self._credentials = credentials
        self._auth_status = result.status
        self._stats["successful_authentications"] += 1
        self._authenticated = True

        if credentials.mode == AuthMode.API_KEY:
            self._persist_api_key(credentials)

        if self.config.cache_credentials:
            await self._cache_credentials(credentials)

        if (
            self.config.auto_refresh
            and credentials.expires_at
            and credentials.refresh_token
        ):
            self._schedule_token_refresh(credentials)

    def _persist_api_key(self, credentials: AuthCredentials) -> None:
        """Persist API key tokens after authentication."""
        self._api_key_manager.persist_api_key(credentials)

    def _record_failed_authentication(self) -> None:
        """Record authentication failure in state and metrics."""

        self._stats["failed_authentications"] += 1
        self._authenticated = False

    def _get_api_key_headers(self) -> dict[str, str]:
        """Generate headers for API key authentication."""
        headers: dict[str, str] = {}
        api_key_value = self._get_api_key_for_internal_use()
        if api_key_value:
            headers["X-API-Key"] = api_key_value
            headers["Authorization"] = f"Bearer {api_key_value}"
        return headers

    async def _get_jwt_headers(self) -> dict[str, str]:
        """Generate headers for JWT token authentication.

        Uses token lock to prevent TOCTOU race on expiration check.
        """
        headers: dict[str, str] = {}
        async with self._token_op_lock:
            if self._current_token:
                if self._current_token.is_expired:
                    refresh_result = await self._refresh_token()
                    if not refresh_result.success:
                        raise AuthenticationError(
                            refresh_result.error_message or "Token refresh failed"
                        )
                headers.update(self._current_token.get_header())
            elif self._credentials and self._credentials.jwt_token:
                headers["Authorization"] = f"Bearer {self._credentials.jwt_token}"
        return headers

    def _get_oauth_headers(self) -> dict[str, str]:
        """Generate headers for OAuth2 authentication."""
        token = self._credentials.jwt_token if self._credentials else ""
        return {"Authorization": f"Bearer {token or ''}"}

    def _get_service_headers(self, target: str) -> dict[str, str]:
        """Generate headers for service-to-service authentication."""
        signature = self._generate_service_signature(target)
        service_key = self._credentials.service_key if self._credentials else ""
        return {
            "Authorization": f"Service {signature}",
            "X-Service-Key": service_key or "",
        }

    def _get_dev_headers(self) -> dict[str, str]:
        """Generate headers for development mode."""
        metadata = self._credentials.metadata if self._credentials else {}
        return {
            "X-Development-Mode": "true",
            "X-Dev-User": metadata.get("dev_user", "developer"),
        }

    def _add_common_headers(self, headers: dict[str, str], target: str) -> None:
        """Add common headers to the request headers dict in-place."""
        headers["X-Client-Version"] = "0.1.0"
        headers["X-Integration-Mode"] = "unified"
        headers.setdefault("Content-Type", "application/json")

        if target in ("cloud", "both"):
            headers["X-Traigent-Client"] = "sdk"
        if target in ("backend", "both"):
            headers["X-Traigent-Service"] = "sdk"
            headers["X-Backend-Integration"] = "true"

    async def get_auth_headers(
        self,
        target: str = "both",  # "cloud", "backend", or "both"
    ) -> dict[str, str]:
        """Get authentication headers for requests.

        Args:
            target: Target service ("cloud", "backend", or "both")

        Returns:
            Dictionary of authentication headers
        """
        if not await self.is_authenticated():
            auth_result = await self.authenticate()
            if not auth_result.success or self._credentials is None:
                self._authenticated = False
                # Fall back to raw API key headers when full auth fails
                # but an API key is available (e.g., local dev mode).
                fallback = self._get_api_key_headers()
                if fallback:
                    self._add_common_headers(fallback, target)
                    return fallback
                return {}

        if not self._credentials:
            return {}

        # Generate headers based on authentication mode
        mode = self._credentials.mode
        if mode == AuthMode.API_KEY:
            headers = self._get_api_key_headers()
        elif mode == AuthMode.JWT_TOKEN:
            headers = await self._get_jwt_headers()
        elif mode == AuthMode.OAUTH2:
            headers = self._get_oauth_headers()
        elif mode == AuthMode.SERVICE_TO_SERVICE:
            headers = self._get_service_headers(target)
        elif mode == AuthMode.DEVELOPMENT:
            headers = self._get_dev_headers()
        else:
            headers = {}

        self._add_common_headers(headers, target)
        return headers

    async def get_headers(self, target: str = "both") -> dict[str, str]:
        """Compatibility wrapper that proxies to :meth:`get_auth_headers`."""

        return await self.get_auth_headers(target)

    async def refresh_token(self) -> bool:
        """Backward-compatible coroutine returning boolean refresh status."""

        if self._credentials and self._credentials.refresh_token:
            result = await self.refresh_authentication()
            return result.success

        if self._get_api_key_for_internal_use():
            result = await self.authenticate()
            return result.success

        return False

    def get_api_key_info(self) -> dict[str, Any] | None:
        """Return non-sensitive API key metadata if available."""
        return self._api_key_manager.get_info()

    def get_owner_fingerprint(self) -> dict[str, Any]:
        """Return sanitized identifiers for the currently authenticated actor.

        The backend now enforces that session-level operations originate from
        the owning user or an administrator.  Exposing a normalized fingerprint
        lets callers persist or compare ownership metadata without handling the
        raw credential material.
        """

        metadata: dict[str, Any] = {}
        if self._credentials and self._credentials.metadata:
            metadata = dict(self._credentials.metadata)

        owner_user_id = (
            metadata.get("owner_user_id")
            or metadata.get("user_id")
            or metadata.get("subject")
            or metadata.get("sub")
        )
        owner_api_key_id = (
            metadata.get("owner_api_key_id")
            or metadata.get("api_key_id")
            or metadata.get("key_id")
        )
        created_by = metadata.get("created_by") or metadata.get("owner")
        if not created_by and owner_user_id:
            created_by = owner_user_id

        owner_scope = metadata.get("owner_scope") or metadata.get("scopes")

        fingerprint: dict[str, Any] = {
            "owner_user_id": owner_user_id,
            "owner_api_key_id": owner_api_key_id,
            "created_by": created_by,
            "owner_scope": owner_scope,
            "credential_source": metadata.get("source") or self._api_key_source,
            "owner_api_key_preview": self._api_key_preview,
            "auth_mode": self._credentials.mode.value if self._credentials else None,
            "metadata_present": bool(metadata),
        }

        return fingerprint

    def is_authenticated(self) -> _AsyncBool:
        """Check if currently authenticated.

        Returns:
            Awaitable/synchronous boolean representing authentication status.
        """

        if self._current_token and not self._current_token.is_expired:
            return _AsyncBool(True)

        if self._auth_status != AuthStatus.AUTHENTICATED or not self._credentials:
            return _AsyncBool(False)

        # Check token expiration
        if self._credentials.expires_at:
            current_time = time.time()
            if current_time >= self._credentials.expires_at:
                self._auth_status = AuthStatus.EXPIRED
                return _AsyncBool(False)

            # Check if refresh is needed
            if (
                (
                    current_time + self.config.token_refresh_threshold
                    >= self._credentials.expires_at
                )
                and self.config.auto_refresh
                and self._credentials.refresh_token
            ):
                if not self._refresh_task or self._refresh_task.done():
                    self._set_refresh_task(asyncio.create_task(self._refresh_token()))

        return _AsyncBool(True)

    async def refresh_authentication(self) -> AuthResult:
        """Refresh authentication credentials.

        Returns:
            AuthResult with refresh status
        """
        if not self._credentials or not self._credentials.refresh_token:
            return AuthResult(
                success=False,
                status=AuthStatus.INVALID,
                error_message="No refresh token available",
            )

        return await self._refresh_token()

    async def logout(self) -> bool:
        """Logout and clear authentication state.

        Returns:
            True if logout successful
        """
        try:
            # Cancel refresh task if running
            if self._refresh_task and not self._refresh_task.done():
                self._refresh_task.cancel()

            self.clear()

            # Clear cached credentials
            if self.config.cache_credentials and self.config.credentials_file:
                cache_path = Path(self.config.credentials_file)
                if cache_path.exists():
                    cache_path.unlink()

            logger.info("Successfully logged out")
            return True

        except Exception as e:
            logger.error(f"Logout failed: {e}")
            return False

    # Authentication Mode Implementations

    async def _authenticate_api_key(self, credentials: AuthCredentials) -> AuthResult:
        """Authenticate using API key."""
        api_key_value = credentials.api_key or self._get_api_key_for_internal_use()

        if not api_key_value:
            return AuthResult(
                success=False,
                status=AuthStatus.INVALID,
                error_message="API key not provided",
            )

        # Validate API key format
        if not self._validate_key_format(api_key_value):
            return AuthResult(
                success=False,
                status=AuthStatus.INVALID,
                error_message="Invalid API key format",
            )

        # For API key auth, we trust the key if format is valid
        # Real validation would happen server-side
        self._clear_secure_tokens()
        expires_at = credentials.expires_at
        expires_dt: datetime | None
        if isinstance(expires_at, datetime):
            expires_dt = expires_at
        elif isinstance(expires_at, (int, float)):
            expires_dt = datetime.fromtimestamp(expires_at, tz=UTC)
        else:
            expires_dt = None

        self._set_api_key_token(
            api_key_value,
            source=credentials.metadata.get("source"),
            expires_at=expires_dt,
        )

        # Ensure credentials continue to reference the key for downstream usage
        credentials.api_key = api_key_value

        # Include both headers for backward compatibility
        headers = {
            "X-API-Key": api_key_value,
            "Authorization": f"Bearer {api_key_value}",
        }

        return AuthResult(
            success=True,
            status=AuthStatus.AUTHENTICATED,
            credentials=credentials,
            headers=headers,
        )

    async def _authenticate_jwt(self, credentials: AuthCredentials) -> AuthResult:
        """Authenticate using JWT token."""
        if not credentials.jwt_token:
            return AuthResult(
                success=False,
                status=AuthStatus.INVALID,
                error_message="JWT token not provided",
            )

        # Validate JWT token
        try:
            # Use secure JWT validation
            # Determine validation mode based on environment
            import os

            from traigent.security.jwt_validator import (
                ValidationMode,
                get_secure_jwt_validator,
            )

            validation_mode = (
                ValidationMode.DEVELOPMENT
                if os.getenv("TRAIGENT_ENV") == "development"
                else ValidationMode.PRODUCTION
            )

            validator = get_secure_jwt_validator(validation_mode)
            result = validator.validate_token(credentials.jwt_token)

            # Log any warnings
            if result.warnings:
                for warning in result.warnings:
                    logger.warning(f"JWT validation warning: {warning}")

            if not result.valid:
                return AuthResult(
                    success=False,
                    status=AuthStatus.INVALID,
                    error_message=f"JWT validation failed: {result.error}",
                )

            # Update expiration time from validated token
            if result.expires_at:
                credentials.expires_at = result.expires_at

            headers = {"Authorization": f"Bearer {credentials.jwt_token}"}

            expires_in: int | None = None
            if credentials.expires_at is not None:
                if isinstance(credentials.expires_at, datetime):
                    expiry_ts = credentials.expires_at.timestamp()
                else:
                    expiry_ts = float(credentials.expires_at)
                expires_in = max(0, int(expiry_ts - time.time()))

            return AuthResult(
                success=True,
                status=AuthStatus.AUTHENTICATED,
                credentials=credentials,
                headers=headers,
                expires_in=expires_in,
            )

        except Exception as e:
            return AuthResult(
                success=False,
                status=AuthStatus.INVALID,
                error_message=f"Invalid JWT token: {e}",
            )

    async def _authenticate_oauth2(self, credentials: AuthCredentials) -> AuthResult:
        """Authenticate using OAuth2 flow."""
        if not credentials.client_id or not credentials.client_secret:
            return AuthResult(
                success=False,
                status=AuthStatus.INVALID,
                error_message="OAuth2 client credentials not provided",
            )

        # If we have a valid access token, use it
        if credentials.jwt_token:
            return await self._authenticate_jwt(credentials)

        # Otherwise, perform OAuth2 client credentials flow
        try:
            token_data = await self._oauth2_client_credentials_flow(credentials)

            # Update credentials with token
            credentials.jwt_token = token_data["access_token"]
            credentials.refresh_token = token_data.get("refresh_token")
            credentials.expires_at = time.time() + token_data.get("expires_in", 3600)

            headers = {"Authorization": f"Bearer {credentials.jwt_token}"}

            return AuthResult(
                success=True,
                status=AuthStatus.AUTHENTICATED,
                credentials=credentials,
                headers=headers,
                expires_in=token_data.get("expires_in", 3600),
            )

        except Exception as e:
            return AuthResult(
                success=False,
                status=AuthStatus.INVALID,
                error_message=f"OAuth2 authentication failed: {e}",
            )

    async def _authenticate_service_to_service(
        self, credentials: AuthCredentials
    ) -> AuthResult:
        """Authenticate using service-to-service mechanism."""
        if not credentials.service_key:
            return AuthResult(
                success=False,
                status=AuthStatus.INVALID,
                error_message="Service key not provided",
            )

        # Generate service authentication signature
        signature = self._generate_service_signature("both")

        headers = {
            "Authorization": f"Service {signature}",
            "X-Service-Key": credentials.service_key,
        }

        return AuthResult(
            success=True,
            status=AuthStatus.AUTHENTICATED,
            credentials=credentials,
            headers=headers,
        )

    async def _authenticate_development(
        self, credentials: AuthCredentials
    ) -> AuthResult:
        """Authenticate using development mode."""
        # Development mode - always succeed for testing
        headers = {
            "X-Development-Mode": "true",
            "X-Dev-User": credentials.metadata.get("dev_user", "developer"),
        }

        return AuthResult(
            success=True,
            status=AuthStatus.AUTHENTICATED,
            credentials=credentials,
            headers=headers,
        )

    # Helper Methods

    async def _load_credentials(
        self, mode: AuthMode | None = None
    ) -> AuthCredentials | None:
        """Delegate to CredentialResolver.load_credentials."""
        return await self._credential_resolver.load_credentials(mode)

    async def _load_cached_credentials(self) -> AuthCredentials | None:
        """Delegate to CredentialResolver.load_cached."""
        return await self._credential_resolver.load_cached()

    async def _load_env_credentials(self, mode: AuthMode) -> AuthCredentials | None:
        """Delegate to CredentialResolver.load_from_env."""
        return await self._credential_resolver.load_from_env(mode)

    async def _cache_credentials(self, credentials: AuthCredentials) -> None:
        """Delegate to CredentialResolver.cache."""
        await self._credential_resolver.cache(credentials)

    def _encrypt_credentials(self, credentials: AuthCredentials) -> dict[str, Any]:
        """Delegate to CredentialResolver.encrypt."""
        return self._credential_resolver.encrypt(credentials)

    def _decrypt_credentials(self, encrypted_data: dict[str, Any]) -> dict[str, Any]:
        """Delegate to CredentialResolver.decrypt."""
        return self._credential_resolver.decrypt(encrypted_data)

    def _generate_service_signature(self, target: str) -> str:
        """Generate service-to-service authentication signature."""
        if not self._credentials or not self._credentials.service_key:
            return ""

        timestamp = str(int(time.time()))
        nonce = str(uuid.uuid4())

        # Create signature payload
        payload = f"{target}:{timestamp}:{nonce}"

        # Generate HMAC signature
        signature = hmac.new(
            self._credentials.service_key.encode(), payload.encode(), hashlib.sha256
        ).hexdigest()

        # Return formatted signature
        return f"{timestamp}:{nonce}:{signature}"

    async def _oauth2_client_credentials_flow(
        self, credentials: AuthCredentials
    ) -> dict[str, Any]:
        """Perform OAuth2 client credentials flow."""
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp not available for OAuth2 flow") from None

        token_url = f"{self.config.cloud_base_url}/oauth/token"

        data = {
            "grant_type": "client_credentials",
            "client_id": credentials.client_id,
            "client_secret": credentials.client_secret,
            "scope": (
                " ".join(credentials.scopes) if credentials.scopes else "read write"
            ),
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(token_url, data=data) as response:
                if response.status == 200:
                    return cast(dict[str, Any], await response.json())
                else:
                    error_text = await response.text()
                    raise RuntimeError(
                        f"OAuth2 flow failed: {response.status} {error_text}"
                    )

    async def _refresh_token(self) -> AuthResult:
        """Refresh authentication token.

        Delegates to TokenManager but tracks stats locally.
        """
        self._stats["token_refreshes"] += 1
        return await self._token_manager.refresh_access_token()

    async def _refresh_oauth2_token(self) -> AuthResult:
        """Refresh OAuth2 access token.

        Delegates to TokenManager.
        """
        return await self._token_manager.refresh_oauth2()

    async def _authenticate_with_login_dict(
        self, credentials: dict[str, str]
    ) -> AuthResult:
        """Authenticate using interactive login credentials.

        Delegates to PasswordAuthHandler.
        """
        return await self._password_auth_handler.authenticate(credentials)

    def _should_rate_limit_login(self) -> bool:
        """Check if login attempts should be rate limited.

        Delegates to PasswordAuthHandler.
        """
        return self._password_auth_handler._should_rate_limit()

    # Legacy compatibility wrappers -------------------------------------------------

    def _validate_credentials(self, credentials: dict[str, str]) -> bool:
        """Backward-compatible alias for legacy tests."""
        return self._validate_login_credentials(credentials)

    def _should_rate_limit(self) -> bool:
        """Backward-compatible alias for legacy tests."""
        return self._should_rate_limit_login()

    def _get_rate_limit_wait(self) -> float:
        """Calculate exponential backoff delay with jitter.

        Delegates to PasswordAuthHandler.
        """
        return self._password_auth_handler._get_rate_limit_wait()

    def _record_failure(self) -> None:
        """Record failed authentication attempt.

        Delegates to PasswordAuthHandler.
        """
        self._password_auth_handler._record_failure()

    def _is_dev_mode_enabled(self) -> bool:
        """Return True when running in an explicitly non-production mode.

        Delegates to PasswordAuthHandler.
        """
        return self._password_auth_handler._is_dev_mode_enabled()

    def _build_dev_token_payload(
        self, credentials: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """Generate mock tokens for development-only fallback.

        Delegates to PasswordAuthHandler.
        """
        return self._password_auth_handler._build_dev_token_payload(credentials)

    def _reset_failure_tracking(self) -> None:
        """Reset failure tracking after successful login.

        Delegates to PasswordAuthHandler.
        """
        self._password_auth_handler._reset_failure_tracking()

    def _validate_login_credentials(self, credentials: dict[str, str]) -> bool:
        """Validate email/password credentials without logging sensitive values.

        Delegates to PasswordAuthHandler.
        """
        return self._password_auth_handler._validate_credentials(credentials)

    async def _perform_password_authentication(
        self, credentials: dict[str, str]
    ) -> dict[str, Any] | None:
        """Perform backend authentication using resilient HTTP client."""
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp not available for authentication") from None

        from traigent.cloud.resilient_client import ResilientClient
        from traigent.config.backend_config import BackendConfig

        backend_api_url = BackendConfig.get_backend_api_url()
        login_url = f"{backend_api_url}/auth/login"

        client = ResilientClient(
            max_retries=MAX_RETRIES,
            base_delay=1.0,
            max_delay=10.0,
            jitter_factor=0.1,
        )

        async def perform_login():
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    login_url,
                    json={
                        "email": credentials["email"],
                        "password": credentials["password"],
                    },
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status == 401:
                        raise InvalidCredentialsError("Invalid credentials")
                    if response.status == 429:
                        error_msg = await response.text()
                        raise Exception(f"429 Rate Limited: {error_msg}")
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"{response.status}: {error_text}")

                    data = await response.json()
                    if not data.get("success"):
                        raise ValueError(data.get("error", "Authentication failed"))

                    token_data = data.get("data", {})
                    access_token = token_data.get("access_token", "")
                    expires_in = token_data.get("expires_in", 3600)

                    if access_token and "." in access_token:
                        try:
                            parts = access_token.split(".")
                            if len(parts) >= 2:
                                payload = parts[1] + "=" * (4 - len(parts[1]) % 4)
                                decoded = base64.urlsafe_b64decode(payload)
                                jwt_data = json.loads(decoded)
                                if "exp" in jwt_data:
                                    expires_in = max(0, jwt_data["exp"] - time.time())
                        except Exception as exc:
                            logger.debug(f"Failed to parse JWT expiry: {exc}")

                    token_data["expires_in"] = expires_in
                    return token_data

        try:
            return await client.execute_with_retry(
                perform_login, operation_name="backend_authentication"
            )
        except InvalidCredentialsError:
            if self._is_dev_mode_enabled():
                logger.warning(
                    "Dev mode enabled - returning mock tokens despite invalid credentials"
                )
                return self._build_dev_token_payload(credentials)
            raise
        except Exception as exc:
            if self._is_dev_mode_enabled():
                logger.warning(
                    "Dev mode enabled - using mock tokens because backend login failed: %s",
                    exc,
                )
                return self._build_dev_token_payload(credentials)

            logger.error(f"Authentication error: {exc}")
            return None

    async def _refresh_jwt_token_secure(self, refresh_token_value: str) -> AuthResult:
        """Refresh JWT access token using secure stored refresh token.

        Delegates to TokenManager.
        """
        return await self._token_manager.refresh_jwt_secure(refresh_token_value)

    def _build_credentials_from_token_data(
        self, token_data: dict[str, Any]
    ) -> AuthCredentials:
        """Construct AuthCredentials from token payload.

        Delegates to TokenManager, but also updates APIKey on AuthManager.
        """
        credentials = self._token_manager.build_credentials_from_token_data(token_data)

        # AuthManager-specific: update _api_key if present
        api_key = token_data.get("api_key") or token_data.get("apiKey")
        if api_key and self._validate_key_format(api_key):
            self._api_key = APIKey(
                key=api_key,
                name="cli",
                created_at=datetime.now(UTC),
                permissions={"optimize": True, "analytics": True, "billing": True},
            )

        return credentials

    def _store_secure_tokens(self, token_data: dict[str, Any]) -> None:
        """Store access and refresh tokens using SecureToken wrappers.

        Delegates to TokenManager.
        """
        self._token_manager.store_tokens(token_data)

    def _clear_secure_tokens(self) -> None:
        """Clear secure token storage.

        Delegates to TokenManager and also clears API key token.
        """
        self._token_manager.clear_tokens()
        self._clear_api_key_token()

    def _schedule_token_refresh(self, credentials: AuthCredentials) -> None:
        """Schedule automatic token refresh.

        Delegates to TokenManager.
        """
        self._token_manager.schedule_refresh(credentials)

    def _set_refresh_task(self, task: asyncio.Task[Any]) -> None:
        """Track background refresh tasks and surface failures.

        Delegates to TokenManager.
        """
        self._token_manager._set_refresh_task(task)

    # Utility Methods

    def get_auth_status(self) -> AuthStatus:
        """Get current authentication status."""
        return self._auth_status

    def get_credentials_info(self) -> dict[str, Any] | None:
        """Get non-sensitive credentials information."""
        if not self._credentials:
            return None

        return {
            "mode": self._credentials.mode.value,
            "has_api_key": bool(self._credentials.api_key),
            "has_jwt_token": bool(self._credentials.jwt_token),
            "has_refresh_token": bool(self._credentials.refresh_token),
            "expires_at": self._credentials.expires_at,
            "scopes": self._credentials.scopes,
            "metadata": self._credentials.metadata,
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get authentication statistics."""
        return {
            **self._stats,
            "auth_status": self._auth_status.value,
            "has_credentials": bool(self._credentials),
            "last_refresh_attempt": self._last_refresh_attempt,
        }


# AuthenticationError now imported from traigent.utils.exceptions

# ============================================================================
# Helper Functions
# ============================================================================


def get_auth_headers() -> dict[str, str]:
    """Get authentication headers for API requests (legacy compatibility).

    Returns:
        Headers dict with authentication
    """
    from traigent.cloud.credential_manager import CredentialManager

    return CredentialManager.get_auth_headers()


def get_auth_manager() -> AuthManager:
    """Create a new :class:`AuthManager` instance."""

    # Could implement singleton pattern if needed
    return AuthManager()


# ============================================================================
# Security Audit Functions
# ============================================================================


def log_auth_event(
    event_type: str, success: bool, metadata: dict[str, Any] | None = None
) -> None:
    """Log authentication events for security audit.

    Args:
        event_type: Type of auth event (login, refresh, logout)
        success: Whether the event was successful
        metadata: Additional context (no sensitive data)
    """
    # Create audit log entry
    audit_entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "event_type": event_type,
        "success": success,
        "metadata": metadata or {},
    }

    # Never log sensitive data
    metadata_dict = cast(dict[str, Any], audit_entry["metadata"])
    if "password" in metadata_dict:
        del metadata_dict["password"]
    if "token" in metadata_dict:
        del metadata_dict["token"]

    # Log as JSON for easy parsing
    logger.info(f"AUDIT: {json.dumps(audit_entry)}")


def constant_time_compare(a: str, b: str) -> bool:
    """Compare two strings in constant time to prevent timing attacks.

    Args:
        a: First string
        b: Second string

    Returns:
        True if strings are equal
    """
    return hmac.compare_digest(a.encode(), b.encode())
