"""Token management module for TraiGent authentication.

This module handles secure token storage, refresh scheduling, and lifecycle management.
Extracted from AuthManager to follow Single Responsibility Principle.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Security CONC-Quality-Maintainability FUNC-SEC-TOKEN-MGMT REQ-SEC-010 REQ-CLOUD-009 SYNC-CloudHybrid

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from traigent.cloud._aiohttp_compat import AIOHTTP_AVAILABLE, aiohttp

if TYPE_CHECKING:
    from traigent.cloud.auth import (
        AuthCredentials,
        AuthResult,
        SecureToken,
        UnifiedAuthConfig,
    )

logger = logging.getLogger(__name__)


class TokenManager:
    """Manages secure token lifecycle and refresh scheduling.

    This class handles:
    - Secure storage of access and refresh tokens
    - Automatic token refresh scheduling
    - Token expiration tracking
    - Building credentials from token data
    """

    def __init__(
        self,
        config: UnifiedAuthConfig,
        *,
        validate_key_format_fn: Any = None,
        set_api_key_token_fn: Any = None,
    ) -> None:
        """Initialize token manager.

        Args:
            config: Unified authentication configuration
            validate_key_format_fn: Callback to validate API key format
            set_api_key_token_fn: Callback to set API key token
        """
        self.config = config
        self._validate_key_format = validate_key_format_fn
        self._set_api_key_token = set_api_key_token_fn

        # Token state
        self._current_token: SecureToken | None = None
        self._refresh_token_secure: SecureToken | None = None
        self._refresh_task: asyncio.Task[None] | None = None
        self._last_refresh_attempt: float = 0.0

        # Callbacks for credential operations (set by AuthManager)
        self._get_credentials_fn: Any = None
        self._set_credentials_fn: Any = None
        self._cache_credentials_fn: Any = None
        self._auth_lock: asyncio.Lock | None = None

    def set_callbacks(
        self,
        *,
        get_credentials: Any,
        set_credentials: Any,
        cache_credentials: Any,
        auth_lock: asyncio.Lock,
        validate_key_format: Any = None,
        set_api_key_token: Any = None,
    ) -> None:
        """Set callback functions for credential operations.

        Args:
            get_credentials: Callback to get current credentials
            set_credentials: Callback to set credentials
            cache_credentials: Callback to cache credentials
            auth_lock: Lock for authentication operations
            validate_key_format: Callback to validate API key format
            set_api_key_token: Callback to set API key token
        """
        self._get_credentials_fn = get_credentials
        self._set_credentials_fn = set_credentials
        self._cache_credentials_fn = cache_credentials
        self._auth_lock = auth_lock
        if validate_key_format is not None:
            self._validate_key_format = validate_key_format
        if set_api_key_token is not None:
            self._set_api_key_token = set_api_key_token

    @property
    def current_token(self) -> SecureToken | None:
        """Get current access token."""
        return self._current_token

    @property
    def refresh_token_secure(self) -> SecureToken | None:
        """Get current refresh token."""
        return self._refresh_token_secure

    @property
    def refresh_task(self) -> asyncio.Task[None] | None:
        """Get current refresh task."""
        return self._refresh_task

    @property
    def last_refresh_attempt(self) -> float:
        """Get timestamp of last refresh attempt."""
        return self._last_refresh_attempt

    def store_tokens(self, token_data: dict[str, Any]) -> None:
        """Store access and refresh tokens using SecureToken wrappers.

        Args:
            token_data: Dictionary containing access_token, refresh_token, and expires_in
        """
        from traigent.cloud.auth import SecureToken, TokenExpiredError

        existing_refresh_token: str | None = None
        if self._refresh_token_secure:
            try:
                existing_refresh_token = self._refresh_token_secure.get_value()
            except TokenExpiredError:
                existing_refresh_token = None

        self.clear_tokens()

        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token") or existing_refresh_token
        expires_in = token_data.get("expires_in", 3600)

        if access_token:
            try:
                self._current_token = SecureToken(
                    _value=access_token,
                    _expires_at=time.time() + expires_in,
                )
            except ValueError:
                logger.debug("Skipping secure storage for invalid access token format")

        if refresh_token:
            refresh_expires = token_data.get("refresh_expires_in")
            if not refresh_expires:
                refresh_expires = expires_in * 24
            try:
                self._refresh_token_secure = SecureToken(
                    _value=refresh_token,
                    _expires_at=time.time() + refresh_expires,
                )
            except ValueError:
                # S3 fix: If SecureToken creation fails and we had a valid existing
                # refresh token, restore it instead of losing it entirely
                logger.debug("Skipping secure storage for invalid refresh token format")
                if existing_refresh_token and refresh_token != existing_refresh_token:
                    try:
                        self._refresh_token_secure = SecureToken(
                            _value=existing_refresh_token,
                            _expires_at=time.time() + expires_in * 24,
                        )
                        logger.debug("Restored existing refresh token after new token failed")
                    except ValueError:
                        logger.warning("Could not restore existing refresh token either")
        else:
            self._refresh_token_secure = None

    def clear_tokens(self) -> None:
        """Clear secure token storage."""
        if self._current_token:
            self._current_token.clear()
            self._current_token = None
        if self._refresh_token_secure:
            self._refresh_token_secure.clear()
            self._refresh_token_secure = None

    def schedule_refresh(self, credentials: AuthCredentials) -> None:
        """Schedule automatic token refresh.

        Args:
            credentials: Credentials containing expiration time
        """
        if not credentials.expires_at:
            return

        # Calculate refresh time (refresh before expiry)
        if isinstance(credentials.expires_at, datetime):
            expiry_ts = credentials.expires_at.timestamp()
        else:
            expiry_ts = float(credentials.expires_at)

        refresh_time = expiry_ts - self.config.token_refresh_threshold
        delay = max(0, refresh_time - time.time())

        # Cancel existing refresh task
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()

        # Schedule new refresh task
        async def refresh_task() -> None:
            await asyncio.sleep(delay)
            await self.refresh_access_token()

        self._set_refresh_task(asyncio.create_task(refresh_task()))

    def _set_refresh_task(self, task: asyncio.Task[None]) -> None:
        """Track background refresh tasks and surface failures.

        Args:
            task: The asyncio task to track
        """

        def _finalize(fut: asyncio.Task[None]) -> None:
            try:
                fut.result()
            except asyncio.CancelledError:  # pragma: no cover - cancellation path
                logger.debug("Token refresh task cancelled")
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error("❌ Background token refresh failed", exc_info=exc)
            finally:
                if self._refresh_task is fut:
                    self._refresh_task = None

        task.add_done_callback(_finalize)
        self._refresh_task = task

    async def refresh_access_token(self) -> AuthResult:
        """Refresh authentication token.

        Returns:
            AuthResult with refresh status
        """
        from traigent.cloud.auth import (
            AuthMode,
            AuthResult,
            AuthStatus,
            TokenExpiredError,
        )

        credentials = self._get_credentials_fn() if self._get_credentials_fn else None
        if not credentials:
            return AuthResult(
                success=False,
                status=AuthStatus.INVALID,
                error_message="No credentials available",
            )

        refresh_token_value: str | None = None
        if self._refresh_token_secure:
            try:
                refresh_token_value = self._refresh_token_secure.get_value()
            except TokenExpiredError:
                refresh_token_value = None

        if not refresh_token_value:
            refresh_token_value = credentials.refresh_token

        if not refresh_token_value:
            return AuthResult(
                success=False,
                status=AuthStatus.INVALID,
                error_message="No refresh token available",
            )

        lock = self._auth_lock or asyncio.Lock()
        async with lock:
            try:
                self._last_refresh_attempt = time.time()

                mode = credentials.mode
                if mode == AuthMode.JWT_TOKEN:
                    return await self.refresh_jwt_secure(refresh_token_value)
                if mode == AuthMode.OAUTH2:
                    return await self.refresh_oauth2()
                if mode == AuthMode.API_KEY:
                    return AuthResult(
                        success=True,
                        status=AuthStatus.AUTHENTICATED,
                        credentials=credentials,
                    )

                return AuthResult(
                    success=False,
                    status=AuthStatus.INVALID,
                    error_message=f"Token refresh not supported for {mode}",
                )

            except Exception as exc:
                logger.error(f"Token refresh failed: {exc}")
                return AuthResult(
                    success=False, status=AuthStatus.INVALID, error_message=str(exc)
                )

    async def refresh_jwt_secure(self, refresh_token_value: str) -> AuthResult:
        """Refresh JWT access token using secure stored refresh token.

        Args:
            refresh_token_value: The refresh token to use

        Returns:
            AuthResult with new credentials on success
        """
        from traigent.cloud.auth import AuthResult, AuthStatus

        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp not available for token refresh")

        from traigent.cloud.resilient_client import ResilientClient
        from traigent.config.backend_config import BackendConfig

        backend_api_url = BackendConfig.get_backend_api_url()
        refresh_url = f"{backend_api_url}/auth/refresh"

        client = ResilientClient(
            max_retries=3,
            base_delay=1.0,
            max_delay=10.0,
            jitter_factor=0.1,
        )

        async def perform_refresh():
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    refresh_url,
                    json={"refresh_token": refresh_token_value},
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status == 401:
                        raise ValueError("Refresh token invalid or expired")
                    if response.status == 429:
                        error_msg = await response.text()
                        raise Exception(f"429 Rate Limited: {error_msg}")
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"{response.status}: {error_text}")

                    data = await response.json()
                    if not data.get("success"):
                        raise ValueError(data.get("error", "Token refresh failed"))

                    payload = data.get("data", {})
                    if "refresh_token" not in payload or not payload["refresh_token"]:
                        payload["refresh_token"] = refresh_token_value
                    return payload

        try:
            token_data = await client.execute_with_retry(
                perform_refresh, operation_name="token_refresh"
            )

            if not token_data:
                return AuthResult(
                    success=False,
                    status=AuthStatus.INVALID,
                    error_message="Token refresh returned no data",
                )

            self.store_tokens(token_data)
            updated_credentials = self.build_credentials_from_token_data(token_data)

            if self._set_credentials_fn:
                self._set_credentials_fn(updated_credentials, AuthStatus.AUTHENTICATED)

            if self.config.auto_refresh and updated_credentials.refresh_token:
                self.schedule_refresh(updated_credentials)

            headers: dict[str, str] = {}
            if updated_credentials.jwt_token:
                headers["Authorization"] = f"Bearer {updated_credentials.jwt_token}"

            if self.config.cache_credentials and self._cache_credentials_fn:
                await self._cache_credentials_fn(updated_credentials)

            return AuthResult(
                success=True,
                status=AuthStatus.AUTHENTICATED,
                credentials=updated_credentials,
                headers=headers,
                expires_in=token_data.get("expires_in"),
            )

        except ValueError as exc:
            self.clear_tokens()
            if self._set_credentials_fn:
                self._set_credentials_fn(None, AuthStatus.INVALID)
            return AuthResult(
                success=False,
                status=AuthStatus.INVALID,
                error_message=str(exc),
            )
        except Exception as exc:
            logger.error(f"Token refresh error: {type(exc).__name__}: {exc}")
            return AuthResult(
                success=False,
                status=AuthStatus.INVALID,
                error_message=str(exc),
            )

    async def refresh_oauth2(self) -> AuthResult:
        """Refresh OAuth2 access token.

        Returns:
            AuthResult with refresh status
        """
        from traigent.cloud.auth import AuthResult, AuthStatus

        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp not available for token refresh") from None

        credentials = self._get_credentials_fn() if self._get_credentials_fn else None
        if not credentials or not credentials.refresh_token:
            return AuthResult(
                success=False,
                status=AuthStatus.UNAUTHENTICATED,
                error_message="No refresh token available",
            )

        token_url = f"{self.config.cloud_base_url}/oauth/token"

        if credentials.client_id is None or credentials.client_secret is None:
            return AuthResult(
                success=False,
                status=AuthStatus.INVALID,
                error_message="OAuth2 client credentials missing",
            )

        data = {
            "grant_type": "refresh_token",
            "refresh_token": credentials.refresh_token,
            "client_id": credentials.client_id,
            "client_secret": credentials.client_secret,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(token_url, data=data) as response:
                if response.status == 200:
                    token_data = await response.json()

                    # Update credentials
                    credentials.jwt_token = token_data["access_token"]
                    if "refresh_token" in token_data:
                        credentials.refresh_token = token_data["refresh_token"]
                    credentials.expires_at = time.time() + token_data.get(
                        "expires_in", 3600
                    )

                    if self._set_credentials_fn:
                        self._set_credentials_fn(credentials, AuthStatus.AUTHENTICATED)

                    # Cache updated credentials
                    if self.config.cache_credentials and self._cache_credentials_fn:
                        await self._cache_credentials_fn(credentials)

                    return AuthResult(
                        success=True,
                        status=AuthStatus.AUTHENTICATED,
                        credentials=credentials,
                        expires_in=token_data.get("expires_in", 3600),
                    )
                else:
                    error_text = await response.text()
                    raise RuntimeError(
                        f"Token refresh failed: {response.status} {error_text}"
                    )

    def build_credentials_from_token_data(
        self, token_data: dict[str, Any]
    ) -> AuthCredentials:
        """Construct AuthCredentials from token payload.

        Args:
            token_data: Token response data

        Returns:
            AuthCredentials populated with token data
        """
        from traigent.cloud.auth import (
            AuthCredentials,
            AuthMode,
            TokenExpiredError,
        )

        expires_in = token_data.get("expires_in")
        expires_at = time.time() + expires_in if expires_in else None
        refresh_token = token_data.get("refresh_token")
        if not refresh_token and self._refresh_token_secure:
            try:
                refresh_token = self._refresh_token_secure.get_value()
            except TokenExpiredError:
                refresh_token = None

        metadata: dict[str, Any] = {}
        credentials = self._get_credentials_fn() if self._get_credentials_fn else None
        if credentials and credentials.metadata:
            metadata.update(credentials.metadata)
        if token_data.get("user") is not None:
            metadata["user"] = token_data.get("user", {})

        api_key = token_data.get("api_key") or token_data.get("apiKey")
        expires_at_dt = (
            datetime.fromtimestamp(expires_at, timezone.utc) if expires_at else None
        )

        if api_key and self._validate_key_format and self._validate_key_format(api_key):
            if self._set_api_key_token:
                self._set_api_key_token(api_key, source="cli", expires_at=expires_at_dt)
            metadata.setdefault("source", "cli")

        return AuthCredentials(
            mode=AuthMode.JWT_TOKEN,
            jwt_token=token_data.get("access_token"),
            refresh_token=refresh_token,
            expires_at=expires_at,
            metadata=metadata,
            api_key=api_key,
        )

    def get_authorization_header(self) -> dict[str, str]:
        """Get authorization header from current token.

        Returns:
            Dictionary with Authorization header, or empty dict if no valid token

        Raises:
            TokenExpiredError: If token has expired
        """
        if self._current_token and not self._current_token.is_expired:
            return self._current_token.get_header()
        return {}
