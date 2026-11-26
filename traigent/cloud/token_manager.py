"""Token management module for Traigent authentication.

This module handles secure token storage, refresh scheduling, and lifecycle management.
Extracted from AuthManager to follow Single Responsibility Principle.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Security CONC-Quality-Maintainability FUNC-SEC-TOKEN-MGMT REQ-SEC-010 REQ-CLOUD-009 SYNC-CloudHybrid

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from traigent.cloud._aiohttp_compat import AIOHTTP_AVAILABLE, aiohttp
from traigent.cloud.url_security import (
    same_host_redirect_trace_configs,
    validate_cloud_base_url_async,
)
from traigent.core.constants import MAX_RETRIES
from traigent.labels.transport_gate import gate_session
from traigent.utils.exceptions import NonRetryableError, RateLimitError

if TYPE_CHECKING:
    from traigent.cloud.auth import (
        AuthCredentials,
        AuthResult,
        SecureToken,
        UnifiedAuthConfig,
    )

logger = logging.getLogger(__name__)

_GENERIC_REFRESH_ERROR = "Token refresh failed"


def _oauth2_credentials_failure(
    credentials: AuthCredentials | None,
) -> AuthResult | None:
    """Failure result when OAuth2 refresh credentials are unusable, else None."""
    from traigent.cloud.auth import AuthResult, AuthStatus

    if not credentials or not credentials.refresh_token:
        return AuthResult(
            success=False,
            status=AuthStatus.UNAUTHENTICATED,
            error_message="No refresh token available",
        )

    if credentials.client_id is None or credentials.client_secret is None:
        return AuthResult(
            success=False,
            status=AuthStatus.INVALID,
            error_message="OAuth2 client credentials missing",
        )
    return None


async def _oauth2_non_200_outcome(response: Any) -> tuple[AuthResult, bool]:
    """Failure result for a non-200 OAuth2 refresh, and whether it is ambiguous."""
    from traigent.cloud.auth import AuthResult, AuthStatus

    await response.text()  # drain body; do not propagate upstream content
    logger.warning(
        "OAuth2 token refresh failed with HTTP %s",
        response.status,
    )
    # 5xx/408 may come from a server or proxy after the
    # token was already rotated: ambiguous. Other 4xx are
    # definitive rejections.
    ambiguous = response.status >= 500 or response.status == 408
    return (
        AuthResult(
            success=False,
            status=AuthStatus.INVALID,
            error_message=f"OAuth2 token refresh failed with HTTP {response.status}",
        ),
        ambiguous,
    )


class RefreshRejectedError(ValueError):
    """The server explicitly rejected the refresh token (HTTP 401 / success=false).

    A dedicated type so the single-shot helper can pass exactly this through;
    any other ValueError (e.g. a JSON decode error) is an ambiguous failure.
    """


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
        no_egress: bool = False,
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
        self.no_egress = bool(no_egress)
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
        self._clear_cached_credentials_fn: Any = None
        self._auth_lock: asyncio.Lock | None = None
        # Latch-on-read hook (set by AuthManager); see APIKeyManager.read_guard.
        self.read_guard: Any = None
        self.guard_active: Any = None  # () -> bool; True when reads must be latched

    def _guarding(self) -> bool:
        return self.read_guard is not None and (
            self.guard_active is None or bool(self.guard_active())
        )

    def set_callbacks(
        self,
        *,
        get_credentials: Any,
        set_credentials: Any,
        cache_credentials: Any,
        auth_lock: asyncio.Lock,
        validate_key_format: Any = None,
        set_api_key_token: Any = None,
        clear_cached_credentials: Any = None,
    ) -> None:
        """Set callback functions for credential operations.

        Args:
            get_credentials: Callback to get current credentials
            set_credentials: Callback to set credentials
            cache_credentials: Callback to cache credentials
            auth_lock: Lock for authentication operations
            validate_key_format: Callback to validate API key format
            set_api_key_token: Callback to set API key token
            clear_cached_credentials: Optional sync callback that removes the
                on-disk credentials cache (called when refresh state is invalidated)
        """
        self._clear_cached_credentials_fn = clear_cached_credentials
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
        token = self._current_token
        if token is not None and self._guarding():
            value = getattr(token, "_value", None)
            if value and not token.is_expired:
                self.read_guard({"Authorization": f"Bearer {value}"})
        return token

    @property
    def refresh_token_secure(self) -> SecureToken | None:
        """Get current refresh token (latched under its own scope on a flagged manager)."""
        token = self._refresh_token_secure
        if token is not None and self._guarding():
            value = getattr(token, "_value", None)
            if value and not token.is_expired:
                self.read_guard({"Authorization": f"Bearer {value}"})
        return token

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

        if not refresh_token:
            self._refresh_token_secure = None
            return

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
            if not (existing_refresh_token and refresh_token != existing_refresh_token):
                return
            try:
                self._refresh_token_secure = SecureToken(
                    _value=existing_refresh_token,
                    _expires_at=time.time() + expires_in * 24,
                )
                logger.debug("Restored existing refresh token after new token failed")
            except ValueError:
                logger.warning("Could not restore existing refresh token either")

    def clear_tokens(self) -> None:
        """Clear secure token storage."""
        if self._current_token:
            self._current_token.clear()
            self._current_token = None
        if self._refresh_token_secure:
            self._refresh_token_secure.clear()
            self._refresh_token_secure = None

    def _invalidate_refresh_state(self) -> None:
        """Drop the refresh token everywhere this manager holds it."""
        from traigent.cloud.auth import AuthStatus

        self.clear_tokens()
        if self._set_credentials_fn:
            self._set_credentials_fn(None, AuthStatus.INVALID)
        if self._clear_cached_credentials_fn:
            # The encrypted cache would otherwise keep (and reload) the old
            # refresh token. Best effort: never mask the original failure.
            try:
                self._clear_cached_credentials_fn()
            except Exception as exc:
                logger.warning(
                    "Could not clear cached credentials: %s", type(exc).__name__
                )

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
                if fut.cancelled():
                    logger.debug("Token refresh task cancelled")
                    return
                fut.result()
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

        lock = self._auth_lock or asyncio.Lock()
        async with lock:
            # Read credentials and the refresh token under the lock so a caller
            # that waited never reuses a token an earlier refresh invalidated.
            credentials = (
                self._get_credentials_fn() if self._get_credentials_fn else None
            )
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
                logger.error("Token refresh failed: %s", type(exc).__name__)
                return AuthResult(
                    success=False,
                    status=AuthStatus.INVALID,
                    error_message=_GENERIC_REFRESH_ERROR,
                )

    async def refresh_jwt_secure(self, refresh_token_value: str) -> AuthResult:
        """Refresh JWT access token using secure stored refresh token.

        Args:
            refresh_token_value: The refresh token to use

        Returns:
            AuthResult with new credentials on success
        """
        from traigent.cloud.auth import AuthResult, AuthStatus
        from traigent.cloud.client import cloud_backend_egress_disabled

        if cloud_backend_egress_disabled(self.no_egress):
            return AuthResult(
                success=False,
                status=AuthStatus.INVALID,
                error_message=_GENERIC_REFRESH_ERROR,
            )

        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp not available for token refresh")

        from traigent.cloud.resilient_client import (
            ResilientClient,
            single_shot_unkeyed_auth_call,
        )
        from traigent.config.backend_config import BackendConfig

        try:
            backend_api_url = await validate_cloud_base_url_async(
                BackendConfig.build_api_base(
                    self.config.backend_base_url
                    or BackendConfig.get_cloud_backend_url()
                ),
                purpose="JWT token refresh",
            )
        except ValueError as exc:
            logger.warning("Rejected JWT token refresh URL: %s", exc)
            return AuthResult(
                success=False,
                status=AuthStatus.INVALID,
                error_message=str(exc),
            )

        refresh_url = f"{backend_api_url}/auth/refresh"

        client = ResilientClient(
            max_retries=MAX_RETRIES,
            base_delay=1.0,
            max_delay=10.0,
            jitter_factor=0.1,
        )

        async def perform_refresh():
            async with gate_session(
                aiohttp.ClientSession(
                    trace_configs=same_host_redirect_trace_configs(), trust_env=True
                )
            ) as session:
                async with session.post(
                    refresh_url,
                    json={"refresh_token": refresh_token_value},
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=30),
                    # A 307/308 would make aiohttp replay the POST (and the
                    # refresh token) elsewhere; a 3xx is a failure (#1975).
                    allow_redirects=False,
                ) as response:
                    return await TokenManager._parse_jwt_refresh_response(
                        response, refresh_token_value
                    )

        token_data: dict[str, Any] | None = None
        try:
            # Refresh is a non-idempotent POST with no Idempotency-Key (token
            # rotation): never replay it on an ambiguous failure (#1975).
            try:
                token_data = await client.execute_with_retry(
                    single_shot_unkeyed_auth_call(
                        perform_refresh, passthrough=(RefreshRejectedError,)
                    ),
                    operation_name="token_refresh",
                )
            except asyncio.CancelledError:
                # The POST may already have rotated the token server-side.
                self._invalidate_refresh_state()
                raise

            if not token_data:
                return AuthResult(
                    success=False,
                    status=AuthStatus.INVALID,
                    error_message="Token refresh returned no data",
                )

            return await self._commit_refreshed_jwt(token_data)

        except ValueError as exc:
            self._invalidate_refresh_state()
            return AuthResult(
                success=False,
                status=AuthStatus.INVALID,
                error_message=str(exc),
            )
        except Exception as exc:
            if token_data or isinstance(exc, NonRetryableError):
                # Server already returned a 2xx token body (the old refresh
                # token is consumed) or the outcome is ambiguous. Fail closed.
                # Drop the old token so the next refresh reports "No
                # credentials available" and the caller must re-authenticate,
                # instead of resending a token that refresh-token-reuse
                # detection would treat as theft.
                self._invalidate_refresh_state()
            logger.error("Token refresh error: %s", type(exc).__name__)
            return AuthResult(
                success=False,
                status=AuthStatus.INVALID,
                error_message=_GENERIC_REFRESH_ERROR,
            )

    async def _run_post_refresh_steps(
        self, updated_credentials: AuthCredentials
    ) -> None:
        """Non-fatal follow-ups after a committed refresh (schedule, cache)."""
        try:
            if self.config.auto_refresh and updated_credentials.refresh_token:
                self.schedule_refresh(updated_credentials)

            if self.config.cache_credentials and self._cache_credentials_fn:
                await self._cache_credentials_fn(updated_credentials)
        except Exception as post_exc:
            logger.warning(
                "Post-refresh step failed after the new token was installed; "
                "keeping it: %s",
                type(post_exc).__name__,
            )

    async def _commit_refreshed_jwt(self, token_data: dict[str, Any]) -> AuthResult:
        """Install refreshed JWT credentials and build the success result."""
        from traigent.cloud.auth import AuthResult, AuthStatus

        # Latch BEFORE committing: the callback refuses (raises) on a flagged manager
        # that cannot latch, and then neither credentials nor the secure store change.
        updated_credentials = self.build_credentials_from_token_data(token_data)

        if self._set_credentials_fn:
            self._set_credentials_fn(updated_credentials, AuthStatus.AUTHENTICATED)
        self.store_tokens(token_data)

        # Committed: the new token is installed and usable. A later
        # failure (auto-refresh scheduling, the cache callback) must not
        # invalidate it, so those steps are non-fatal.
        headers: dict[str, str] = {}
        if updated_credentials.jwt_token:
            headers["Authorization"] = f"Bearer {updated_credentials.jwt_token}"

        await self._run_post_refresh_steps(updated_credentials)

        return AuthResult(
            success=True,
            status=AuthStatus.AUTHENTICATED,
            credentials=updated_credentials,
            headers=headers,
            expires_in=token_data.get("expires_in"),
        )

    @staticmethod
    async def _parse_jwt_refresh_response(
        response: Any, refresh_token_value: str
    ) -> dict[str, Any]:
        if response.status == 401:
            raise RefreshRejectedError("Refresh token invalid or expired")
        if response.status == 429:
            await response.text()  # drain body; do not propagate upstream content
            raise RateLimitError("Token refresh failed with HTTP 429")
        if response.status != 200:
            await response.text()  # drain body; do not propagate upstream content
            raise RuntimeError(f"Token refresh failed with HTTP {response.status}")
        data = await response.json()
        if not data.get("success"):
            raise RefreshRejectedError(_GENERIC_REFRESH_ERROR)

        payload = data.get("data", {})
        if "refresh_token" not in payload or not payload["refresh_token"]:
            payload["refresh_token"] = refresh_token_value
        return cast(dict[str, Any], payload)

    async def _build_oauth2_token_url(self) -> str:
        from traigent.config.backend_config import BackendConfig

        cloud_base_url = await validate_cloud_base_url_async(
            self.config.cloud_base_url or BackendConfig.get_cloud_backend_url(),
            purpose="OAuth2 cloud_base_url",
        )
        return f"{cloud_base_url}/oauth/token"

    async def _oauth2_token_url_or_failure(
        self,
    ) -> tuple[str | None, AuthResult | None]:
        from traigent.cloud.auth import AuthResult, AuthStatus

        try:
            return await self._build_oauth2_token_url(), None
        except ValueError as exc:
            logger.warning("Rejected unsafe OAuth2 cloud_base_url: %s", exc)
            return None, AuthResult(
                success=False,
                status=AuthStatus.INVALID,
                error_message=f"OAuth2 cloud_base_url rejected (private or loopback host not allowed): {exc}",
            )

    async def _apply_oauth2_token_response(
        self,
        credentials: AuthCredentials,
        token_data: dict[str, Any],
    ) -> AuthResult:
        from traigent.cloud.auth import AuthResult, AuthStatus

        # Build a fresh candidate: ``credentials`` aliases the manager's stored object, which
        # must stay untouched unless the latch-aware setter accepts the new token.
        candidate = dataclasses.replace(
            credentials,
            scopes=list(credentials.scopes),
            metadata=dict(credentials.metadata),
            jwt_token=token_data["access_token"],
            refresh_token=token_data.get("refresh_token", credentials.refresh_token),
            expires_at=time.time() + token_data.get("expires_in", 3600),
        )
        credentials = candidate

        if self._set_credentials_fn:
            self._set_credentials_fn(credentials, AuthStatus.AUTHENTICATED)

        if self.config.cache_credentials and self._cache_credentials_fn:
            await self._cache_credentials_fn(credentials)

        return AuthResult(
            success=True,
            status=AuthStatus.AUTHENTICATED,
            credentials=credentials,
            expires_in=token_data.get("expires_in", 3600),
        )

    def _oauth2_egress_blocked(self) -> bool:
        from traigent.cloud.client import (
            CloudEgressBlockedError,
            raise_if_cloud_egress_disabled,
        )

        try:
            raise_if_cloud_egress_disabled(
                "OAuth2 token refresh", no_egress=self.no_egress
            )
        except CloudEgressBlockedError:
            return True
        return False

    def _oauth2_refresh_failure(
        self, exc: Exception, token_data: dict[str, Any] | None
    ) -> AuthResult:
        from traigent.cloud.auth import AuthResult, AuthStatus

        if token_data is not None or isinstance(exc, NonRetryableError):
            # Ambiguous outcome, or the server already consumed the old
            # refresh token: drop it everywhere so it is never resent.
            self._invalidate_refresh_state()
        logger.error("OAuth2 token refresh failed: %s", type(exc).__name__)
        return AuthResult(
            success=False,
            status=AuthStatus.INVALID,
            error_message="OAuth2 token refresh failed",
        )

    async def refresh_oauth2(self) -> AuthResult:
        """Refresh OAuth2 access token.

        Returns:
            AuthResult with refresh status
        """
        from traigent.cloud.auth import AuthResult, AuthStatus

        if self._oauth2_egress_blocked():
            return AuthResult(
                success=False,
                status=AuthStatus.INVALID,
                error_message=_GENERIC_REFRESH_ERROR,
            )

        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp not available for token refresh") from None

        credentials = self._get_credentials_fn() if self._get_credentials_fn else None
        unusable = _oauth2_credentials_failure(credentials)
        if unusable is not None:
            return unusable
        credentials = cast("AuthCredentials", credentials)

        token_url, url_failure = await self._oauth2_token_url_or_failure()
        if url_failure is not None:
            return url_failure
        token_url = cast(str, token_url)

        data = {
            "grant_type": "refresh_token",
            "refresh_token": credentials.refresh_token,
            "client_id": credentials.client_id,
            "client_secret": credentials.client_secret,
        }

        # refresh_oauth2 advertises an ``AuthResult`` return contract and is
        # called via delegation from AuthManager.refresh_oauth2_token (see
        # traigent/cloud/auth.py). Network errors, JSON parse errors, and
        # non-200 responses must be reported as AuthResult(success=False,
        # status=INVALID) so callers can inspect the result and decide how
        # to react (e.g. surface to the user, retry, or clear stored
        # credentials). Propagating exceptions would break that contract.
        # NOTE: this branch returns the INVALID AuthResult without actively
        # clearing self credentials; if specific failure modes (e.g. 401
        # invalid_grant) should also invalidate stored credentials, that
        # behavior should be added explicitly here.
        from traigent.cloud.resilient_client import (
            ResilientClient,
            single_shot_unkeyed_auth_call,
        )

        ambiguous_status = False

        async def perform_refresh() -> AuthResult | dict[str, Any]:
            nonlocal ambiguous_status
            async with gate_session(
                aiohttp.ClientSession(
                    trace_configs=same_host_redirect_trace_configs(), trust_env=True
                )
            ) as session:
                async with session.post(
                    token_url,
                    data=data,
                    # A 307/308 would replay the POST (and the refresh token
                    # and client secret) elsewhere (#1975).
                    allow_redirects=False,
                ) as response:
                    if response.status != 200:
                        rejection, ambiguous_status = await _oauth2_non_200_outcome(
                            response
                        )
                        return rejection
                    return cast(dict[str, Any], await response.json())

        client = ResilientClient(
            max_retries=MAX_RETRIES,
            base_delay=1.0,
            max_delay=10.0,
            jitter_factor=0.1,
        )
        token_data: dict[str, Any] | None = None
        try:
            # Non-idempotent POST that may rotate the refresh token: never
            # replay it on an ambiguous failure (#1975).
            try:
                outcome = await client.execute_with_retry(
                    single_shot_unkeyed_auth_call(perform_refresh),
                    operation_name="oauth2_token_refresh",
                )
            except asyncio.CancelledError:
                # The POST may already have rotated the token server-side.
                self._invalidate_refresh_state()
                raise
            if isinstance(outcome, AuthResult):
                if ambiguous_status:
                    self._invalidate_refresh_state()
                return outcome
            token_data = outcome
            return await self._apply_oauth2_token_response(credentials, token_data)
        except Exception as exc:
            return self._oauth2_refresh_failure(exc, token_data)

    def build_credentials_from_token_data(
        self, token_data: dict[str, Any]
    ) -> AuthCredentials:
        """Construct AuthCredentials from token payload.

        Args:
            token_data: Token response data

        Returns:
            AuthCredentials populated with token data
        """
        from traigent.cloud.auth import AuthCredentials, AuthMode, TokenExpiredError

        expires_in = token_data.get("expires_in")
        expires_at = time.time() + expires_in if expires_in else None
        refresh_token = token_data.get("refresh_token")
        if not refresh_token and self._refresh_token_secure:
            try:
                refresh_token = self._refresh_token_secure.get_value()
            except TokenExpiredError:
                refresh_token = None
            if refresh_token and self._guarding():
                # handed out inside AuthCredentials: latch it like the guarded getter does
                self.read_guard({"Authorization": f"Bearer {refresh_token}"})

        metadata, api_key = TokenManager._credentials_metadata(
            self, token_data, expires_at
        )

        return AuthCredentials(
            mode=AuthMode.JWT_TOKEN,
            jwt_token=token_data.get("access_token"),
            refresh_token=refresh_token,
            expires_at=expires_at,
            metadata=metadata,
            api_key=api_key,
        )

    def _credentials_metadata(
        self, token_data: dict[str, Any], expires_at: float | None
    ) -> tuple[dict[str, Any], Any]:
        metadata: dict[str, Any] = {}
        credentials = self._get_credentials_fn() if self._get_credentials_fn else None
        if credentials and credentials.metadata:
            metadata.update(credentials.metadata)
        if token_data.get("user") is not None:
            metadata["user"] = token_data.get("user", {})

        api_key = token_data.get("api_key") or token_data.get("apiKey")
        expires_at_dt = datetime.fromtimestamp(expires_at, UTC) if expires_at else None

        if api_key and self._validate_key_format and self._validate_key_format(api_key):
            if self._set_api_key_token:
                self._set_api_key_token(api_key, source="cli", expires_at=expires_at_dt)
            metadata.setdefault("source", "cli")
        return metadata, api_key

    def get_authorization_header(self) -> dict[str, str]:
        """Get authorization header from current token.

        Returns:
            Dictionary with Authorization header, or empty dict if no valid token

        Raises:
            TokenExpiredError: If token has expired
        """
        if self._current_token and not self._current_token.is_expired:
            header = cast(dict[str, str], self._current_token.get_header())
            if self._guarding():
                self.read_guard(header)
            return header
        return {}
