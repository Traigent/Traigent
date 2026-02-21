"""Password authentication handler for Traigent.

This module handles email/password authentication with rate limiting and
resilient backend communication.
Extracted from AuthManager to follow Single Responsibility Principle.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Security FUNC-SECURITY REQ-SEC-010

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import secrets
import time
import uuid
from typing import TYPE_CHECKING, Any, cast

from traigent.cloud._aiohttp_compat import AIOHTTP_AVAILABLE, aiohttp
from traigent.core.constants import MAX_RETRIES

if TYPE_CHECKING:
    from traigent.cloud.auth import AuthResult

logger = logging.getLogger(__name__)


class PasswordAuthHandler:
    """Handles password-based authentication with rate limiting.

    This class manages:
    - Email/password validation
    - Rate limiting of login attempts
    - Backend authentication calls
    - Development mode mocking
    """

    def __init__(self) -> None:
        """Initialize password auth handler."""
        self._failed_attempts = 0
        self._last_failure_time = 0.0

        # Callbacks
        self._build_credentials_fn: Any = None
        self._store_tokens_fn: Any = None

    def set_callbacks(
        self,
        *,
        build_credentials: Any,
        store_tokens: Any,
    ) -> None:
        """Set callback functions for credential operations.

        Args:
            build_credentials: Callback to build credentials from token data
            store_tokens: Callback to store secure tokens
        """
        self._build_credentials_fn = build_credentials
        self._store_tokens_fn = store_tokens

    async def authenticate(self, credentials: dict[str, str]) -> AuthResult:
        """Authenticate using interactive login credentials.

        Args:
            credentials: Dictionary containing 'email' and 'password'

        Returns:
            AuthResult with authentication status
        """
        # Import at runtime to avoid circular import
        from traigent.cloud.auth import AuthResult, AuthStatus, InvalidCredentialsError

        try:
            if self._should_rate_limit():
                wait_time = self._get_rate_limit_wait()
                logger.warning(
                    f"Rate limited: waiting {wait_time}s after failed attempts"
                )
                await asyncio.sleep(wait_time)

            if not self._validate_credentials(credentials):
                self._record_failure()
                return AuthResult(
                    success=False,
                    status=AuthStatus.INVALID,
                    error_message="Invalid credential format",
                )

            token_data = await self._perform_authentication(credentials)
            if not token_data:
                self._record_failure()
                return AuthResult(
                    success=False,
                    status=AuthStatus.UNAUTHENTICATED,
                    error_message="Authentication failed",
                )

            # Success path
            auth_credentials = None
            if self._build_credentials_fn:
                auth_credentials = self._build_credentials_fn(token_data)

            if self._store_tokens_fn:
                self._store_tokens_fn(token_data)

            self._reset_failure_tracking()

            headers: dict[str, str] = {}
            if auth_credentials:
                if auth_credentials.jwt_token:
                    headers["Authorization"] = f"Bearer {auth_credentials.jwt_token}"
                if auth_credentials.api_key:
                    headers["X-API-Key"] = auth_credentials.api_key

            return AuthResult(
                success=True,
                status=AuthStatus.AUTHENTICATED,
                credentials=auth_credentials,
                headers=headers,
                expires_in=token_data.get("expires_in"),
            )

        except InvalidCredentialsError as exc:
            self._record_failure()
            return AuthResult(
                success=False, status=AuthStatus.INVALID, error_message=str(exc)
            )
        except Exception as exc:
            logger.error(f"Authentication error: {type(exc).__name__}: {exc}")
            self._record_failure()
            return AuthResult(
                success=False, status=AuthStatus.INVALID, error_message=str(exc)
            )

    def _should_rate_limit(self) -> bool:
        """Check if login attempts should be rate limited."""
        if self._failed_attempts >= 3:
            if time.time() - self._last_failure_time < 60:
                return True
        return False

    def _get_rate_limit_wait(self) -> float:
        """Calculate exponential backoff delay with jitter."""
        wait_time = min(2**self._failed_attempts, 60)
        jitter = secrets.randbelow(1000) / 1000
        return cast(float, wait_time + jitter)

    def _record_failure(self) -> None:
        """Record failed authentication attempt."""
        self._failed_attempts += 1
        self._last_failure_time = time.time()
        logger.warning(f"Authentication failure #{self._failed_attempts}")

    def _reset_failure_tracking(self) -> None:
        """Reset failure tracking after successful login."""
        self._failed_attempts = 0
        self._last_failure_time = 0.0

    def _validate_credentials(self, credentials: dict[str, str]) -> bool:
        """Validate email/password credentials without logging sensitive values."""
        if self._is_dev_mode_enabled():
            logger.warning(
                "Development mode enabled - skipping strict credential validation"
            )
            return True

        required_fields = {"email", "password"}
        if not all(field in credentials for field in required_fields):
            logger.error("Missing required credential fields")
            return False

        email = credentials.get("email", "")
        if "@" not in email or len(email) < 5:
            logger.error("Invalid email format")
            return False

        password = credentials.get("password", "")
        if len(password) < 8:
            logger.error("Password does not meet minimum requirements")
            return False

        return True

    def _is_dev_mode_enabled(self) -> bool:
        """Return True when running in an explicitly non-production mode."""
        env = os.getenv("TRAIGENT_ENV", "").strip().lower()
        if env in {"dev", "development", "local"}:
            return True

        if os.getenv("TRAIGENT_DEV_MODE", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return True

        try:
            from traigent.config.backend_config import BackendConfig

            if BackendConfig.is_local_backend():
                return True
        except Exception as e:
            logger.debug(f"Could not check local backend status: {e}")

        return False

    async def _perform_authentication(
        self, credentials: dict[str, str]
    ) -> dict[str, Any] | None:
        """Perform backend authentication using resilient HTTP client."""
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp not available for authentication") from None

        # Import at runtime to avoid circular import
        from traigent.cloud.auth import InvalidCredentialsError
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

    def _build_dev_token_payload(
        self, credentials: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """Generate mock tokens for development-only fallback."""
        email = (credentials or {}).get("email") or "dev@traigent.local"
        user_id = (credentials or {}).get("user_id") or f"dev-{uuid.uuid4()}"
        dev_key_suffix = "".join(
            secrets.choice(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
            )
            for _ in range(61)
        )
        return {
            "access_token": f"dev-access-{uuid.uuid4()}",
            "refresh_token": f"dev-refresh-{uuid.uuid4()}",
            "api_key": f"tg_{dev_key_suffix}",
            "user": {"email": email, "id": user_id},
            "expires_in": 3600,
            "dev_mode": True,
        }
