"""API key management module for Traigent authentication.

This module handles API key lifecycle, validation, rotation, and secure storage.
Extracted from AuthManager to follow Single Responsibility Principle.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Security CONC-Quality-Maintainability FUNC-SEC-APIKEY-MGMT REQ-SEC-010 REQ-CLOUD-009 SYNC-CloudHybrid

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from traigent.cloud.auth import (
        APIKey,
        AuthCredentials,
        SecureToken,
        UnifiedAuthConfig,
    )

logger = logging.getLogger(__name__)

# Default TTL for API key tokens in seconds (30 days)
API_KEY_TOKEN_TTL = 30 * 24 * 60 * 60

# Track sources that have already been warned about to reduce log noise
_warned_api_key_sources: set[str] = set()


class APIKeyManager:
    """Manages API key lifecycle, validation, and rotation.

    This class handles:
    - Secure storage of API key tokens
    - API key format validation
    - Expiration and rotation tracking
    - Status monitoring and health checks
    """

    def __init__(self, config: UnifiedAuthConfig) -> None:
        """Initialize API key manager.

        Args:
            config: Unified authentication configuration
        """
        self.config = config

        # API key state
        self._api_key_token: SecureToken | None = None
        self._api_key_preview: str | None = None
        self._api_key_source: str | None = None
        self._api_key_expiry: datetime | None = None
        self._api_key_last_rotated: datetime | None = None
        self._api_key: APIKey | None = None

        # Callbacks for credential operations (set by AuthManager)
        self._get_credentials_fn: Any = None
        self._get_provided_credentials_fn: Any = None
        self._get_last_auth_result_fn: Any = None

    def set_callbacks(
        self,
        *,
        get_credentials: Any = None,
        get_provided_credentials: Any = None,
        get_last_auth_result: Any = None,
    ) -> None:
        """Set callback functions for credential operations.

        Args:
            get_credentials: Callback to get current credentials
            get_provided_credentials: Callback to get provided credentials
            get_last_auth_result: Callback to get last auth result
        """
        if get_credentials is not None:
            self._get_credentials_fn = get_credentials
        if get_provided_credentials is not None:
            self._get_provided_credentials_fn = get_provided_credentials
        if get_last_auth_result is not None:
            self._get_last_auth_result_fn = get_last_auth_result

    @property
    def api_key_token(self) -> SecureToken | None:
        """Get current API key token."""
        return self._api_key_token

    @property
    def api_key_preview(self) -> str | None:
        """Get API key preview (masked)."""
        return self._api_key_preview

    @property
    def api_key_source(self) -> str | None:
        """Get API key source."""
        return self._api_key_source

    @property
    def api_key_expiry(self) -> datetime | None:
        """Get API key expiry datetime."""
        return self._api_key_expiry

    @property
    def api_key_last_rotated(self) -> datetime | None:
        """Get timestamp of last API key rotation."""
        return self._api_key_last_rotated

    @property
    def api_key(self) -> APIKey | None:
        """Get the APIKey object."""
        return self._api_key

    @staticmethod
    def mask_key(api_key: str) -> str:
        """Mask an API key for display, showing only first/last 4 characters.

        Args:
            api_key: The API key to mask

        Returns:
            Masked API key string (e.g., "tg_a****xyz1")
        """
        if not api_key:
            return ""
        if len(api_key) <= 8:
            return "*" * len(api_key)
        return f"{api_key[:4]}{'*' * (len(api_key) - 8)}{api_key[-4:]}"

    def set_token(
        self,
        api_key: str,
        source: str | None = None,
        expires_at: datetime | None = None,
    ) -> None:
        """Set or update the API key token with secure storage.

        Args:
            api_key: The API key value
            source: Source of the API key (e.g., "env", "cli", "explicit")
            expires_at: Optional expiration datetime
        """
        from traigent.cloud.auth import SecureToken

        if not api_key:
            self.clear_token()
            return

        try:
            ttl_seconds: float = API_KEY_TOKEN_TTL
            if self.config.api_key_default_ttl_days:
                ttl_seconds = max(60.0, self.config.api_key_default_ttl_days * 86400)
            self._api_key_token = SecureToken(
                _value=api_key,
                _expires_at=time.time() + ttl_seconds,
            )
        except ValueError as e:
            # Only warn once per source to reduce log noise; suppress in offline mode
            from traigent.config.backend_config import SIGNUP_URL
            from traigent.utils.env_config import is_backend_offline

            warn_key = f"{source}:{len(api_key)}"
            if warn_key not in _warned_api_key_sources and not is_backend_offline():
                logger.warning(
                    "Ignoring invalid API key (length=%d, source=%s): %s. "
                    "Continuing without cloud authentication. "
                    "Get a valid key at %s",
                    len(api_key),
                    source,
                    e,
                    SIGNUP_URL,
                )
                _warned_api_key_sources.add(warn_key)
            else:
                logger.debug(
                    "Ignoring invalid API key (length=%d, source=%s): %s",
                    len(api_key),
                    source,
                    e,
                )
            self._api_key_token = None
            self._api_key_preview = None
        else:
            self._api_key_preview = self.mask_key(api_key)
            logger.info(
                "✅ API key token set successfully (source=%s, preview=%s)",
                source,
                self._api_key_preview,
            )

        self._api_key_source = source
        self._api_key_last_rotated = datetime.now(UTC)

        if expires_at is None and self.config.api_key_default_ttl_days:
            expires_at = datetime.now(UTC) + timedelta(
                days=self.config.api_key_default_ttl_days
            )

        if expires_at is not None:
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=UTC)
            self._api_key_expiry = expires_at

    def clear_token(self) -> None:
        """Clear all API key state securely."""
        if self._api_key_token:
            self._api_key_token.clear()
        self._api_key_token = None
        self._api_key_preview = None
        self._api_key_source = None
        self._api_key_expiry = None
        self._api_key_last_rotated = None

    def clear_api_key_object(self) -> None:
        """Clear the APIKey object (separate from token)."""
        self._api_key = None

    def get_preview(self) -> str | None:
        """Return a masked preview of the currently configured API key.

        Returns:
            Masked API key preview or None if no key is set
        """
        return self._api_key_preview

    def has_key(self) -> bool:
        """Return True if an API key has been configured.

        Returns:
            True if API key is available, False otherwise
        """
        credentials = self._get_credentials_fn() if self._get_credentials_fn else None
        return self._api_key_token is not None or (
            credentials is not None and bool(credentials.api_key)
        )

    def _extract_valid_key(self, credentials: Any) -> str | None:
        """Return api_key from credentials if it passes format validation."""
        if (
            credentials
            and credentials.api_key
            and self.validate_format(credentials.api_key)
        ):
            return str(credentials.api_key)
        return None

    def get_key_for_internal_use(self) -> str | None:
        """Retrieve the API key value for internal operations.

        Checks multiple sources in order of priority:
        1. Secure token storage
        2. Current credentials
        3. Provided credentials
        4. Last authentication result

        Returns:
            API key string or None if not available
        """
        from traigent.cloud.auth import TokenExpiredError

        if self._api_key_token and not self._api_key_token.is_expired:
            try:
                return self._api_key_token.get_value()
            except TokenExpiredError:
                pass

        # Fall back through credential sources
        for fetch_fn in (
            self._get_credentials_fn,
            self._get_provided_credentials_fn,
        ):
            if fetch_fn is None:
                continue
            key = self._extract_valid_key(fetch_fn())
            if key:
                return key

        # Last auth result has an extra .credentials indirection
        if self._get_last_auth_result_fn:
            last_result = self._get_last_auth_result_fn()
            if last_result:
                return self._extract_valid_key(last_result.credentials)

        return None

    def validate_format(self, key: str | None) -> bool:
        """Validate API key format.

        Accepts keys with valid prefixes and lengths, containing only
        alphanumerics or underscores after the prefix.
        Returns False for any deviation so callers can reject suspicious values early.

        Supported formats:
        - `tg_`: Standard Traigent API keys (64 characters total)
        - `uk_`: User/utility keys issued by the backend (46 characters total)

        Args:
            key: API key to validate

        Returns:
            True if format is valid, False otherwise
        """
        if not key or not isinstance(key, str):
            return False

        # Define valid prefixes and their expected total lengths
        prefix_lengths = {
            "tg_": 64,  # Standard Traigent API keys
            "uk_": 46,  # User/utility keys from backend
        }

        # Find matching prefix
        matched_prefix = None
        for prefix in prefix_lengths:
            if key.startswith(prefix):
                matched_prefix = prefix
                break

        if matched_prefix is None:
            return False

        expected_length = prefix_lengths[matched_prefix]
        if len(key) != expected_length:
            return False

        allowed = set(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"  # pragma: allowlist secret
        )
        return all(char in allowed for char in key[3:])

    def get_status(self) -> dict[str, Any]:
        """Return structured API key status for monitoring.

        Returns:
            Dictionary with state, preview, expires_at, and days_remaining
        """
        if not self.has_key():
            return {
                "state": "missing",
                "preview": None,
                "expires_at": None,
                "days_remaining": None,
            }

        expires_at = self._api_key_expiry
        preview = self.get_preview()

        if not expires_at:
            return {
                "state": "unknown",
                "preview": preview,
                "expires_at": None,
                "days_remaining": None,
            }

        now = datetime.now(UTC)
        delta = expires_at - now
        days_remaining = delta.total_seconds() / 86400

        if days_remaining <= 0:
            state = "expired"
        elif days_remaining <= self.config.api_key_critical_days:
            state = "critical"
        elif days_remaining <= self.config.api_key_warning_days:
            state = "warning"
        else:
            state = "ok"

        return {
            "state": state,
            "preview": preview,
            "expires_at": expires_at.isoformat(),
            "days_remaining": days_remaining,
        }

    def check_rotation(self) -> bool:
        """Log rotation guidance and return True when the key is healthy.

        Returns:
            True if key is healthy, False if rotation is needed
        """
        status = self.get_status()
        state = status["state"]

        if state == "ok":
            return True

        if state == "warning":
            logger.warning(
                "API key approaching rotation threshold (preview=%s, days_remaining=%.2f)",
                status["preview"],
                status["days_remaining"],
            )
            return False

        if state == "critical":
            logger.error(
                "API key rotation required immediately (preview=%s, days_remaining=%.2f)",
                status["preview"],
                status["days_remaining"],
            )
            return False

        if state == "expired":
            logger.error(
                "API key expired and must be rotated (preview=%s)",
                status["preview"],
            )
            return False

        logger.warning(
            "API key status unknown; recommend rotation (preview=%s)", status["preview"]
        )
        return False

    def persist_api_key(self, credentials: AuthCredentials) -> None:
        """Persist API key tokens after authentication.

        Args:
            credentials: Credentials containing the API key
        """
        from traigent.cloud.auth import APIKey

        api_key_value = credentials.api_key or self.get_key_for_internal_use()
        if not api_key_value:
            return

        source = None
        if credentials.metadata:
            source = credentials.metadata.get("source")

        self.set_token(api_key_value, source=source)

        if self._api_key is None or self._api_key.key != api_key_value:
            now = datetime.now(UTC)
            self._api_key = APIKey(
                key=api_key_value,
                name="default",
                created_at=now,
                expires_at=now + timedelta(days=365),
                permissions={
                    "optimize": True,
                    "analytics": True,
                    "billing": True,
                },
            )

    def get_info(self) -> dict[str, Any] | None:
        """Return non-sensitive API key metadata if available.

        Returns:
            Dictionary with API key info or None if not available
        """
        from typing import cast

        if self._api_key:
            created_at_dt = cast(datetime, self._api_key.created_at)
            expires_at_dt = cast(datetime | None, self._api_key.expires_at)
            return {
                "name": self._api_key.name,
                "created_at": created_at_dt.isoformat(),
                "expires_at": (expires_at_dt.isoformat() if expires_at_dt else None),
                "permissions": self._api_key.permissions,
                "is_valid": self._api_key.is_valid(),
            }

        api_key_value = self.get_key_for_internal_use()

        if api_key_value and self.validate_format(api_key_value):
            return {
                "name": "environment",
                "created_at": None,
                "expires_at": None,
                "permissions": {"optimize": True, "analytics": True},
                "is_valid": True,
                "preview": self.get_preview(),
            }

        return None

    def set_api_key_object(self, api_key: APIKey) -> None:
        """Set the APIKey object directly.

        Args:
            api_key: APIKey object to set
        """
        self._api_key = api_key
