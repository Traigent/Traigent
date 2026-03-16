"""Centralized backend configuration for Traigent SDK.

This module provides a single source of truth for backend URL and API key configuration,
replacing hardcoded URLs scattered throughout the codebase.
"""

# Traceability: CONC-CloudService CONC-Security FUNC-CLOUD-HYBRID FUNC-SECURITY REQ-CLOUD-009 REQ-SEC-010 SYNC-CloudHybrid CONC-Layer-Core

from __future__ import annotations

import logging
import os
from typing import Any
from urllib.parse import urlparse, urlunparse

logger = logging.getLogger(__name__)

#: Single source of truth for the default local backend URL.
#: Import this constant instead of repeating the literal string.
DEFAULT_LOCAL_URL = "http://localhost:5000"

#: Single source of truth for the default cloud portal URL.
#: Import this constant instead of repeating the literal string.
DEFAULT_CLOUD_URL = "https://portal.traigent.ai"

#: Signup/login page — derived from the portal base.
SIGNUP_URL = f"{DEFAULT_CLOUD_URL}/login"


def get_no_credentials_hint() -> str:
    """One-liner appended to missing-credentials warnings."""
    return f"Sign up at {SIGNUP_URL} or run 'traigent auth login'."


class BackendConfig:
    """Centralized backend configuration management.

    This class provides a single source of truth for backend configuration,
    prioritizing environment variables and providing sensible defaults.
    """

    # Default backend URLs (overridable via environment variables)
    _FALLBACK_LOCAL_URL = DEFAULT_LOCAL_URL
    DEFAULT_CLOUD_URL = DEFAULT_CLOUD_URL  # Re-export module constant
    DEFAULT_PROD_URL = DEFAULT_CLOUD_URL  # Backward-compatible alias
    _DEFAULT_API_PATH = "/api/v1"

    @classmethod
    def _get_default_local_url(cls) -> str:
        """Resolve default local URL with environment overrides."""
        return os.environ.get("TRAIGENT_DEFAULT_LOCAL_URL") or cls._FALLBACK_LOCAL_URL

    @classmethod
    def get_default_local_url(cls) -> str:
        """Public accessor for the default local backend URL."""
        return cls._get_default_local_url()

    @classmethod
    def _normalize_origin(cls, value: str | None) -> str | None:
        """Return sanitized origin (scheme + host) for the backend."""

        if not value:
            return None

        value = value.strip()
        if not value:
            return None

        # Ensure scheme is present; default to https
        if "://" not in value:
            value = f"https://{value}"

        parsed = urlparse(value)

        # If parsing failed to produce a netloc (e.g., value was bare host)
        if not parsed.netloc and parsed.path:
            parsed = urlparse(f"https://{parsed.path}")

        if not parsed.scheme or not parsed.netloc:
            logger.warning("Failed to normalize backend origin from '%s'", value)
            return None

        normalized = urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))
        return normalized.rstrip("/")

    @classmethod
    def _extract_origin_and_path(
        cls, value: str | None
    ) -> tuple[str | None, str | None]:
        """Return origin and path components from a URL value."""

        if not value:
            return (None, None)

        value = value.strip()
        if not value:
            return (None, None)

        if "://" not in value:
            value = f"https://{value}"

        parsed = urlparse(value)
        if not parsed.netloc and parsed.path:
            parsed = urlparse(f"https://{parsed.path}")

        if not parsed.scheme or not parsed.netloc:
            logger.warning("Failed to parse backend url '%s'", value)
            return (None, None)

        origin = urlunparse((parsed.scheme, parsed.netloc, "", "", "", "")).rstrip("/")
        path = parsed.path.rstrip("/") or None
        return (origin, path)

    @classmethod
    def _get_configured_backend_origin(cls) -> str | None:
        """Return an explicitly configured backend origin, if any."""
        backend_env = os.environ.get("TRAIGENT_BACKEND_URL")
        api_env = os.environ.get("TRAIGENT_API_URL")

        origin = cls._normalize_origin(backend_env)
        if origin:
            logger.debug("Using TRAIGENT_BACKEND_URL: %s", origin)
            return origin

        api_origin, _ = cls._extract_origin_and_path(api_env)
        if api_origin:
            logger.debug("Using TRAIGENT_API_URL origin: %s", api_origin)
            return api_origin

        # Check stored CLI credentials for backend URL
        try:
            from traigent.cloud.credential_manager import CredentialManager

            raw_url = CredentialManager.get_stored_backend_url()
            stored_url = cls._normalize_origin(raw_url) if raw_url else None
            if stored_url:
                logger.debug(
                    "Using backend URL from stored CLI credentials: %s", stored_url
                )
                return stored_url
        except Exception:
            pass

        return None

    @classmethod
    def get_configured_backend_url(cls) -> str | None:
        """Return an explicitly configured backend origin, if available."""

        return cls._get_configured_backend_origin()

    @classmethod
    def get_backend_url(cls) -> str:
        """Get backend origin URL from environment, stored credentials, or local default.

        Priority:
            1. TRAIGENT_BACKEND_URL environment variable
            2. TRAIGENT_API_URL environment variable (origin extracted)
            3. Stored CLI credentials (backend_url from ``traigent auth login``)
            4. Local default URL
        """

        configured_origin = cls._get_configured_backend_origin()
        if configured_origin:
            return configured_origin

        default_local = cls._get_default_local_url()
        logger.debug("Using default local URL: %s", default_local)
        return default_local

    @classmethod
    def get_cloud_backend_url(cls) -> str:
        """Get backend origin URL for cloud-facing entry points."""

        configured_origin = cls._get_configured_backend_origin()
        if configured_origin:
            return configured_origin

        logger.debug("Using default cloud URL: %s", cls.DEFAULT_CLOUD_URL)
        return cls.DEFAULT_CLOUD_URL

    @classmethod
    def _build_api_url(cls, backend_origin: str) -> str:
        """Return the API base URL, respecting explicit API-path overrides."""

        api_env = os.environ.get("TRAIGENT_API_URL")
        if api_env:
            origin, path = cls._extract_origin_and_path(api_env)
            if origin:
                api_path = path or cls._DEFAULT_API_PATH
                return f"{origin}{api_path}"

        return f"{backend_origin}{cls._DEFAULT_API_PATH}"

    @classmethod
    def get_backend_api_url(cls) -> str:
        """Return the base API URL (including version path)."""

        return cls._build_api_url(cls.get_backend_url())

    @classmethod
    def get_cloud_api_url(cls) -> str:
        """Return the cloud API base URL (including version path)."""

        return cls._build_api_url(cls.get_cloud_backend_url())

    @classmethod
    def normalize_backend_origin(cls, value: str | None) -> str | None:
        """Public accessor for origin normalization."""

        return cls._normalize_origin(value)

    @classmethod
    def get_default_api_path(cls) -> str:
        """Return the default API path segment."""

        return cls._DEFAULT_API_PATH

    @classmethod
    def build_api_base(cls, origin: str | None = None) -> str:
        """Compose an API base URL from an origin."""

        resolved_origin = (origin or cls.get_backend_url()).rstrip("/")
        return f"{resolved_origin}{cls._DEFAULT_API_PATH}"

    @classmethod
    def split_api_url(cls, value: str | None) -> tuple[str | None, str | None]:
        """Public helper returning origin/path components for an API URL."""

        return cls._extract_origin_and_path(value)

    @classmethod
    def is_local_origin(cls, value: str | None) -> bool:
        """Return True when the provided backend origin targets localhost."""

        normalized = cls._normalize_origin(value)
        if not normalized:
            return False

        parsed = urlparse(normalized)
        hostname = (parsed.hostname or "").lower()
        return hostname in {"localhost", "127.0.0.1", "::1"}

    @classmethod
    def get_api_key(cls) -> str | None:
        """Get API key from environment or stored CLI credentials.

        Priority:
            1. TRAIGENT_API_KEY environment variable
            2. Stored CLI credentials (api_key only, not JWT)

        Returns:
            str | None: API key if configured, None otherwise
        """
        api_key = os.environ.get("TRAIGENT_API_KEY")
        if api_key:
            logger.info(
                "✅ Using API key from TRAIGENT_API_KEY (length=%d)", len(api_key)
            )
            return api_key

        # Fall through to CLI-stored credentials — api_key only, not JWT
        try:
            from traigent.cloud.credential_manager import CredentialManager

            stored_key = CredentialManager.get_stored_api_key_only()
            if stored_key:
                logger.info(
                    "✅ Using API key from stored CLI credentials (length=%d)",
                    len(stored_key),
                )
                return stored_key
        except Exception:
            pass

        # Only warn if not in offline mode - offline mode doesn't need API keys
        from traigent.utils.env_config import is_backend_offline

        if not is_backend_offline():
            logger.warning(
                "No API key found (checked TRAIGENT_API_KEY and stored credentials). %s",
                get_no_credentials_hint(),
            )
        return None

    @classmethod
    def has_auth_credentials(cls) -> bool:
        """Check whether any non-empty backend credentials are available.

        This broader predicate is intended for UX warnings that should stay
        silent when the user is already authenticated via either API key or JWT.
        """
        if os.environ.get("TRAIGENT_API_KEY") or os.environ.get("TRAIGENT_JWT_TOKEN"):
            return True

        try:
            from traigent.cloud.credential_manager import CredentialManager

            stored_creds = CredentialManager.get_credentials()
        except Exception:
            return False

        return bool(stored_creds.get("api_key") or stored_creds.get("jwt_token"))

    @classmethod
    def is_local_backend(cls) -> bool:
        """Check if using local backend URL.

        Returns:
            bool: True if backend URL points to localhost
        """
        return cls.is_local_origin(cls.get_backend_url())

    @classmethod
    def requires_authentication(cls) -> bool:
        """Check if backend requires authentication.

        Local backends typically don't require authentication,
        while production backends always do.

        Returns:
            bool: True if authentication is required
        """
        # Local backends may not require authentication
        if cls.is_local_backend():
            # Check if explicitly enabled
            return os.environ.get("REQUIRE_LOCAL_AUTH", "false").lower() == "true"

        # Production always requires authentication
        return True

    @classmethod
    def get_config_summary(cls) -> dict[str, Any]:
        """Get current configuration summary for debugging.

        Returns:
            dict: Configuration summary (sanitized for logging)
        """
        api_key = cls.get_api_key()
        return {
            "backend_url": cls.get_backend_url(),
            "backend_api_url": cls.get_backend_api_url(),
            "api_key_configured": api_key is not None,
            "api_key_prefix": api_key[:8] + "..." if api_key else None,
            "is_local": cls.is_local_backend(),
            "requires_auth": cls.requires_authentication(),
            "environment": os.environ.get("TRAIGENT_ENV", "production"),
            "configured_via": next(
                (
                    env_var
                    for env_var in (
                        "TRAIGENT_BACKEND_URL",
                        "TRAIGENT_API_URL",
                    )
                    if os.environ.get(env_var)
                ),
                "default",
            ),
            "api_key_env": next(
                (
                    env_var
                    for env_var in ("TRAIGENT_API_KEY",)
                    if os.environ.get(env_var)
                ),
                None,
            ),
        }
