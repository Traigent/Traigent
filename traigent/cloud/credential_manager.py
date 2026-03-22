"""Credential management for Traigent SDK.

Provides unified access to authentication credentials from multiple sources:
1. Environment variables (highest priority)
2. CLI stored credentials (secure storage)
3. Explicit dev-mode credentials (development only)
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability CONC-Quality-Security FUNC-CLOUD-HYBRID FUNC-SECURITY REQ-CLOUD-009 REQ-SEC-010 SYNC-CloudHybrid

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

from traigent.config.backend_config import BackendConfig

from traigent.utils.logging import get_logger

logger = get_logger(__name__)

# Constants matching CLI auth
TRAIGENT_CONFIG_DIR = Path.home() / ".traigent"
CREDENTIALS_FILE = TRAIGENT_CONFIG_DIR / "credentials.json"


class CredentialManager:
    """Manages authentication credentials for Traigent SDK."""

    @classmethod
    def get_api_key(cls) -> str | None:
        """Get API key from available sources.

        Priority order:
        1. TRAIGENT_API_KEY environment variable
        2. Stored credentials from CLI auth
        3. Explicit dev-mode credentials

        Returns:
            API key or None if not found
        """
        # Check environment variables first (highest priority)
        api_key = cls._get_env_api_key()
        if api_key:
            logger.debug("Using API key from environment variable")
            return api_key

        # Check for CLI stored credentials
        stored_creds = cls._load_cli_credentials()
        if stored_creds:
            # Prefer API key over JWT
            if stored_creds.get("api_key"):
                logger.debug("Using API key from CLI credentials")
                return cast(str, stored_creds["api_key"])
            elif stored_creds.get("jwt_token"):
                logger.debug("Using JWT token from CLI credentials")
                return cast(str, stored_creds["jwt_token"])

        # Development fallback
        if cls._is_development_environment():
            logger.debug("Using default test credentials (development only)")
            return "test_api_key_for_development"

        logger.debug("No API key found in any source")
        return None

    @classmethod
    def get_credentials(cls) -> dict[str, Any]:
        """Get complete credentials including tokens and user info.

        Returns:
            Dictionary with credentials or empty dict
        """
        # Environment variables
        api_key = cls._get_env_api_key()
        if api_key:
            return {
                "api_key": api_key,
                "backend_url": BackendConfig.get_backend_url(),
                "source": "environment",
            }

        # CLI stored credentials
        stored_creds = cls._load_cli_credentials()
        if stored_creds:
            stored_creds["source"] = "cli"
            return stored_creds

        # Development fallback
        if cls._is_development_environment():
            return {
                "api_key": "test_api_key_for_development",  # pragma: allowlist secret
                "backend_url": BackendConfig.get_backend_url(),
                "source": "development",
            }

        return {}

    @classmethod
    def _load_cli_credentials(cls) -> dict[str, Any] | None:
        """Load credentials stored by CLI auth command.

        Reads from the local credentials file only. Keyring is not used
        by the SDK to avoid triggering OS credential prompts (e.g. macOS
        Keychain) during normal library usage.

        Returns:
            Stored credentials or None
        """
        if CREDENTIALS_FILE.exists():
            try:
                with open(CREDENTIALS_FILE) as f:
                    return cast(dict[str, Any], json.load(f))
            except Exception as e:
                logger.debug(f"Failed to load credentials file: {e}")

        return None

    @classmethod
    def get_stored_api_key_only(cls) -> str | None:
        """Return the stored API key from CLI credentials, or None.

        Unlike :meth:`get_api_key`, this does **not** fall back to JWT tokens
        or development credentials.  It is safe to call from code paths that
        must not mix credential types (e.g. ``BackendConfig.get_api_key``).
        """
        stored_creds = cls._load_cli_credentials()
        if stored_creds and stored_creds.get("api_key"):
            return cast(str, stored_creds["api_key"])
        return None

    @classmethod
    def get_stored_backend_url(cls) -> str | None:
        """Return the stored backend URL from CLI credentials, or None."""
        stored_creds = cls._load_cli_credentials()
        if stored_creds and stored_creds.get("backend_url"):
            return cast(str, stored_creds["backend_url"])
        return None

    @classmethod
    def _is_development_environment(cls) -> bool:
        """Check if running in development environment.

        Returns:
            True if in development mode
        """
        # Only explicit development toggles may enable the fallback credential.
        return any(
            [
                os.environ.get("TRAIGENT_DEV_MODE", "").lower() in ("true", "1", "yes"),
                os.environ.get("TRAIGENT_GENERATE_MOCKS", "").lower()
                in ("true", "1", "yes"),
            ]
        )

    @classmethod
    def is_authenticated(cls) -> bool:
        """Check if any valid credentials are available.

        Returns:
            True if authenticated
        """
        return cls.get_api_key() is not None

    @classmethod
    def get_auth_headers(cls) -> dict[str, str]:
        """Get authentication headers for API requests.

        Returns:
            Headers dict with authentication
        """
        api_key = cls.get_api_key()
        if not api_key:
            return {}

        # Determine if it's a JWT token or API key
        if "." in api_key and len(api_key.split(".")) == 3:
            # Looks like a JWT token
            return {"Authorization": f"Bearer {api_key}"}
        else:
            # API key
            return {"X-API-Key": api_key}

    @staticmethod
    def _get_env_api_key() -> str | None:
        """Return API key from supported environment variables."""

        return os.environ.get("TRAIGENT_API_KEY")

    @classmethod
    def clear_credentials(cls) -> bool:
        """Clear stored credentials (for logout).

        Returns:
            True if cleared successfully
        """
        success = True

        if CREDENTIALS_FILE.exists():
            try:
                CREDENTIALS_FILE.unlink()
            except Exception as e:
                logger.error(f"Failed to delete credentials file: {e}")
                success = False

        return success
