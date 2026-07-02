"""API Key Management for Traigent SDK.

This module provides secure API key storage with best practice warnings.
"""

# Traceability: CONC-Security FUNC-SECURITY REQ-SEC-010 SYNC-CloudHybrid CONC-Layer-Core

from __future__ import annotations

import threading
import warnings

from traigent.config.provider_support import resolve_api_key_from_env


class APIKeyManager:
    """Manages API keys with security best practices."""

    def __init__(self) -> None:
        self._keys: dict[str, str] = {}
        self._warned = False
        self._lock = threading.RLock()

    def set_api_key(self, provider: str, key: str, source: str = "code") -> None:
        """Set an API key with security warnings.

        Args:
            provider: The provider name (e.g., 'openai', 'anthropic')
            key: The API key
            source: Where the key came from ('code', 'env', 'config')
        """
        should_warn = False
        with self._lock:
            if not self._warned and source == "code":
                self._warned = True
                should_warn = True

            # Store key but mark it as sensitive
            self._keys[provider] = key

        if should_warn:
            warnings.warn(
                "API keys detected in code. For production use, consider using:\n"
                "1. Environment variables: OPENAI_API_KEY, ANTHROPIC_API_KEY\n"
                "2. Configuration file with proper permissions\n"
                "3. Key management service\n"
                "See: https://docs.traigent.com/security/api-keys",
                UserWarning,
                stacklevel=2,
            )

    def get_api_key(self, provider: str) -> str | None:
        """Get API key, checking environment variables first.

        Args:
            provider: The provider name

        Returns:
            The API key if found, None otherwise
        """
        # Priority: Environment > Explicitly set > Config file.
        # The env-var chain per provider is derived from the canonical
        # provider-support table (traigent/config/provider_support.py) so the
        # key map can no longer drift from the validator's env reading. This is
        # what gives google (GOOGLE_API_KEY/GEMINI_API_KEY) and mistral
        # (MISTRAL_API_KEY) first-class key-manager support (#1568) and keeps
        # the HuggingFace alias chain HF_TOKEN -> HUGGING_FACE_HUB_TOKEN ->
        # HF_API_KEY intact.
        env_key = resolve_api_key_from_env(provider)
        if env_key:
            return env_key

        # Return explicitly set key
        with self._lock:
            return self._keys.get(provider)

    def __repr__(self) -> str:
        """Prevent accidental key exposure in logs."""
        return f"<APIKeyManager with {len(self._keys)} keys>"

    def __str__(self) -> str:
        """Prevent accidental key exposure in string conversion."""
        return self.__repr__()


# Global instance
_API_KEY_MANAGER = APIKeyManager()
