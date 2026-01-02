"""API Key Management for Traigent SDK.

This module provides secure API key storage with best practice warnings.
"""

# Traceability: CONC-Security FUNC-SECURITY REQ-SEC-010 SYNC-CloudHybrid CONC-Layer-Core

from __future__ import annotations

import os
import threading
import warnings


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
        # Priority: Environment > Explicitly set > Config file
        env_names = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "cohere": "COHERE_API_KEY",
            "huggingface": "HF_API_KEY",
        }

        # Check environment first (most secure)
        if provider in env_names:
            env_key = os.getenv(env_names[provider])
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
