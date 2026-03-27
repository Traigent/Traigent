"""Credential resolution module for Traigent authentication.

This module handles credential loading, caching, and resolution from multiple sources.
Extracted from AuthManager to follow Single Responsibility Principle.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Security CONC-Quality-Maintainability FUNC-SEC-CRED-RESOLVE REQ-SEC-010 REQ-CLOUD-009 SYNC-CloudHybrid

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from traigent.cloud.auth import AuthCredentials, AuthMode, UnifiedAuthConfig

logger = logging.getLogger(__name__)


class CredentialResolver:
    """Resolves credentials from multiple sources with caching.

    This class handles:
    - Credential resolution from provided, cached, or environment sources
    - Secure credential caching with encryption
    - Environment variable and credential manager integration
    """

    def __init__(self, config: UnifiedAuthConfig) -> None:
        """Initialize credential resolver.

        Args:
            config: Unified authentication configuration
        """
        self.config = config

        # Callbacks for credential operations (set by AuthManager)
        self._get_provided_credentials_fn: Any = None
        self._set_api_key_token_fn: Any = None
        self._get_api_key_token_fn: Any = None
        self._increment_cache_hits_fn: Any = None

    def set_callbacks(
        self,
        *,
        get_provided_credentials: Any = None,
        set_api_key_token: Any = None,
        get_api_key_token: Any = None,
        increment_cache_hits: Any = None,
    ) -> None:
        """Set callback functions for credential operations.

        Args:
            get_provided_credentials: Callback to get provided credentials
            set_api_key_token: Callback to set API key token
            get_api_key_token: Callback to get API key token
            increment_cache_hits: Callback to increment cache hits stat
        """
        if get_provided_credentials is not None:
            self._get_provided_credentials_fn = get_provided_credentials
        if set_api_key_token is not None:
            self._set_api_key_token_fn = set_api_key_token
        if get_api_key_token is not None:
            self._get_api_key_token_fn = get_api_key_token
        if increment_cache_hits is not None:
            self._increment_cache_hits_fn = increment_cache_hits

    async def resolve(
        self,
        credentials: AuthCredentials | dict[str, Any] | None,
        mode: AuthMode | None,
    ) -> AuthCredentials | dict[str, Any] | None:
        """Return explicit credentials or load them from configuration.

        Args:
            credentials: Explicitly provided credentials, or None
            mode: Authentication mode to use for loading

        Returns:
            Resolved credentials or None if not available
        """
        if credentials is not None:
            return credentials
        return await self.load_credentials(mode)

    async def load_credentials(
        self, mode: AuthMode | None = None
    ) -> AuthCredentials | None:
        """Load credentials from cache or environment.

        Priority order:
        1. Explicitly provided credentials
        2. Cached credentials (if caching enabled)
        3. Environment variables / credential manager

        Args:
            mode: Authentication mode to load credentials for

        Returns:
            Loaded credentials or None if not available
        """

        target_mode = mode or self.config.default_mode

        # Check for provided credentials first
        provided: AuthCredentials | None = (
            self._get_provided_credentials_fn()
            if self._get_provided_credentials_fn
            else None
        )

        if provided:
            # Always prefer explicitly provided credentials
            api_key_token = (
                self._get_api_key_token_fn() if self._get_api_key_token_fn else None
            )
            if provided.api_key and api_key_token is None:
                expires_meta = (
                    provided.metadata.get("expires_at") if provided.metadata else None
                )
                expires_at = None
                if isinstance(expires_meta, str):
                    try:
                        expires_at = datetime.fromisoformat(expires_meta)
                    except ValueError:
                        expires_at = None
                if self._set_api_key_token_fn:
                    self._set_api_key_token_fn(
                        provided.api_key,
                        source=(
                            provided.metadata.get("source")
                            if provided.metadata
                            else None
                        ),
                        expires_at=expires_at,
                    )
            return provided

        # Try to load from cache first
        if self.config.cache_credentials and self.config.credentials_file:
            cached = await self.load_cached()
            if cached and cached.mode == target_mode:
                if self._increment_cache_hits_fn:
                    self._increment_cache_hits_fn()
                return cached

        # Try to load from environment
        return await self.load_from_env(target_mode)

    async def load_cached(self) -> AuthCredentials | None:
        """Load credentials from cache file with security checks.

        Returns:
            Cached credentials or None if not available/invalid
        """
        from traigent.cloud.auth import AuthCredentials

        try:
            if not self.config.credentials_file:
                return None

            from traigent.security.crypto_utils import SecureFileManager

            # Use secure file reading with permission checks
            creds_path = Path(self.config.credentials_file).expanduser().resolve()
            data = SecureFileManager.read_secure_file(
                str(creds_path), base_dir=creds_path.parent
            )

            # Decrypt credentials
            decrypted_data = self.decrypt(data)

            return AuthCredentials(**decrypted_data)

        except FileNotFoundError:
            # File not found is expected for first run
            return None
        except Exception as e:
            logger.warning(f"Failed to load cached credentials: {e}")
            return None

    async def load_from_env(self, mode: AuthMode) -> AuthCredentials | None:
        """Load credentials from environment variables or credential manager.

        Args:
            mode: Authentication mode to load credentials for

        Returns:
            Credentials loaded from environment or None
        """
        from traigent.cloud.auth import AuthCredentials, AuthMode

        try:
            from traigent.cloud.credential_manager import CredentialManager

            manager_creds = CredentialManager.get_credentials()

            if mode == AuthMode.API_KEY:
                api_key = None
                metadata: dict[str, Any] = {}
                backend_url = None
                expires_at: datetime | None = None
                if manager_creds and manager_creds.get("api_key"):
                    api_key = manager_creds["api_key"]
                    metadata["source"] = manager_creds.get("source", "cli")
                    backend_url = manager_creds.get("backend_url")
                    expires_meta = manager_creds.get("expires_at")
                    if isinstance(expires_meta, str):
                        try:
                            expires_at = datetime.fromisoformat(expires_meta)
                        except ValueError:
                            expires_at = None
                else:
                    api_key = CredentialManager.get_api_key()
                    if not api_key:
                        api_key = os.getenv("TRAIGENT_API_KEY")
                        if api_key:
                            metadata["source"] = "environment"

                if api_key:
                    if self._set_api_key_token_fn:
                        self._set_api_key_token_fn(
                            api_key,
                            source=metadata.get("source"),
                            expires_at=expires_at,
                        )
                    creds = AuthCredentials(
                        mode=mode,
                        api_key=api_key,
                        backend_url=backend_url,
                        metadata=metadata,
                        expires_at=(
                            expires_at.timestamp()
                            if isinstance(expires_at, datetime)
                            else None
                        ),
                    )
                    return creds

            elif mode == AuthMode.JWT_TOKEN:
                jwt_token = None
                refresh_token = None
                metadata = {}  # Reset for this branch
                backend_url = None
                if manager_creds and manager_creds.get("jwt_token"):
                    jwt_token = manager_creds.get("jwt_token")
                    refresh_token = manager_creds.get("refresh_token")
                    metadata["source"] = manager_creds.get("source", "cli")
                    backend_url = manager_creds.get("backend_url")
                else:
                    jwt_token = os.getenv("TRAIGENT_JWT_TOKEN")

                if jwt_token:
                    return AuthCredentials(
                        mode=mode,
                        jwt_token=jwt_token,
                        refresh_token=refresh_token,
                        backend_url=backend_url,
                        metadata=metadata,
                    )

            elif mode == AuthMode.OAUTH2:
                client_id = os.getenv("TRAIGENT_CLIENT_ID")
                client_secret = os.getenv("TRAIGENT_CLIENT_SECRET")
                if client_id and client_secret:
                    return AuthCredentials(
                        mode=mode, client_id=client_id, client_secret=client_secret
                    )

            elif mode == AuthMode.SERVICE_TO_SERVICE:
                service_key = os.getenv("TRAIGENT_SERVICE_KEY")
                if service_key:
                    return AuthCredentials(mode=mode, service_key=service_key)

            elif mode == AuthMode.DEVELOPMENT:
                dev_user = os.getenv("TRAIGENT_DEV_USER", "developer")
                return AuthCredentials(mode=mode, metadata={"dev_user": dev_user})

            return None

        except Exception as e:
            logger.warning(f"Failed to load environment credentials: {e}")
            return None

    async def cache(self, credentials: AuthCredentials) -> None:
        """Cache credentials to file with secure permissions.

        Args:
            credentials: Credentials to cache
        """
        try:
            if not self.config.credentials_file:
                return

            from traigent.security.crypto_utils import SecureFileManager

            # Encrypt credentials
            encrypted_data = self.encrypt(credentials)

            # Write with secure permissions from the start
            creds_path = Path(self.config.credentials_file).expanduser().resolve()
            SecureFileManager.write_secure_file(
                str(creds_path),
                encrypted_data,
                base_dir=creds_path.parent,
            )

        except Exception as e:
            logger.warning(f"Failed to cache credentials: {e}")

    def encrypt(self, credentials: AuthCredentials) -> dict[str, Any]:
        """Secure credential encryption using AES-256.

        Args:
            credentials: Credentials to encrypt

        Returns:
            Encrypted credential data
        """
        from traigent.security.crypto_utils import get_credential_storage

        data = {
            "mode": credentials.mode.value,
            "api_key": credentials.api_key,
            "jwt_token": credentials.jwt_token,
            "refresh_token": credentials.refresh_token,
            "client_id": credentials.client_id,
            "client_secret": credentials.client_secret,
            "service_key": credentials.service_key,
            "expires_at": credentials.expires_at,
            "scopes": credentials.scopes,
            "metadata": credentials.metadata,
        }

        # Use secure AES encryption
        crypto = get_credential_storage()
        return crypto.encrypt_credentials(data)

    def decrypt(self, encrypted_data: dict[str, Any]) -> dict[str, Any]:
        """Secure credential decryption using AES-256.

        Args:
            encrypted_data: Encrypted credential data

        Returns:
            Decrypted credential data
        """
        from traigent.security.crypto_utils import get_credential_storage

        # Use secure AES decryption
        crypto = get_credential_storage()
        return crypto.decrypt_credentials(encrypted_data)
