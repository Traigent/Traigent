"""Shared backend helper classes for the cloud client.

This module contains lightweight configuration and manager classes that are
used by the high-level `backend_client`. Extracting them keeps the main client
implementation focused on orchestration logic while these helpers encapsulate
authentication, configuration synchronisation, and simple trial generation
behaviour.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability FUNC-CLOUD-HYBRID REQ-CLOUD-009 SYNC-CloudHybrid

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from traigent.cloud.auth import (
    AuthenticationError,
    AuthManager,
    AuthMode,
    UnifiedAuthConfig,
)
from traigent.cloud.billing import UsageTracker
from traigent.cloud.client import CloudServiceError
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BackendClientConfig:
    """Configuration container for backend integrations."""

    backend_base_url: str | None = None
    api_base_url: str | None = None
    use_mcp: bool = False
    mcp_server_path: str | None = None
    enable_session_sync: bool = True
    session_sync_interval: float = 5.0
    backend_explicitly_set: bool = field(init=False, repr=False, default=False)
    api_explicitly_set: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        """Populate missing configuration using global backend settings."""
        from traigent.config.backend_config import (
            BackendConfig,
            get_no_credentials_hint,
        )

        backend_env = os.environ.get("TRAIGENT_BACKEND_URL")
        api_env = os.environ.get("TRAIGENT_API_URL")

        # Track if backend_base_url was explicitly provided
        self.backend_explicitly_set = self.backend_base_url is not None
        self.api_explicitly_set = self.api_base_url is not None

        if self.backend_base_url is not None:
            normalized = BackendConfig.normalize_backend_origin(self.backend_base_url)
            self.backend_base_url = normalized or self.backend_base_url.rstrip("/")
        elif backend_env or api_env:
            self.backend_base_url = BackendConfig.get_backend_url().rstrip("/")
        else:
            self.backend_base_url = BackendConfig.get_backend_url().rstrip("/")

        # Warn when defaulting to cloud without any credentials.
        # Skip if the caller explicitly passed the URL (not our default).
        # Use a silent predicate so this path does not emit duplicate warnings.
        if (
            not self.backend_explicitly_set
            and self.backend_base_url
            and self.backend_base_url.rstrip("/")
            == BackendConfig.DEFAULT_PROD_URL.rstrip("/")
            and not BackendConfig.has_auth_credentials()
        ):
            logger.warning(
                "Defaulting to Traigent cloud (%s) but no credentials found. %s "
                "Set TRAIGENT_ENV=development for local mode.",
                self.backend_base_url,
                get_no_credentials_hint(),
            )

        if self.api_base_url is not None:
            if "://" not in self.api_base_url.strip():
                origin = BackendConfig.normalize_backend_origin(self.api_base_url)
                if origin:
                    self.api_base_url = BackendConfig.build_api_base(origin)
            else:
                parsed_origin, path = BackendConfig.split_api_url(self.api_base_url)
                if parsed_origin:
                    api_path = path or BackendConfig.get_default_api_path()
                    self.api_base_url = f"{parsed_origin}{api_path}"
        elif api_env and not self.backend_explicitly_set:
            self.api_base_url = BackendConfig.get_backend_api_url()
            # Only derive backend_base_url from API URL if not explicitly provided
            self.backend_base_url = BackendConfig.get_backend_url().rstrip("/")
        elif self.backend_base_url is not None:
            self.api_base_url = BackendConfig.build_api_base(self.backend_base_url)


class BackendAuthManager:
    """Handles authentication headers and basic rate limiting."""

    def __init__(
        self,
        api_key: str | None,
        rate_limit_calls: int = 100,
        rate_limit_period: float = 60.0,
    ) -> None:
        auth_cls: type[AuthManager]
        backend_client_module = sys.modules.get("traigent.cloud.backend_client")
        if backend_client_module and hasattr(backend_client_module, "AuthManager"):
            auth_cls = backend_client_module.AuthManager
        else:
            auth_cls = AuthManager

        jwt_token = os.getenv("TRAIGENT_JWT_TOKEN")
        if jwt_token:
            self.auth = auth_cls(
                config=UnifiedAuthConfig(default_mode=AuthMode.JWT_TOKEN)
            )
        else:
            self.auth = auth_cls(api_key=api_key)
        self.rate_limit_calls = rate_limit_calls
        self.rate_limit_period = rate_limit_period
        self._request_times: list[float] = []

    def has_api_key(self) -> bool:
        """Delegate to internal AuthManager to check if API key is configured."""
        return self.auth.has_api_key()

    async def augment_headers(
        self, headers: dict[str, str], target: str = "backend"
    ) -> dict[str, str]:
        """Merge authentication headers into the provided dictionary.

        B4 ROUND 4: ``AuthenticationError`` (and ``InvalidCredentialsError``)
        must propagate. Previously this method swallowed every auth-side
        exception and returned the caller's headers unchanged -- which
        meant a backend-rejected key produced an unauthenticated request
        instead of surfacing the auth failure. That is a parallel
        fail-open path to the one in ``backend_client._ensure_session``.
        """

        merged_headers = dict(headers)
        # Auth errors are intentionally NOT caught here: callers must see
        # the rejection rather than receive a silently-unauthenticated
        # header dict. Other unexpected errors are wrapped so callers see a
        # cloud-layer failure with diagnostic context instead of a raw helper
        # exception.
        try:
            auth_headers = await self.auth.get_headers(target=target)
        except AuthenticationError:
            raise
        except Exception as exc:
            logger.warning("Could not augment backend auth headers: %s", exc)
            raise CloudServiceError(
                f"Failed to augment backend auth headers: {exc}"
            ) from exc

        for key, value in auth_headers.items():
            if key not in merged_headers:
                merged_headers[key] = value

        return merged_headers

    async def check_rate_limit(self) -> None:
        """Enforce request rate limits using a simple sliding window."""

        current_time = time.time()
        # Remove requests outside of the current window.
        self._request_times = [
            timestamp
            for timestamp in self._request_times
            if current_time - timestamp < self.rate_limit_period
        ]

        if len(self._request_times) >= self.rate_limit_calls:
            sleep_time = self.rate_limit_period - (
                current_time - self._request_times[0]
            )
            if sleep_time > 0:
                logger.warning(
                    "Rate limit exceeded. Sleeping for %.2f seconds", sleep_time
                )
                await asyncio.sleep(sleep_time)

        self._request_times.append(current_time)

    def generate_request_nonce(self) -> str:
        """Create a unique nonce for request deduplication."""

        return f"{int(time.time() * 1000)}_{secrets.token_hex(8)}"


class BackendSessionManager:
    """Backend session manager.

    This component is intentionally fail-closed until it is wired to concrete
    backend endpoints. It must not synthesize local session payloads that look
    like successful backend operations.
    """

    def __init__(
        self, auth_manager: BackendAuthManager, backend_config: BackendClientConfig
    ) -> None:
        self.auth_manager = auth_manager
        self.backend_config = backend_config

    async def create_session(
        self,
        session_id: str,
        session_config: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a backend session.

        Raises:
            NotImplementedError: until a real backend endpoint is configured.
        """
        raise NotImplementedError(
            "BackendSessionManager.create_session is not wired to a backend "
            "endpoint. Use the canonical backend client session APIs instead."
        )


class BackendTrialManager:
    """Backend trial manager.

    Trial suggestions must come from a real optimizer/backend service. This
    manager fails closed rather than returning placeholder configurations.
    """

    def __init__(
        self, auth_manager: BackendAuthManager, backend_config: BackendClientConfig
    ) -> None:
        self.auth_manager = auth_manager
        self.backend_config = backend_config
        self.usage_tracker = UsageTracker()

    async def get_next_privacy_trial(
        self,
        session_id: str,
        trial_count: int = 1,
    ) -> list[dict[str, Any]]:
        """Return privacy trial suggestions from the backend."""
        raise NotImplementedError(
            "BackendTrialManager.get_next_privacy_trial is not wired to a "
            "backend optimizer endpoint."
        )

    def generate_trial_id(
        self, session_id: str, config: dict[str, Any], metadata: dict[str, Any] | None
    ) -> str:
        """Create a deterministic trial identifier based on configuration."""

        if metadata and "trial_id" in metadata:
            trial_id = metadata["trial_id"]
            if isinstance(trial_id, str):
                return trial_id

        config_hash = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()[:8]
        timestamp = int(time.time() * 1000)
        return f"{session_id}_trial_{timestamp}_{config_hash}"
