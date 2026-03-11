"""Configuration for the Traigent prompt management client."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from traigent.config.backend_config import BackendConfig
from traigent.config.tenant import TENANT_ENV_VAR, TENANT_HEADER_NAME

MAX_TIMEOUT_SECONDS = 600.0


@dataclass
class PromptManagementConfig:
    """Configuration for SDK-side prompt management requests."""

    backend_origin: str = field(default_factory=BackendConfig.get_backend_url)
    api_key: str | None = field(default_factory=BackendConfig.get_api_key)
    tenant_id: str | None = field(default_factory=lambda: os.getenv(TENANT_ENV_VAR))
    api_path: str = "/api/v1beta/prompts"
    request_timeout: float = 10.0
    extra_headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.backend_origin = self.backend_origin.rstrip("/")
        self.api_path = "/" + self.api_path.strip("/")
        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be greater than 0")
        if self.request_timeout > MAX_TIMEOUT_SECONDS:
            raise ValueError(
                f"request_timeout must be less than or equal to {MAX_TIMEOUT_SECONDS}"
            )

    def build_headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "traigent-prompts/0.1",
            **self.extra_headers,
        }
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        if self.tenant_id:
            headers[TENANT_HEADER_NAME] = self.tenant_id
        return headers
