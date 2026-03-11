"""Configuration for the Traigent observability client."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from traigent.config.backend_config import BackendConfig
from traigent.config.project import read_optional_project_env, scope_api_path
from traigent.config.tenant import TENANT_ENV_VAR, TENANT_HEADER_NAME, read_optional_env

MAX_BATCH_SIZE = 10_000
MAX_QUEUE_SIZE = 1_000_000
MAX_BUFFER_AGE_SECONDS = 3600.0
MAX_TIMEOUT_SECONDS = 600.0


@dataclass
class ObservabilityConfig:
    """Configuration for SDK-side observability delivery."""

    backend_origin: str = field(default_factory=BackendConfig.get_backend_url)
    api_key: str | None = field(default_factory=BackendConfig.get_api_key)
    tenant_id: str | None = field(
        default_factory=lambda: read_optional_env(TENANT_ENV_VAR)
    )
    project_id: str | None = field(default_factory=read_optional_project_env)
    api_path: str = "/api/v1beta/observability"
    batch_size: int = 100
    max_buffer_age: float = 5.0
    max_queue_size: int = 10_000
    flush_timeout: float = 30.0
    request_timeout: float = 10.0
    enable_atexit_flush: bool = True
    default_environment: str | None = field(
        default_factory=lambda: (
            os.getenv("TRAIGENT_ENVIRONMENT") or os.getenv("ENVIRONMENT")
        )
    )
    default_release: str | None = field(
        default_factory=lambda: os.getenv("TRAIGENT_RELEASE")
    )
    extra_headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.backend_origin = self.backend_origin.rstrip("/")
        self.tenant_id = (
            self.tenant_id.strip() or None if self.tenant_id is not None else None
        )
        self.project_id = (
            self.project_id.strip() or None if self.project_id is not None else None
        )
        self.api_path = scope_api_path(self.api_path, self.project_id)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        if self.batch_size > MAX_BATCH_SIZE:
            raise ValueError(
                f"batch_size must be less than or equal to {MAX_BATCH_SIZE}"
            )
        if self.max_buffer_age <= 0:
            raise ValueError("max_buffer_age must be greater than 0")
        if self.max_buffer_age > MAX_BUFFER_AGE_SECONDS:
            raise ValueError(
                f"max_buffer_age must be less than or equal to {MAX_BUFFER_AGE_SECONDS}"
            )
        if self.max_queue_size <= 0:
            raise ValueError("max_queue_size must be greater than 0")
        if self.max_queue_size > MAX_QUEUE_SIZE:
            raise ValueError(
                f"max_queue_size must be less than or equal to {MAX_QUEUE_SIZE}"
            )
        if self.flush_timeout <= 0:
            raise ValueError("flush_timeout must be greater than 0")
        if self.flush_timeout > MAX_TIMEOUT_SECONDS:
            raise ValueError(
                f"flush_timeout must be less than or equal to {MAX_TIMEOUT_SECONDS}"
            )
        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be greater than 0")
        if self.request_timeout > MAX_TIMEOUT_SECONDS:
            raise ValueError(
                f"request_timeout must be less than or equal to {MAX_TIMEOUT_SECONDS}"
            )

    @property
    def ingest_url(self) -> str:
        return f"{self.backend_origin}{self.api_path}/ingest"

    def build_headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "traigent-observability/0.1",
            **self.extra_headers,
        }
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        if self.tenant_id:
            headers[TENANT_HEADER_NAME] = self.tenant_id
        return headers
