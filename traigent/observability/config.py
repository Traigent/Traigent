"""Configuration for the Traigent observability client."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from traigent.cloud.url_security import validate_cloud_base_url
from traigent.config.backend_config import BackendConfig
from traigent.config.project import read_optional_project_env, scope_api_path
from traigent.config.tenant import TENANT_ENV_VAR, TENANT_HEADER_NAME, read_optional_env
from traigent.utils.env_config import (
    is_backend_offline,
    is_truthy,
    resolve_environment_label,
)

MAX_BATCH_SIZE = 10_000
MAX_QUEUE_SIZE = 1_000_000
MAX_BATCH_BYTES = 10 * 1024 * 1024
MAX_BUFFER_AGE_SECONDS = 3600.0
MAX_TIMEOUT_SECONDS = 600.0
OBSERVABILITY_CONTENT_MODES = frozenset({"metadata", "redacted", "record"})

_EXECUTION_CONTEXT_ENV_VARS = {
    "agent_id": "TRAIGENT_AGENT_ID",
    "agent_version": "TRAIGENT_AGENT_VERSION",
    "release_id": "TRAIGENT_RELEASE_ID",
    "deployment_id": "TRAIGENT_DEPLOYMENT_ID",
    "code_revision": "TRAIGENT_CODE_REVISION",
    "configuration_id": "TRAIGENT_CONFIGURATION_ID",
    "configuration_version": "TRAIGENT_CONFIGURATION_VERSION",
    "prompt_id": "TRAIGENT_PROMPT_ID",
    "prompt_version": "TRAIGENT_PROMPT_VERSION",
    "toolset_id": "TRAIGENT_TOOLSET_ID",
    "toolset_version": "TRAIGENT_TOOLSET_VERSION",
    "evaluator_id": "TRAIGENT_EVALUATOR_ID",
    "evaluator_version": "TRAIGENT_EVALUATOR_VERSION",
    "dataset_id": "TRAIGENT_DATASET_ID",
    "dataset_version": "TRAIGENT_DATASET_VERSION",
    "experiment_run_id": "TRAIGENT_EXPERIMENT_RUN_ID",
    "configuration_run_id": "TRAIGENT_CONFIGURATION_RUN_ID",
    "optimization_run_id": "TRAIGENT_OPTIMIZATION_RUN_ID",
    "intervention_id": "TRAIGENT_INTERVENTION_ID",
}


def nonblank_credential(value: str | None) -> bool:
    """A credential value counts only when it carries non-whitespace content.

    Used by both header construction and the missing-credential guard so a
    whitespace-only api_key/jwt can neither authenticate a request nor
    overwrite working auth supplied via ``extra_headers``.
    """
    return bool(value and value.strip())


def _read_observability_offline_mode() -> bool:
    return is_backend_offline() or is_truthy(os.getenv("TRAIGENT_DISABLE_TELEMETRY"))


def _read_observability_content_mode() -> str:
    raw_mode = os.getenv("TRAIGENT_OBSERVABILITY_CONTENT")
    if raw_mode is None:
        return (
            "record"
            if is_truthy(os.getenv("TRAIGENT_OBSERVABILITY_CAPTURE_CONTENT"))
            else "metadata"
        )

    mode = raw_mode.strip().lower()
    if mode not in OBSERVABILITY_CONTENT_MODES:
        allowed = ", ".join(sorted(OBSERVABILITY_CONTENT_MODES))
        raise ValueError(f"TRAIGENT_OBSERVABILITY_CONTENT must be one of: {allowed}")
    return mode


def _read_default_execution_context() -> dict[str, str | None]:
    """Read only explicit Traigent lineage identifiers from the environment."""
    context: dict[str, str | None] = {}
    for field_name, environment_name in _EXECUTION_CONTEXT_ENV_VARS.items():
        raw_value = os.getenv(environment_name)
        if raw_value is not None and raw_value.strip():
            context[field_name] = raw_value.strip()
    if "release_id" not in context:
        release = os.getenv("TRAIGENT_RELEASE")
        if release is not None and release.strip():
            context["release_id"] = release.strip()
    return context


@dataclass
class ObservabilityConfig:
    """Configuration for SDK-side observability delivery.

    ``health_callback`` receives batch-level local-drop and warning snapshots
    after internal transport locks are released. It may run on background
    threads, and exceptions raised by the callback are swallowed.
    """

    backend_origin: str = field(default_factory=BackendConfig.get_backend_url)
    api_key: str | None = field(default_factory=BackendConfig.get_api_key)
    jwt_token: str | None = field(
        default_factory=lambda: os.getenv("TRAIGENT_JWT_TOKEN")
    )
    tenant_id: str | None = field(
        default_factory=lambda: read_optional_env(TENANT_ENV_VAR)
    )
    project_id: str | None = field(default_factory=read_optional_project_env)
    api_path: str = "/api/v1beta/observability"
    batch_size: int = 100
    max_buffer_age: float = 5.0
    max_queue_size: int = 10_000
    max_batch_bytes: int = 4 * 1024 * 1024
    flush_timeout: float = 30.0
    request_timeout: float = 10.0
    enable_atexit_flush: bool = True
    offline_mode: bool = field(default_factory=_read_observability_offline_mode)
    content_mode: str = field(default_factory=_read_observability_content_mode)
    health_callback: Callable[[str, dict[str, Any]], None] | None = None
    default_environment: str | None = field(
        default_factory=lambda: resolve_environment_label(default=None)
    )
    default_release: str | None = field(
        default_factory=lambda: os.getenv("TRAIGENT_RELEASE")
    )
    default_execution_context: dict[str, str | None] = field(
        default_factory=_read_default_execution_context
    )
    extra_headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.backend_origin = validate_cloud_base_url(
            self.backend_origin,
            purpose="observability backend",
        )
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
        if self.max_batch_bytes <= 0:
            raise ValueError("max_batch_bytes must be greater than 0")
        if self.max_batch_bytes > MAX_BATCH_BYTES:
            raise ValueError(
                f"max_batch_bytes must be less than or equal to {MAX_BATCH_BYTES}"
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
        self.content_mode = self.content_mode.strip().lower()
        if self.content_mode not in OBSERVABILITY_CONTENT_MODES:
            allowed = ", ".join(sorted(OBSERVABILITY_CONTENT_MODES))
            raise ValueError(f"content_mode must be one of: {allowed}")

    @property
    def ingest_url(self) -> str:
        return f"{self.backend_origin}{self.api_path}/ingest"

    def build_headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "traigent-observability/0.1",
            **self.extra_headers,
        }
        # Blank explicit credentials are treated as absent so they can never
        # overwrite working auth supplied through extra_headers.
        if nonblank_credential(self.api_key):
            headers["X-API-Key"] = self.api_key
        if nonblank_credential(self.jwt_token):
            headers["Authorization"] = f"Bearer {self.jwt_token}"
        if self.tenant_id:
            headers[TENANT_HEADER_NAME] = self.tenant_id
        return headers
