"""Configuration for the Traigent project management client."""

from __future__ import annotations

from dataclasses import dataclass, field

from traigent.cloud.url_security import validate_cloud_base_url
from traigent.config.backend_config import BackendConfig
from traigent.config.tenant import TENANT_ENV_VAR, TENANT_HEADER_NAME, read_optional_env

MAX_TIMEOUT_SECONDS = 600.0


@dataclass
class ProjectManagementConfig:
    backend_origin: str = field(default_factory=BackendConfig.get_backend_url)
    api_key: str | None = field(default_factory=BackendConfig.get_api_key)
    tenant_id: str | None = field(
        default_factory=lambda: read_optional_env(TENANT_ENV_VAR)
    )
    api_path: str = "/api/v1beta/projects"
    request_timeout: float = 10.0
    extra_headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        backend_origin = self.backend_origin.rstrip("/")
        self.backend_origin = validate_cloud_base_url(
            backend_origin, purpose="project management request"
        )
        self.tenant_id = (
            self.tenant_id.strip() or None if self.tenant_id is not None else None
        )
        self.api_path = "/" + self.api_path.strip("/")
        if self.request_timeout <= 0 or self.request_timeout > MAX_TIMEOUT_SECONDS:
            raise ValueError(
                f"request_timeout must be greater than 0 and less than or equal to {MAX_TIMEOUT_SECONDS}"
            )

    def build_headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "traigent-projects/0.1",
            **self.extra_headers,
        }
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        if self.tenant_id:
            headers[TENANT_HEADER_NAME] = self.tenant_id
        return headers
