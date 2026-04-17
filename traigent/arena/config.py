"""Configuration for the Agent Arena SDK client."""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlparse  # stdlib, not urllib.request

from traigent.config.backend_config import BackendConfig
from traigent.config.project import read_optional_project_env
from traigent.config.tenant import TENANT_ENV_VAR, TENANT_HEADER_NAME, read_optional_env

MAX_TIMEOUT_SECONDS = 600.0
PROJECT_HEADER_NAME = "X-Project-Id"


@dataclass
class ArenaConfig:
    """Configuration for brokered Agent Arena API calls."""

    backend_origin: str = field(default_factory=BackendConfig.get_backend_url)
    api_key: str | None = field(default_factory=BackendConfig.get_api_key)
    tenant_id: str | None = field(
        default_factory=lambda: read_optional_env(TENANT_ENV_VAR)
    )
    project_id: str | None = field(default_factory=read_optional_project_env)
    api_path: str = "/api/v1beta/arena"
    request_timeout: float = 15.0
    extra_headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.backend_origin = self.backend_origin.rstrip("/")
        parsed_origin = urlparse(self.backend_origin)
        if parsed_origin.scheme not in {"http", "https"} or not parsed_origin.netloc:
            raise ValueError("backend_origin must be an absolute http or https URL")
        self.api_path = "/" + self.api_path.strip("/")
        if self.request_timeout <= 0 or self.request_timeout > MAX_TIMEOUT_SECONDS:
            raise ValueError(
                f"request_timeout must be greater than 0 and less than or equal to {MAX_TIMEOUT_SECONDS}"
            )

    def build_headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "traigent-arena/0.1",
            **self.extra_headers,
        }
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        if self.tenant_id:
            headers[TENANT_HEADER_NAME] = self.tenant_id
        if self.project_id:
            headers[PROJECT_HEADER_NAME] = self.project_id
        return headers
