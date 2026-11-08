"""Configuration for the Traigent prompt management client."""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import quote

from traigent.cloud.url_security import validate_cloud_base_url
from traigent.config.backend_config import BackendConfig
from traigent.config.project import read_optional_project_env
from traigent.config.tenant import TENANT_ENV_VAR, TENANT_HEADER_NAME, read_optional_env
from traigent.utils.exceptions import ConfigurationError

MAX_TIMEOUT_SECONDS = 600.0


@dataclass
class PromptManagementConfig:
    """Configuration for SDK-side prompt management requests.

    Prompts live inside a project, and the project id is part of every request
    path. ``project_id`` defaults to the ``TRAIGENT_PROJECT_ID`` environment
    variable. There is no fallback project: when neither is set, building a
    request raises :class:`~traigent.utils.exceptions.ConfigurationError`.
    """

    backend_origin: str = field(default_factory=BackendConfig.get_backend_url)
    api_key: str | None = field(default_factory=BackendConfig.get_api_key)
    tenant_id: str | None = field(
        default_factory=lambda: read_optional_env(TENANT_ENV_VAR)
    )
    project_id: str | None = field(default_factory=read_optional_project_env)
    request_timeout: float = 10.0
    extra_headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        backend_origin = self.backend_origin.rstrip("/")
        self.backend_origin = validate_cloud_base_url(
            backend_origin, purpose="prompt management request"
        )
        self.tenant_id = (
            self.tenant_id.strip() or None if self.tenant_id is not None else None
        )
        self.project_id = (
            self.project_id.strip() or None if self.project_id is not None else None
        )
        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be greater than 0")
        if self.request_timeout > MAX_TIMEOUT_SECONDS:
            raise ValueError(
                f"request_timeout must be less than or equal to {MAX_TIMEOUT_SECONDS}"
            )

    @property
    def api_path(self) -> str:
        """Project-scoped prompts path, for example ``/api/v1/projects/<id>/prompts``.

        Raises:
            ConfigurationError: If no project id is configured.
        """
        if not self.project_id:
            raise ConfigurationError(
                "Prompt management needs a project id: pass project_id= to "
                "PromptManagementConfig or set TRAIGENT_PROJECT_ID"
            )
        return f"/api/v1/projects/{quote(self.project_id, safe='')}/prompts"

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
