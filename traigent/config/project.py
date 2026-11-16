"""Shared project-path constants for SDK clients."""

from __future__ import annotations

from urllib.parse import quote

from traigent.config.tenant import read_optional_env

PROJECT_ENV_VAR = "TRAIGENT_PROJECT_ID"


def scope_api_path(api_path: str, project_id: str | None) -> str:
    """Rewrite canonical API roots to project-prefixed v1beta paths."""
    normalized = "/" + api_path.strip("/")
    normalized_project_id = project_id.strip() if project_id else ""
    if not normalized_project_id:
        return normalized

    project_segment = quote(normalized_project_id, safe="")
    if normalized.startswith("/api/v1beta/projects/"):
        return normalized
    if normalized == "/api/v1beta":
        return f"/api/v1beta/projects/{project_segment}"
    if normalized.startswith("/api/v1beta/"):
        suffix = normalized[len("/api/v1beta") :]
        return f"/api/v1beta/projects/{project_segment}{suffix}"
    if normalized == "/api/v1/analytics" or normalized.startswith("/api/v1/analytics/"):
        return normalized
    if normalized == "/api/v1/optimization-comparisons" or normalized.startswith(
        "/api/v1/optimization-comparisons/"
    ):
        return normalized
    if normalized == "/api/v1/measures":
        return f"/api/v1beta/projects/{project_segment}/measures"
    if normalized.startswith("/api/v1/"):
        suffix = normalized[len("/api/v1") :]
        return f"/api/v1beta/projects/{project_segment}{suffix}"
    return normalized


def read_optional_project_env() -> str | None:
    """Return the active project env value or ``None`` when blank."""
    return read_optional_env(PROJECT_ENV_VAR)


PROJECT_HEADER = "X-Project-Id"


def apply_project_header(headers: dict[str, str]) -> None:
    """Add ``X-Project-Id`` in place when a project is configured.

    A header the caller already set (any case) is never overridden, and nothing
    is added when no project is configured.
    """
    if any(name.lower() == PROJECT_HEADER.lower() for name in headers):
        return
    project_id = read_optional_project_env()
    if project_id:
        headers[PROJECT_HEADER] = project_id
