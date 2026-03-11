"""Shared project-path constants for SDK clients."""

from __future__ import annotations

from urllib.parse import quote

PROJECT_ENV_VAR = "TRAIGENT_PROJECT_ID"


def scope_api_path(api_path: str, project_id: str | None) -> str:
    """Rewrite canonical API roots to project-prefixed v1beta paths."""
    normalized = "/" + api_path.strip("/")
    if not project_id:
        return normalized

    project_segment = quote(project_id.strip(), safe="")
    if normalized.startswith("/api/v1beta/projects/"):
        return normalized
    if normalized == "/api/v1beta":
        return f"/api/v1beta/projects/{project_segment}"
    if normalized.startswith("/api/v1beta/"):
        suffix = normalized[len("/api/v1beta") :]
        return f"/api/v1beta/projects/{project_segment}{suffix}"
    if normalized == "/api/v1/measures":
        return f"/api/v1beta/projects/{project_segment}/measures"
    if normalized.startswith("/api/v1/"):
        suffix = normalized[len("/api/v1") :]
        return f"/api/v1beta/projects/{project_segment}{suffix}"
    return normalized
