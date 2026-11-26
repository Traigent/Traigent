"""Shared project-path constants for SDK clients."""

from __future__ import annotations

from collections.abc import Callable
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


def apply_project_header(
    headers: dict[str, str], project_id: str | None = None
) -> None:
    """Add ``X-Project-Id`` in place when a project is configured.

    ``project_id`` (an explicit client option) wins over ``TRAIGENT_PROJECT_ID``.
    A header the caller already set (any case) is never overridden, and nothing
    is added when no project is configured. Call this only for requests to the
    Traigent backend, never for provider or other third-party clients.
    """
    if any(name.lower() == PROJECT_HEADER.lower() for name in headers):
        return
    explicit = project_id.strip() if isinstance(project_id, str) else ""
    resolved = explicit or read_optional_project_env()
    if resolved:
        headers[PROJECT_HEADER] = resolved


PROJECT_CONTEXT_REQUIRED_CODES = frozenset(
    {"PROJECT_CONTEXT_REQUIRED", "PROJECT_CONTEXT_REQUIRED_FOR_WRITE"}
)
PROJECT_CONTEXT_REQUIRED_MESSAGE = (
    "this call needs a project: set TRAIGENT_PROJECT_ID or use a project-bound key"
)


def project_header_for(
    explicit: object, blank_error: Callable[[], Exception]
) -> dict[str, str]:
    """The ONE place that decides the ``X-Project-Id`` of a flat best-config/deploy call.

    ``explicit`` is the client's own project (argument or attribute).  ``None`` means
    "not given" and reads ``TRAIGENT_PROJECT_ID``.  An explicitly passed blank (or a
    non-string, or one with control characters) raises ``blank_error()`` before any request and never falls back.
    With no project configured the result is empty: no header is sent and the backend
    decides (a project-bound key succeeds).  A configured project is sent exactly as
    configured (surrounding whitespace trimmed), never empty, never another id.
    """
    if explicit is None:
        resolved = read_optional_project_env() or ""
        if not resolved:
            return {}
        raw = resolved
    elif isinstance(explicit, str):
        raw = explicit
    else:
        raise blank_error()
    # No shape check: the backend accepts more shapes than the reactive paths do.
    # Only an empty value or a header-unsafe one (control characters, checked on the
    # raw value BEFORE trimming, so a trailing CR/LF/TAB is refused) is rejected.
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in raw):
        raise blank_error()
    resolved = raw.strip()
    if not resolved:
        raise blank_error()
    return {PROJECT_HEADER: resolved}


def is_project_context_required(status: int, body: object) -> bool:
    """True for a 422 whose top-level ``error_code`` is a project-context code.

    Only the status and the code are read, never a message or error text.
    """
    if status != 422 or not isinstance(body, dict):
        return False
    code = body.get("error_code")
    return isinstance(code, str) and code in PROJECT_CONTEXT_REQUIRED_CODES
