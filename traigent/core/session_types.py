"""Structured types for backend session creation outcomes.

Leaf module with no cloud-package imports so core orchestration code can
depend on it even when the optional cloud package is unavailable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any


class SessionCreationFailureReason(Enum):
    """Why backend session creation failed."""

    AUTH = "auth"  # 401/403 — non-retryable
    NO_API_KEY = "no_api_key"  # known before any HTTP call  # pragma: allowlist secret
    SESSION_FAILED = "session_failed"  # transient: 5xx, connection, timeout


class SessionCreationFailureClassification(Enum):
    """User-facing classification for a failed backend session creation."""

    MISSING_PERMISSION = "missing-permission"
    INVALID_OR_REVOKED_KEY = "invalid-or-revoked-key"
    KEY_NOT_FOUND = "key-not-found"
    BACKEND_UNREACHABLE = "backend-unreachable"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SessionCreationFailureDetail:
    """Structured backend response details for failed session creation."""

    status_code: int | None = None
    error_code: str | None = None
    message: str | None = None
    missing_permissions: tuple[str, ...] = ()
    raw_body: str | None = None

    @classmethod
    def from_http_response(
        cls, status_code: int, raw_body: str | None
    ) -> SessionCreationFailureDetail:
        """Parse the backend's structured session-creation error response."""

        payload: dict[str, Any] = {}
        if raw_body:
            try:
                parsed = json.loads(raw_body)
            except (TypeError, ValueError):
                parsed = None
            if isinstance(parsed, dict):
                payload = parsed

        details = payload.get("details")
        if not isinstance(details, dict):
            details = {}

        raw_missing = details.get("missing_permissions")
        if isinstance(raw_missing, (list, tuple, set)):
            missing_permissions = tuple(str(item) for item in raw_missing)
        else:
            missing_permissions = ()

        error_code = payload.get("error_code") or payload.get("code")
        message = payload.get("message") or payload.get("error")

        return cls(
            status_code=status_code,
            error_code=str(error_code) if error_code else None,
            message=str(message) if message else None,
            missing_permissions=missing_permissions,
            raw_body=raw_body,
        )

    def one_line_summary(self) -> str:
        """Return a compact summary suitable for exception/log details."""

        parts: list[str] = []
        if self.status_code is not None:
            parts.append(f"HTTP {self.status_code}")
        if self.error_code:
            parts.append(self.error_code)
        if self.message:
            parts.append(self.message)
        if self.missing_permissions:
            parts.append(
                "missing_permissions=" + ",".join(sorted(set(self.missing_permissions)))
            )
        if not parts and self.raw_body:
            parts.append(self.raw_body[:200])
        return " | ".join(parts) or "unknown"


class SessionCreationHTTPError(Exception):
    """HTTP error raised while creating a backend tracking session."""

    def __init__(self, detail: SessionCreationFailureDetail) -> None:
        super().__init__(detail.one_line_summary())
        self.detail = detail


@dataclass
class SessionCreationResult:
    """Structured outcome of a session creation attempt."""

    session_id: str
    backend_connected: bool
    failure_reason: SessionCreationFailureReason | None = None
    failure_detail: str | None = None
    failure_response: SessionCreationFailureDetail | None = None

    def __str__(self) -> str:
        """Return the session ID string for backwards compatibility.

        v0.10.x returned a plain str from create_session(); code that does
        ``session_id = create_session()`` and passes the result as a string
        argument will continue to work without changes.
        """
        return self.session_id

    def __post_init__(self) -> None:
        if not self.backend_connected and self.failure_reason is None:
            raise ValueError("failure_reason is required when backend_connected=False")
        if self.backend_connected and self.failure_reason is not None:
            raise ValueError("failure_reason must be None when backend_connected=True")

    @classmethod
    def connected(cls, session_id: str) -> SessionCreationResult:
        """Factory for a successful backend session."""
        return cls(session_id=session_id, backend_connected=True)

    @classmethod
    def fallback(
        cls,
        session_id: str,
        reason: SessionCreationFailureReason,
        detail: str | None = None,
        failure_response: SessionCreationFailureDetail | None = None,
    ) -> SessionCreationResult:
        """Factory for a local-fallback session after backend failure."""
        return cls(
            session_id=session_id,
            backend_connected=False,
            failure_reason=reason,
            failure_detail=detail,
            failure_response=failure_response,
        )
