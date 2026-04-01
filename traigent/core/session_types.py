"""Structured types for backend session creation outcomes.

Leaf module with no cloud-package imports so core orchestration code can
depend on it even when the optional cloud package is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SessionCreationFailureReason(Enum):
    """Why backend session creation failed."""

    AUTH = "auth"  # 401/403 — non-retryable
    NO_API_KEY = "no_api_key"  # known before any HTTP call  # pragma: allowlist secret
    SESSION_FAILED = "session_failed"  # transient: 5xx, connection, timeout


@dataclass
class SessionCreationResult:
    """Structured outcome of a session creation attempt."""

    session_id: str
    backend_connected: bool
    failure_reason: SessionCreationFailureReason | None = None
    failure_detail: str | None = None

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
    ) -> SessionCreationResult:
        """Factory for a local-fallback session after backend failure."""
        return cls(
            session_id=session_id,
            backend_connected=False,
            failure_reason=reason,
            failure_detail=detail,
        )
