"""Structured types for session creation results.

Leaf module with no cloud-package imports — safe to import from anywhere
without circular import risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SessionCreationFailureReason(Enum):
    """Why backend session creation failed."""

    AUTH = "auth"  # 401/403 — non-retryable
    NO_API_KEY = "no_api_key"  # known before any HTTP call
    SESSION_FAILED = "session_failed"  # transient: 5xx, connection, timeout


@dataclass
class SessionCreationResult:
    """Structured outcome of a session creation attempt.

    Invariants enforced by ``__post_init__``:
    * ``backend_connected=False`` requires a non-None ``failure_reason``.
    * ``backend_connected=True`` requires ``failure_reason is None``.
    """

    session_id: str
    backend_connected: bool
    failure_reason: SessionCreationFailureReason | None = None
    failure_detail: str | None = None

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
