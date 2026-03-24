"""Compatibility re-export for backend session creation types."""

from traigent.core.session_types import (
    SessionCreationFailureReason,
    SessionCreationResult,
)

__all__ = ["SessionCreationFailureReason", "SessionCreationResult"]
