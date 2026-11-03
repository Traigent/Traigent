"""Closed, content-free errors for the execution-snapshot store."""

from __future__ import annotations

from enum import StrEnum

__all__ = ["SnapshotErrorCode", "SnapshotStoreError"]


class SnapshotErrorCode(StrEnum):
    """Why a store operation failed. Closed list; never carries provider text."""

    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    TOO_LARGE = "too_large"
    AUTH = "auth"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    CORRUPT = "corrupt"
    UNSUPPORTED_BACKEND = "unsupported_backend"
    UNSAFE_DESTINATION = "unsafe_destination"


class SnapshotStoreError(Exception):
    """A snapshot store operation failed.

    Holds only a code, the snapshot part path (our own layout, never customer
    text) and a backend name. Raise sites use ``raise ... from None`` so no
    provider exception, response, URL or credential is reachable from it.
    """

    def __init__(
        self,
        code: SnapshotErrorCode,
        part: str | None = None,
        backend: str | None = None,
    ) -> None:
        self.code = code
        self.part = part
        self.backend = backend
        super().__init__(self._describe())

    def _describe(self) -> str:
        text = f"snapshot store error: {self.code.value}"
        if self.backend:
            text += f" (backend={self.backend})"
        if self.part:
            text += f" (part={self.part})"
        return text

    def __reduce__(
        self,
    ) -> tuple[type, tuple[SnapshotErrorCode, str | None, str | None]]:
        return (type(self), (self.code, self.part, self.backend))
