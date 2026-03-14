"""HTTP error types for Traigent hybrid wrapper servers.

Service implementations can raise these exceptions from execute/evaluate handlers
to return explicit HTTP status codes and structured error payloads.
"""

from __future__ import annotations

from typing import Any


class HybridAPIError(Exception):
    """Base exception for explicit Hybrid API HTTP responses."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        error_code: str,
        details: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.details = details
        self.headers = headers or {}


class BadRequestError(HybridAPIError):
    """Invalid request payload or unsupported argument values."""

    def __init__(
        self,
        message: str,
        *,
        error_code: str = "INVALID_REQUEST",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=400,
            error_code=error_code,
            details=details,
        )


class UnauthorizedError(HybridAPIError):
    """Authentication required or token invalid."""

    def __init__(
        self,
        message: str = "Authentication required",
        *,
        error_code: str = "UNAUTHORIZED",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=401,
            error_code=error_code,
            details=details,
        )


class RequestTimeoutError(HybridAPIError):
    """Request exceeded server-side timeout budget."""

    def __init__(
        self,
        message: str,
        *,
        error_code: str = "REQUEST_TIMEOUT",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=408,
            error_code=error_code,
            details=details,
        )


class RateLimitError(HybridAPIError):
    """Caller exceeded allowed request rate."""

    def __init__(
        self,
        message: str = "Too many requests",
        *,
        retry_after: int | float | None = None,
        error_code: str = "RATE_LIMITED",
        details: dict[str, Any] | None = None,
    ) -> None:
        headers: dict[str, str] = {}
        if retry_after is not None:
            headers["Retry-After"] = str(retry_after)
        super().__init__(
            message,
            status_code=429,
            error_code=error_code,
            details=details,
            headers=headers,
        )


class ServiceUnavailableError(HybridAPIError):
    """Dependency outage or temporary service unavailability."""

    def __init__(
        self,
        message: str = "Service unavailable",
        *,
        retry_after: int | float | None = None,
        error_code: str = "SERVICE_UNAVAILABLE",
        details: dict[str, Any] | None = None,
    ) -> None:
        headers: dict[str, str] = {}
        if retry_after is not None:
            headers["Retry-After"] = str(retry_after)
        super().__init__(
            message,
            status_code=503,
            error_code=error_code,
            details=details,
            headers=headers,
        )
