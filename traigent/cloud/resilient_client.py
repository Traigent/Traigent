"""Resilient HTTP client with retry logic and security best practices.

Provides automatic retry with exponential backoff for transient failures
while maintaining security and SOC2 compliance.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability CONC-Quality-Security FUNC-CLOUD-HYBRID FUNC-SECURITY REQ-CLOUD-009 REQ-SEC-010 SYNC-CloudHybrid

import asyncio
import logging
import secrets
import time
from enum import Enum
from hashlib import sha256
from typing import Any, Awaitable, Callable, TypeVar, cast

from traigent.utils.exceptions import NonRetryableError, RetryableError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ErrorType(Enum):
    """Classification of error types for retry logic."""

    NETWORK = "network"  # Connection errors, timeouts
    AUTH = "auth"  # Authentication/authorization errors (don't retry)
    RATE_LIMIT = "rate_limit"  # Rate limiting (retry with backoff)
    SERVER = "server"  # 5xx errors (retry)
    CLIENT = "client"  # 4xx errors except auth/rate limit (don't retry)
    UNKNOWN = "unknown"  # Unexpected errors


class ResilientClient:
    """HTTP client with automatic retry and security features."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter_factor: float = 0.1,
    ) -> None:
        """Initialize resilient client.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds for exponential backoff
            max_delay: Maximum delay between retries
            jitter_factor: Random jitter factor (0-1) to prevent thundering herd
        """
        self.max_retries = min(max_retries, 10)  # Cap at 10 for safety
        self.base_delay = max(base_delay, 0.1)  # Minimum 100ms
        self.max_delay = min(max_delay, 300)  # Cap at 5 minutes
        self.jitter_factor = min(max(jitter_factor, 0), 1)  # Clamp to 0-1

        # Track retry statistics for monitoring
        self._retry_stats: dict[str, int | float] = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_retries": 0,
            "retry_successes": 0,
        }

        logger.info(f"ResilientClient initialized with max_retries={self.max_retries}")

    def classify_error(self, error: Exception) -> ErrorType:
        """Classify error type to determine if retry is appropriate.

        Args:
            error: The exception that occurred

        Returns:
            ErrorType classification
        """
        error_str = str(error).lower()
        error_type_name = type(error).__name__.lower()

        # Network errors - always retry
        network_indicators = [
            "connection",
            "timeout",
            "network",
            "dns",
            "refused",
            "reset",
            "broken pipe",
            "ssl",
        ]
        if any(
            ind in error_str or ind in error_type_name for ind in network_indicators
        ):
            return ErrorType.NETWORK

        # Authentication errors - never retry
        auth_indicators = ["401", "unauthorized", "authentication", "forbidden", "403"]
        if any(ind in error_str for ind in auth_indicators):
            return ErrorType.AUTH

        # Rate limiting - retry with backoff
        rate_limit_indicators = ["429", "rate limit", "too many requests"]
        if any(ind in error_str for ind in rate_limit_indicators):
            return ErrorType.RATE_LIMIT

        # Server errors - retry
        server_indicators = [
            "500",
            "502",
            "503",
            "504",
            "server error",
            "internal error",
        ]
        if any(ind in error_str for ind in server_indicators):
            return ErrorType.SERVER

        # Client errors - don't retry (except specific cases above)
        client_indicators = [
            "400",
            "404",
            "405",
            "406",
            "409",
            "410",
            "bad request",
            "not found",
        ]
        if any(ind in error_str for ind in client_indicators):
            return ErrorType.CLIENT

        return ErrorType.UNKNOWN

    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if request should be retried.

        Args:
            error: The error that occurred
            attempt: Current attempt number (0-based)

        Returns:
            True if should retry
        """
        if attempt >= self.max_retries:
            return False

        error_type = self.classify_error(error)

        # Never retry auth errors
        if error_type == ErrorType.AUTH:
            logger.debug("Auth error - not retrying")
            return False

        # Never retry client errors (except rate limit)
        if error_type == ErrorType.CLIENT:
            logger.debug("Client error - not retrying")
            return False

        # Retry network, rate limit, and server errors
        if error_type in (ErrorType.NETWORK, ErrorType.RATE_LIMIT, ErrorType.SERVER):
            logger.debug(f"{error_type.value} error - will retry")
            return True

        # For unknown errors, be conservative and don't retry
        logger.debug("Unknown error type - not retrying")
        return False

    def calculate_delay(self, attempt: int, error: Exception) -> float:
        """Calculate delay before next retry with exponential backoff.

        Args:
            attempt: Current attempt number (0-based)
            error: The error that occurred

        Returns:
            Delay in seconds
        """
        # Base exponential backoff: 2^attempt * base_delay
        delay: float = min(self.base_delay * (2**attempt), self.max_delay)

        # Special handling for rate limiting
        error_type = self.classify_error(error)
        if error_type == ErrorType.RATE_LIMIT:
            # For rate limiting, use longer delays
            delay = min(delay * 2, self.max_delay)

            # Check if server provided Retry-After header
            error_str = str(error)
            if "retry-after" in error_str.lower():
                # Try to parse retry-after value (simplified)
                try:
                    import re

                    match = re.search(
                        r"retry-after[:\s]+(\d+)", error_str, re.IGNORECASE
                    )
                    if match:
                        server_delay = int(match.group(1))
                        delay = min(server_delay, self.max_delay)
                        logger.debug(f"Using server-provided retry-after: {delay}s")
                except Exception as e:
                    logger.debug(f"Could not parse retry-after header: {e}")

        # Add jitter to prevent thundering herd
        if self.jitter_factor > 0:
            jitter: float = secrets.SystemRandom().uniform(
                0, delay * self.jitter_factor
            )
            delay += jitter

        return delay

    async def execute_with_retry(
        self,
        operation: Callable[..., Awaitable[T]],
        *args: Any,
        operation_name: str | None = None,
        **kwargs: Any,
    ) -> T:
        """Execute operation with automatic retry on failure.

        Args:
            operation: Async function to execute
            *args: Positional arguments for operation
            operation_name: Name for logging (optional)
            **kwargs: Keyword arguments for operation

        Returns:
            Result from successful operation

        Raises:
            Exception: If all retries exhausted
        """
        op_name = operation_name or operation.__name__
        op_hash = sha256(f"{op_name}{time.time()}".encode()).hexdigest()[:8]

        self._retry_stats["total_requests"] += 1
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                # Log attempt (without sensitive data)
                if attempt > 0:
                    self._retry_stats["total_retries"] += 1
                    logger.info(
                        f"Retry {attempt}/{self.max_retries} for {op_name} [{op_hash}]"
                    )

                # Execute operation
                result = await operation(*args, **kwargs)

                # Success
                if attempt > 0:
                    self._retry_stats["retry_successes"] += 1
                    logger.info(f"Retry successful for {op_name} [{op_hash}]")

                self._retry_stats["successful_requests"] += 1
                return result

            except Exception as e:
                last_error = e

                # Never log sensitive data in errors
                safe_error = self._sanitize_error(e)
                logger.warning(f"Operation {op_name} failed: {safe_error}")

                # Check if we should retry
                if not self.should_retry(e, attempt):
                    logger.error(f"Non-retryable error for {op_name}: {safe_error}")
                    break

                if attempt < self.max_retries:
                    # Calculate delay
                    delay = self.calculate_delay(attempt, e)
                    logger.info(f"Waiting {delay:.1f}s before retry {attempt + 1}")
                    await asyncio.sleep(delay)

        # All retries exhausted
        self._retry_stats["failed_requests"] += 1
        logger.error(f"All retries exhausted for {op_name} [{op_hash}]")

        if last_error:
            raise last_error from None
        else:
            raise RuntimeError(f"Operation {op_name} failed without error")

    def _sanitize_error(self, error: Exception) -> str:
        """Sanitize error message to remove sensitive data.

        Args:
            error: The error to sanitize

        Returns:
            Safe error message
        """
        error_str = str(error)

        # Remove potential sensitive data patterns
        sensitive_patterns = [
            r"[Bb]earer\s+[A-Za-z0-9\-_\.]+",  # Bearer tokens
            r"[Aa]pi[_\-]?[Kk]ey[:\s]+[A-Za-z0-9\-_]+",  # API keys
            r"[Pp]assword[:\s]+[^\s]+",  # Passwords
            r"[Tt]oken[:\s]+[A-Za-z0-9\-_\.]+",  # Tokens
            r"[Ss]ecret[:\s]+[^\s]+",  # Secrets
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}",  # Emails
        ]

        import re

        safe_error = error_str
        for pattern in sensitive_patterns:
            safe_error = re.sub(pattern, "[REDACTED]", safe_error)

        # Also sanitize the error type if it contains sensitive info
        error_type = type(error).__name__
        if error_type != safe_error:
            return f"{error_type}: {safe_error}"
        return safe_error

    def get_statistics(self) -> dict[str, Any]:
        """Get retry statistics for monitoring.

        Returns:
            Dictionary of statistics
        """
        stats = self._retry_stats.copy()

        # Calculate success rate
        if stats["total_requests"] > 0:
            stats["success_rate"] = (
                stats["successful_requests"] / stats["total_requests"]
            ) * 100
            stats["retry_rate"] = (
                stats["total_retries"] / stats["total_requests"]
            ) * 100
        else:
            stats["success_rate"] = 0
            stats["retry_rate"] = 0

        # Calculate retry effectiveness
        if stats["total_retries"] > 0:
            stats["retry_success_rate"] = (
                stats["retry_successes"] / stats["total_retries"]
            ) * 100
        else:
            stats["retry_success_rate"] = 0

        return stats

    def reset_statistics(self) -> None:
        """Reset statistics counters."""
        self._retry_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_retries": 0,
            "retry_successes": 0,
        }
        logger.debug("Retry statistics reset")


# Example usage wrapper for backend operations
async def resilient_backend_request(
    client: ResilientClient,
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    data: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Make resilient request to backend with retry logic.

    Args:
        client: ResilientClient instance
        method: HTTP method (GET, POST, etc.)
        url: Request URL
        headers: Request headers
        data: Request data
        timeout: Request timeout

    Returns:
        Response data

    Raises:
        Exception: If request fails after all retries
    """

    async def make_request() -> dict[str, Any]:
        try:
            import aiohttp
        except ImportError as exc:  # pragma: no cover - environment specific
            raise RuntimeError(
                "aiohttp is required for resilient backend requests"
            ) from exc

        request_timeout = aiohttp.ClientTimeout(total=timeout)

        async with aiohttp.ClientSession(timeout=request_timeout) as session:
            async with session.request(
                method,
                url,
                headers=headers,
                json=data if isinstance(data, dict) else None,
                data=None if isinstance(data, dict) else data,
            ) as response:
                status = response.status

                if status >= 500:
                    raise RetryableError(f"Server error: HTTP {status}")
                if status == 429:
                    retry_after_header = response.headers.get("Retry-After")
                    retry_after_value = None
                    if retry_after_header:
                        try:
                            retry_after_value = float(retry_after_header)
                        except ValueError:
                            logger.debug(
                                "Non-numeric Retry-After header encountered: %s",
                                retry_after_header,
                            )
                    raise RetryableError("Rate limited", retry_after_value) from None
                if status in {401, 403}:
                    raise NonRetryableError("Authentication failed")
                if status >= 400:
                    raise NonRetryableError(f"Client error: HTTP {status}")

                try:
                    payload = await response.json()
                except aiohttp.ContentTypeError:
                    payload_text = await response.text()
                    payload = {"raw_response": payload_text}

                return cast(dict[str, Any], payload)
        #         return await response.json()

        # For now, raise not implemented
        raise NotImplementedError("Backend request not yet implemented") from None

    return await client.execute_with_retry(
        make_request, operation_name=f"{method} request"
    )
