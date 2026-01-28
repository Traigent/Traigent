"""Exception handling utilities for pause-on-error behavior."""

from __future__ import annotations

import errno
import logging
import math
import socket

from traigent.utils.exceptions import (
    NetworkError,
    QuotaExceededError,
    RateLimitError,
    ServiceError,
    ServiceUnavailableError,
    TraigentConnectionError,
)
from traigent.utils.retry import ServiceUnavailableError as RetryServiceUnavailableError
from traigent.utils.user_prompts import (
    print_budget_prompt,
    print_network_error_prompt,
    print_vendor_error_prompt,
)

logger = logging.getLogger(__name__)

# User prompt constants
_PROMPT_SELECT_OPTION = "Select option [1/2]: "
_PROMPT_ABORTED = "\n  Aborted."
_PROMPT_INVALID_OPTION = "  Invalid option. Please enter 1 or 2."

# Network-related errno codes
NETWORK_ERRNO_CODES = {
    errno.ENETUNREACH,  # Network unreachable
    errno.ECONNREFUSED,  # Connection refused
    errno.ECONNRESET,  # Connection reset by peer
    errno.ECONNABORTED,  # Connection aborted
    errno.EHOSTUNREACH,  # No route to host
    errno.ETIMEDOUT,  # Connection timed out
    errno.ENETDOWN,  # Network is down
    errno.ENOTCONN,  # Transport endpoint not connected
}

# Strong network error indicators in error messages
# Used to identify network issues from generic ConnectionError or other exceptions
_NETWORK_MSG_INDICATORS = (
    "network is unreachable",
    "network unreachable",
    "no route to host",
    "host unreachable",
    "name resolution",
    "getaddrinfo failed",
    "temporary failure in name resolution",
    "nodename nor servname provided",
)

# HTTP client library network error class names (lowercase)
# These are specific class names, not message patterns, so false positives are unlikely
_NETWORK_CLASS_INDICATORS = (
    "clientconnectorerror",  # aiohttp
    "newconnectionerror",  # urllib3
    "serverconnectionerror",  # aiohttp server connection
    "connecterror",  # httpx ConnectError
    "connectionerror",  # openai APIConnectionError, Python ConnectionError subclasses
)

# Default connectivity check endpoints (IP:port pairs)
# Uses well-known DNS servers that don't require DNS resolution
# Multiple fallbacks for enterprise networks that may block specific IPs
_DEFAULT_CONNECTIVITY_ENDPOINTS = (
    ("1.1.1.1", 53),  # Cloudflare DNS
    ("8.8.8.8", 53),  # Google DNS
    ("9.9.9.9", 53),  # Quad9 DNS
)

NETWORK_ERROR_EXPLANATIONS = {
    "connection_lost": (
        "Network connection lost",
        "Your network connection appears to be down.\n"
        "Check your internet connection (WiFi, Ethernet, or airplane mode).",
    ),
    "dns_failure": (
        "DNS resolution failed",
        "Unable to resolve the server address.\n"
        "Check your DNS settings or internet connection.",
    ),
    "connection_refused": (
        "Connection refused",
        "The server refused the connection.\n"
        "The service may be down or your network may be blocking access.",
    ),
    "connection_timeout": (
        "Connection timed out",
        "The connection attempt timed out.\n"
        "This may indicate network issues or server unavailability.",
    ),
}

VENDOR_ERROR_EXPLANATIONS = {
    "rate_limit": (
        "Rate limit exceeded",
        "Rate limit exceeded.\n"
        "Wait a moment before resuming, or reduce parallelism.",
    ),
    "quota_exhausted": (
        "Quota exhausted",
        "Quota exhausted.\n" "Check your provider dashboard or billing limits.",
    ),
    "service_unavailable": (
        "Service unavailable",
        "Service unavailable.\n" "Try resuming in a few moments.",
    ),
    "api_error": (
        "API error",
        "An error occurred communicating with your LLM provider.",
    ),
}


class VendorPauseError(Exception):
    """Raised when a vendor error should pause the optimization loop."""

    def __init__(
        self,
        error: Exception,
        *,
        partial_cost: float | None = None,
    ) -> None:
        super().__init__(str(error))
        self.error = error
        self.partial_cost = partial_cost


class NetworkPauseError(Exception):
    """Raised when a network error should pause the optimization loop."""

    def __init__(
        self,
        error: Exception,
        *,
        partial_cost: float | None = None,
    ) -> None:
        super().__init__(str(error))
        self.error = error
        self.partial_cost = partial_cost


def classify_network_error(error: Exception) -> str:
    """Classify network error into user-friendly category."""
    # Check for DNS-related errors
    if isinstance(error, socket.gaierror):
        return "dns_failure"

    # Check for timeout
    if isinstance(error, (socket.timeout, TimeoutError)):
        return "connection_timeout"

    # Check OSError with errno
    if isinstance(error, OSError) and error.errno is not None:
        if error.errno == errno.ECONNREFUSED:
            return "connection_refused"
        if error.errno == errno.ETIMEDOUT:
            return "connection_timeout"
        if error.errno in NETWORK_ERRNO_CODES:
            return "connection_lost"

    # Check error message for indicators
    error_msg = str(error).lower()
    if (
        "dns" in error_msg
        or "name resolution" in error_msg
        or "getaddrinfo" in error_msg
    ):
        return "dns_failure"
    if "refused" in error_msg:
        return "connection_refused"
    if "timeout" in error_msg or "timed out" in error_msg:
        return "connection_timeout"

    return "connection_lost"


def is_network_exception(error: Exception) -> bool:
    """Check whether an error is a network/connectivity failure.

    This detects errors caused by network issues on the user's side,
    such as airplane mode, WiFi disconnection, or DNS failures.

    Note: Vendor errors (rate limits, quota) take precedence. If an error
    is detected as a vendor error, it should NOT be classified as network.

    This function is intentionally conservative to avoid false positives,
    since a false positive will pause the entire optimization run.
    """
    # Vendor errors take precedence - don't misclassify rate limits/quota
    # as network errors even if they have connection-related messaging
    if is_vendor_exception(error):
        return False

    # Direct network error types from traigent
    if isinstance(error, (NetworkError, TraigentConnectionError)):
        return True

    # DNS resolution failures - strong signal of network issues
    if isinstance(error, socket.gaierror):
        return True

    # Socket timeout - strong signal of network issues
    if isinstance(error, socket.timeout):
        return True

    # OSError with network-related errno - strong signal
    if isinstance(error, OSError) and error.errno in NETWORK_ERRNO_CODES:
        return True

    # For generic ConnectionError, require additional evidence via errno or message
    # to avoid false positives from non-network connection issues
    if isinstance(error, ConnectionError):
        # Check if it has a network-related errno
        if hasattr(error, "errno") and error.errno in NETWORK_ERRNO_CODES:
            return True
        # Check error message for network indicators
        error_msg = str(error).lower()
        if any(indicator in error_msg for indicator in _NETWORK_MSG_INDICATORS):
            return True
        # Generic ConnectionError without network evidence - don't classify as network
        return False

    # Check error message for strong network indicators
    error_msg = str(error).lower()
    if any(indicator in error_msg for indicator in _NETWORK_MSG_INDICATORS):
        return True
    # Also check for HTTP client library indicators in message
    if any(indicator in error_msg for indicator in _NETWORK_CLASS_INDICATORS):
        return True

    # Check for specific HTTP client library network error classes
    # These are specific to actual network connectivity issues
    error_class = type(error).__name__.lower()
    return any(indicator in error_class for indicator in _NETWORK_CLASS_INDICATORS)


def classify_vendor_error(error: Exception) -> str:
    """Classify vendor error into user-friendly category."""
    if isinstance(error, RateLimitError):
        return "rate_limit"
    if isinstance(error, QuotaExceededError):
        return "quota_exhausted"
    if isinstance(error, (ServiceUnavailableError, RetryServiceUnavailableError)):
        return "service_unavailable"
    if isinstance(error, ServiceError) and error.status_code in {429, 503}:
        return "rate_limit" if error.status_code == 429 else "service_unavailable"

    error_msg = str(error).lower()
    if "rate" in error_msg and "limit" in error_msg:
        return "rate_limit"
    if "quota" in error_msg or "budget" in error_msg or "exceeded" in error_msg:
        return "quota_exhausted"
    if "unavailable" in error_msg or "503" in error_msg:
        return "service_unavailable"
    return "api_error"


def is_vendor_exception(error: Exception) -> bool:
    """Check whether an error looks like a vendor/provider failure."""
    if isinstance(
        error,
        (
            RateLimitError,
            QuotaExceededError,
            ServiceUnavailableError,
            RetryServiceUnavailableError,
        ),
    ):
        return True
    if isinstance(error, ServiceError) and error.status_code in {429, 503}:
        return True
    error_msg = str(error).lower()
    indicators = (
        "rate limit",
        "too many requests",
        "quota",
        "insufficient quota",
        "service unavailable",
        "temporarily unavailable",
        "503",
        "429",
    )
    return any(indicator in error_msg for indicator in indicators)


def _log_vendor_details(error: Exception) -> None:
    retry_after = getattr(error, "retry_after", None)
    status_code = getattr(error, "status_code", None)
    if retry_after is not None or status_code is not None:
        logger.info(
            "Vendor error details: status_code=%s retry_after=%s",
            status_code,
            retry_after,
        )


def handle_budget_limit_reached(
    current_limit: float, spent: float
) -> tuple[bool, float]:
    """Handle budget limit reached exception."""
    while True:
        print_budget_prompt(current_limit, spent)
        try:
            choice = input(_PROMPT_SELECT_OPTION).strip()
        except (EOFError, KeyboardInterrupt):
            print(_PROMPT_ABORTED)
            return (False, current_limit)

        if choice == "2":
            return (False, current_limit)

        if choice == "1":
            while True:
                add_amount = _get_raise_amount()
                if add_amount is None:
                    break
                new_limit = current_limit + add_amount
                if not math.isfinite(new_limit):
                    print(
                        '  Invalid amount. Please enter a finite number, or type "stop" to cancel.'
                    )
                    continue
                if new_limit <= spent:
                    print(
                        f"  Please enter a higher amount (over ${spent - current_limit:.2f}), "
                        'or type "stop" to cancel.'
                    )
                    continue
                print(f"New budget limit: ${new_limit:.2f}")
                print("Continuing optimization...")
                return (True, new_limit)
            continue

        print(_PROMPT_INVALID_OPTION)


def _get_raise_amount() -> float | None:
    """Get amount to add from user with validation."""
    while True:
        try:
            add_input = input("  Enter amount to add: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(_PROMPT_ABORTED)
            return None

        if add_input.lower() == "stop":
            print("  Returning to options...\n")
            return None

        try:
            add_amount = float(add_input)
            if not math.isfinite(add_amount):
                print(
                    '  Invalid input. Please enter a finite number, or type "stop" to cancel.'
                )
                continue
            if add_amount <= 0:
                print(
                    '  Amount must be positive. Please try again, or type "stop" to cancel.'
                )
                continue
            return add_amount
        except ValueError:
            print('  Invalid input. Please enter a number, or type "stop" to cancel.')


def handle_vendor_exception(error: Exception) -> bool:
    """Handle vendor exception with user-friendly explanation."""
    error_type = classify_vendor_error(error)
    title, explanation = VENDOR_ERROR_EXPLANATIONS.get(
        error_type,
        ("Unknown error", "An unexpected error occurred."),
    )

    _log_vendor_details(error)
    print_vendor_error_prompt(title, explanation)

    while True:
        try:
            choice = input(_PROMPT_SELECT_OPTION).strip()
        except (EOFError, KeyboardInterrupt):
            print(_PROMPT_ABORTED)
            return False
        if choice == "1":
            return True
        if choice == "2":
            return False
        print(_PROMPT_INVALID_OPTION)


def _log_network_details(error: Exception) -> None:
    """Log network error details for debugging.

    Only logs error type and errno - NOT the full message, which may
    contain sensitive information (URLs with tokens, API keys, etc.).
    """
    error_type = type(error).__name__
    error_errno = getattr(error, "errno", None)
    logger.debug(
        "Network error details: type=%s errno=%s",
        error_type,
        error_errno,
    )


def _check_network_connectivity(
    timeout: float = 2.0,
    endpoints: tuple[tuple[str, int], ...] | None = None,
) -> bool:
    """Check if network connectivity is available.

    Attempts lightweight TCP connections to well-known DNS servers.
    Uses multiple fallback endpoints for enterprise networks that may
    block specific IPs.

    Args:
        timeout: Connection timeout in seconds per endpoint.
        endpoints: Optional tuple of (host, port) pairs to try.
            Defaults to Cloudflare, Google, and Quad9 DNS servers.

    Returns:
        True if network is available (any endpoint reachable), False otherwise.
    """
    if endpoints is None:
        endpoints = _DEFAULT_CONNECTIVITY_ENDPOINTS

    for host, port in endpoints:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect((host, port))
            sock.close()
            return True
        except OSError:
            continue  # Try next endpoint

    return False


def is_network_available(timeout: float = 2.0) -> bool:
    """Public wrapper for lightweight connectivity checks."""
    return _check_network_connectivity(timeout=timeout)


def _wait_for_network_recovery(
    timeout_seconds: float = 180.0,
    poll_interval: float = 3.0,
) -> bool:
    """Wait for network connectivity to be restored.

    Args:
        timeout_seconds: Maximum time to wait (default 3 minutes).
        poll_interval: How often to check connectivity (seconds).

    Returns:
        True if network was restored, False if timeout reached.
    """
    import time

    start_time = time.monotonic()

    try:
        while (time.monotonic() - start_time) < timeout_seconds:
            remaining = int(timeout_seconds - (time.monotonic() - start_time))
            print(
                f"  Waiting for network... ({remaining}s remaining)    ",
                end="\r",
                flush=True,
            )

            if _check_network_connectivity():
                print("\n  Network restored!                              ")
                return True

            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print(_PROMPT_ABORTED)
        return False

    print("\n  Network wait timeout reached.                  ")
    return False


def handle_network_exception(
    error: Exception,
    *,
    network_wait_timeout: float = 180.0,
    poll_interval: float = 3.0,
) -> bool:
    """Handle network exception with user-friendly explanation.

    Shows a prompt and waits for user input. If user chooses to resume,
    waits for network connectivity to be restored (with timeout).

    Flow:
    1. Show prompt with [1] Resume / [2] Stop options
    2. If user chooses Resume → wait for network (up to network_wait_timeout)
    3. If network restored → return True (continue optimization)
    4. If timeout reached → show prompt again
    5. If user chooses Stop → return False

    Note:
        This function uses synchronous I/O (input() and time.sleep()) and will
        block the event loop. This is acceptable because it is only called when
        parallel_trials <= 1, meaning no concurrent evaluation tasks need to
        continue. Telemetry and logging may be stalled during user interaction.

    Args:
        error: The network exception that occurred.
        network_wait_timeout: How long to wait for network after user resumes (seconds).
        poll_interval: How often to check network status (seconds).

    Returns:
        True if resuming (network restored), False to stop.
    """
    error_type = classify_network_error(error)
    title, explanation = NETWORK_ERROR_EXPLANATIONS.get(
        error_type,
        ("Network error", "A network error occurred. Check your connection."),
    )

    _log_network_details(error)

    try:
        while True:
            print_network_error_prompt(title, explanation)

            # Get user choice
            while True:
                try:
                    choice = input(_PROMPT_SELECT_OPTION).strip()
                except (EOFError, KeyboardInterrupt):
                    print(_PROMPT_ABORTED)
                    return False

                if choice == "1":
                    # User wants to resume - wait for network
                    break
                if choice == "2":
                    return False
                print(_PROMPT_INVALID_OPTION)

            # Wait for network to come back
            print("\n  You chose to resume. Waiting for network connection...")
            if _wait_for_network_recovery(network_wait_timeout, poll_interval):
                print("  Continuing optimization...")
                return True

            # Network didn't come back within timeout - show prompt again
            print("\n  Network not restored within timeout. Please check your connection.")
            print("  Showing options again...\n")
    except KeyboardInterrupt:
        print(_PROMPT_ABORTED)
        return False
