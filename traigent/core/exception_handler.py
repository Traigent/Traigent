"""Vendor error classification and interactive pause prompts.

Provides error classification for vendor/provider errors (rate limits, quota
exhaustion, service unavailability) and an injectable prompt adapter for
interactive pause-and-resume during optimization runs.

The TerminalPausePrompt is the default implementation, using ``input()`` for
interactive terminals and auto-stopping in non-interactive environments (CI,
pipes, scripts).
"""

from __future__ import annotations

import sys
from enum import Enum
from typing import Protocol

from traigent.utils.logging import get_logger

logger = get_logger(__name__)


class VendorErrorCategory(Enum):
    """Categories of vendor errors that can trigger interactive pause."""

    RATE_LIMIT = "rate_limit"
    QUOTA_EXHAUSTED = "quota_exhausted"
    INSUFFICIENT_FUNDS = "insufficient_funds"
    SERVICE_UNAVAILABLE = "service_unavailable"


class PausePromptAdapter(Protocol):
    """Protocol for interactive pause prompts. Injectable for testing."""

    def prompt_vendor_pause(
        self, error: Exception, category: VendorErrorCategory
    ) -> str:
        """Prompt user after vendor error. Returns 'resume' or 'stop'."""
        ...

    def prompt_budget_pause(self, accumulated: float, limit: float) -> str:
        """Prompt user after budget limit. Returns 'raise:<amount>' or 'stop'."""
        ...


_VENDOR_ERROR_PATTERNS: list[tuple[str, VendorErrorCategory]] = [
    ("rate limit", VendorErrorCategory.RATE_LIMIT),
    ("rate_limit", VendorErrorCategory.RATE_LIMIT),
    ("ratelimit", VendorErrorCategory.RATE_LIMIT),
    ("status 429", VendorErrorCategory.RATE_LIMIT),
    ("code: 429", VendorErrorCategory.RATE_LIMIT),
    ("code 429", VendorErrorCategory.RATE_LIMIT),
    ("http 429", VendorErrorCategory.RATE_LIMIT),
    ("error 429", VendorErrorCategory.RATE_LIMIT),
    ("429 too many", VendorErrorCategory.RATE_LIMIT),
    ("too many requests", VendorErrorCategory.RATE_LIMIT),
    ("status 402", VendorErrorCategory.INSUFFICIENT_FUNDS),
    ("code: 402", VendorErrorCategory.INSUFFICIENT_FUNDS),
    ("code 402", VendorErrorCategory.INSUFFICIENT_FUNDS),
    ("http 402", VendorErrorCategory.INSUFFICIENT_FUNDS),
    ("error 402", VendorErrorCategory.INSUFFICIENT_FUNDS),
    ("402 payment required", VendorErrorCategory.INSUFFICIENT_FUNDS),
    ("insufficient_funds", VendorErrorCategory.INSUFFICIENT_FUNDS),
    ("insufficient funds", VendorErrorCategory.INSUFFICIENT_FUNDS),
    ("insufficient credits", VendorErrorCategory.INSUFFICIENT_FUNDS),
    ("billing hard limit", VendorErrorCategory.INSUFFICIENT_FUNDS),
    ("exceeded your current billing", VendorErrorCategory.INSUFFICIENT_FUNDS),
    ("payment required", VendorErrorCategory.INSUFFICIENT_FUNDS),
    ("quota", VendorErrorCategory.QUOTA_EXHAUSTED),
    ("quota_exceeded", VendorErrorCategory.QUOTA_EXHAUSTED),
    ("insufficient_quota", VendorErrorCategory.QUOTA_EXHAUSTED),
    ("service unavailable", VendorErrorCategory.SERVICE_UNAVAILABLE),
    ("status 503", VendorErrorCategory.SERVICE_UNAVAILABLE),
    ("code: 503", VendorErrorCategory.SERVICE_UNAVAILABLE),
    ("code 503", VendorErrorCategory.SERVICE_UNAVAILABLE),
    ("http 503", VendorErrorCategory.SERVICE_UNAVAILABLE),
    ("error 503", VendorErrorCategory.SERVICE_UNAVAILABLE),
    ("503 service", VendorErrorCategory.SERVICE_UNAVAILABLE),
    ("overloaded", VendorErrorCategory.SERVICE_UNAVAILABLE),
    ("server_error", VendorErrorCategory.SERVICE_UNAVAILABLE),
]


def classify_vendor_error(exc: Exception) -> VendorErrorCategory | None:
    """Classify an exception as a vendor-pausable error category.

    Checks against known Traigent exception types first, then falls back
    to pattern-matching the exception message string.

    Args:
        exc: The exception to classify.

    Returns:
        VendorErrorCategory if the error is vendor-pausable, None otherwise.
    """
    from traigent.utils.exceptions import (
        InsufficientFundsError,
        QuotaExceededError,
        RateLimitError,
        ServiceUnavailableError,
    )

    if isinstance(exc, RateLimitError):
        return VendorErrorCategory.RATE_LIMIT
    if isinstance(exc, InsufficientFundsError):
        return VendorErrorCategory.INSUFFICIENT_FUNDS
    if isinstance(exc, QuotaExceededError):
        return VendorErrorCategory.QUOTA_EXHAUSTED
    if isinstance(exc, ServiceUnavailableError):
        return VendorErrorCategory.SERVICE_UNAVAILABLE

    # Fallback: pattern-match the error message
    msg = str(exc).lower()
    for pattern, category in _VENDOR_ERROR_PATTERNS:
        if pattern in msg:
            return category

    return None


_CATEGORY_DESCRIPTIONS: dict[VendorErrorCategory, str] = {
    VendorErrorCategory.RATE_LIMIT: (
        "The LLM provider is rate-limiting requests. "
        "This usually resolves after a short wait."
    ),
    VendorErrorCategory.QUOTA_EXHAUSTED: (
        "The LLM provider quota has been exhausted. "
        "Check your billing dashboard or wait for the quota to reset."
    ),
    VendorErrorCategory.INSUFFICIENT_FUNDS: (
        "Your LLM provider API key has insufficient credits. "
        "Please top up your account at your provider's billing page and retry."
    ),
    VendorErrorCategory.SERVICE_UNAVAILABLE: (
        "The LLM provider service is temporarily unavailable. "
        "This is usually a transient issue."
    ),
}


class TerminalPausePrompt:
    """Default terminal-based prompt implementation.

    Uses ``input()`` for interactive prompts. In non-interactive mode
    (``sys.stdin.isatty()`` is False), automatically returns ``"stop"``.
    """

    def prompt_vendor_pause(
        self, error: Exception, category: VendorErrorCategory
    ) -> str:
        """Prompt user to resume or stop after a vendor error.

        Args:
            error: The original vendor exception.
            category: Classified error category.

        Returns:
            ``"resume"`` or ``"stop"``.
        """
        if not sys.stdin.isatty():
            logger.info(
                "Non-interactive mode: auto-stopping after vendor error (%s)",
                category.value,
            )
            return "stop"

        description = _CATEGORY_DESCRIPTIONS.get(category, str(error))
        print(f"\n⚠️  Vendor error: {category.value}")
        print(f"   {description}")
        print(f"   Error: {error}")

        # Insufficient funds is non-recoverable — stop immediately
        if category == VendorErrorCategory.INSUFFICIENT_FUNDS:
            print()
            print("Stopping optimization (insufficient funds cannot be resolved by retrying).")
            return "stop"

        print()
        print("Options:")
        print("  [r] Resume optimization")
        print("  [s] Stop and return partial results")

        try:
            choice = input("\nYour choice (r/s): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return "stop"

        if choice in ("r", "resume"):
            return "resume"
        return "stop"

    def prompt_budget_pause(self, accumulated: float, limit: float) -> str:
        """Prompt user to raise budget or stop after budget limit reached.

        Args:
            accumulated: Total cost accumulated so far (USD).
            limit: Current cost limit (USD).

        Returns:
            ``"raise:<new_limit>"`` or ``"stop"``.
        """
        if not sys.stdin.isatty():
            logger.info(
                "Non-interactive mode: auto-stopping at budget limit ($%.2f / $%.2f)",
                accumulated,
                limit,
            )
            return "stop"

        print(f"\n💰 Budget limit reached: ${accumulated:.2f} / ${limit:.2f}")
        print()
        print("Options:")
        print("  Enter a new limit (e.g. 5.0) to raise and continue")
        print("  [s] Stop and return partial results")

        try:
            choice = input("\nNew limit or 's' to stop: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return "stop"

        if choice in ("s", "stop", ""):
            return "stop"

        try:
            new_limit = float(choice)
            if new_limit <= accumulated:
                print(
                    f"  New limit must be greater than current spend (${accumulated:.2f})"
                )
                return "stop"
            if not (0 < new_limit < float("inf")):
                print("  Invalid amount. Stopping.")
                return "stop"
            return f"raise:{new_limit}"
        except ValueError:
            print("  Invalid input. Stopping.")
            return "stop"
