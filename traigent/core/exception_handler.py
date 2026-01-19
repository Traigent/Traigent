"""Exception handling utilities for pause-on-error behavior."""

from __future__ import annotations

import logging
import math
from traigent.utils.exceptions import (
    QuotaExceededError,
    RateLimitError,
    ServiceError,
    ServiceUnavailableError,
)
from traigent.utils.retry import ServiceUnavailableError as RetryServiceUnavailableError
from traigent.utils.user_prompts import print_budget_prompt, print_vendor_error_prompt

logger = logging.getLogger(__name__)

VENDOR_ERROR_EXPLANATIONS = {
    "rate_limit": (
        "Rate limit exceeded",
        "Rate limit exceeded.\n"
        "Wait a moment before resuming, or reduce parallelism.",
    ),
    "quota_exhausted": (
        "Quota exhausted",
        "Quota exhausted.\n"
        "Check your provider dashboard or billing limits.",
    ),
    "service_unavailable": (
        "Service unavailable",
        "Service unavailable.\n"
        "Try resuming in a few moments.",
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
            choice = input("Select option [1/2]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Aborted.")
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
                        "  Invalid amount. Please enter a finite number, or type \"stop\" to cancel."
                    )
                    continue
                if new_limit <= spent:
                    print(
                        f"  Please enter a higher amount (over ${spent - current_limit:.2f}), "
                        "or type \"stop\" to cancel."
                    )
                    continue
                print(f"New budget limit: ${new_limit:.2f}")
                print("Continuing optimization...")
                return (True, new_limit)
            continue

        print("  Invalid option. Please enter 1 or 2.")


def _get_raise_amount() -> float | None:
    """Get amount to add from user with validation."""
    while True:
        try:
            add_input = input("  Enter amount to add: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Aborted.")
            return None

        if add_input.lower() == "stop":
            print("  Returning to options...\n")
            return None

        try:
            add_amount = float(add_input)
            if not math.isfinite(add_amount):
                print(
                    "  Invalid input. Please enter a finite number, or type \"stop\" to cancel."
                )
                continue
            if add_amount <= 0:
                print(
                    "  Amount must be positive. Please try again, or type \"stop\" to cancel."
                )
                continue
            return add_amount
        except ValueError:
            print("  Invalid input. Please enter a number, or type \"stop\" to cancel.")


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
            choice = input("Select option [1/2]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Aborted.")
            return False
        if choice == "1":
            return True
        if choice == "2":
            return False
        print("  Invalid option. Please enter 1 or 2.")
