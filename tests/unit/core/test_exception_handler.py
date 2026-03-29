"""Tests for traigent.core.exception_handler module."""

from __future__ import annotations

from unittest.mock import patch

from traigent.core.exception_handler import (
    TerminalPausePrompt,
    VendorErrorCategory,
    classify_vendor_error,
)
from traigent.utils.exceptions import (
    InsufficientFundsError,
    QuotaExceededError,
    RateLimitError,
    ServiceUnavailableError,
)

# ---------------------------------------------------------------------------
# classify_vendor_error
# ---------------------------------------------------------------------------


class TestClassifyVendorError:
    def test_rate_limit_error(self):
        exc = RateLimitError("Rate limit exceeded")
        assert classify_vendor_error(exc) == VendorErrorCategory.RATE_LIMIT

    def test_quota_exceeded_error(self):
        exc = QuotaExceededError("Quota exhausted")
        assert classify_vendor_error(exc) == VendorErrorCategory.QUOTA_EXHAUSTED

    def test_service_unavailable_error(self):
        exc = ServiceUnavailableError("Service down")
        assert classify_vendor_error(exc) == VendorErrorCategory.SERVICE_UNAVAILABLE

    def test_insufficient_funds_error(self):
        exc = InsufficientFundsError("Insufficient funds")
        assert classify_vendor_error(exc) == VendorErrorCategory.INSUFFICIENT_FUNDS

    def test_generic_402_in_message(self):
        exc = RuntimeError("HTTP status 402 Payment Required")
        assert classify_vendor_error(exc) == VendorErrorCategory.INSUFFICIENT_FUNDS

    def test_generic_insufficient_funds_in_message(self):
        exc = RuntimeError("Error: insufficient funds on this account")
        assert classify_vendor_error(exc) == VendorErrorCategory.INSUFFICIENT_FUNDS

    def test_generic_billing_hard_limit_in_message(self):
        exc = RuntimeError("You exceeded your current billing hard limit")
        assert classify_vendor_error(exc) == VendorErrorCategory.INSUFFICIENT_FUNDS

    def test_generic_429_in_message(self):
        exc = RuntimeError("HTTP 429 Too Many Requests")
        assert classify_vendor_error(exc) == VendorErrorCategory.RATE_LIMIT

    def test_generic_quota_in_message(self):
        exc = RuntimeError("insufficient_quota for this model")
        assert classify_vendor_error(exc) == VendorErrorCategory.QUOTA_EXHAUSTED

    def test_generic_503_in_message(self):
        exc = RuntimeError("HTTP 503 Service Unavailable")
        assert classify_vendor_error(exc) == VendorErrorCategory.SERVICE_UNAVAILABLE

    def test_non_vendor_error_returns_none(self):
        exc = ValueError("Invalid argument")
        assert classify_vendor_error(exc) is None

    def test_generic_exception_no_match(self):
        exc = RuntimeError("Something unexpected happened")
        assert classify_vendor_error(exc) is None


# ---------------------------------------------------------------------------
# TerminalPausePrompt
# ---------------------------------------------------------------------------


class TestTerminalPausePromptVendor:
    def test_non_interactive_returns_stop(self):
        prompt = TerminalPausePrompt()
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            result = prompt.prompt_vendor_pause(
                RuntimeError("rate limit"), VendorErrorCategory.RATE_LIMIT
            )
        assert result == "stop"

    def test_interactive_resume(self):
        prompt = TerminalPausePrompt()
        with (
            patch("sys.stdin") as mock_stdin,
            patch("builtins.input", return_value="r"),
        ):
            mock_stdin.isatty.return_value = True
            result = prompt.prompt_vendor_pause(
                RuntimeError("rate limit"), VendorErrorCategory.RATE_LIMIT
            )
        assert result == "resume"

    def test_interactive_stop(self):
        prompt = TerminalPausePrompt()
        with (
            patch("sys.stdin") as mock_stdin,
            patch("builtins.input", return_value="s"),
        ):
            mock_stdin.isatty.return_value = True
            result = prompt.prompt_vendor_pause(
                RuntimeError("rate limit"), VendorErrorCategory.RATE_LIMIT
            )
        assert result == "stop"

    def test_eof_returns_stop(self):
        prompt = TerminalPausePrompt()
        with (
            patch("sys.stdin") as mock_stdin,
            patch("builtins.input", side_effect=EOFError),
        ):
            mock_stdin.isatty.return_value = True
            result = prompt.prompt_vendor_pause(
                RuntimeError("rate limit"), VendorErrorCategory.RATE_LIMIT
            )
        assert result == "stop"

    def test_keyboard_interrupt_returns_stop(self):
        prompt = TerminalPausePrompt()
        with (
            patch("sys.stdin") as mock_stdin,
            patch("builtins.input", side_effect=KeyboardInterrupt),
        ):
            mock_stdin.isatty.return_value = True
            result = prompt.prompt_vendor_pause(
                RuntimeError("rate limit"), VendorErrorCategory.RATE_LIMIT
            )
        assert result == "stop"

    def test_insufficient_funds_auto_stops(self):
        """Insufficient funds is non-recoverable — should auto-stop without prompting."""
        prompt = TerminalPausePrompt()
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            result = prompt.prompt_vendor_pause(
                RuntimeError("insufficient funds"), VendorErrorCategory.INSUFFICIENT_FUNDS
            )
        assert result == "stop"


class TestTerminalPausePromptBudget:
    def test_non_interactive_returns_stop(self):
        prompt = TerminalPausePrompt()
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            result = prompt.prompt_budget_pause(1.50, 2.00)
        assert result == "stop"

    def test_interactive_raise_limit(self):
        prompt = TerminalPausePrompt()
        with (
            patch("sys.stdin") as mock_stdin,
            patch("builtins.input", return_value="5.0"),
        ):
            mock_stdin.isatty.return_value = True
            result = prompt.prompt_budget_pause(1.50, 2.00)
        assert result == "raise:5.0"

    def test_interactive_stop(self):
        prompt = TerminalPausePrompt()
        with (
            patch("sys.stdin") as mock_stdin,
            patch("builtins.input", return_value="s"),
        ):
            mock_stdin.isatty.return_value = True
            result = prompt.prompt_budget_pause(1.50, 2.00)
        assert result == "stop"

    def test_invalid_input_returns_stop(self):
        prompt = TerminalPausePrompt()
        with (
            patch("sys.stdin") as mock_stdin,
            patch("builtins.input", return_value="abc"),
        ):
            mock_stdin.isatty.return_value = True
            result = prompt.prompt_budget_pause(1.50, 2.00)
        assert result == "stop"

    def test_limit_below_accumulated_returns_stop(self):
        prompt = TerminalPausePrompt()
        with (
            patch("sys.stdin") as mock_stdin,
            patch("builtins.input", return_value="1.0"),
        ):
            mock_stdin.isatty.return_value = True
            result = prompt.prompt_budget_pause(1.50, 2.00)
        assert result == "stop"

    def test_eof_returns_stop(self):
        prompt = TerminalPausePrompt()
        with (
            patch("sys.stdin") as mock_stdin,
            patch("builtins.input", side_effect=EOFError),
        ):
            mock_stdin.isatty.return_value = True
            result = prompt.prompt_budget_pause(1.50, 2.00)
        assert result == "stop"

    def test_keyboard_interrupt_returns_stop(self):
        prompt = TerminalPausePrompt()
        with (
            patch("sys.stdin") as mock_stdin,
            patch("builtins.input", side_effect=KeyboardInterrupt),
        ):
            mock_stdin.isatty.return_value = True
            result = prompt.prompt_budget_pause(1.50, 2.00)
        assert result == "stop"

    def test_infinity_input_returns_stop(self):
        prompt = TerminalPausePrompt()
        with (
            patch("sys.stdin") as mock_stdin,
            patch("builtins.input", return_value="inf"),
        ):
            mock_stdin.isatty.return_value = True
            result = prompt.prompt_budget_pause(1.50, 2.00)
        assert result == "stop"

    def test_zero_input_returns_stop(self):
        prompt = TerminalPausePrompt()
        with (
            patch("sys.stdin") as mock_stdin,
            patch("builtins.input", return_value="0"),
        ):
            mock_stdin.isatty.return_value = True
            result = prompt.prompt_budget_pause(1.50, 2.00)
        assert result == "stop"

    def test_empty_input_returns_stop(self):
        prompt = TerminalPausePrompt()
        with (
            patch("sys.stdin") as mock_stdin,
            patch("builtins.input", return_value=""),
        ):
            mock_stdin.isatty.return_value = True
            result = prompt.prompt_budget_pause(1.50, 2.00)
        assert result == "stop"
