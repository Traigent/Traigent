"""Rate limit simulation infrastructure for testing.

This module provides tools for simulating rate limit behavior during tests,
including RPM/TPM limits, burst limits, daily quotas, and chaos mode.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RateLimitType(Enum):
    """Types of rate limits that can be simulated."""

    RPM = "rpm"  # Requests per minute
    TPM = "tpm"  # Tokens per minute
    BURST = "burst"  # Burst/concurrent requests
    DAILY = "daily"  # Daily quota
    CHAOS = "chaos"  # Random failures


@dataclass
class RateLimitConfig:
    """Configuration for rate limit simulation.

    Attributes:
        rpm_limit: Maximum requests per minute (None = unlimited)
        tpm_limit: Maximum tokens per minute (None = unlimited)
        burst_limit: Maximum concurrent/burst requests (None = unlimited)
        daily_limit: Maximum requests per day (None = unlimited)
        retry_after: Suggested retry delay in seconds
        error_probability: Probability of random rate limit (0.0-1.0, for chaos mode)
        provider: Provider name for error message formatting
        window_seconds: Time window for RPM/TPM limits (default 60s)
        reset_on_success: Whether to reset failure counters on success
    """

    rpm_limit: int | None = None
    tpm_limit: int | None = None
    burst_limit: int | None = None
    daily_limit: int | None = None
    retry_after: float = 1.0
    error_probability: float = 0.0
    provider: str = "openai"
    window_seconds: float = 60.0
    reset_on_success: bool = False


@dataclass
class RateLimitState:
    """Tracks current state of rate limit counters.

    Attributes:
        request_count: Number of requests in current window
        token_count: Number of tokens in current window
        daily_count: Number of requests today
        window_start: Start time of current window
        day_start: Start time of current day
        consecutive_limits: Number of consecutive rate limits hit
    """

    request_count: int = 0
    token_count: int = 0
    daily_count: int = 0
    window_start: float = field(default_factory=time.monotonic)
    day_start: float = field(default_factory=time.monotonic)
    consecutive_limits: int = 0


@dataclass
class RateLimitResult:
    """Result of a rate limit check.

    Attributes:
        limited: Whether the request was rate limited
        limit_type: Type of rate limit that was triggered
        retry_after: Suggested retry delay in seconds
        message: Human-readable error message
        remaining: Remaining requests/tokens in current window
    """

    limited: bool
    limit_type: RateLimitType | None = None
    retry_after: float | None = None
    message: str = ""
    remaining: int | None = None


class RateLimitSimulator:
    """Simulates rate limit behavior for testing.

    This class provides a configurable rate limit simulator that can
    enforce various types of limits (RPM, TPM, burst, daily) and inject
    random failures for chaos testing.

    Example:
        ```python
        config = RateLimitConfig(rpm_limit=10, retry_after=1.0)
        simulator = RateLimitSimulator(config)

        # Check if request should be rate limited
        result = simulator.check_rate_limit(token_count=100)
        if result.limited:
            raise RateLimitError(result.message, retry_after=result.retry_after)

        # Record successful request
        simulator.record_request(token_count=100)
        ```
    """

    def __init__(
        self,
        config: RateLimitConfig | None = None,
        random_seed: int | None = None,
    ):
        """Initialize the rate limit simulator.

        Args:
            config: Rate limit configuration
            random_seed: Seed for random number generator (for reproducibility)
        """
        self.config = config or RateLimitConfig()
        self._state = RateLimitState()
        self._random = random.Random(random_seed)

    @property
    def request_count(self) -> int:
        """Current request count in window."""
        return self._state.request_count

    @property
    def token_count(self) -> int:
        """Current token count in window."""
        return self._state.token_count

    @property
    def daily_count(self) -> int:
        """Current daily request count."""
        return self._state.daily_count

    def check_rate_limit(self, token_count: int = 0) -> RateLimitResult:
        """Check if a request should be rate limited.

        Args:
            token_count: Number of tokens for this request

        Returns:
            RateLimitResult indicating whether request is limited
        """
        # Check if window needs reset
        self._maybe_reset_window()

        # Check chaos mode first (random failures)
        if self.config.error_probability > 0:
            if self._random.random() < self.config.error_probability:
                self._state.consecutive_limits += 1
                return RateLimitResult(
                    limited=True,
                    limit_type=RateLimitType.CHAOS,
                    retry_after=self.config.retry_after,
                    message=self._format_error_message(RateLimitType.CHAOS),
                )

        # Check RPM limit
        if self.config.rpm_limit is not None:
            if self._state.request_count >= self.config.rpm_limit:
                self._state.consecutive_limits += 1
                remaining = 0
                return RateLimitResult(
                    limited=True,
                    limit_type=RateLimitType.RPM,
                    retry_after=self._calculate_retry_after(RateLimitType.RPM),
                    message=self._format_error_message(RateLimitType.RPM),
                    remaining=remaining,
                )

        # Check TPM limit
        if self.config.tpm_limit is not None:
            projected_tokens = self._state.token_count + token_count
            if projected_tokens > self.config.tpm_limit:
                self._state.consecutive_limits += 1
                remaining = max(0, self.config.tpm_limit - self._state.token_count)
                return RateLimitResult(
                    limited=True,
                    limit_type=RateLimitType.TPM,
                    retry_after=self._calculate_retry_after(RateLimitType.TPM),
                    message=self._format_error_message(RateLimitType.TPM),
                    remaining=remaining,
                )

        # Check burst limit
        if self.config.burst_limit is not None:
            if self._state.request_count >= self.config.burst_limit:
                self._state.consecutive_limits += 1
                return RateLimitResult(
                    limited=True,
                    limit_type=RateLimitType.BURST,
                    retry_after=self.config.retry_after,
                    message=self._format_error_message(RateLimitType.BURST),
                    remaining=0,
                )

        # Check daily limit
        if self.config.daily_limit is not None:
            if self._state.daily_count >= self.config.daily_limit:
                self._state.consecutive_limits += 1
                return RateLimitResult(
                    limited=True,
                    limit_type=RateLimitType.DAILY,
                    retry_after=self._calculate_retry_after(RateLimitType.DAILY),
                    message=self._format_error_message(RateLimitType.DAILY),
                    remaining=0,
                )

        # Not rate limited - calculate remaining
        remaining = None
        if self.config.rpm_limit is not None:
            remaining = self.config.rpm_limit - self._state.request_count - 1

        return RateLimitResult(
            limited=False,
            limit_type=None,
            retry_after=None,
            message="",
            remaining=remaining,
        )

    def record_request(self, token_count: int = 0) -> None:
        """Record a successful request.

        Args:
            token_count: Number of tokens used in this request
        """
        self._state.request_count += 1
        self._state.token_count += token_count
        self._state.daily_count += 1

        if self.config.reset_on_success:
            self._state.consecutive_limits = 0

    def reset_window(self) -> None:
        """Manually reset the rate limit window."""
        self._state.request_count = 0
        self._state.token_count = 0
        self._state.window_start = time.monotonic()

    def reset_daily(self) -> None:
        """Manually reset the daily counter."""
        self._state.daily_count = 0
        self._state.day_start = time.monotonic()

    def reset_all(self) -> None:
        """Reset all counters and state."""
        self._state = RateLimitState()

    def get_stats(self) -> dict[str, Any]:
        """Get current rate limit statistics.

        Returns:
            Dictionary with current counter values and limits
        """
        return {
            "request_count": self._state.request_count,
            "token_count": self._state.token_count,
            "daily_count": self._state.daily_count,
            "consecutive_limits": self._state.consecutive_limits,
            "rpm_limit": self.config.rpm_limit,
            "tpm_limit": self.config.tpm_limit,
            "burst_limit": self.config.burst_limit,
            "daily_limit": self.config.daily_limit,
            "window_elapsed": time.monotonic() - self._state.window_start,
        }

    def _maybe_reset_window(self) -> None:
        """Reset window counters if window has elapsed."""
        elapsed = time.monotonic() - self._state.window_start
        if elapsed >= self.config.window_seconds:
            self._state.request_count = 0
            self._state.token_count = 0
            self._state.window_start = time.monotonic()

    def _calculate_retry_after(self, limit_type: RateLimitType) -> float:
        """Calculate appropriate retry delay based on limit type.

        Args:
            limit_type: Type of rate limit that was triggered

        Returns:
            Retry delay in seconds
        """
        base_retry = self.config.retry_after

        if limit_type == RateLimitType.DAILY:
            # For daily limits, suggest waiting longer
            return max(base_retry, 60.0)

        if limit_type in (RateLimitType.RPM, RateLimitType.TPM):
            # Calculate time until window resets
            elapsed = time.monotonic() - self._state.window_start
            remaining = self.config.window_seconds - elapsed
            return max(base_retry, remaining)

        return base_retry

    def _format_error_message(self, limit_type: RateLimitType) -> str:
        """Format error message for rate limit type.

        Args:
            limit_type: Type of rate limit that was triggered

        Returns:
            Human-readable error message
        """
        provider = self.config.provider.capitalize()

        messages = {
            RateLimitType.RPM: (
                f"{provider} API rate limit exceeded. "
                f"Limit: {self.config.rpm_limit} requests per minute."
            ),
            RateLimitType.TPM: (
                f"{provider} API token limit exceeded. "
                f"Limit: {self.config.tpm_limit} tokens per minute."
            ),
            RateLimitType.BURST: (
                f"{provider} API burst limit exceeded. "
                f"Limit: {self.config.burst_limit} concurrent requests."
            ),
            RateLimitType.DAILY: (
                f"{provider} API daily quota exceeded. "
                f"Limit: {self.config.daily_limit} requests per day."
            ),
            RateLimitType.CHAOS: (
                f"{provider} API temporarily unavailable. Please retry."
            ),
        }

        return messages.get(limit_type, "Rate limit exceeded.")


class ProviderRateLimitSimulator(RateLimitSimulator):
    """Provider-specific rate limit simulator with realistic defaults.

    This class provides pre-configured rate limit settings that match
    real-world provider limits for more realistic testing.
    """

    # Default limits by provider (conservative estimates)
    PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
        "openai": {
            "rpm_limit": 60,
            "tpm_limit": 90000,
            "retry_after": 1.0,
        },
        "anthropic": {
            "rpm_limit": 60,
            "tpm_limit": 100000,
            "retry_after": 1.0,
        },
        "gemini": {
            "rpm_limit": 60,
            "tpm_limit": 120000,
            "retry_after": 1.0,
        },
        "cohere": {
            "rpm_limit": 100,
            "tpm_limit": 100000,
            "retry_after": 1.0,
        },
    }

    def __init__(
        self,
        provider: str = "openai",
        config_overrides: dict[str, Any] | None = None,
        random_seed: int | None = None,
    ):
        """Initialize provider-specific rate limit simulator.

        Args:
            provider: Provider name (openai, anthropic, gemini, cohere)
            config_overrides: Override specific config values
            random_seed: Seed for random number generator
        """
        defaults = self.PROVIDER_DEFAULTS.get(provider, {})
        config_dict = {**defaults, "provider": provider}

        if config_overrides:
            config_dict.update(config_overrides)

        config = RateLimitConfig(**config_dict)
        super().__init__(config, random_seed)


class SequentialRateLimiter:
    """Rate limiter that fails on specific call numbers.

    Useful for testing deterministic retry behavior where you want
    to fail on specific calls (e.g., fail on call 3, succeed on call 4).

    Example:
        ```python
        limiter = SequentialRateLimiter(
            fail_on_calls=[2, 3],  # Fail on 2nd and 3rd calls
            retry_after=0.5,
        )

        for i in range(5):
            result = limiter.check_rate_limit()
            if result.limited:
                print(f"Call {i+1} rate limited")
            else:
                limiter.record_request()
                print(f"Call {i+1} succeeded")
        ```
    """

    def __init__(
        self,
        fail_on_calls: list[int] | None = None,
        retry_after: float = 1.0,
        provider: str = "openai",
    ):
        """Initialize sequential rate limiter.

        Args:
            fail_on_calls: List of 1-indexed call numbers to fail
            retry_after: Retry delay in seconds
            provider: Provider name for error messages
        """
        self.fail_on_calls = set(fail_on_calls or [])
        self.retry_after = retry_after
        self.provider = provider
        self._call_count = 0

    @property
    def call_count(self) -> int:
        """Number of calls made so far."""
        return self._call_count

    def check_rate_limit(self, token_count: int = 0) -> RateLimitResult:
        """Check if this call should be rate limited.

        Args:
            token_count: Ignored for sequential limiter

        Returns:
            RateLimitResult indicating whether call is limited
        """
        next_call = self._call_count + 1

        if next_call in self.fail_on_calls:
            return RateLimitResult(
                limited=True,
                limit_type=RateLimitType.RPM,
                retry_after=self.retry_after,
                message=f"{self.provider.capitalize()} API rate limit exceeded.",
            )

        return RateLimitResult(limited=False)

    def record_request(self, token_count: int = 0) -> None:
        """Record a request (successful or failed).

        Args:
            token_count: Ignored for sequential limiter
        """
        self._call_count += 1

    def reset(self) -> None:
        """Reset the call counter."""
        self._call_count = 0
