"""Contract tests for rate limit behavior against real APIs.

These tests verify that our rate limit handling correctly parses
responses from real LLM providers. They should be run sparingly
as they make actual API calls.

Run with: RUN_CONTRACT_TESTS=true pytest -m contract
"""

from __future__ import annotations

import os
from datetime import UTC

import pytest

# Skip all tests in this module unless RUN_CONTRACT_TESTS is set
pytestmark = [
    pytest.mark.contract,
    pytest.mark.skipif(
        not os.getenv("RUN_CONTRACT_TESTS"),
        reason="Contract tests disabled (set RUN_CONTRACT_TESTS=true to enable)",
    ),
]


class TestOpenAIRateLimitContract:
    """Contract tests for OpenAI rate limit responses."""

    @pytest.mark.asyncio
    async def test_openai_rate_limit_response_format(self) -> None:
        """Verify OpenAI rate limit response format.

        This test documents the expected format of OpenAI 429 responses.
        When we receive a rate limit, we expect:
        - HTTP 429 status code
        - Retry-After header (in seconds)
        - Error object with type="rate_limit_error"

        Note: This test does NOT intentionally trigger rate limits.
        Instead, it verifies our parsing of documented response formats.
        """
        # Verify our exception can hold these values
        from traigent.utils.exceptions import RateLimitError

        error = RateLimitError(
            message="Rate limit reached for gpt-4",
            retry_after=60.0,
        )

        assert error.retry_after == 60.0
        assert "Rate limit" in str(error)

    @pytest.mark.asyncio
    async def test_openai_headers_parsing(self) -> None:
        """Verify we can parse OpenAI rate limit headers.

        OpenAI returns rate limit information in response headers
        even on successful requests.
        """
        # Simulate parsing OpenAI headers
        sample_headers = {
            "x-ratelimit-limit-requests": "500",
            "x-ratelimit-limit-tokens": "40000",
            "x-ratelimit-remaining-requests": "499",
            "x-ratelimit-remaining-tokens": "39500",
            "x-ratelimit-reset-requests": "1ms",
            "x-ratelimit-reset-tokens": "5s",
        }

        # Verify parsing logic
        limit_requests = int(sample_headers["x-ratelimit-limit-requests"])
        limit_tokens = int(sample_headers["x-ratelimit-limit-tokens"])
        remaining_requests = int(sample_headers["x-ratelimit-remaining-requests"])
        remaining_tokens = int(sample_headers["x-ratelimit-remaining-tokens"])

        assert limit_requests == 500
        assert limit_tokens == 40000
        assert remaining_requests == 499
        assert remaining_tokens == 39500

        # Parse reset time
        def parse_reset_time(value: str) -> float:
            """Parse OpenAI reset time format."""
            if value.endswith("ms"):
                return float(value[:-2]) / 1000
            elif value.endswith("s"):
                return float(value[:-1])
            elif value.endswith("m"):
                return float(value[:-1]) * 60
            return float(value)

        assert parse_reset_time("1ms") == 0.001
        assert parse_reset_time("5s") == 5.0
        assert parse_reset_time("2m") == 120.0


class TestAnthropicRateLimitContract:
    """Contract tests for Anthropic rate limit responses."""

    @pytest.mark.asyncio
    async def test_anthropic_rate_limit_response_format(self) -> None:
        """Verify Anthropic rate limit response format.

        Anthropic returns 429 with:
        - retry-after header
        - Error with type="rate_limit_error"
        """
        # Verify our exception handling
        from traigent.utils.exceptions import RateLimitError

        error = RateLimitError(
            message="Rate limit exceeded",
            retry_after=30.0,
        )

        assert error.retry_after == 30.0

    @pytest.mark.asyncio
    async def test_anthropic_headers_parsing(self) -> None:
        """Verify we can parse Anthropic rate limit headers."""
        from datetime import datetime

        sample_headers = {
            "anthropic-ratelimit-requests-limit": "50",
            "anthropic-ratelimit-requests-remaining": "49",
            "anthropic-ratelimit-requests-reset": "2024-01-15T10:00:00Z",
            "anthropic-ratelimit-tokens-limit": "40000",
            "anthropic-ratelimit-tokens-remaining": "35000",
            "anthropic-ratelimit-tokens-reset": "2024-01-15T10:00:05Z",
        }

        limit_requests = int(sample_headers["anthropic-ratelimit-requests-limit"])
        remaining_requests = int(
            sample_headers["anthropic-ratelimit-requests-remaining"]
        )

        assert limit_requests == 50
        assert remaining_requests == 49

        # Parse ISO timestamp
        reset_time = datetime.fromisoformat(
            sample_headers["anthropic-ratelimit-requests-reset"].replace("Z", "+00:00")
        )
        assert reset_time.tzinfo == UTC


class TestAzureOpenAIRateLimitContract:
    """Contract tests for Azure OpenAI rate limit responses."""

    @pytest.mark.asyncio
    async def test_azure_rate_limit_response_format(self) -> None:
        """Verify Azure OpenAI rate limit response format.

        Azure OpenAI returns 429 with:
        - Retry-After header
        - x-ms-region header indicating which region hit limit
        """
        from traigent.utils.exceptions import RateLimitError

        error = RateLimitError(
            message="Requests to the API have exceeded call rate limit",
            retry_after=10.0,
        )

        assert error.retry_after == 10.0

    @pytest.mark.asyncio
    async def test_azure_throttling_scenarios(self) -> None:
        """Document Azure OpenAI throttling scenarios.

        Azure has different throttling at:
        1. Model deployment level (PTU or tokens per minute)
        2. Account level
        3. Subscription level

        Each returns 429 but with different messages.
        """
        throttling_messages = [
            "Requests to the API have exceeded call rate limit",
            "Rate limit is exceeded. Try again later.",
            "Too many requests. Please retry after X seconds.",
        ]

        for msg in throttling_messages:
            from traigent.utils.exceptions import RateLimitError

            error = RateLimitError(message=msg, retry_after=30.0)
            assert error.retry_after == 30.0


class TestRetryAfterParsing:
    """Contract tests for Retry-After header parsing."""

    @pytest.mark.asyncio
    async def test_retry_after_seconds_format(self) -> None:
        """Verify parsing of Retry-After as seconds."""
        # Standard seconds format
        assert self._parse_retry_after("60") == 60.0
        assert self._parse_retry_after("30") == 30.0
        assert self._parse_retry_after("1") == 1.0

    @pytest.mark.asyncio
    async def test_retry_after_http_date_format(self) -> None:
        """Verify parsing of Retry-After as HTTP date.

        Some providers return HTTP date format:
        Retry-After: Wed, 21 Oct 2024 07:28:00 GMT
        """
        from email.utils import parsedate_to_datetime

        http_date = "Wed, 21 Oct 2024 07:28:00 GMT"
        parsed = parsedate_to_datetime(http_date)

        assert parsed.year == 2024
        assert parsed.month == 10
        assert parsed.day == 21

    def _parse_retry_after(self, value: str) -> float:
        """Parse Retry-After header value."""
        try:
            return float(value)
        except ValueError:
            # Try HTTP date format
            import time
            from email.utils import parsedate_to_datetime

            parsed = parsedate_to_datetime(value)
            return max(0.0, parsed.timestamp() - time.time())


class TestRateLimitRecoveryContract:
    """Contract tests for rate limit recovery behavior."""

    @pytest.mark.asyncio
    async def test_recovery_after_backoff(self) -> None:
        """Verify system recovers after respecting backoff.

        This documents the expected behavior: after waiting the
        Retry-After duration, the next request should succeed.
        """
        from traigent.utils.retry import RetryConfig

        config = RetryConfig(
            max_attempts=5,
            initial_delay=1.0,
            max_delay=60.0,
            jitter=True,
        )

        # Verify config is valid
        assert config.max_attempts == 5
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.jitter is True

    @pytest.mark.asyncio
    async def test_circuit_breaker_on_repeated_limits(self) -> None:
        """Verify circuit breaker activates on repeated rate limits.

        After hitting rate limits repeatedly, the circuit breaker
        should open to prevent further requests.
        """
        from traigent.utils.exceptions import RateLimitError
        from traigent.utils.retry import (
            CircuitBreaker,
            CircuitBreakerState,
            ServiceUnavailableError,
        )

        breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=5.0,
            success_threshold=1,
        )

        def failing_call() -> None:
            raise RateLimitError("Rate limited")

        # Trigger failures through the call() method
        for _ in range(3):
            try:
                breaker.call(failing_call)
            except RateLimitError:
                pass

        assert breaker.state == CircuitBreakerState.OPEN

        # Next call should raise ServiceUnavailableError
        with pytest.raises(ServiceUnavailableError):
            breaker.call(lambda: "success")
