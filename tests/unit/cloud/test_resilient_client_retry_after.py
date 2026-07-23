"""Regression tests for bounded Retry-After handling."""

import pytest

from traigent.cloud.resilient_client import ResilientClient
from traigent.utils.exceptions import RetryableError


def _client() -> ResilientClient:
    return ResilientClient(base_delay=1.0, max_delay=30.0, jitter_factor=0.0)


def test_structured_retry_after_overrides_rate_limit_backoff():
    delay = _client().calculate_delay(3, RetryableError("Rate limited", retry_after=5))
    assert delay == 5.0


@pytest.mark.parametrize(
    ("retry_after", "expected_delay"),
    [
        (-4, 0.0),
        ("NaN", 2.0),
        ("Inf", 2.0),
        (999999, 30.0),
        ("not-a-number", 2.0),
    ],
)
def test_retry_after_adversarial_values_are_safe(retry_after, expected_delay):
    """Malformed values never produce a negative, NaN, or infinite sleep."""
    delay = _client().calculate_delay(
        0, RetryableError("Rate limited", retry_after=retry_after)
    )
    assert delay == expected_delay
