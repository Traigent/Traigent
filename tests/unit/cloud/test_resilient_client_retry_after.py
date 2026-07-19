"""Regression tests for Retry-After honoring in ResilientClient (#1974).

calculate_delay used to gate its Retry-After recovery on the substring
"retry-after" being present in str(error). But resilient_backend_request raises
RetryableError("Rate limited", retry_after_value) where the numeric header
lives on the .retry_after attribute, not in the message — so the parsed value
was silently discarded and the client fell back to its own exponential backoff.
These tests pin that the structured retry_after attribute is now honored.
"""

from traigent.cloud.resilient_client import ResilientClient
from traigent.utils.exceptions import RetryableError


def _no_jitter_client(**kwargs) -> ResilientClient:
    return ResilientClient(jitter_factor=0.0, **kwargs)


def test_structured_retry_after_is_honored():
    client = _no_jitter_client(base_delay=1.0, max_delay=300.0)
    # Message is just "Rate limited"; the header value is on .retry_after.
    err = RetryableError("Rate limited", retry_after=42)

    delay = client.calculate_delay(attempt=0, error=err)

    assert delay == 42.0, "server Retry-After must override the exponential backoff"


def test_structured_retry_after_overrides_doubled_backoff():
    # attempt=3 would otherwise give a large doubled delay; server value wins.
    client = _no_jitter_client(base_delay=1.0, max_delay=300.0)
    err = RetryableError("Rate limited", retry_after=5)

    delay = client.calculate_delay(attempt=3, error=err)

    assert delay == 5.0


def test_retry_after_capped_at_max_delay():
    client = _no_jitter_client(base_delay=1.0, max_delay=30.0)
    err = RetryableError("Rate limited", retry_after=99999)

    delay = client.calculate_delay(attempt=0, error=err)

    assert delay == 30.0


def test_missing_retry_after_falls_back_to_backoff():
    client = _no_jitter_client(base_delay=1.0, max_delay=300.0)
    err = RetryableError("Rate limited")  # retry_after defaults to None

    delay = client.calculate_delay(attempt=0, error=err)

    # base_delay * 2^0 = 1.0, then doubled for rate-limit = 2.0
    assert delay == 2.0


def test_message_embedded_retry_after_still_parsed():
    """Back-compat: header scraped from the message when no structured value."""
    client = _no_jitter_client(base_delay=1.0, max_delay=300.0)
    err = Exception("429 Too Many Requests; Retry-After: 17")

    delay = client.calculate_delay(attempt=0, error=err)

    assert delay == 17.0


def test_unparsable_structured_retry_after_falls_back():
    client = _no_jitter_client(base_delay=1.0, max_delay=300.0)
    err = RetryableError("Rate limited", retry_after="not-a-number")

    delay = client.calculate_delay(attempt=0, error=err)

    # Falls back to doubled exponential backoff rather than raising.
    assert delay == 2.0
