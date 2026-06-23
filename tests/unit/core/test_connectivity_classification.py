"""A 4xx client error must not be treated as a transient connectivity failure.

Regression (found in a live demo): a trial-result submission rejected with
HTTP 400 (e.g. an unknown objective metric like ``cost_usd``) was wrapped in
``CloudBrainUnavailableError`` and classified as connectivity, silently
degrading the run to local-only and leaving a PENDING experiment on the
backend. A permanent 4xx is now surfaced loudly (no fallback); transient
5xx/network still fall back.
"""

import pytest

from traigent.core.execution_policy_runtime import (
    CloudBrainUnavailableError,
    exception_is_connectivity,
)


@pytest.mark.parametrize(
    "exc,expected",
    [
        # permanent client errors -> NOT connectivity (surface loudly)
        (
            CloudBrainUnavailableError(
                "next-trial",
                "Failed to submit trial result: HTTP 400 - Invalid request data",
            ),
            False,
        ),
        (CloudBrainUnavailableError("next-trial", "HTTP 404 not found"), False),
        (CloudBrainUnavailableError("next-trial", "HTTP 422 unprocessable"), False),
        (CloudBrainUnavailableError("next-trial", "HTTP 429 too many requests"), False),
        # transient -> connectivity (allow local fallback)
        (
            CloudBrainUnavailableError(
                "session-create", "backend unavailable: HTTP 503"
            ),
            True,
        ),
        (CloudBrainUnavailableError("next-trial", "HTTP 500 internal error"), True),
        (CloudBrainUnavailableError("session-create", "connection refused"), True),
        (ConnectionError("connection refused"), True),
        (TimeoutError("timed out"), True),
    ],
)
def test_connectivity_classification(exc, expected):
    assert exception_is_connectivity(exc) is expected
