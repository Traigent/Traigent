"""#2552 round 6: non-quota session-create errors show status and code only."""

from __future__ import annotations

import logging
from unittest.mock import Mock

import pytest

from traigent.cloud.api_operations import ApiOperations
from traigent.cloud.client import CloudServiceError
from traigent.core.backend_session_manager import (
    _classify_session_creation_failure,
    _format_untracked_warning_block,
)
from traigent.core.execution_policy_runtime import fallback_reason_from_session_result
from traigent.core.session_types import (
    SessionCreationFailureClassification,
    SessionCreationFailureDetail,
    SessionCreationFailureReason,
    SessionCreationResult,
)

CANARY = "customer text here"
BODIES = [
    f'{{"error_code": "bad_request", "message": "{CANARY}"}}',
    f'{{"error_code": "made_up_{CANARY.replace(" ", "_")}", "message": "{CANARY}"}}',
    CANARY,
]


def _clean(text: str) -> None:
    assert "customer" not in text.lower()


def _raise(status: int, body: str) -> CloudServiceError:
    ops = ApiOperations(Mock())
    with pytest.raises(CloudServiceError) as info:
        ops._handle_session_error(status, body)
    return info.value


@pytest.mark.parametrize("status", [400, 500])
@pytest.mark.parametrize("body", BODIES)
def test_no_backend_text_on_any_surface(status, body, caplog):
    with caplog.at_level(logging.DEBUG):
        exc = _raise(status, body)
    _clean(str(exc))
    _clean(" ".join(r.getMessage() for r in caplog.records))

    detail = exc.session_creation_failure
    assert isinstance(detail, SessionCreationFailureDetail)
    summary = detail.one_line_summary()
    _clean(summary)
    assert summary.startswith(f"HTTP {status}")

    result = SessionCreationResult.fallback(
        session_id="local",
        reason=SessionCreationFailureReason.SESSION_FAILED,
        detail=summary,
        failure_response=detail,
        degraded=True,
    )
    _clean(fallback_reason_from_session_result(result))
    classification = _classify_session_creation_failure(
        SessionCreationFailureReason.SESSION_FAILED, detail, failure_detail=summary
    )
    _clean(_format_untracked_warning_block(result, classification, aborting=False))


def test_auth_error_message_has_no_backend_text():
    from traigent.cloud.client import AuthenticationError

    ops = ApiOperations(Mock())
    with pytest.raises(AuthenticationError) as info:
        ops._handle_session_error(403, BODIES[0])
    _clean(str(info.value))


def test_503_still_classified_as_connectivity():
    exc = _raise(503, BODIES[0])
    detail = exc.session_creation_failure
    assert (
        _classify_session_creation_failure(
            SessionCreationFailureReason.SESSION_FAILED, detail
        )
        is SessionCreationFailureClassification.BACKEND_UNREACHABLE
    )


def test_400_classification_unchanged():
    exc = _raise(400, BODIES[0])
    assert (
        _classify_session_creation_failure(
            SessionCreationFailureReason.SESSION_FAILED, exc.session_creation_failure
        )
        is SessionCreationFailureClassification.UNKNOWN
    )
