"""Review round 3: malformed next-trial replies, bare 408, CI example, 401 evidence."""

from __future__ import annotations

import json

import pytest

from traigent.cli.auth_commands import KeyCheckState, _key_check_from_response
from traigent.core.session_types import normalize_next_trial_payload


@pytest.mark.parametrize(
    "payload",
    [
        {"has_next": None, "trial": None},
        {"has_next": "false", "trial": {"trial_id": "t1", "configuration": {}}},
        {"has_next": False, "error": "session not found"},
    ],
)
def test_malformed_legacy_next_trial_is_left_for_the_loud_paths(payload) -> None:
    assert normalize_next_trial_payload(payload) is payload
    assert "should_continue" not in payload


def test_wellformed_legacy_false_still_normalises() -> None:
    out = normalize_next_trial_payload({"has_next": False})
    assert out["should_continue"] is False


def test_bare_408_session_create_is_connectivity() -> None:
    from traigent.core.execution_policy_runtime import (
        session_failure_is_connectivity,
    )
    from traigent.core.session_types import (
        SessionCreationFailureDetail,
        SessionCreationFailureReason,
        SessionCreationResult,
    )

    detail = SessionCreationFailureDetail.from_http_response(408, "")
    result = SessionCreationResult.fallback(
        session_id="local_x",
        reason=SessionCreationFailureReason.SESSION_FAILED,
        detail="d",
        failure_response=detail,
    )
    assert session_failure_is_connectivity(result) is True


def test_ci_approval_example_creates_the_token_directory(tmp_path) -> None:
    from traigent.core.ci_approval import _approval_required_message

    msg = _approval_required_message(tmp_path / "nested" / "approval.token")
    assert "mkdir -p" in msg


@pytest.mark.parametrize(
    ("body", "headers"),
    [
        ("<html>Please sign in via SSO</html>", {}),
        ("", {"Server": "corp-proxy", "cf-ray": "abc"}),
    ],
)
def test_proxy_401_is_not_a_verdict_on_the_key(body, headers) -> None:
    assert (
        _key_check_from_response(401, body, headers).state
        is KeyCheckState.INDETERMINATE
    )


def test_traigent_401_is_a_rejection() -> None:
    body = json.dumps({"error": "invalid api key", "error_code": "invalid_key"})
    assert _key_check_from_response(401, body, {}).state is KeyCheckState.REJECTED


@pytest.mark.parametrize(
    "body", ['{"valid": true, "data": null}', '{"valid": "false"}']
)
def test_odd_200_shapes_are_indeterminate(body) -> None:
    assert _key_check_from_response(200, body, {}).state is KeyCheckState.INDETERMINATE
