"""#2510 item 2: the local-fallback notice must not tell a user to set
TRAIGENT_REQUIRE_CLOUD=1 when it is already set."""

from __future__ import annotations

import pytest

from traigent.core.execution_policy_runtime import local_fallback_notice
from traigent.core.session_types import SessionCreationFailureReason


@pytest.mark.parametrize(
    "failure_reason", [None, SessionCreationFailureReason.NO_API_KEY]
)
def test_notice_does_not_recommend_an_already_set_require_cloud(
    monkeypatch, failure_reason
) -> None:
    monkeypatch.setenv("TRAIGENT_REQUIRE_CLOUD", "1")
    notice = local_fallback_notice("HTTP 503 during trial submission", failure_reason)
    assert "TRAIGENT_REQUIRE_CLOUD=1 to fail" not in notice
    assert "TRAIGENT_REQUIRE_CLOUD is set" in notice


@pytest.mark.parametrize(
    "failure_reason", [None, SessionCreationFailureReason.NO_API_KEY]
)
def test_notice_still_recommends_require_cloud_when_unset(
    monkeypatch, failure_reason
) -> None:
    monkeypatch.delenv("TRAIGENT_REQUIRE_CLOUD", raising=False)
    notice = local_fallback_notice("backend unavailable", failure_reason)
    assert "TRAIGENT_REQUIRE_CLOUD=1 to fail instead of falling back." in notice
