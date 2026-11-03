"""#2855: a quota 429 surfaces allowlisted structured facts, never backend text."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from tests.unit.cloud.test_quota_refusal_all_paths import _hybrid_ops
from tests.unit.cloud.test_session_operations_validation import FakeClient
from traigent.cloud.client import CloudServiceError, SessionQuotaRefusedError
from traigent.cloud.session_operations import SessionOperations
from traigent.cloud.session_types import SessionCreationFailureDetail

FREE_TEXT = "customer text here: Upgrade your plan or wait"


def _body(**overrides) -> str:
    payload = {
        "error_code": "quota_exceeded",
        "resource_type": "optimization_samples",
        "current_usage": 408,
        "limit": 500,
        "reset_at": "2026-12-01T00:00:00Z",
        "upgrade_url": "/billing",
        "enforcement_behavior": "block",
        "requested_units": 396,
        "message": FREE_TEXT,
    }
    payload.update(overrides)
    return json.dumps({k: v for k, v in payload.items() if v is not ...})


def _detail(**overrides) -> SessionCreationFailureDetail:
    return SessionCreationFailureDetail.from_http_response(429, _body(**overrides))


class TestMessage:
    def test_message_names_resource_limit_usage_requested_and_reset(self):
        err = SessionQuotaRefusedError.from_detail(_detail())
        text = str(err)
        assert "optimization_samples" in text
        assert "408" in text and "500" in text and "396" in text
        assert "2026-12-01 00:00 UTC" in text
        assert "wait until 2026-12-01 00:00 UTC" in text.lower() or (
            "Wait until 2026-12-01 00:00 UTC" in text
        )
        assert "upgrade" in text.lower()
        assert "customer text" not in text

    def test_without_reset_at_keeps_upgrade_remedy_only(self):
        text = str(SessionQuotaRefusedError.from_detail(_detail(reset_at=None)))
        assert "Upgrade your plan" in text
        assert "ait until" not in text
        assert "optimization_samples" in text

    def test_requested_units_absent_is_tolerated(self):
        text = str(SessionQuotaRefusedError.from_detail(_detail(requested_units=...)))
        assert "408" in text and "requested" not in text

    def test_body_without_facts_matches_legacy_message(self):
        detail = SessionCreationFailureDetail.from_http_response(
            429, json.dumps({"error_code": "quota_exceeded", "message": FREE_TEXT})
        )
        text = str(SessionQuotaRefusedError.from_detail(detail))
        assert text.startswith(
            "Plan quota/limit reached (HTTP 429, error_code=quota_exceeded). "
        )
        assert "Upgrade your plan" in text


class TestAllowlisting:
    def test_unknown_resource_type_is_dropped_not_echoed(self):
        text = str(
            SessionQuotaRefusedError.from_detail(_detail(resource_type="evil <b>x</b>"))
        )
        assert "evil" not in text and "resource" not in text

    @pytest.mark.parametrize(
        "bad", ["lots", True, float("inf"), -3, None, {"a": 1}, [1]]
    )
    def test_non_finite_or_non_numeric_numbers_are_dropped(self, bad):
        body = _body(limit=bad, current_usage=bad, requested_units=bad)
        # json.dumps writes inf as Infinity, which json.loads accepts.
        detail = SessionCreationFailureDetail.from_http_response(429, body)
        text = str(SessionQuotaRefusedError.from_detail(detail))
        assert "Infinity" not in text and "lots" not in text
        assert "usage=" not in text and "requested=" not in text
        assert "of limit" not in text and "limit=" not in text

    @pytest.mark.parametrize(
        "bad", ["next tuesday", "<script>", 12345, "2026-13-45T00:00:00Z", "x" * 500]
    )
    def test_bad_reset_at_is_dropped_and_remedy_falls_back(self, bad):
        text = str(SessionQuotaRefusedError.from_detail(_detail(reset_at=bad)))
        assert "script" not in text and "next tuesday" not in text
        assert "ait until" not in text and "Upgrade your plan" in text

    def test_reset_at_is_rerendered_by_sdk_not_echoed(self):
        text = str(
            SessionQuotaRefusedError.from_detail(
                _detail(reset_at="2026-12-01T02:00:00+02:00")
            )
        )
        assert "2026-12-01 00:00 UTC" in text
        assert "+02:00" not in text

    def test_free_text_never_stored_on_error_detail(self):
        err = SessionQuotaRefusedError.from_detail(_detail())
        failure = err.session_creation_failure
        assert failure.message is None and failure.raw_body is None
        assert FREE_TEXT not in repr(failure)
        assert err.quota_facts.resource_type == "optimization_samples"


@pytest.mark.backend_online
class TestEveryRaiser:
    @pytest.mark.asyncio
    async def test_hybrid_create_raises_with_facts(self):
        ops = _hybrid_ops(429, _body())
        with patch("traigent.cloud.session_operations.AIOHTTP_AVAILABLE", True):
            with pytest.raises(SessionQuotaRefusedError) as info:
                await ops.create_hybrid_session("p", {"a": [1]}, {"max_trials": 1})
        assert "optimization_samples" in str(info.value) and "500" in str(info.value)
        assert "customer text" not in str(info.value)

    def test_governed_create_raises_with_facts(self):
        exc = CloudServiceError("boom customer text here")
        exc.session_creation_failure = _detail()
        client = FakeClient()

        async def create_via_api(_request):
            raise exc

        client._create_traigent_session_via_api = create_via_api
        ops = SessionOperations(client)
        with patch(
            "traigent.cloud.session_operations.is_backend_offline", return_value=False
        ):
            with pytest.raises(SessionQuotaRefusedError) as info:
                ops.create_session(
                    "fn", {"a": [1, 2]}, promotion_policy={"min_trials": 1}
                )
        assert "optimization_samples" in str(info.value)
        assert "customer text" not in str(info.value)

    @pytest.mark.asyncio
    async def test_typed_client_create_raises_with_facts(self):
        from traigent.cloud.client import TraigentCloudClient

        response = Mock(status=429)
        response.text = AsyncMock(return_value=_body())
        with pytest.raises(SessionQuotaRefusedError) as info:
            await TraigentCloudClient._raise_session_creation_response_error(response)
        assert "408" in str(info.value) and "2026-12-01 00:00 UTC" in str(info.value)


class TestUntrackedBanner:
    def test_banner_lines_and_remedy_carry_facts_not_text(self):
        from traigent.core.backend_session_manager import (
            _untracked_detail_lines,
            _untracked_remedy_line,
        )
        from traigent.core.session_types import (
            SessionCreationFailureClassification,
            SessionCreationFailureReason,
            SessionCreationResult,
        )

        detail = _detail()
        lines = "\n".join(_untracked_detail_lines(detail))
        assert "resource=optimization_samples" in lines and "500" in lines
        assert "customer text" not in lines
        result = SessionCreationResult(
            session_id="s",
            backend_connected=False,
            failure_reason=SessionCreationFailureReason.SESSION_FAILED,
            failure_response=detail,
        )
        remedy = _untracked_remedy_line(
            result, SessionCreationFailureClassification.QUOTA_EXCEEDED
        )
        assert "Wait until 2026-12-01 00:00 UTC" in remedy


@pytest.mark.parametrize(
    "hostile",
    [
        {"resource_type": []},
        {"resource_type": {}},
        {"resource_type": ["api_calls"], "limit": 5},
        {"reset_at": "0001-01-01T00:00:00+05:00"},
        {"reset_at": "9999-12-31T23:59:59-05:00"},
        {"limit": 10**400},
        {"limit": -(10**400)},
        {"current_usage": 10**400, "limit": 5},
    ],
)
def test_hostile_quota_body_never_crashes_fact_parsing(hostile):
    from traigent.core.session_types import QuotaRefusalFacts

    QuotaRefusalFacts.from_payload(hostile)  # must not raise


def test_huge_number_is_rendered_bounded():
    from traigent.core.session_types import _fmt_quota_number

    assert len(_fmt_quota_number(1e300)) < 20
