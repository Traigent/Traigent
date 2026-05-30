"""Tests for the backend-backed GuidancePlanProvider.

Uses a fake sync post_json (and a fake async post) so no live backend is needed.
Asserts correct paths + request serialization, GuidancePlan parsing, the
fail-closed behavior on a missing/garbled response, and the async->sync bridge.
"""

from __future__ import annotations

import pytest

from traigent.generation import (
    BackendGuidanceError,
    BackendGuidanceProvider,
    GuidanceAction,
    GuidanceResultItem,
    GuidanceResultSubmission,
)
from traigent.generation.models import GuidancePlanRequest, PlanKind

_PLAN_JSON = {
    "plan_id": "plan_1",
    "policy_version": "gp-2026.05",
    "plan_kind": "benchmark_guide",
    "items": [
        {"seed_ref": "ex_a_0", "action": "generate_harder", "coarse_priority": "high"}
    ],
    "plan_budget": {"total_generations": 10},
    "plan_token": "gp1.x.y",
    "expires_at": "2026-05-30T00:00:00Z",
}


class _RecordingPost:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, path: str, body: dict) -> dict:
        self.calls.append((path, body))
        return self.response


def test_get_guidance_plan_posts_and_parses() -> None:
    post = _RecordingPost(_PLAN_JSON)
    provider = BackendGuidanceProvider("sess_42", post)
    plan = provider.get_guidance_plan(
        GuidancePlanRequest(plan_kind=PlanKind.BENCHMARK_GUIDE)
    )
    assert post.calls[0][0] == "/api/v1/sessions/sess_42/guidance-plan"
    assert post.calls[0][1]["plan_kind"] == "benchmark_guide"
    assert plan.plan_id == "plan_1"
    assert plan.items[0].action is GuidanceAction.GENERATE_HARDER


def test_get_guidance_plan_fails_closed_on_empty_response() -> None:
    provider = BackendGuidanceProvider("s", _RecordingPost({}))
    with pytest.raises(BackendGuidanceError):
        provider.get_guidance_plan(
            GuidancePlanRequest(plan_kind=PlanKind.PROMPT_REWRITE)
        )


def test_get_guidance_plan_fails_closed_on_malformed_response() -> None:
    provider = BackendGuidanceProvider(
        "s", _RecordingPost({"plan_id": "x", "plan_kind": "bogus"})
    )
    with pytest.raises(BackendGuidanceError):
        provider.get_guidance_plan(
            GuidancePlanRequest(plan_kind=PlanKind.BENCHMARK_GUIDE)
        )


def test_submit_results_posts_content_free_payload() -> None:
    post = _RecordingPost({})
    provider = BackendGuidanceProvider("sess_42", post)
    provider.submit_guidance_results(
        GuidanceResultSubmission(
            plan_id="plan_1",
            plan_token="gp1.x.y",
            results=[
                GuidanceResultItem(
                    "ex_a_0", GuidanceAction.GENERATE_HARDER, 2, ["ex_b_0"]
                )
            ],
        )
    )
    path, body = post.calls[0]
    assert path == "/api/v1/sessions/sess_42/guidance-results"
    assert body["results"][0]["generated_count"] == 2


def test_requires_session_id() -> None:
    with pytest.raises(BackendGuidanceError):
        BackendGuidanceProvider("", _RecordingPost(_PLAN_JSON))


def test_from_async_post_bridges_to_sync() -> None:
    seen: list[tuple[str, dict]] = []

    async def async_post(path: str, body: dict) -> dict:
        seen.append((path, body))
        return _PLAN_JSON

    provider = BackendGuidanceProvider.from_async_post("sess_9", async_post)
    plan = provider.get_guidance_plan(
        GuidancePlanRequest(plan_kind=PlanKind.BENCHMARK_GUIDE)
    )
    assert plan.plan_id == "plan_1"
    assert seen[0][0] == "/api/v1/sessions/sess_9/guidance-plan"
