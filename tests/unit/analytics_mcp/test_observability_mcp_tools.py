"""Unit coverage for read-only, content-free observability MCP tools."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


def _install_fake_client(monkeypatch, reader: AsyncMock) -> None:
    from traigent.analytics_mcp import tools as tools_mod

    class _FakeClient:
        backend_url = "https://backend.example.com"

        async def __aenter__(self):
            return reader

        async def __aexit__(self, *exc):
            return False

    async def _new_client():
        return _FakeClient()

    monkeypatch.setattr(tools_mod, "_new_analytics_client", _new_client)


def _analysis_insights(*, with_signals: bool = False) -> dict[str, object]:
    return {
        "project_id": "project-1",
        "start_time": "2026-07-01T00:00:00Z",
        "end_time": "2026-07-02T00:00:00Z",
        "content_included": False,
        "conformance": {
            "baseline_type": "observed_dominant_variant",
            "baseline_variant_id": "variant-1",
            "analyzed_trace_count": 10,
            "sampled_trace_count": 10,
            "total_trace_count": 10,
            "analysis_coverage": 1.0,
            "sample_coverage": 1.0,
            "conforming_trace_count": 8 if with_signals else 10,
            "conformance_rate": 0.8 if with_signals else 1.0,
            "alternate_trace_count": 2 if with_signals else 0,
            "alternate_rate": 0.2 if with_signals else 0.0,
            "alternate_variant_count": 1 if with_signals else 0,
            "deviations": (
                [
                    {
                        "variant_id": "variant-2",
                        "trace_count": 2,
                        "failed_trace_count": 1,
                        "representative_trace_id": "trace-2",
                        "evidence_trace_ids": ["trace-2"],
                        "share": 0.2,
                    }
                ]
                if with_signals
                else []
            ),
            "sample_truncated": False,
            "interpretation": "PRIVACY_CANARY_ALLOWED_INTERPRETATION",
        },
        "recommendations": (
            [
                {
                    "id": "recommendation-1",
                    "category": "tool_reliability",
                    "priority": "medium",
                    "confidence": 0.7,
                    "subject": "search.lookup",
                    "evidence": {
                        "normalized_tool_id": "search.lookup",
                        "attempt_count": 10,
                        "failure_count": 2,
                    },
                    "suggested_action": "PRIVACY_CANARY_ALLOWED_SUGGESTED_ACTION",
                    "measurement": {
                        "comparison": "before_after_cohorts",
                        "metrics": ["error_rate"],
                        "intervention_context_key": "intervention_id",
                    },
                }
            ]
            if with_signals
            else []
        ),
        "limitations": ["PRIVACY_CANARY_ALLOWED_LIMITATION"],
        "generated_at": "2026-07-02T00:00:00Z",
    }


@pytest.mark.asyncio
async def test_observability_search_requires_project_and_bounded_window(
    monkeypatch,
) -> None:
    from traigent.analytics_mcp.tools import observability_search_traces_tool

    reader = AsyncMock()
    _install_fake_client(monkeypatch, reader)

    missing = await observability_search_traces_tool(
        "", "2026-07-01T00:00:00Z", "2026-07-02T00:00:00Z"
    )
    too_wide = await observability_search_traces_tool(
        "project-1", "2026-05-01T00:00:00Z", "2026-07-02T00:00:00Z"
    )
    oversized_page = await observability_search_traces_tool(
        "project-1",
        "2026-07-01T00:00:00Z",
        "2026-07-02T00:00:00Z",
        per_page=101,
    )

    assert missing["code"] == "invalid_input"
    assert too_wide["code"] == "invalid_input"
    assert "31 days" in too_wide["message"]
    assert oversized_page["code"] == "invalid_input"
    reader.search_observability_traces.assert_not_called()


@pytest.mark.asyncio
async def test_observability_search_propagates_only_bounded_filters(
    monkeypatch,
) -> None:
    from traigent.analytics_mcp.tools import observability_search_traces_tool

    reader = AsyncMock()
    reader.search_observability_traces.return_value = {"items": [], "total": 0}
    _install_fake_client(monkeypatch, reader)

    result = await observability_search_traces_tool(
        "project-1",
        "2026-07-01T00:00:00Z",
        "2026-07-02T00:00:00Z",
        page=2,
        per_page=25,
        status="failed",
        environment="prod",
        release="release-7",
    )

    assert result["ok"] is True
    reader.search_observability_traces.assert_awaited_once_with(
        "project-1",
        start_time="2026-07-01T00:00:00Z",
        end_time="2026-07-02T00:00:00Z",
        page=2,
        per_page=25,
        status="failed",
        environment="prod",
        release="release-7",
    )


@pytest.mark.asyncio
async def test_issue_tools_validate_and_propagate(monkeypatch) -> None:
    from traigent.analytics_mcp.tools import (
        observability_get_issue_tool,
        observability_list_issues_tool,
    )

    reader = AsyncMock()
    reader.list_observability_issues.return_value = {"items": []}
    reader.get_observability_issue.return_value = {"issue": {"id": "issue-1"}}
    _install_fake_client(monkeypatch, reader)

    invalid = await observability_list_issues_tool(
        "project-1", detector_family="semantic_guess"
    )
    listed = await observability_list_issues_tool(
        "project-1",
        page=2,
        per_page=20,
        state="open",
        detector_family="retry",
        severity="error",
    )
    detail = await observability_get_issue_tool(
        "project-1", "issue-1", occurrence_page=3, occurrences_per_page=10
    )

    assert invalid["code"] == "invalid_input"
    assert listed["ok"] is True
    assert detail["ok"] is True
    reader.list_observability_issues.assert_awaited_once_with(
        "project-1",
        page=2,
        per_page=20,
        state="open",
        detector_family="retry",
        severity="error",
        search=None,
    )
    reader.get_observability_issue.assert_awaited_once_with(
        "project-1",
        "issue-1",
        occurrence_page=3,
        occurrences_per_page=10,
    )


@pytest.mark.asyncio
async def test_mcp_edge_reapplies_content_free_projection(monkeypatch) -> None:
    from traigent.analytics_mcp.tools import observability_get_issue_tool

    reader = AsyncMock()
    reader.get_observability_issue.return_value = {
        "issue": {
            "id": "issue-1",
            "detector_family": "retry",
            "error_text": "PRIVACY_CANARY_ERROR",
        },
        "occurrences": [
            {
                "id": "occ-1",
                "trace_id": "trace-1",
                "input_data": "PRIVACY_CANARY_INPUT",
            }
        ],
    }
    _install_fake_client(monkeypatch, reader)

    result = await observability_get_issue_tool("project-1", "issue-1")

    assert result["ok"] is True
    assert "PRIVACY_CANARY" not in str(result)
    assert result["observability_issue"] == {
        "issue": {"id": "issue-1", "detector_family": "retry"},
        "occurrences": [{"id": "occ-1", "trace_id": "trace-1"}],
    }


@pytest.mark.asyncio
async def test_trace_slice_and_tool_analysis_enforce_bounds(monkeypatch) -> None:
    from traigent.analytics_mcp.tools import (
        observability_get_tool_analysis_tool,
        observability_get_trace_slice_tool,
    )

    reader = AsyncMock()
    reader.get_observability_trace_slice.return_value = {"items": []}
    reader.get_observability_tool_analysis.return_value = {"items": []}
    _install_fake_client(monkeypatch, reader)

    bad_slice = await observability_get_trace_slice_tool(
        "project-1", "trace-1", limit=501
    )
    bad_tools = await observability_get_tool_analysis_tool(
        "project-1",
        "2026-07-01T00:00:00",
        "2026-07-02T00:00:00Z",
    )
    good_slice = await observability_get_trace_slice_tool(
        "project-1", "trace-1", cursor="cursor-1", limit=300
    )

    assert bad_slice["code"] == "invalid_input"
    assert bad_tools["code"] == "invalid_input"
    assert good_slice["ok"] is True
    reader.get_observability_trace_slice.assert_awaited_once_with(
        "project-1", "trace-1", cursor="cursor-1", limit=300
    )


@pytest.mark.asyncio
async def test_analysis_insights_tool_enforces_bounds_and_reprojects(
    monkeypatch,
) -> None:
    from traigent.analytics_mcp.tools import observability_get_analysis_insights_tool

    reader = AsyncMock()
    reader.get_observability_analysis_insights.return_value = _analysis_insights(
        with_signals=True
    )
    _install_fake_client(monkeypatch, reader)

    too_wide = await observability_get_analysis_insights_tool(
        "project-1", "2026-05-01T00:00:00Z", "2026-07-02T00:00:00Z"
    )
    oversized = await observability_get_analysis_insights_tool(
        "project-1",
        "2026-07-01T00:00:00Z",
        "2026-07-02T00:00:00Z",
        limit=101,
    )
    result = await observability_get_analysis_insights_tool(
        "project-1",
        "2026-07-01T00:00:00Z",
        "2026-07-02T00:00:00Z",
        limit=25,
    )

    assert too_wide["code"] == "invalid_input"
    assert oversized["code"] == "invalid_input"
    assert result["ok"] is True
    assert "PRIVACY_CANARY" not in str(result)
    projected = result["observability_analysis_insights"]
    assert "interpretation" not in projected["conformance"]
    assert "suggested_action" not in projected["recommendations"][0]
    assert "limitations" not in projected
    reader.get_observability_analysis_insights.assert_awaited_once_with(
        "project-1",
        start_time="2026-07-01T00:00:00Z",
        end_time="2026-07-02T00:00:00Z",
        limit=25,
    )


@pytest.mark.asyncio
async def test_analysis_insights_tool_rejects_allowed_key_at_wrong_nesting(
    monkeypatch,
) -> None:
    from traigent.analytics_mcp.tools import observability_get_analysis_insights_tool

    reader = AsyncMock()
    payload = _analysis_insights(with_signals=True)
    conformance = payload["conformance"]
    assert isinstance(conformance, dict)
    conformance["subject"] = "PRIVACY_CANARY_ALLOWED_KEY_WRONG_NESTING"
    reader.get_observability_analysis_insights.return_value = payload
    _install_fake_client(monkeypatch, reader)

    result = await observability_get_analysis_insights_tool(
        "project-1",
        "2026-07-01T00:00:00Z",
        "2026-07-02T00:00:00Z",
    )

    assert result["ok"] is False
    assert result["code"] == "malformed_response"
    assert "PRIVACY_CANARY" not in str(result)


@pytest.mark.asyncio
async def test_cohort_compare_rejects_content_and_propagates_closed_shape(
    monkeypatch,
) -> None:
    from traigent.analytics_mcp.tools import observability_compare_cohorts_tool

    reader = AsyncMock()
    reader.compare_observability_cohorts.return_value = {"deltas": []}
    _install_fake_client(monkeypatch, reader)
    base = {
        "start_time": "2026-07-01T00:00:00Z",
        "end_time": "2026-07-02T00:00:00Z",
        "trace_statuses": ["completed"],
        "sample_limit": 100,
    }

    rejected = await observability_compare_cohorts_tool(
        "project-1",
        {**base, "raw_prompt": "PRIVACY_CANARY"},
        base,
        ["error_rate"],
    )
    accepted = await observability_compare_cohorts_tool(
        "project-1",
        base,
        {**base, "execution_context": {"release_id": "release-2"}},
        ["error_rate", "cost_usd"],
    )

    assert rejected["code"] == "invalid_input"
    assert "PRIVACY_CANARY" not in str(rejected)
    assert accepted["ok"] is True
    call = reader.compare_observability_cohorts.await_args.kwargs
    assert call["metrics"] == ["error_rate", "cost_usd"]
    assert call["reference"]["variant_ids"] == []
    assert call["comparison"]["execution_context"] == {
        "schema_version": "1.0",
        "release_id": "release-2",
    }


@pytest.mark.asyncio
async def test_related_changes_calls_lineage_reader(monkeypatch) -> None:
    from traigent.analytics_mcp.tools import observability_get_related_changes_tool

    reader = AsyncMock()
    reader.get_observability_related_changes.return_value = {"links": []}
    _install_fake_client(monkeypatch, reader)

    result = await observability_get_related_changes_tool("project-1", "trace-1")

    assert result["ok"] is True
    reader.get_observability_related_changes.assert_awaited_once_with(
        "project-1", "trace-1"
    )


@pytest.mark.asyncio
async def test_change_brief_composes_projected_evidence_without_causal_claims(
    monkeypatch,
) -> None:
    from traigent.analytics_mcp.tools import observability_build_change_brief_tool

    reader = AsyncMock()
    reader.get_observability_trace_analysis.return_value = {
        "project_id": "project-1",
        "trace_id": "trace-1",
        "analysis_status": "completed",
        "failure_code": None,
        "repeat_groups": [{"id": "repeat-1", "input_data": "PRIVACY_CANARY"}],
        "tool_summaries": [
            {
                "normalized_tool_id": "search.lookup",
                "attempt_count": 2,
                "failure_count": 1,
                "retry_count": 1,
                "fallback_count": 0,
                "raw_error": "PRIVACY_CANARY",
            }
        ],
        "issue_ids": ["issue-1"],
    }
    reader.get_observability_trace_slice.return_value = {
        "project_id": "project-1",
        "trace_id": "trace-1",
        "projection_mode": "content_free",
        "content_included": False,
        "items": [
            {
                "observation_id": "obs-1",
                "normalized_tool_id": "search.lookup",
                "status": "failed",
                "output_data": "PRIVACY_CANARY",
            }
        ],
    }
    reader.get_observability_related_changes.return_value = {
        "project_id": "project-1",
        "trace_id": "trace-1",
        "execution_context": {"release_id": "release-2"},
        "links": [
            {
                "resource_type": "release",
                "resource_id": "release-2",
                "resource_version": "v2",
                "metadata": "PRIVACY_CANARY",
            }
        ],
    }
    reader.get_observability_tool_analysis.return_value = {
        "project_id": "project-1",
        "items": [{"normalized_tool_id": "search.lookup", "failure_count": 1}],
    }
    reader.get_observability_analysis_insights.return_value = _analysis_insights(
        with_signals=True
    )
    _install_fake_client(monkeypatch, reader)

    result = await observability_build_change_brief_tool(
        "project-1",
        "trace-1",
        "2026-07-01T00:00:00Z",
        "2026-07-02T00:00:00Z",
    )

    assert result["ok"] is True
    assert "PRIVACY_CANARY" not in str(result)
    brief = result["observability_change_brief"]
    assert brief["assessment"] == "evidence_only_non_causal"
    assert {item["category"] for item in brief["hypotheses"]} >= {
        "tool_execution_reliability",
        "loop_control",
        "change_attribution",
        "structural_conformance",
    }
    assert any(
        item["action"] == "test_tool_contract_change"
        for item in brief["recommendations"]
    )
    assert any(
        item["action"] == "evaluate_aggregate_recommendation"
        for item in brief["recommendations"]
    )
    reader.get_observability_trace_slice.assert_awaited_once_with(
        "project-1", "trace-1", cursor=None, limit=200
    )
    reader.get_observability_tool_analysis.assert_awaited_once_with(
        "project-1",
        start_time="2026-07-01T00:00:00Z",
        end_time="2026-07-02T00:00:00Z",
        limit=50,
    )
    reader.get_observability_analysis_insights.assert_awaited_once_with(
        "project-1",
        start_time="2026-07-01T00:00:00Z",
        end_time="2026-07-02T00:00:00Z",
        limit=20,
    )


@pytest.mark.asyncio
async def test_change_brief_handles_missing_evidence_and_validates_comparison(
    monkeypatch,
) -> None:
    from traigent.analytics_mcp.tools import observability_build_change_brief_tool

    reader = AsyncMock()
    reader.get_observability_trace_analysis.return_value = {
        "analysis_status": "completed",
        "repeat_groups": [],
        "tool_summaries": [],
        "issue_ids": [],
    }
    reader.get_observability_trace_slice.return_value = {"items": []}
    reader.get_observability_related_changes.return_value = {"links": []}
    reader.get_observability_tool_analysis.return_value = {"items": []}
    reader.get_observability_analysis_insights.return_value = _analysis_insights()
    _install_fake_client(monkeypatch, reader)

    invalid = await observability_build_change_brief_tool(
        "project-1",
        "trace-1",
        "2026-07-01T00:00:00Z",
        "2026-07-02T00:00:00Z",
        reference={
            "start_time": "2026-07-01T00:00:00Z",
            "end_time": "2026-07-02T00:00:00Z",
        },
    )
    assert invalid["code"] == "invalid_input"
    reader.get_observability_trace_analysis.assert_not_called()

    result = await observability_build_change_brief_tool(
        "project-1",
        "trace-1",
        "2026-07-01T00:00:00Z",
        "2026-07-02T00:00:00Z",
    )
    brief = result["observability_change_brief"]
    assert brief["hypotheses"] == [
        {
            "category": "insufficient_evidence",
            "assessment": "unverified",
            "statement": "The available content-free evidence does not identify a concrete failure mechanism.",
            "evidence": {},
        }
    ]
    assert any(
        item["action"] == "improve_trace_instrumentation"
        for item in brief["recommendations"]
    )


@pytest.mark.asyncio
async def test_change_brief_runs_bounded_before_after_comparison(monkeypatch) -> None:
    from traigent.analytics_mcp.tools import observability_build_change_brief_tool

    reader = AsyncMock()
    reader.get_observability_trace_analysis.return_value = {
        "analysis_status": "completed",
        "repeat_groups": [],
        "tool_summaries": [],
        "issue_ids": [],
    }
    reader.get_observability_trace_slice.return_value = {"items": [{"id": "obs-1"}]}
    reader.get_observability_related_changes.return_value = {"links": []}
    reader.get_observability_tool_analysis.return_value = {"items": []}
    reader.get_observability_analysis_insights.return_value = _analysis_insights()
    reader.compare_observability_cohorts.return_value = {
        "matched_pair_count": 12,
        "deltas": [{"metric": "error_rate", "absolute_delta": -0.1}],
    }
    _install_fake_client(monkeypatch, reader)
    reference = {
        "start_time": "2026-07-01T00:00:00Z",
        "end_time": "2026-07-02T00:00:00Z",
        "execution_context": {"release_id": "release-1"},
    }
    comparison = {
        "start_time": "2026-07-02T00:00:00Z",
        "end_time": "2026-07-03T00:00:00Z",
        "execution_context": {"release_id": "release-2"},
    }

    result = await observability_build_change_brief_tool(
        "project-1",
        "trace-1",
        "2026-07-01T00:00:00Z",
        "2026-07-03T00:00:00Z",
        reference=reference,
        comparison=comparison,
    )

    reader.compare_observability_cohorts.assert_awaited_once()
    call = reader.compare_observability_cohorts.await_args.kwargs
    assert call["metrics"] == [
        "error_rate",
        "retry_rate",
        "fallback_rate",
        "latency_ms",
        "cost_usd",
    ]
    brief = result["observability_change_brief"]
    assert brief["evidence"]["cohort_comparison"]["matched_pair_count"] == 12
    assert any(
        item["action"] == "review_measured_cohort_deltas"
        for item in brief["recommendations"]
    )


@pytest.mark.asyncio
async def test_all_observability_tools_are_registered() -> None:
    pytest.importorskip("mcp")
    from traigent.analytics_mcp.server import create_server
    from traigent.analytics_mcp.tools import ANALYTICS_TOOL_NAMES

    expected = {
        "observability_search_traces",
        "observability_list_issues",
        "observability_get_issue",
        "observability_get_trace_slice",
        "observability_get_tool_analysis",
        "observability_get_analysis_insights",
        "observability_compare_cohorts",
        "observability_get_related_changes",
        "observability_build_change_brief",
    }
    assert expected.issubset(ANALYTICS_TOOL_NAMES)
    registered = {tool.name for tool in await create_server().list_tools()}
    assert expected.issubset(registered)
