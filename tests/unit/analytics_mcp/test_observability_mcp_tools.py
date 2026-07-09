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
async def test_observability_search_propagates_only_bounded_filters(monkeypatch) -> None:
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
        "observability_compare_cohorts",
        "observability_get_related_changes",
    }
    assert expected.issubset(ANALYTICS_TOOL_NAMES)
    registered = {tool.name for tool in await create_server().list_tools()}
    assert expected.issubset(registered)
