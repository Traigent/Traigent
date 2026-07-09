"""Transport and privacy tests for content-free observability analytics reads."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


def _client():
    from traigent.cloud.analytics_client import BackendAnalyticsClient

    return BackendAnalyticsClient(
        backend_url="http://localhost:5000",
        api_key="uk_test_key",  # pragma: allowlist secret
    )


def _install_get(client, data: object) -> AsyncMock:
    response = MagicMock()
    response.json.return_value = {"success": True, "message": "ok", "data": data}
    response.raise_for_status = MagicMock()
    http = AsyncMock()
    http.get.return_value = response
    client._client = http
    return http


@pytest.mark.asyncio
async def test_trace_search_uses_canonical_project_route_and_strips_content() -> None:
    client = _client()
    http = _install_get(
        client,
        {
            "items": [
                {
                    "id": "trace-1",
                    "name": "PRIVACY_CANARY_NAME",
                    "status": "failed",
                    "metadata": {"secret": "PRIVACY_CANARY_METADATA"},
                    "input_data": "PRIVACY_CANARY_INPUT",
                    "output_data": "PRIVACY_CANARY_OUTPUT",
                    "started_at": "2026-07-01T12:00:00Z",
                    "observation_count": 4,
                    "total_tokens": 12,
                }
            ],
            "pagination": {
                "page": 1,
                "per_page": 10,
                "total": 1,
                "has_next": False,
            },
        },
    )

    result = await client.search_observability_traces(
        "project/one",
        start_time="2026-07-01T00:00:00Z",
        end_time="2026-07-02T00:00:00Z",
        per_page=10,
    )

    assert result["items"] == [
        {
            "id": "trace-1",
            "status": "failed",
            "started_at": "2026-07-01T12:00:00Z",
            "observation_count": 4,
            "total_tokens": 12,
        }
    ]
    assert "PRIVACY_CANARY" not in str(result)
    assert http.get.await_args.args[0] == (
        "/api/v1beta/projects/project%2Fone/observability/traces"
    )
    assert http.get.await_args.kwargs["params"]["per_page"] == "10"


@pytest.mark.asyncio
async def test_issue_projection_drops_unknown_nested_content_fields() -> None:
    client = _client()
    _install_get(
        client,
        {
            "issue": {
                "id": "issue-1",
                "project_id": "project-1",
                "detector_family": "explicit_error",
                "error_text": "PRIVACY_CANARY_ERROR",
            },
            "occurrences": [
                {
                    "id": "occ-1",
                    "trace_id": "trace-1",
                    "evidence": [
                        {
                            "evidence_type": "explicit_error",
                            "observation_id": "obs-1",
                            "copied_input": "PRIVACY_CANARY_PROMPT",
                        }
                    ],
                }
            ],
            "occurrence_page": 1,
            "occurrences_per_page": 10,
            "total_occurrences": 1,
            "variant_ids": [],
            "generated_at": "2026-07-02T00:00:00Z",
        },
    )

    result = await client.get_observability_issue(
        "project-1", "issue-1", occurrences_per_page=10
    )

    assert result["issue"] == {
        "id": "issue-1",
        "project_id": "project-1",
        "detector_family": "explicit_error",
    }
    assert result["occurrences"][0]["evidence"] == [
        {"evidence_type": "explicit_error", "observation_id": "obs-1"}
    ]
    assert "PRIVACY_CANARY" not in str(result)


@pytest.mark.asyncio
async def test_trace_slice_requires_server_content_free_markers() -> None:
    from traigent.cloud.analytics_client import AnalyticsClientError

    client = _client()
    _install_get(
        client,
        {
            "project_id": "project-1",
            "trace_id": "trace-1",
            "projection_mode": "raw",
            "content_included": True,
            "items": [],
            "next_cursor": None,
            "has_more": False,
            "generated_at": "2026-07-02T00:00:00Z",
        },
    )

    with pytest.raises(AnalyticsClientError, match="content-free markers"):
        await client.get_observability_trace_slice("project-1", "trace-1")


@pytest.mark.asyncio
async def test_cohort_compare_posts_sanitized_closed_contract() -> None:
    client = _client()
    response = MagicMock()
    response.json.return_value = {
        "success": True,
        "message": "ok",
        "data": {
            "project_id": "project-1",
            "reference": {"trace_count": 10, "metrics": []},
            "comparison": {"trace_count": 9, "metrics": []},
            "matched_pair_count": 8,
            "deltas": [],
            "generated_at": "2026-07-02T00:00:00Z",
            "raw_examples": ["PRIVACY_CANARY"],
        },
    }
    response.raise_for_status = MagicMock()
    http = AsyncMock()
    http.post.return_value = response
    client._client = http
    cohort = {
        "start_time": "2026-07-01T00:00:00Z",
        "end_time": "2026-07-02T00:00:00Z",
        "trace_statuses": ["completed"],
        "sample_limit": 100,
    }

    result = await client.compare_observability_cohorts(
        "project-1",
        reference=cohort,
        comparison=cohort,
        metrics=["error_rate"],
    )

    assert "PRIVACY_CANARY" not in str(result)
    assert http.post.await_args.args[0].endswith("/analysis/cohorts/compare")
    body = http.post.await_args.kwargs["json"]
    assert set(body) == {"reference", "comparison", "metrics"}
    assert body["reference"]["variant_ids"] == []


@pytest.mark.asyncio
async def test_client_rejects_unbounded_windows_before_transport() -> None:
    client = _client()
    client._client = AsyncMock()

    with pytest.raises(ValueError, match="31 days"):
        await client.get_observability_tool_analysis(
            "project-1",
            start_time="2026-05-01T00:00:00Z",
            end_time="2026-07-01T00:00:00Z",
        )

    client._client.get.assert_not_called()
