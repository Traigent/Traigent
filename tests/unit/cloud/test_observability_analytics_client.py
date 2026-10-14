"""Transport and privacy tests for content-free observability analytics reads."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

# SDK #2033: opt into the connected/backend code paths (see pyproject markers).
pytestmark = pytest.mark.backend_online


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


def _analysis_insights_payload() -> dict[str, object]:
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
            "conforming_trace_count": 8,
            "conformance_rate": 0.8,
            "alternate_trace_count": 2,
            "alternate_rate": 0.2,
            "alternate_variant_count": 1,
            "deviations": [
                {
                    "variant_id": "variant-2",
                    "trace_count": 2,
                    "failed_trace_count": 1,
                    "representative_trace_id": "trace-2",
                    "evidence_trace_ids": ["trace-2"],
                    "share": 0.2,
                }
            ],
            "sample_truncated": False,
            "interpretation": "Descriptive baseline; not a correctness assertion.",
        },
        "recommendations": [
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
                "suggested_action": "Test timeout and retry changes.",
                "measurement": {
                    "comparison": "before_after_cohorts",
                    "metrics": ["error_rate", "latency_ms"],
                    "intervention_context_key": "intervention_id",
                },
            }
        ],
        "limitations": ["Observed structure is not an intended workflow."],
        "generated_at": "2026-07-02T00:00:00Z",
    }


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
                    "status": "success",
                    "metadata": {
                        "secret": "PRIVACY_CANARY_METADATA"  # pragma: allowlist secret
                    },
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
            "status": "completed",
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
async def test_issue_projection_fails_closed_on_unknown_nested_content_fields() -> None:
    from traigent.cloud.analytics_client import AnalyticsClientError

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

    with pytest.raises(AnalyticsClientError) as exc_info:
        await client.get_observability_issue(
            "project-1", "issue-1", occurrences_per_page=10
        )

    assert "PRIVACY_CANARY" not in str(exc_info.value)


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
async def test_cohort_compare_rejects_unknown_response_fields() -> None:
    from traigent.cloud.analytics_client import AnalyticsClientError

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

    with pytest.raises(AnalyticsClientError) as exc_info:
        await client.compare_observability_cohorts(
            "project-1",
            reference=cohort,
            comparison=cohort,
            metrics=["error_rate"],
        )

    assert "PRIVACY_CANARY" not in str(exc_info.value)
    assert http.post.await_args.args[0].endswith("/analysis/cohorts/compare")
    body = http.post.await_args.kwargs["json"]
    assert set(body) == {"reference", "comparison", "metrics"}
    assert body["reference"]["variant_ids"] == []


@pytest.mark.asyncio
async def test_analysis_insights_validates_and_projects_strict_contract() -> None:
    client = _client()
    http = _install_get(client, _analysis_insights_payload())

    result = await client.get_observability_analysis_insights(
        "project/one",
        start_time="2026-07-01T00:00:00Z",
        end_time="2026-07-02T00:00:00Z",
        limit=25,
    )

    assert result["content_included"] is False
    assert result["conformance"]["baseline_variant_id"] == "variant-1"
    assert result["recommendations"][0]["suggested_action"].startswith("Test")
    assert http.get.await_args.args[0].endswith(
        "/projects/project%2Fone/observability/analysis/insights"
    )
    assert http.get.await_args.kwargs["params"]["limit"] == "25"


@pytest.mark.asyncio
async def test_analysis_insights_rejects_missing_nested_keys_and_content_marker() -> (
    None
):
    from traigent.cloud.analytics_client import AnalyticsClientError

    client = _client()
    payload = _analysis_insights_payload()
    payload["content_included"] = True
    _install_get(client, payload)
    with pytest.raises(AnalyticsClientError, match="content_included must be false"):
        await client.get_observability_analysis_insights(
            "project-1",
            start_time="2026-07-01T00:00:00Z",
            end_time="2026-07-02T00:00:00Z",
        )

    payload = _analysis_insights_payload()
    conformance = payload["conformance"]
    assert isinstance(conformance, dict)
    del conformance["interpretation"]
    _install_get(client, payload)
    with pytest.raises(
        AnalyticsClientError, match="missing required key.*interpretation"
    ):
        await client.get_observability_analysis_insights(
            "project-1",
            start_time="2026-07-01T00:00:00Z",
            end_time="2026-07-02T00:00:00Z",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["conformance"].__setitem__(
                "subject", "PRIVACY_CANARY_ALLOWED_KEY_WRONG_NESTING"
            ),
            "unsupported key",
        ),
        (
            lambda payload: payload["conformance"].__setitem__("alternate_rate", 1.1),
            "alternate_rate must be a finite number",
        ),
        (
            lambda payload: payload["recommendations"][0].__setitem__(
                "category", "raw prompt text"
            ),
            "category must be one of",
        ),
        (
            lambda payload: payload["recommendations"][0]["measurement"].__setitem__(
                "metrics", ["error_rate", "error_rate"]
            ),
            "metrics must contain unique values",
        ),
        (
            lambda payload: payload["conformance"]["deviations"][0].__setitem__(
                "evidence_trace_ids", ["trace-2", "trace-2"]
            ),
            "evidence_trace_ids must contain unique values",
        ),
    ],
)
async def test_analysis_insights_fails_closed_on_malformed_aggregate_values(
    mutate,
    message: str,
) -> None:
    from traigent.cloud.analytics_client import AnalyticsClientError

    client = _client()
    payload = _analysis_insights_payload()
    mutate(payload)
    _install_get(client, payload)

    with pytest.raises(AnalyticsClientError, match=message):
        await client.get_observability_analysis_insights(
            "project-1",
            start_time="2026-07-01T00:00:00Z",
            end_time="2026-07-02T00:00:00Z",
        )


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

    with pytest.raises(ValueError, match="between 1 and 100"):
        await client.get_observability_analysis_insights(
            "project-1",
            start_time="2026-07-01T00:00:00Z",
            end_time="2026-07-02T00:00:00Z",
            limit=101,
        )

    client._client.get.assert_not_called()


_TRACE_SEARCH_WINDOW = {
    "start_time": "2026-07-01T00:00:00Z",
    "end_time": "2026-07-02T00:00:00Z",
}


def _install_transport(client) -> list:
    """Route the client through a real httpx stack so the wire query is visible."""
    import httpx

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "success": True,
                "message": "ok",
                "data": {
                    "items": [],
                    "pagination": {
                        "page": 1,
                        "per_page": 50,
                        "total": 0,
                        "has_next": False,
                    },
                },
            },
        )

    client._client = httpx.AsyncClient(
        base_url="http://localhost:5000", transport=httpx.MockTransport(handler)
    )
    return requests


@pytest.mark.asyncio
async def test_trace_search_encodes_new_filters_with_repeated_tag_keys() -> None:
    client = _client()
    requests = _install_transport(client)

    await client.search_observability_traces(
        "project-1",
        **_TRACE_SEARCH_WINDOW,
        model="gpt-4o-mini",
        tags=[" support,billing ", "vip"],
        outcome_kind=" csat ",
        has_outcome=True,
    )

    assert len(requests) == 1
    params = requests[0].url.params
    query = requests[0].url.query.decode()
    # A tag containing a comma travels verbatim as its own repeated key; it is
    # never comma-joined with its siblings (the server would split that).
    assert params.get_list("tags") == ["support,billing", "vip"]
    assert query.count("tags=") == 2
    assert "tags=support%2Cbilling&tags=vip" in query
    assert params["model"] == "gpt-4o-mini"
    assert params["outcome_kind"] == "csat"
    assert params["has_outcome"] == "true"
    assert params["start_time_from"] == "2026-07-01T00:00:00Z"
    assert params["start_time_to"] == "2026-07-02T00:00:00Z"
    for unsupported in ("search", "user_id", "session_id", "q"):
        assert unsupported not in params


@pytest.mark.asyncio
async def test_trace_search_sends_has_outcome_false_and_omits_unset_filters() -> None:
    client = _client()
    requests = _install_transport(client)

    await client.search_observability_traces(
        "project-1", **_TRACE_SEARCH_WINDOW, tags=[], has_outcome=False
    )
    await client.search_observability_traces("project-1", **_TRACE_SEARCH_WINDOW)

    assert len(requests) == 2
    explicit_false, unset = (request.url.params for request in requests)
    assert explicit_false["has_outcome"] == "false"
    for absent in ("model", "tags", "outcome_kind"):
        assert absent not in explicit_false
    for absent in ("model", "tags", "outcome_kind", "has_outcome"):
        assert absent not in unset
    assert set(unset.keys()) == {
        "start_time_from",
        "start_time_to",
        "page",
        "per_page",
    }


@pytest.mark.asyncio
async def test_trace_search_accepts_filter_bounds_and_outcome_vocabulary() -> None:
    from traigent.cloud.analytics_client import OBSERVABILITY_OUTCOME_KINDS

    assert OBSERVABILITY_OUTCOME_KINDS == ("conversion", "resolution", "csat", "custom")
    client = _client()
    requests = _install_transport(client)
    tags = [f"tag-{index}" for index in range(19)] + ["t" * 255]

    await client.search_observability_traces(
        "project-1", **_TRACE_SEARCH_WINDOW, model="m" * 255, tags=tags
    )
    for kind in OBSERVABILITY_OUTCOME_KINDS:
        await client.search_observability_traces(
            "project-1", **_TRACE_SEARCH_WINDOW, outcome_kind=kind, has_outcome=True
        )

    assert requests[0].url.params["model"] == "m" * 255
    assert requests[0].url.params.get_list("tags") == tags
    assert [request.url.params["outcome_kind"] for request in requests[1:]] == list(
        OBSERVABILITY_OUTCOME_KINDS
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("filters", "message"),
    [
        ({"model": "m" * 256}, "model"),
        ({"outcome_kind": "refund"}, "outcome_kind"),
        ({"outcome_kind": "CSAT"}, "outcome_kind"),
        ({"tags": [f"tag-{index}" for index in range(21)]}, "tags"),
        ({"tags": ["t" * 256]}, "tags"),
        ({"tags": ["vip", " vip "]}, "tags"),
        ({"has_outcome": "true"}, "has_outcome"),
        ({"has_outcome": 1}, "has_outcome"),
        (
            {"outcome_kind": "conversion", "has_outcome": False},
            r"(?s)(?=.*outcome_kind)(?=.*has_outcome)",
        ),
    ],
)
async def test_trace_search_rejects_invalid_filters_before_transport(
    filters: dict[str, object], message: str
) -> None:
    client = _client()
    client._client = AsyncMock()

    with pytest.raises(ValueError, match=message):
        await client.search_observability_traces(
            "project-1", **_TRACE_SEARCH_WINDOW, **filters
        )

    client._client.get.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tags",
    ["vip", [1], [["a"]], 5],
    ids=["bare-string", "int-value", "nested-list", "int"],
)
async def test_trace_search_rejects_non_list_tags_before_transport(
    tags: object,
) -> None:
    """A bare string must not be iterated into per-character tags."""
    client = _client()
    client._client = AsyncMock()

    with pytest.raises(ValueError, match="tags"):
        await client.search_observability_traces(
            "project-1", **_TRACE_SEARCH_WINDOW, tags=tags
        )

    client._client.get.assert_not_called()


@pytest.mark.asyncio
async def test_trace_search_sends_single_comma_tag_as_one_key() -> None:
    """A lone tag containing a comma is still exactly one ``tags`` key.

    The server keeps repeated (``getlist``) values verbatim and only splits a
    bare comma string, so one key carrying ``support,billing`` stays one tag.
    """
    client = _client()
    requests = _install_transport(client)

    await client.search_observability_traces(
        "project-1", **_TRACE_SEARCH_WINDOW, tags=["support,billing"]
    )

    assert len(requests) == 1
    assert requests[0].url.params.get_list("tags") == ["support,billing"]
    assert requests[0].url.query.decode().count("tags=") == 1
