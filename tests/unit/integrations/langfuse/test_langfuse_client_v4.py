"""Tests for the Langfuse v4 (Observations API v2) read path.

Fixtures follow the response shape documented at
https://langfuse.com/faq/all/deprecated-api-migration and
https://langfuse.com/docs/api-and-data-platform/features/observations-api :
``{"data": [...observation rows...], "meta": {"cursor": "..."}}``, cost fields
(``totalCost``) returned as strings, flat ``inputUsage``/``outputUsage``/
``totalUsage`` token counts, ``latency`` per row, no trace objects.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from traigent.integrations.langfuse.client import (
    AIOHTTP_AVAILABLE,
    REQUESTS_AVAILABLE,
    LangfuseClient,
)

pytestmark = pytest.mark.skipif(not REQUESTS_AVAILABLE, reason="requests not installed")

ROOT = {
    "id": "obs-root",
    "traceId": "t1",
    "type": "SPAN",
    "name": "workflow",
    "traceName": "my-workflow",
    "parentObservationId": None,
    "isRootObservation": True,
    "sessionId": "sess-1",
    "userId": "user-1",
    "metadata": {"env": "test"},
    "startTime": "2026-10-01T10:00:00.000Z",
    "endTime": "2026-10-01T10:00:03.000Z",
    "latency": 3.0,
    "totalCost": "0",
}
GEN_A = {
    "id": "obs-a",
    "traceId": "t1",
    "type": "GENERATION",
    "name": "grader",
    "parentObservationId": "obs-root",
    "model": "gpt-4o",
    "inputUsage": 100,
    "outputUsage": 50,
    "totalUsage": 150,
    "totalCost": "0.0015",
    "latency": 1.5,
    "metadata": {"langgraph_node": "grader"},
}
GEN_B = {
    "id": "obs-b",
    "traceId": "t1",
    "type": "GENERATION",
    "name": "generator",
    "parentObservationId": "obs-root",
    "inputUsage": 10,
    "outputUsage": 5,
    "totalUsage": 15,
    "totalCost": "0.0005",
    "latency": 0.5,
    "metadata": {"langgraph_node": "generator"},
}


def _resp(payload, status=200):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload
    return r


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    c = LangfuseClient(host="https://cloud.langfuse.com")
    c._sdk_client = None
    return c


def test_trace_metrics_from_two_pages_use_cursor(client):
    pages = [
        _resp({"data": [GEN_A, ROOT], "meta": {"cursor": "c1"}}),
        _resp({"data": [GEN_B], "meta": {}}),
    ]
    with patch("requests.get", side_effect=pages) as get:
        metrics = client.get_trace_metrics("t1")

    assert get.call_count == 2
    first, second = (c.kwargs for c in get.call_args_list)
    assert get.call_args_list[0].args[0].endswith("/api/public/v2/observations")
    assert first["params"]["traceId"] == "t1"
    assert first["params"]["limit"] == "1000"
    assert "usage" in first["params"]["fields"]
    assert "fromStartTime" in first["params"] and "toStartTime" in first["params"]
    assert "cursor" not in first["params"]
    assert second["params"]["cursor"] == "c1"

    assert metrics is not None
    assert metrics.observations_partial is False
    assert len(metrics.observations) == 3
    assert metrics.total_cost == pytest.approx(0.002)
    assert metrics.total_tokens == 165
    assert metrics.total_input_tokens == 110
    assert metrics.total_output_tokens == 55
    assert metrics.total_latency_ms == pytest.approx(5000.0)
    assert metrics.per_agent_costs["grader"] == pytest.approx(0.0015)
    assert metrics.per_agent_tokens["generator"] == 15
    keys = metrics.to_measures_dict(prefix="langfuse_")
    assert keys["langfuse_total_cost"] == pytest.approx(0.002)
    assert "langfuse_grader_latency_ms" in keys


def test_trace_reconstructed_from_root_observation(client):
    with patch("requests.get", return_value=_resp({"data": [GEN_A, ROOT], "meta": {}})):
        trace = client.get_trace("t1")
    assert trace is not None
    assert trace["id"] == "t1"
    assert trace["name"] == "my-workflow"
    assert trace["sessionId"] == "sess-1"
    assert trace["userId"] == "user-1"
    assert trace["metadata"] == {"env": "test"}


def test_empty_trace_returns_none(client):
    with patch("requests.get", return_value=_resp({"data": [], "meta": {}})):
        assert client.get_trace("missing") is None
        assert client.get_trace_metrics("missing") is None


def test_v2_404_falls_back_to_legacy_and_sticks(client):
    legacy_trace = {"id": "t1", "name": "legacy", "observations": []}
    with patch(
        "requests.get",
        side_effect=[_resp({}, 404), _resp(legacy_trace)],
    ) as get:
        assert client.get_trace("t1") == legacy_trace
    assert client._legacy_api is True
    assert get.call_args_list[1].args[0].endswith("/api/public/traces/t1")

    with patch("requests.get", return_value=_resp(legacy_trace)) as get:
        client.get_trace("t1")
    assert get.call_args.args[0].endswith("/api/public/traces/t1")


def test_v2_404_observations_fall_back_to_legacy_page_api(client):
    legacy = {"data": [{"id": "o", "name": "x"}], "meta": {"totalItems": 1}}
    with patch("requests.get", side_effect=[_resp({}, 404), _resp(legacy)]) as get:
        obs = client.get_observations_for_trace("t1")
    assert len(obs) == 1
    assert get.call_args_list[1].args[0].endswith("/api/public/observations")
    assert get.call_args_list[1].kwargs["params"]["page"] == "1"


def test_max_pages_marks_partial(client):
    pages = [
        _resp({"data": [GEN_A], "meta": {"cursor": "c1"}}),
        _resp({"data": [GEN_B], "meta": {"cursor": "c2"}}),
    ]
    with patch("requests.get", side_effect=pages):
        obs = client._get_observations_http("t1", max_pages=2)
    assert [o.id for o in obs] == ["obs-a", "obs-b"]
    assert client._observations_partial_by_trace["t1"] is True


def test_cursor_cycle_stops_with_unique_partial_rows(client):
    pages = [
        _resp({"data": [GEN_A], "meta": {"cursor": "c1"}}),
        _resp({"data": [GEN_A, GEN_B], "meta": {"cursor": "c1"}}),
    ]
    with patch("requests.get", side_effect=pages) as get:
        obs = client._get_observations_http("t1")
    assert get.call_count == 2
    assert [o.id for o in obs] == ["obs-a", "obs-b"]
    assert client._observations_partial_by_trace["t1"] is True


def test_two_distinct_pages_are_not_partial(client):
    pages = [
        _resp({"data": [GEN_A], "meta": {"cursor": "c1"}}),
        _resp({"data": [GEN_B], "meta": {}}),
    ]
    with patch("requests.get", side_effect=pages):
        obs = client._get_observations_http("t1")
    assert [o.id for o in obs] == ["obs-a", "obs-b"]
    assert client._observations_partial_by_trace["t1"] is False


def test_window_computed_once_per_fetch(client):
    pages = [
        _resp({"data": [GEN_A], "meta": {"cursor": "c1"}}),
        _resp({"data": [GEN_B], "meta": {}}),
    ]
    with patch("requests.get", side_effect=pages) as get:
        client._get_observations_http("t1")
    first, second = (c.kwargs["params"] for c in get.call_args_list)
    assert first["fromStartTime"] == second["fromStartTime"]
    assert first["toStartTime"] == second["toStartTime"]


def test_lookback_days_configurable(monkeypatch):
    from datetime import datetime, timedelta

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    c = LangfuseClient(v2_lookback_days=7)
    c._sdk_client = None
    with patch("requests.get", return_value=_resp({"data": [ROOT], "meta": {}})) as get:
        c.get_trace("t1")
    params = get.call_args.kwargs["params"]
    span = datetime.fromisoformat(params["toStartTime"]) - datetime.fromisoformat(
        params["fromStartTime"]
    )
    assert span == timedelta(days=7, hours=1)


def test_later_page_404_returns_partial_rows_without_flip(client):
    pages = [
        _resp({"data": [GEN_A], "meta": {"cursor": "c1"}}),
        _resp({}, 404),
    ]
    with patch("requests.get", side_effect=pages):
        obs = client._get_observations_http("t1")
    assert [o.id for o in obs] == ["obs-a"]
    assert client._observations_partial_by_trace["t1"] is True
    assert client._legacy_api is False


def test_v2_success_then_404_does_not_flip_to_legacy(client):
    with patch("requests.get", return_value=_resp({"data": [ROOT], "meta": {}})):
        client.get_trace("t1")
    assert client._v2_confirmed is True
    with patch("requests.get", return_value=_resp({}, 404)) as get:
        assert client.get_trace("t1") is None
    assert client._legacy_api is False
    assert get.call_count == 1
    assert get.call_args.args[0].endswith("/api/public/v2/observations")


def test_get_trace_requests_io_and_decodes_root_json(client):
    root = {**ROOT, "input": '{"q": "hi"}', "output": "plain text"}
    with patch(
        "requests.get", return_value=_resp({"data": [GEN_A, root], "meta": {}})
    ) as get:
        trace = client.get_trace("t1")
    assert get.call_args.kwargs["params"]["fields"].endswith(",io")
    assert trace["input"] == {"q": "hi"}
    assert trace["output"] == "plain text"


def test_metrics_path_does_not_request_io(client):
    with patch("requests.get", return_value=_resp({"data": [ROOT], "meta": {}})) as get:
        client.get_trace_metrics("t1")
    assert "io" not in get.call_args.kwargs["params"]["fields"].split(",")


def test_legacy_observations_use_limit_100(client):
    legacy = {"data": [{"id": "o", "name": "x"}], "meta": {"totalItems": 1}}
    with patch("requests.get", side_effect=[_resp({}, 404), _resp(legacy)]) as get:
        client.get_observations_for_trace("t1")
    assert get.call_args_list[1].kwargs["params"]["limit"] == "100"


def test_multiple_roots_pick_earliest_deterministically(client):
    late = {
        **ROOT,
        "id": "late",
        "traceName": "late",
        "startTime": "2026-10-01T11:00:00Z",
    }
    early = {
        **ROOT,
        "id": "early",
        "traceName": "early",
        "startTime": "2026-10-01T09:00:00Z",
    }
    with patch(
        "requests.get", return_value=_resp({"data": [late, GEN_A, early], "meta": {}})
    ):
        trace = client.get_trace("t1")
    assert trace["name"] == "early"


def test_no_root_leaves_root_fields_unset(client):
    child = {
        **GEN_A,
        "traceName": "child-name",
        "sessionId": "s",
        "userId": "u",
        "input": "x",
        "output": "y",
    }
    with patch("requests.get", return_value=_resp({"data": [child], "meta": {}})):
        trace = client.get_trace("t1")
    assert trace is not None
    assert trace["name"] is None
    assert trace["sessionId"] is None and trace["userId"] is None
    assert trace["input"] is None and trace["output"] is None
    assert not trace["metadata"]


def test_string_cost_and_flat_usage_parsed(client):
    obs = client._dict_to_observation(GEN_A)
    assert obs.cost == pytest.approx(0.0015)
    assert (obs.input_tokens, obs.output_tokens, obs.total_tokens) == (100, 50, 150)
    assert obs.latency_ms == pytest.approx(1500.0)


@pytest.mark.skipif(not AIOHTTP_AVAILABLE, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_async_two_pages_and_trace_reconstruction(client):
    payloads = [
        {"data": [GEN_A, ROOT], "meta": {"cursor": "c1"}},
        {"data": [GEN_B], "meta": {}},
    ]
    seen: list[dict] = []

    class Resp:
        status = 200

        def __init__(self, payload):
            self.payload = payload

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_a):
            return False

        def raise_for_status(self):
            return None

        async def json(self):
            return self.payload

    class Session:
        def __init__(self, **_k):
            self.i = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_a):
            return False

        def get(self, url, **kwargs):
            assert url.endswith("/api/public/v2/observations")
            seen.append(kwargs["params"])
            self.i += 1
            return Resp(payloads[self.i - 1])

    with patch(
        "traigent.integrations.langfuse.client.aiohttp.ClientSession", new=Session
    ):
        metrics = await client.get_trace_metrics_async("t1")

    assert [("cursor" in p) for p in seen] == [False, True]
    assert seen[1]["cursor"] == "c1"
    assert metrics is not None
    assert metrics.trace_name == "my-workflow"
    assert metrics.total_tokens == 165
    assert metrics.observations_partial is False


def _fake_aiohttp(script):
    """Build a fake ClientSession; ``script`` items are payloads, (status, payload) or exceptions."""

    class Resp:
        def __init__(self, status, payload):
            self.status = status
            self.payload = payload

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_a):
            return False

        def raise_for_status(self):
            return None

        async def json(self):
            return self.payload

    class Session:
        i = 0

        def __init__(self, **_k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_a):
            return False

        def get(self, url, **kwargs):
            item = script[Session.i]
            Session.i += 1
            if isinstance(item, BaseException):
                raise item
            status, payload = item if isinstance(item, tuple) else (200, item)
            return Resp(status, payload)

    return Session


@pytest.mark.skipif(not AIOHTTP_AVAILABLE, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_async_timeout_after_first_page_returns_partial(client):

    session = _fake_aiohttp(
        [{"data": [GEN_A], "meta": {"cursor": "c1"}}, TimeoutError()]
    )
    with patch(
        "traigent.integrations.langfuse.client.aiohttp.ClientSession", new=session
    ):
        obs = await client.get_observations_for_trace_async("t1")
    assert [o.id for o in obs] == ["obs-a"]
    assert client._observations_partial_by_trace["t1"] is True


@pytest.mark.skipif(not AIOHTTP_AVAILABLE, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_async_first_page_timeout_returns_none(client):

    session = _fake_aiohttp([TimeoutError()])
    with patch(
        "traigent.integrations.langfuse.client.aiohttp.ClientSession", new=session
    ):
        assert await client.get_trace_async("t1") is None


@pytest.mark.skipif(not AIOHTTP_AVAILABLE, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_async_later_page_404_partial_no_flip(client):
    session = _fake_aiohttp([{"data": [GEN_A], "meta": {"cursor": "c1"}}, (404, {})])
    with patch(
        "traigent.integrations.langfuse.client.aiohttp.ClientSession", new=session
    ):
        obs = await client.get_observations_for_trace_async("t1")
    assert [o.id for o in obs] == ["obs-a"]
    assert client._legacy_api is False


@pytest.mark.skipif(not AIOHTTP_AVAILABLE, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_async_cursor_cycle_partial(client):
    session = _fake_aiohttp(
        [
            {"data": [GEN_A], "meta": {"cursor": "c1"}},
            {"data": [GEN_A], "meta": {"cursor": "c1"}},
        ]
    )
    with patch(
        "traigent.integrations.langfuse.client.aiohttp.ClientSession", new=session
    ):
        obs = await client.get_observations_for_trace_async("t1")
    assert [o.id for o in obs] == ["obs-a"]
    assert client._observations_partial_by_trace["t1"] is True
