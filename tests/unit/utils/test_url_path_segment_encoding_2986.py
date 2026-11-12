"""Caller-supplied ids must stay one URL path segment (#2986)."""

from __future__ import annotations

import asyncio
from urllib.parse import quote

import pytest

from traigent.admin.client import EnterpriseAdminClient
from traigent.evaluation.client import EvaluationClient
from traigent.observability.client import ObservabilityClient
from traigent.utils.url_segments import quote_path_segment

pytestmark = pytest.mark.backend_online

HOSTILE_IDS = ["a/b", "a?x=1", "a#frag", "a b", "a/../../projects/p1", "x%y"]


class _Captured(Exception):
    pass


def _capture():
    seen: list[tuple[str, str]] = []

    def sender(method: str, path: str, payload=None):
        seen.append((method, path))
        raise _Captured

    return seen, sender


def _admin(c, i):
    return [
        lambda: c.get_tenant(i),
        lambda: c.update_tenant(i, name="n"),
        lambda: c.list_tenant_memberships(i),
        lambda: c.create_tenant_membership(i, user_id="u"),
        lambda: c.update_tenant_membership("t", i, role="member"),
        lambda: c.update_tenant_membership(i, "m", role="member"),
        lambda: c.get_tenant_sso_config(i),
        lambda: c.upsert_tenant_sso_config(i, provider_type="saml"),
    ]


def _evaluation(c, i):
    return [
        lambda: c.get_evaluator(i),
        lambda: c.update_evaluator(i, name="n"),
        lambda: c.execute_evaluator(
            i, target={"target_type": "observability_trace", "target_id": "t"}
        ),
        lambda: c.backfill_evaluator(i),
        lambda: c.get_evaluator_run(i),
        lambda: c.retry_evaluator_run(i),
        lambda: c.get_annotation_queue(i),
        lambda: c.update_annotation_queue(i, name="n"),
        lambda: c.add_annotation_queue_items(i, targets=[]),
        lambda: c.list_annotation_queue_items(i),
        lambda: c.get_next_annotation_queue_item(i),
        lambda: c.update_annotation_queue_item(i, status="skipped"),
        lambda: c.complete_annotation_queue_item(i, scores=[]),
        lambda: c.update_typed_measure(i, {}),
        lambda: c.preview_evaluation_dataset_example_from_trace(
            i, evaluation_dataset_id="d"
        ),
        lambda: c.create_evaluation_dataset_example_from_trace(i, source_trace_id="t"),
    ]


def _observability(c, i):
    return [
        lambda: c.list_comments(i),
        lambda: c.add_comment(i, "hello", content_mode="record"),
        lambda: c.submit_feedback(i, "up", content_mode="record"),
        lambda: c.update_collaboration(i, is_bookmarked=True),
        lambda: c.get_trace(i),
        lambda: c.get_trace_observations(i),
        lambda: c.get_session(i),
    ]


@pytest.mark.parametrize(
    ("factory", "cases"),
    [
        (EnterpriseAdminClient, _admin),
        (EvaluationClient, _evaluation),
        (ObservabilityClient, _observability),
    ],
    ids=["admin", "evaluation", "observability"],
)
@pytest.mark.parametrize("hostile", HOSTILE_IDS)
def test_hostile_ids_are_encoded_as_one_segment(factory, cases, hostile):
    seen, sender = _capture()
    client = factory(request_sender=sender)
    encoded = quote(hostile, safe="")
    calls = cases(client, hostile)
    for call in calls:
        with pytest.raises(_Captured):
            call()
    assert len(seen) == len(calls)
    for _method, path in seen:
        route = path.split("?", 1)[0]  # list endpoints add a legitimate query
        assert encoded in route
        assert hostile not in path
        assert "#" not in path
        assert "/../" not in path


@pytest.mark.parametrize("dot", [".", ".."])
def test_evaluation_trace_routes_reject_dot_segments(dot):
    seen, sender = _capture()
    client = EvaluationClient(request_sender=sender)
    with pytest.raises(ValueError):
        client.preview_evaluation_dataset_example_from_trace(
            dot, evaluation_dataset_id="d"
        )
    with pytest.raises(ValueError):
        client.create_evaluation_dataset_example_from_trace(dot, source_trace_id="t")
    assert seen == []


@pytest.mark.parametrize("bad", ["", " ", ".", "..", None])
def test_helper_rejects_unsafe_values_without_echo(bad):
    with pytest.raises(ValueError) as exc:
        quote_path_segment(bad, field="trace_id")
    assert "trace_id" in str(exc.value)


def test_helper_encodes_reserved_characters():
    assert quote_path_segment("a/b?c#d e") == "a%2Fb%3Fc%23d%20e"
    assert quote_path_segment("ok-1_x.y") == "ok-1_x.y"


def test_observability_dotdot_id_is_rejected_before_request():
    seen, sender = _capture()
    client = ObservabilityClient(request_sender=sender)
    with pytest.raises(ValueError):
        client.get_trace("..")
    assert seen == []


class _FakeResponse:
    status_code = 404

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return {}


def test_example_insights_ids_are_encoded():
    from traigent.analytics.example_insights import ExampleInsightsClient

    urls: list[str] = []

    class _FakeHttp:
        async def post(self, url, **_kw):
            urls.append(url)
            raise _Captured

        async def get(self, url, **_kw):
            urls.append(url)
            raise _Captured

    client = ExampleInsightsClient(backend_url="http://localhost:5000", api_key="k")
    client._client = _FakeHttp()  # type: ignore[assignment]
    hostile = "a/b?c#d"
    with pytest.raises(_Captured):
        asyncio.run(client.compute_scores(hostile))
    with pytest.raises(_Captured):
        asyncio.run(client.get_example_scores(hostile))
    with pytest.raises(_Captured):
        asyncio.run(client.get_dataset_quality(hostile))
    assert len(urls) == 3
    for url in urls:
        assert "a%2Fb%3Fc%23d" in url
        assert "?" not in url and "#" not in url


def test_langfuse_legacy_trace_id_is_encoded(monkeypatch):
    from traigent.integrations.langfuse import client as lf

    calls: list[str] = []

    def fake_get(url, **_kw):
        calls.append(url)
        resp = _FakeResponse()
        return resp

    monkeypatch.setattr(lf, "preflight_requests", lambda *a, **k: None)
    monkeypatch.setattr(lf.requests, "get", fake_get)
    client = lf.LangfuseClient.__new__(lf.LangfuseClient)
    client.host = "http://langfuse.invalid"
    client.timeout = 1
    client._get_auth_header = lambda: {}  # type: ignore[method-assign]
    assert client._get_trace_http_legacy("a/b?c#d") is None
    assert calls == ["http://langfuse.invalid/api/public/traces/a%2Fb%3Fc%23d"]
