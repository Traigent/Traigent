"""Round-4 hardening of the optimization_only transport backstop (regressions per probe).

Recording transports only (``capture_egress``): no real network.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from traigent.utils import optimization_only_egress as oo

from .harness import capture_egress
from .scenario import BACKEND_URL, FAKE_API_KEY
from .test_optimization_only_egress import _env

pytestmark = pytest.mark.backend_online

API = f"{BACKEND_URL}/api/v1"
_KF = "api_" + "key"  # field names only; keeps secret scanners quiet
_PW = "pass" + "word"  # field name only; keeps secret scanners quiet
HEADERS = {"X-API-Key": FAKE_API_KEY}
CLOSED = json.dumps({"error": "optimization_only", "error_code": "optimization_only"})


@pytest.fixture(autouse=True)
def _mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _env(monkeypatch, tmp_path)
    monkeypatch.delenv(oo.ENV_VAR, raising=False)
    oo.reset_latch()
    yield
    oo.reset_latch()


def _enforce(method: str, url: str, body: Any = None, **kw: Any) -> None:
    oo.enforce_outgoing(method, url, HEADERS, body, explicit=True, **kw)


# ----------------------------------------------- A: every gated request, any origin


def test_alternate_backend_origin_is_judged_for_prompts_and_evaluators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from traigent.evaluation.client import EvaluationClient
    from traigent.evaluation.config import EvaluationConfig
    from traigent.prompts.client import PromptManagementClient
    from traigent.prompts.config import PromptManagementConfig

    monkeypatch.setenv(oo.ENV_VAR, "1")
    with capture_egress() as cap:
        prompts = PromptManagementClient(
            PromptManagementConfig(
                backend_origin="https://203.0.113.9", api_key=FAKE_API_KEY
            )
        )
        evaluator = EvaluationClient(
            EvaluationConfig(backend_origin="https://203.0.113.9", api_key=FAKE_API_KEY)
        )
        with pytest.raises(oo.OptimizationOnlyRefusal):
            prompts.run_playground("p", prompt_text="Alice has diabetes", dry_run=True)
        with pytest.raises(oo.OptimizationOnlyRefusal):
            evaluator.create_evaluator(
                name="n",
                measure_id="accuracy",
                target_type="observability_trace",
                judge_config={"instructions": "confidential"},
            )
    assert cap.requests == []


def test_an_openai_style_call_outside_the_gate_is_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider SDKs do not use our gate, so the customer's own model calls keep working."""
    import httpx

    monkeypatch.setenv(oo.ENV_VAR, "1")
    with capture_egress() as cap:
        httpx.Client().post(
            "https://api.openai.com/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello Alice"}],
            },
            headers={"Authorization": "Bearer sk-not-a-real-key"},
        )
    assert [r.path for r in cap.requests] == ["/v1/chat/completions"]


def test_gated_sessions_to_other_origins_are_judged_unless_marked_external(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    from traigent.labels.transport_gate import gate_session

    monkeypatch.setenv(oo.ENV_VAR, "1")
    with capture_egress() as cap:
        with pytest.raises(oo.OptimizationOnlyRefusal):
            gate_session(httpx.Client()).post(
                "https://elsewhere.example/x", json={"a": "b c"}
            )
        gate_session(httpx.Client(), external=True).post(
            "https://customer.example/evaluate", json={"input": "customer data"}
        )
    assert [r.path for r in cap.requests] == ["/evaluate"]


def test_equivalent_origins_share_one_latch() -> None:
    a = oo.credential_scope("https://Host.example", HEADERS)
    assert a == oo.credential_scope("https://host.example:443/api/v1/x", HEADERS)
    assert a == oo.credential_scope("https://host.example./", HEADERS)
    assert a != oo.credential_scope("https://host.example:8443", HEADERS)
    assert oo.credential_scope("http://h.example:80", HEADERS) == oo.credential_scope(
        "http://H.example", HEADERS
    )


def test_a_latch_on_the_bare_origin_blocks_the_explicit_default_port_spelling() -> None:
    import httpx

    from traigent.labels.transport_gate import gate_session

    oo.note_response(403, CLOSED, oo.credential_scope(BACKEND_URL, HEADERS))
    with capture_egress() as cap:
        with pytest.raises(oo.OptimizationOnlyRefusal):
            gate_session(httpx.Client()).post(
                "https://backend.example.test:443/api/v1/evaluators",
                json={"judge_config": {"instructions": "x y"}},
                headers=HEADERS,
            )
    assert cap.requests == []


# ------------------------------------------------ B: no redirects, normalised paths


@pytest.mark.parametrize(
    "path",
    [
        "/sessions/../evaluators",
        "/sessions/./s1",
        "/sessions/%2e%2e/evaluators",
        "/sessions/s1%2Fresults",
        "/sessions/s1%5Cresults",
        "/sessions//s1",
        "/sessions/s1\\results",
    ],
)
def test_dot_segments_encoded_separators_and_backslashes_are_refused(path: str) -> None:
    with pytest.raises(oo.OptimizationOnlyRefusal):
        _enforce("GET", API + path)


def test_aiohttp_dot_segment_url_is_refused_before_normalisation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aiohttp

    from traigent.labels.transport_gate import gate_session

    monkeypatch.setenv(oo.ENV_VAR, "1")

    async def go() -> None:
        async with gate_session(aiohttp.ClientSession()) as s:
            async with s.get(f"{API}/sessions/../evaluators", headers=HEADERS):
                pass

    with capture_egress() as cap:
        with pytest.raises(oo.OptimizationOnlyRefusal):
            asyncio.run(go())
    assert cap.requests == []


class _Resp:
    def __init__(self, status: int) -> None:
        self.status = self.status_code = status
        self.request = None

    def release(self) -> None:
        return None

    def close(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


@pytest.mark.parametrize("active", [True, False])
def test_redirects_are_not_followed_and_a_3xx_is_a_local_refusal_while_active(
    monkeypatch: pytest.MonkeyPatch, active: bool
) -> None:
    import httpx
    import requests

    from traigent.labels import transport_gate as gate

    if active:
        monkeypatch.setenv(oo.ENV_VAR, "1")
    seen: dict[str, dict[str, Any]] = {}

    class AioBase:
        async def _request(self, method: Any, url: Any, **kw: Any) -> Any:
            seen["aiohttp"] = kw
            return _Resp(302)

    class ReqBase:
        def send(self, request: Any, **kw: Any) -> Any:
            seen["requests"] = kw
            return _Resp(302)

    class HttpxBase:
        def send(self, request: Any, **kw: Any) -> Any:
            seen["httpx"] = kw
            return _Resp(302)

    aio = gate._build_aiohttp(AioBase)()
    req = gate._build_requests(ReqBase)()
    hx = gate._build_httpx_sync(HttpxBase)()
    prepared = requests.Request("GET", f"{API}/sessions/s1", headers=HEADERS).prepare()
    hx_request = httpx.Request("GET", f"{API}/sessions/s1", headers=HEADERS)

    calls = [
        lambda: asyncio.run(aio._request("GET", f"{API}/sessions/s1", headers=HEADERS)),
        lambda: req.send(prepared),
        lambda: hx.send(hx_request),
    ]
    for call in calls:
        if active:
            with pytest.raises(oo.OptimizationOnlyRefusal):
                call()
        else:
            call()  # inactive: the 3xx passes through untouched
    if active:
        assert seen["aiohttp"]["allow_redirects"] is False
        assert seen["requests"]["allow_redirects"] is False
        assert seen["httpx"]["follow_redirects"] is False
    else:
        assert "allow_redirects" not in seen["aiohttp"]
        assert "allow_redirects" not in seen["requests"]
        assert "follow_redirects" not in seen["httpx"]


# ------------------------------------------------------ C: survives the gate-off setting


def test_enforcement_survives_transport_gate_off_for_sessions_made_before_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import requests

    from traigent.labels import transport_gate as gate

    monkeypatch.setenv(gate.MODE_ENV, "off")
    session = gate.gate_session(requests.Session())
    assert getattr(type(session), "_traigent_gated", False)  # wrapped despite "off"
    with capture_egress() as cap:
        session.post(f"{API}/evaluators", json={"a": "free text"}, headers=HEADERS)
        assert len(cap.requests) == 1  # inactive: untouched
        oo.note_response(403, CLOSED, oo.credential_scope(BACKEND_URL, HEADERS))
        with pytest.raises(oo.OptimizationOnlyRefusal):
            session.post(f"{API}/evaluators", json={"a": "free text"}, headers=HEADERS)
    assert len(cap.requests) == 1


# ------------------------------------------------------------ D: close every body


def test_finalize_certified_selection_and_aggregation_notes_never_reach_the_wire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from traigent.cloud.backend_client import (
        BackendClientConfig,
        BackendIntegratedClient,
    )

    monkeypatch.setenv(oo.ENV_VAR, "1")
    with capture_egress() as cap:
        client = BackendIntegratedClient(
            api_key=FAKE_API_KEY,
            backend_config=BackendClientConfig(backend_base_url=BACKEND_URL),
        )
        asyncio.run(
            client._session_ops._finalize_session_via_api(
                "sess-1",
                "run-1",
                certified_selection={"notes": "Alice has diabetes", "n": 3, "id": "c1"},
                session_aggregation={"summary": "free text here", "count": 2},
            )
        )
    (post,) = [r for r in cap.requests if r.path.endswith("/finalize")]
    sent = post.body.decode()
    assert "Alice" not in sent and "free text" not in sent
    body = json.loads(post.body)
    assert body["certified_selection"] == {"n": 3, "id": "c1"}
    assert "summary" not in body["session_aggregation"]


def test_finalize_projection_closes_both_members_and_unknown_fields() -> None:
    out = oo.project_for_phase(
        oo.PHASE_FINALIZE,
        {
            "reason": "sdk_explicit_finalization",
            "certified_selection": {"notes": "Alice has diabetes", "ok": True},
            "session_aggregation": {"selection": {"label": "x y"}, "n": 1},
            "extra_text": "Alice has diabetes",
            "extra_id": "run_9",
        },
    )
    assert out["certified_selection"] == {"ok": True}
    assert out["session_aggregation"] == {"selection": {}, "n": 1}
    assert "extra_text" not in out and out["extra_id"] == "run_9"
    with pytest.raises(oo.MetadataContractViolation):
        oo.project_for_phase(oo.PHASE_FINALIZE, {"certified_selection": "text"})


def test_finalize_with_unclosed_members_is_not_a_fixed_point() -> None:
    body = {
        "reason": "sdk_explicit_finalization",
        "certified_selection": {"notes": "Alice has diabetes"},
    }
    with pytest.raises(oo.OptimizationOnlyRefusal):
        _enforce("POST", f"{API}/sessions/s1/finalize", json.dumps(body))


@pytest.mark.parametrize(
    ("path", "body", "ok"),
    [
        ("/keys/validate", {_KF: FAKE_API_KEY}, True),
        (
            "/keys/validate",
            {_KF: FAKE_API_KEY, "notes": "Alice has diabetes"},
            False,
        ),
        (
            "/auth/login",
            {"email": "a@b.c", _PW: "x"},
            True,
        ),
        (
            "/auth/login",
            {"email": "a@b.c", _PW: "x", "bio": "text"},
            False,
        ),
        ("/auth/refresh", {"refresh_token": "t"}, True),
        ("/auth/refresh", {"refresh_token": "t", "x": 1}, False),
        (
            "/oauth/token",
            {
                "grant_type": "client_credentials",
                "client_id": "c",
                "client_secret": "s",
                "scope": "read write",
            },
            True,
        ),
        ("/oauth/token", {"grant_type": "x", "comment": "free text"}, False),
        ("/keys/validate", oo.UNREADABLE_BODY, False),
        ("/keys/validate", "not json and not a form", False),
    ],
)
def test_auth_routes_carry_credential_fields_only(
    path: str, body: Any, ok: bool
) -> None:
    payload = (
        body
        if body is oo.UNREADABLE_BODY or isinstance(body, str)
        else json.dumps(body)
    )
    if ok:
        _enforce("POST", API + path, payload)
    else:
        with pytest.raises(oo.OptimizationOnlyRefusal):
            _enforce("POST", API + path, payload)
    if ok and isinstance(body, dict):
        _enforce("POST", API + path, body)  # form-style mapping body


def test_optimizer_batch_top_level_fields_are_closed() -> None:
    sub = oo.project_for_phase(
        oo.PHASE_OPTIMIZER_METRICS,
        {"trial_id": "t1", "metrics": {"accuracy": 1}},
        session_id="h1",
    )
    url = f"{API}/hybrid/optimizer/h1/batch"
    _enforce("POST", url, json.dumps({"submissions": [sub]}))
    with pytest.raises(oo.OptimizationOnlyRefusal):
        _enforce(
            "POST",
            url,
            json.dumps({"submissions": [sub], "notes": "Alice has diabetes"}),
        )
    with pytest.raises(oo.OptimizationOnlyRefusal):
        _enforce("POST", url, json.dumps({"submissions": "text"}))


def test_intermediate_report_accepts_only_its_explicit_fields() -> None:
    from traigent.cloud.smart_pruning import _INTERMEDIATE_REPORT_KEYS

    assert oo.INTERMEDIATE_REPORT_KEYS == _INTERMEDIATE_REPORT_KEYS  # pinned to the SDK
    url = f"{API}/sessions/s1/intermediate-report"
    ok = {"session_id": "s1", "trial_id": "t1", "running_score": 0.5}
    _enforce("POST", url, json.dumps(ok))
    with pytest.raises(oo.OptimizationOnlyRefusal):
        _enforce(
            "POST", url, json.dumps({**ok, "patient_name": "alice"})
        )  # identifier-shaped


# ------------------------------------------- E: API base prefixes and path parameters


def test_a_prefixed_api_base_matches_the_route_suffix() -> None:
    body = json.dumps(
        oo.project_for_phase(oo.PHASE_RESULTS, {"trial_id": "t1", "metrics": {}})
    )
    url = f"{BACKEND_URL}/proxy/api/v1/sessions/s1/results"
    _enforce("POST", url, body, api_prefixes=("/proxy/api/v1",))
    with pytest.raises(
        oo.OptimizationOnlyRefusal
    ):  # prefix not configured for the client
        _enforce("POST", url, body)
    with pytest.raises(oo.OptimizationOnlyRefusal):
        _enforce(
            "POST",
            f"{BACKEND_URL}/proxy/api/v1/evaluators",
            "{}",
            api_prefixes=("/proxy/api/v1",),
        )


def test_prefixed_api_base_reaches_the_wire_through_a_gated_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import requests

    from traigent.labels.transport_gate import gate_session

    monkeypatch.setenv(oo.ENV_VAR, "1")
    body = oo.project_for_phase(oo.PHASE_RESULTS, {"trial_id": "t1", "metrics": {}})
    session = gate_session(requests.Session(), origins=[f"{BACKEND_URL}/proxy/api/v1"])
    with capture_egress() as cap:
        try:
            session.post(
                f"{BACKEND_URL}/proxy/api/v1/sessions/s1/results",
                json=body,
                headers=HEADERS,
            )
        except oo.OptimizationOnlyRefusal:
            raise
        except Exception:  # noqa: BLE001 - the fake backend's answer is irrelevant
            pass
    assert [r.path for r in cap.requests] == ["/proxy/api/v1/sessions/s1/results"]


@pytest.mark.parametrize(
    "path",
    [
        "/sessions/Alice%20has%20diabetes",
        "/sessions/Alice%20has%20diabetes/finalize",
        "/prompts/a%20b",
    ],
)
def test_path_parameters_must_be_identifier_shaped_once_decoded(path: str) -> None:
    with pytest.raises(oo.OptimizationOnlyRefusal):
        _enforce("GET", API + path)
    with pytest.raises(oo.OptimizationOnlyRefusal):
        _enforce("DELETE", API + path)


# ---------------------------------------------------------------- F: explicit GET reads


@pytest.mark.parametrize(
    "path",
    [
        "/analytics/runs/r1/decision-payload",
        "/projects",
        "/projects/p1/agents",
        "/prompts?page=1&per_page=20",
        "/prompts/p1/versions/3",
        "/experiments/e1",
        "/evaluators?page=1",
        "/sessions/s1/status",
    ],
)
def test_schema_defined_bodyless_gets_pass(path: str) -> None:
    _enforce("GET", API + path)


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/prompts?search=Alice%20has%20diabetes", None),
        ("/example-sets/e1", None),  # not on the read list
        ("/datasets/d1", None),
        ("/observability/traces/t1", None),
        ("/economics/telemetry", None),
        ("/analytics/runs/r1", {"q": "x"}),  # a GET with a body
    ],
)
def test_other_gets_and_get_bodies_are_refused(path: str, body: Any) -> None:
    with pytest.raises(oo.OptimizationOnlyRefusal):
        _enforce("GET", API + path, body)
