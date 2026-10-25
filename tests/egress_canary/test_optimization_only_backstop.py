"""optimization_only transport backstop: default-deny at the gate, coverage proof, probes.

Recording transports only (``capture_egress``): no real network, sockets are denied.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from traigent.utils import optimization_only_egress as oo

from .harness import capture_egress
from .http_egress_scan import scan
from .scenario import BACKEND_URL, FAKE_API_KEY
from .test_optimization_only_egress import DIRTY_TYPED_SESSION, _env

pytestmark = pytest.mark.backend_online

API = f"{BACKEND_URL}/api/v1"
CLOSED = json.dumps({"error": "optimization_only", "error_code": "optimization_only"})
KEY_HEADERS = {"X-API-Key": FAKE_API_KEY}


@pytest.fixture(autouse=True)
def _mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _env(monkeypatch, tmp_path)
    monkeypatch.delenv(oo.ENV_VAR, raising=False)
    oo.reset_latch()
    yield
    oo.reset_latch()


def _latch_key() -> str:
    scope = oo.credential_scope(BACKEND_URL, KEY_HEADERS)
    assert scope is not None
    oo.note_response(403, CLOSED, scope)
    return scope


# ----------------------------------------------------------------- coverage proof

#: Raw HTTP constructs that deliberately do NOT pass through the transport gate.
BYPASS_REGISTER: dict[tuple[str, str], str] = {
    ("cloud/production_mcp_client.py", "aiohttp-session"): (
        "false positive: ``ClientSession`` here is the MCP protocol session over an MCP "
        "transport, not an aiohttp HTTP session"
    ),
    ("utils/diagnostics.py", "socket"): (
        "TCP reachability probe for the diagnose command: connects and sends no bytes"
    ),
}


def test_every_http_egress_construct_is_gated_or_a_documented_bypass() -> None:
    constructs = scan()
    assert len(constructs) > 40  # the scan really walks the tree
    ungated = {(c.file, c.kind) for c in constructs if not c.gated}
    undocumented = sorted(ungated - set(BYPASS_REGISTER))
    assert undocumented == [], (
        "a raw aiohttp/requests/httpx/urllib/socket call bypasses the transport gate: "
        f"route it through gate_session/preflight_requests/UrllibGateHandler, or document "
        f"it in BYPASS_REGISTER: {undocumented}"
    )
    stale = sorted(set(BYPASS_REGISTER) - ungated)
    assert stale == [], f"stale bypass register entries: {stale}"


def test_the_scanner_flags_a_new_raw_call(tmp_path: Path) -> None:
    import ast

    from .http_egress_scan import _Visitor

    visitor = _Visitor("new_module.py")
    visitor.visit(
        ast.parse(
            "import requests, httpx\n"
            "def f():\n    requests.post('https://x', json={})\n"
            "def g():\n    return httpx.Client()\n"
            "def h(gate_session):\n    return gate_session(httpx.Client())\n"
        )
    )
    assert [(c.kind, c.gated) for c in visitor.found] == [
        ("requests-call", False),
        ("httpx-client", False),
        ("httpx-client", True),
    ]


# ------------------------------------------- default-deny: unknown routes are refused

UNKNOWN_ROUTES = [
    ("POST", "/prompts/p/playground"),
    ("POST", "/evaluators"),
    ("POST", "/example-sets"),
    ("POST", "/example-sets/e1/upload"),
    ("POST", "/observability/ingest"),
    ("POST", "/observability/otlp/v1/traces"),
    ("POST", "/economics/telemetry"),
    ("POST", "/optimization/plan"),
    ("POST", "/production-requirements"),
    ("POST", "/optimize"),
    ("POST", "/agent/execute"),
    ("POST", "/hybrid/sessions"),
    ("POST", "/labels/objects"),
    ("GET", "/analytics/runs/r1/decision-payload"),
    ("PUT", "/projects/p1"),
]


@pytest.mark.parametrize(("method", "path"), UNKNOWN_ROUTES)
def test_unknown_routes_are_refused_when_the_mode_is_active(
    method: str, path: str
) -> None:
    with pytest.raises(oo.OptimizationOnlyRefusal):
        oo.enforce_outgoing(method, API + path, KEY_HEADERS, {"a": 1}, explicit=True)


def test_nothing_is_enforced_when_the_mode_is_inactive() -> None:
    oo.enforce_outgoing("POST", API + "/evaluators", KEY_HEADERS, {"a": "free text"})
    oo.enforce_outgoing(
        "POST", API + "/evaluators", KEY_HEADERS, {"a": "free text"}, explicit=False
    )


# --------------------- allowlisted routes: a non-fixed-point body is refused, a clean one passes

SID = "sess-1"
ROUTE_CASES: list[tuple[str, str, Any, Any]] = [
    # (method, path, dirty body, projector phase)
    ("POST", "/sessions", DIRTY_TYPED_SESSION, oo.PHASE_SESSION_CREATE_AUTO),
    (
        "POST",
        f"/sessions/{SID}/results",
        {"trial_id": "t1", "metrics": {"patient_diabetes": 1.0}},
        oo.PHASE_RESULTS,
    ),
    (
        "POST",
        f"/sessions/{SID}/next-trial",
        {
            "session_id": SID,
            "previous_results": [{"trial_id": "t", "metrics": {"zz": 1}}],
        },
        oo.PHASE_NEXT_TRIAL,
    ),
    (
        "POST",
        f"/sessions/{SID}/finalize",
        {"reason": "because of Alice", "metadata": {"a": "b"}},
        oo.PHASE_FINALIZE,
    ),
    (
        "POST",
        f"/sessions/{SID}/intermediate-report",
        {"session_id": SID, "trial_id": "t", "objective_name": "free text"},
        oo.PHASE_INTERMEDIATE_REPORT,
    ),
    (
        "PUT",
        "/configuration-runs/c1/status",
        {"status": "failed", "error_message": "Alice has diabetes"},
        oo.PHASE_CONFIGURATION_RUN,
    ),
    (
        "PUT",
        "/configuration-runs/c1/measures",
        {"measures": [{"accuracy": 1.0, "patient_diabetes": 2}]},
        oo.PHASE_CONFIGURATION_RUN,
    ),
    (
        "PUT",
        "/experiment-runs/runs/r1",
        {"status": "COMPLETED", "error_message": "free text"},
        oo.PHASE_CONFIGURATION_RUN,
    ),
    (
        "POST",
        "/experiments/e1/runs",
        {"name": "Alice's run", "metadata": {"agent_id": "a1"}},
        oo.PHASE_PARENT_RUN,
    ),
    (
        "POST",
        "/best-configs",
        {"spec": {"config_id": "cfg-1", "config": {}, "function_ref": "My Function"}},
        oo.PHASE_BEST_CONFIG,
    ),
    (
        "POST",
        "/hybrid/optimizer/h1",
        {"trial_id": "t", "metrics": {"patient_diabetes": 1.0}},
        oo.PHASE_OPTIMIZER_METRICS,
    ),
    (
        "POST",
        "/hybrid/optimizer/h1/batch",
        {"submissions": [{"trial_id": "t", "metrics": {"patient_diabetes": 1.0}}]},
        oo.PHASE_OPTIMIZER_METRICS,
    ),
    (
        "POST",
        f"/director/sessions/{SID}/turn",
        {"session_revision": 1, "report": {"notes": "Alice has diabetes"}},
        None,
    ),
    ("POST", "/director/sessions", {"project_id": "p1", "goal": "x"}, None),
]


@pytest.mark.parametrize(
    ("method", "path", "dirty", "phase"),
    ROUTE_CASES,
    ids=[f"{m} {p}" for m, p, _d, _ph in ROUTE_CASES],
)
def test_allowlisted_route_refuses_a_body_that_is_not_a_fixed_point(
    method: str, path: str, dirty: Any, phase: str | None
) -> None:
    url = API + path
    with pytest.raises(oo.OptimizationOnlyRefusal):
        oo.enforce_outgoing(method, url, KEY_HEADERS, json.dumps(dirty), explicit=True)
    if phase is None:
        return
    if phase == oo.PHASE_INTERMEDIATE_REPORT:
        clean: Any = {**dirty, "objective_name": "accuracy"}
    elif phase == oo.PHASE_BEST_CONFIG:
        spec = {k: v for k, v in dirty["spec"].items() if k != "function_ref"}
        clean = {"spec": spec}
    elif phase == oo.PHASE_OPTIMIZER_METRICS and "submissions" in dirty:
        clean = {
            "submissions": [
                oo.project_for_phase(phase, s, session_id="h1")
                for s in dirty["submissions"]
            ]
        }
    else:
        clean = oo.project_for_phase(phase, dirty, session_id=SID)
    oo.enforce_outgoing(method, url, KEY_HEADERS, json.dumps(clean), explicit=True)


def test_director_allowlisted_shape_passes_and_unreadable_or_query_content_is_refused() -> (
    None
):
    ok = {"session_revision": 1, "report": {"instruction_id": "i1", "status": "done"}}
    oo.enforce_outgoing(
        "POST", f"{API}/director/sessions/{SID}/turn", KEY_HEADERS, ok, explicit=True
    )
    with pytest.raises(oo.OptimizationOnlyRefusal):
        oo.enforce_outgoing(
            "POST",
            f"{API}/sessions/{SID}/results",
            KEY_HEADERS,
            oo.UNREADABLE_BODY,
            explicit=True,
        )
    with pytest.raises(oo.OptimizationOnlyRefusal):
        oo.enforce_outgoing(
            "GET",
            f"{API}/best-configs/cfg-1?function_ref=My%20Function",
            KEY_HEADERS,
            None,
            explicit=True,
        )
    oo.enforce_outgoing(
        "GET",
        f"{API}/best-configs/cfg-1?function_ref=fn_abc",
        KEY_HEADERS,
        None,
        explicit=True,
    )
    with pytest.raises(oo.OptimizationOnlyRefusal):
        oo.enforce_outgoing(
            "GET", f"{API}/sessions/{SID}", KEY_HEADERS, {"a": 1}, explicit=True
        )


# ------------------------- the backstop at every transport wrapper (recording transports)


def _send_aiohttp(url: str, body: Any) -> None:
    import aiohttp

    from traigent.labels.transport_gate import gate_session

    async def go() -> None:
        async with gate_session(aiohttp.ClientSession(), origins=[BACKEND_URL]) as s:
            async with s.post(url, json=body, headers=KEY_HEADERS):
                pass

    asyncio.run(go())


def _send_requests(url: str, body: Any) -> None:
    import requests

    from traigent.labels.transport_gate import gate_session

    gate_session(requests.Session(), origins=[BACKEND_URL]).post(
        url, json=body, headers=KEY_HEADERS
    )


def _send_httpx(url: str, body: Any) -> None:
    import httpx

    from traigent.labels.transport_gate import gate_session

    gate_session(httpx.Client(), origins=[BACKEND_URL]).post(
        url, json=body, headers=KEY_HEADERS
    )


def _send_httpx_async(url: str, body: Any) -> None:
    import httpx

    from traigent.labels.transport_gate import gate_session

    async def go() -> None:
        client = gate_session(httpx.AsyncClient(), origins=[BACKEND_URL])
        await client.post(url, json=body, headers=KEY_HEADERS)

    asyncio.run(go())


def _send_urllib(url: str, body: Any) -> None:
    import urllib.request

    from traigent.cloud.url_security import build_no_redirect_opener

    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers=KEY_HEADERS, method="POST"
    )
    build_no_redirect_opener().open(req, timeout=2)


def _send_module_level_requests(url: str, body: Any) -> None:
    import requests

    from traigent.labels.transport_gate import preflight_requests

    preflight_requests("POST", url, json=body, headers=KEY_HEADERS)
    requests.post(url, json=body, headers=KEY_HEADERS, timeout=2)


TRANSPORTS = {
    "aiohttp": _send_aiohttp,
    "requests-session": _send_requests,
    "httpx-sync": _send_httpx,
    "httpx-async": _send_httpx_async,
    "urllib": _send_urllib,
    "requests-module-level": _send_module_level_requests,
}


@pytest.mark.parametrize("transport", sorted(TRANSPORTS))
def test_every_transport_refuses_unknown_and_unprojected_requests_when_active(
    monkeypatch: pytest.MonkeyPatch, transport: str
) -> None:
    monkeypatch.setenv(oo.ENV_VAR, "1")
    send = TRANSPORTS[transport]
    with capture_egress() as cap:
        with pytest.raises(oo.OptimizationOnlyRefusal):
            send(f"{API}/evaluators", {"judge_config": {"instructions": "secret"}})
        with pytest.raises(oo.OptimizationOnlyRefusal):
            send(f"{API}/sessions/{SID}/results", {"metrics": {"patient_diabetes": 1}})
    assert cap.requests == [], [r.route for r in cap.requests]


@pytest.mark.parametrize("transport", ["aiohttp", "requests-session", "urllib"])
def test_every_transport_lets_a_projected_loop_request_through_when_active(
    monkeypatch: pytest.MonkeyPatch, transport: str
) -> None:
    monkeypatch.setenv(oo.ENV_VAR, "1")
    clean = oo.project_for_phase(
        oo.PHASE_RESULTS, {"trial_id": "t1", "metrics": {"accuracy": 1}}
    )
    with capture_egress() as cap:
        try:
            TRANSPORTS[transport](f"{API}/sessions/{SID}/results", clean)
        except oo.OptimizationOnlyRefusal:
            raise
        except Exception:  # noqa: BLE001 - the fake backend's answer is irrelevant
            pass
    assert [r.path for r in cap.requests] == [f"/api/v1/sessions/{SID}/results"]


@pytest.mark.parametrize("transport", sorted(TRANSPORTS))
def test_mode_off_adds_no_enforcement_and_no_work(
    monkeypatch: pytest.MonkeyPatch, transport: str
) -> None:
    """Mode off, nothing latched: the hook is one cheap check (no origin or scope work)."""
    from traigent.labels import transport_gate as gate

    def boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("enforcement work done while the mode is off")

    monkeypatch.setattr(gate, "backend_origins", boom)
    monkeypatch.setattr(oo, "credential_scope", boom)
    monkeypatch.setattr(oo, "enforce_outgoing", boom)
    with capture_egress() as cap:
        try:
            TRANSPORTS[transport](
                f"{API}/evaluators", {"judge_config": {"i": "free text"}}
            )
        except Exception as exc:  # noqa: BLE001 - only enforcement errors matter here
            assert not isinstance(exc, (oo.OptimizationOnlyRefusal, AssertionError)), (
                exc
            )
    assert any(r.path.endswith("/evaluators") for r in cap.requests)


# ------------------------------------------------ sol's four probes (round 3) as regressions


def test_probe_lazy_env_jwt_latched_then_a_fresh_clients_hybrid_create_sends_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from traigent.cloud.backend_client import (
        BackendClientConfig,
        BackendIntegratedClient,
    )

    jwt = "eyJhbGciOi." + "j" * 40
    monkeypatch.delenv("TRAIGENT_API_KEY", raising=False)
    monkeypatch.delenv("TRAIGENT_JWT_TOKEN", raising=False)
    oo.note_response(
        403,
        CLOSED,
        oo.credential_scope(BACKEND_URL, {"Authorization": f"Bearer {jwt}"}),
    )
    with capture_egress() as cap:
        client = BackendIntegratedClient(
            backend_config=BackendClientConfig(backend_base_url=BACKEND_URL)
        )
        # the JWT only appears in the environment AFTER the client exists (lazy credential)
        monkeypatch.setenv("TRAIGENT_JWT_TOKEN", jwt)
        with pytest.raises(Exception):  # noqa: B017 - refusal (guard or backstop) or auth
            asyncio.run(
                client.create_hybrid_session(
                    problem_statement="Alice has diabetes",
                    search_space={"m": ["a"]},
                    optimization_config={"objectives": ["accuracy"]},
                )
            )
    assert [r for r in cap.requests if "/hybrid/" in r.path] == []
    assert not any(b"Alice" in r.body for r in cap.requests)


def test_probe_latch_set_inside_get_headers_sends_nothing_or_a_projected_body() -> None:
    from traigent.cloud.client import TraigentCloudClient

    async def go() -> None:
        cloud = TraigentCloudClient(api_key=FAKE_API_KEY, base_url=BACKEND_URL)
        real = cloud._get_headers

        async def latching() -> dict[str, str]:
            headers = await real()
            oo.note_response(
                403, CLOSED, oo.credential_scope(cloud.api_base_url, headers)
            )
            return headers

        cloud._get_headers = latching
        await cloud._submit_optimization({"function_name": "Alice has diabetes"})

    from traigent.cloud.client import CloudServiceError

    with capture_egress() as cap:
        with pytest.raises((oo.OptimizationOnlyRefusal, CloudServiceError)) as caught:
            asyncio.run(go())
    assert "route_not_allowed" in str(caught.value) or "hosted_optimization" in str(
        caught.value
    )
    assert [r.path for r in cap.requests if "keys/validate" not in r.path] == []


def test_probe_run_playground_with_the_mode_on_sends_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from traigent.prompts.client import PromptManagementClient

    monkeypatch.setenv(oo.ENV_VAR, "1")
    with capture_egress() as cap:
        with pytest.raises(oo.OptimizationOnlyRefusal):
            PromptManagementClient().run_playground(
                "p", prompt_text="Alice has diabetes", dry_run=True
            )
    assert cap.requests == []


def test_probe_create_evaluator_with_the_mode_on_sends_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from traigent.evaluation.client import EvaluationClient

    monkeypatch.setenv(oo.ENV_VAR, "1")
    with capture_egress() as cap:
        with pytest.raises(oo.OptimizationOnlyRefusal):
            EvaluationClient().create_evaluator(
                name="n",
                measure_id="accuracy",
                target_type="observability_trace",
                judge_config={"instructions": "confidential judge instructions"},
            )
    assert cap.requests == []


def test_probe_run_playground_and_create_evaluator_with_only_a_latch_send_nothing() -> (
    None
):
    from traigent.evaluation.client import EvaluationClient
    from traigent.prompts.client import PromptManagementClient

    _latch_key()  # env unset, no keyword: only the credential's latch is on
    with capture_egress() as cap:
        with pytest.raises(oo.OptimizationOnlyRefusal):
            PromptManagementClient().run_playground(
                "p", prompt_text="Alice has diabetes", dry_run=True
            )
        with pytest.raises(oo.OptimizationOnlyRefusal):
            EvaluationClient().create_evaluator(
                name="n",
                measure_id="accuracy",
                target_type="observability_trace",
                judge_config={"instructions": "confidential"},
            )
    assert cap.requests == []


def test_explicit_keyword_true_reaches_the_backstop_through_gate_session() -> None:
    from traigent.cloud.backend_client import (
        BackendClientConfig,
        BackendIntegratedClient,
    )

    with capture_egress() as cap:
        client = BackendIntegratedClient(
            api_key=FAKE_API_KEY,
            backend_config=BackendClientConfig(backend_base_url=BACKEND_URL),
            optimization_only=True,
        )
        session = client._api_ops  # noqa: F841 - ops share the client's policy
        import aiohttp

        from traigent.labels.transport_gate import gate_session

        async def go() -> None:
            sess = gate_session(
                aiohttp.ClientSession(),
                origins=[BACKEND_URL],
                optimization_only=oo.explicit_of(client),
            )
            async with sess:
                async with sess.post(
                    f"{API}/evaluators", json={"a": "free text"}, headers=KEY_HEADERS
                ):
                    pass

        with pytest.raises(oo.OptimizationOnlyRefusal):
            asyncio.run(go())
    assert cap.requests == []


def test_backstop_covers_a_lazy_credential_the_site_guard_cannot_see() -> None:
    """Review 3 / P1: a JWT that only appears in the FINAL headers (the configured-credential
    lookup returns nothing before authentication) is judged by the transport, by construction."""
    jwt = "eyJhbGciOi." + "j" * 40
    jwt_headers = {"Authorization": f"Bearer {jwt}"}
    oo.note_response(403, CLOSED, oo.credential_scope(BACKEND_URL, jwt_headers))

    class PreAuthClient:
        optimization_only = None

        class auth:  # noqa: N801 - stand-in for an AuthManager that has not authenticated
            @staticmethod
            def configured_credential_headers() -> dict[str, str]:
                return {}

    assert (
        oo.policy_of(PreAuthClient(), None, API).scope is None
    )  # the site guard is blind

    import aiohttp

    from traigent.labels.transport_gate import gate_session

    async def go() -> None:
        async with gate_session(aiohttp.ClientSession(), origins=[BACKEND_URL]) as s:
            async with s.post(
                f"{API}/hybrid/sessions",
                json={"problem_statement": "Alice has diabetes"},
                headers=jwt_headers,
            ):
                pass

    with capture_egress() as cap:
        with pytest.raises(oo.OptimizationOnlyRefusal):
            asyncio.run(go())
    assert cap.requests == []
