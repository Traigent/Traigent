"""optimization_only: what the SDK sends is the contract projection; forbidden operations
make zero transport calls; with the mode off nothing changes.

Everything runs under the egress-capture harness (every transport patched, sockets denied),
so "zero transport calls" means zero captured requests, not a mocked method.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import pytest

from traigent.utils import optimization_only_egress as oo

from . import direct_paths
from .harness import CapturedRequest, capture_egress
from .scenario import BACKEND_URL, FAKE_API_KEY, run_scenario

pytestmark = pytest.mark.backend_online

_SESS = r"/sessions/(?P<sid>[^/]+)"
#: route regex -> (method, phase). Bodies on these routes must be projection fixed points.
ROUTES: list[tuple[str, str, str]] = [
    ("POST", r"/hybrid/sessions$", oo.PHASE_HYBRID_SESSION_CREATE),
    ("POST", r"(?<!/hybrid)/sessions$", oo.PHASE_SESSION_CREATE_AUTO),
    ("POST", _SESS + r"/results$", oo.PHASE_RESULTS),
    ("POST", _SESS + r"/next-trial$", oo.PHASE_NEXT_TRIAL),
    ("POST", _SESS + r"/finalize$", oo.PHASE_FINALIZE),
    ("POST", _SESS + r"/intermediate-report$", oo.PHASE_INTERMEDIATE_REPORT),
    (
        "PUT",
        r"/configuration-runs/[^/]+/(status|measures|summary-stats)$",
        oo.PHASE_CONFIGURATION_RUN,
    ),
    ("PUT", r"/experiment-runs/runs/[^/]+$", oo.PHASE_CONFIGURATION_RUN),
    ("POST", r"/hybrid/optimizer/[^/]+(/batch)?$", oo.PHASE_OPTIMIZER_METRICS),
]
FORBIDDEN_PATH = re.compile(
    r"example-sets|/observability|/otlp|/traces|/agent/(execute|optimize)|/optimize$|/features$"
)


@pytest.fixture(autouse=True)
def _mode(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(oo.ENV_VAR, raising=False)
    oo.reset_latch()
    yield
    oo.reset_latch()


def _phase_for(req: CapturedRequest) -> str | None:
    for method, pattern, phase in ROUTES:
        if req.method == method and re.search(pattern, req.path):
            return phase
    return None


def _bodies(req: CapturedRequest) -> list[Any]:
    try:
        body = json.loads(req.body) if req.body else None
    except ValueError:
        return []
    if isinstance(body, dict) and isinstance(body.get("submissions"), list):
        return list(body["submissions"])
    return [] if body is None else [body]


def _fixed_point(req: CapturedRequest, phase: str, body: Any) -> bool:
    m = re.search(_SESS, req.path)
    sid = m.group("sid") if m else None
    if phase == oo.PHASE_OPTIMIZER_METRICS:
        sid = req.path.rstrip("/").split("/")[-2 if req.path.endswith("/batch") else -1]
    projected = oo.project_for_phase(phase, body, session_id=sid)
    return json.dumps(projected, sort_keys=True) == json.dumps(body, sort_keys=True)


def test_every_body_the_sdk_sends_is_already_the_projection(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(oo.ENV_VAR, "1")
    # ``direct_serializer`` hand-builds a body and POSTs it with a raw aiohttp session,
    # bypassing every SDK send path; the real TraigentCloudClient routes are covered by
    # ``test_cloud_client_routes_send_projected_bodies`` below.
    monkeypatch.delitem(direct_paths.EXERCISERS, "direct_serializer")
    result = run_scenario(
        Path(tmp_path_factory.mktemp("oo-scenario")), direct_paths=True
    )

    checked: dict[str, int] = {}
    problems: list[str] = []
    for req in result.requests:
        phase = _phase_for(req)
        if phase is None:
            continue
        for body in _bodies(req):
            checked[phase] = checked.get(phase, 0) + 1
            try:
                ok = _fixed_point(req, phase, body)
            except oo.MetadataContractViolation:
                ok = False
            if not ok:
                problems.append(f"{req.method} {req.route} ({phase})")
    assert not problems, problems
    # non-vacuous: the core loop and the retry/direct exercisers all went through
    for phase in (
        oo.PHASE_SESSION_CREATE_AUTO,
        oo.PHASE_RESULTS,
        oo.PHASE_NEXT_TRIAL,
        oo.PHASE_FINALIZE,
        oo.PHASE_CONFIGURATION_RUN,
        oo.PHASE_INTERMEDIATE_REPORT,
        oo.PHASE_HYBRID_SESSION_CREATE,
        oo.PHASE_OPTIMIZER_METRICS,
    ):
        assert checked.get(phase, 0) > 0, (phase, checked)

    forbidden = [r.route for r in result.requests if FORBIDDEN_PATH.search(r.path)]
    assert forbidden == [], forbidden
    # forbidden exercisers were refused locally, not by the (fake) server
    assert "OptimizationOnlyRefusal" in (
        result.exerciser_outcomes.get("example_set_upload") or ""
    )
    assert "OptimizationOnlyRefusal" in (
        result.exerciser_outcomes.get("aiohttp_retry_optimize") or ""
    )


# --------------------------------------------------------------- exact bytes, on and off

DIRTY_TYPED_SESSION: dict[str, Any] = {
    "function_name": "opt:accuracy · model (ab12)",
    "configuration_space": {"model": {"type": "categorical", "choices": ["a", "b"]}},
    "objectives": ["accuracy"],
    "run_title": "Alice's secret run",
    "run_description": "free text",
    "dataset_metadata": {"size": 3, "name": "Alice has diabetes"},
    "max_trials": 2,
    "metadata": {
        "function_name": "f",
        "evaluation_set": "my set",
        "notes": "free text",
    },
}


def _client():
    from traigent.cloud.backend_client import (
        BackendClientConfig,
        BackendIntegratedClient,
    )

    return BackendIntegratedClient(
        api_key=FAKE_API_KEY,
        backend_config=BackendClientConfig(backend_base_url=BACKEND_URL),
    )


def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for k, v in {
        "TRAIGENT_API_KEY": FAKE_API_KEY,
        "TRAIGENT_BACKEND_URL": BACKEND_URL,
        "TRAIGENT_OFFLINE_MODE": "false",
        "TRAIGENT_OFFLINE": "false",
        "TRAIGENT_RESULTS_FOLDER": str(tmp_path),
    }.items():
        monkeypatch.setenv(k, v)


async def _create_session_raw(payload: dict[str, Any]) -> None:
    client = _client()
    await client._api_ops._post_session_creation(
        json.loads(json.dumps(payload)), {"Content-Type": "application/json"}, None
    )


def _session_posts(cap) -> list[CapturedRequest]:
    return [
        r for r in cap.requests if r.method == "POST" and r.path.endswith("/sessions")
    ]


@pytest.mark.parametrize("retry", [False, True])
def test_session_create_bytes_equal_projection_when_on(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, retry: bool
) -> None:
    _env(monkeypatch, tmp_path)
    monkeypatch.setenv(oo.ENV_VAR, "1")
    expected = oo.project_for_phase(oo.PHASE_SESSION_CREATE_AUTO, DIRTY_TYPED_SESSION)
    assert expected != DIRTY_TYPED_SESSION
    with capture_egress() as cap:
        if retry:
            cap.backend.inject("POST", r"/sessions$", [503, "connection"])
        for _ in range(3 if retry else 1):
            try:
                asyncio.run(_create_session_raw(DIRTY_TYPED_SESSION))
            except Exception:  # noqa: BLE001 - injected failures are the point
                pass
    posts = _session_posts(cap)
    assert len(posts) == (3 if retry else 1)
    for post in posts:  # first attempt and every re-send carry the same projected bytes
        assert json.loads(post.body) == expected
    sent = posts[0].body.decode()
    assert "Alice" not in sent and "free text" not in sent


def test_session_create_bytes_unchanged_when_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _env(monkeypatch, tmp_path)
    with capture_egress() as cap:
        asyncio.run(_create_session_raw(DIRTY_TYPED_SESSION))
    (post,) = _session_posts(cap)
    assert json.loads(post.body) == DIRTY_TYPED_SESSION  # legacy: byte-for-byte content


def test_explicit_false_beats_env_on_the_wire(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from traigent.cloud.backend_client import (
        BackendClientConfig,
        BackendIntegratedClient,
    )

    _env(monkeypatch, tmp_path)
    monkeypatch.setenv(oo.ENV_VAR, "1")
    client = BackendIntegratedClient(
        api_key=FAKE_API_KEY,
        backend_config=BackendClientConfig(backend_base_url=BACKEND_URL),
        optimization_only=False,
    )
    with capture_egress() as cap:
        asyncio.run(
            client._api_ops._post_session_creation(
                json.loads(json.dumps(DIRTY_TYPED_SESSION)), {}, None
            )
        )
    assert json.loads(_session_posts(cap)[0].body) == DIRTY_TYPED_SESSION


def test_closed_403_latches_and_next_send_is_projected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _env(monkeypatch, tmp_path)
    monkeypatch.setenv(oo.ENV_VAR, "0")
    closed = {
        "success": False,
        "error": "optimization_only",
        "error_code": "optimization_only",
    }
    with capture_egress() as cap:
        asyncio.run(
            _create_session_raw(DIRTY_TYPED_SESSION)
        )  # env says off: legacy bytes
        assert oo.note_response(403, json.dumps(closed))
        asyncio.run(_create_session_raw(DIRTY_TYPED_SESSION))  # latched: projected
    first, second = _session_posts(cap)
    assert json.loads(first.body) == DIRTY_TYPED_SESSION
    assert "Alice" not in second.body.decode()


# --------------------------------------------------------------- forbidden operations


def _zero(coro_fn, monkeypatch, tmp_path, *, expect: type[BaseException] | None):
    _env(monkeypatch, tmp_path)
    monkeypatch.setenv(oo.ENV_VAR, "1")
    with capture_egress() as cap:
        if expect is None:
            asyncio.run(coro_fn())
        else:
            with pytest.raises(expect):
                asyncio.run(coro_fn())
    assert cap.requests == [], [r.route for r in cap.requests]
    assert cap.guard_violations == []


def test_dataset_upload_is_refused_with_zero_transport(monkeypatch, tmp_path) -> None:
    from traigent.cloud.dataset_converter import DatasetConverter
    from traigent.evaluators.base import Dataset, EvaluationExample

    async def go():
        ds = Dataset(
            name="n",
            examples=[
                EvaluationExample(
                    input_data={"q": "text"}, expected_output="a", metadata={}
                )
            ],
        )
        async with DatasetConverter(BACKEND_URL) as conv:
            await conv.upload_sdk_dataset_to_backend(ds, example_set_name="x")

    _zero(go, monkeypatch, tmp_path, expect=oo.OptimizationOnlyRefusal)


def test_hosted_paths_are_refused_with_zero_transport(monkeypatch, tmp_path) -> None:
    from traigent.cloud.client import TraigentCloudClient

    async def go():
        cloud = TraigentCloudClient(api_key=FAKE_API_KEY, base_url=BACKEND_URL)
        for call in (
            lambda: cloud._submit_optimization({"function_name": "f"}),
            lambda: cloud.execute_agent({"name": "a"}, {}, {}, {}),
            lambda: cloud.optimize_agent(None, None, {}),
        ):
            with pytest.raises(oo.OptimizationOnlyRefusal):
                await call()

    _zero(go, monkeypatch, tmp_path, expect=None)


def test_example_feature_upload_is_refused_with_zero_transport(
    monkeypatch, tmp_path
) -> None:
    _env(monkeypatch, tmp_path)
    monkeypatch.setenv(oo.ENV_VAR, "1")
    with capture_egress() as cap:
        with pytest.raises(oo.OptimizationOnlyRefusal):
            _client().upload_example_features(
                "run-1", "difficulty", [{"example_id": "e1"}]
            )
    assert cap.requests == []


def test_best_config_publish_projects_and_fetch_refuses_non_ids(
    monkeypatch, tmp_path
) -> None:
    _env(monkeypatch, tmp_path)
    monkeypatch.setenv(oo.ENV_VAR, "1")
    spec = {
        "schema_version": "traigent.best_config.v1",
        "config_id": "cfg-1",
        "config": {"model": "gpt-4o", "prompt": "config values are an allowed channel"},
        "provenance": {
            "source": "run-1",
            "note": "free text note",
            "at": "2026-01-01T00:00:00Z",
        },
    }
    with capture_egress() as cap:
        _client().publish_best_config_sync(spec, environment="default")
        with pytest.raises(oo.OptimizationOnlyRefusal):
            _client().fetch_best_config_sync("cfg-1", function_ref="My Function Name")
        with pytest.raises(oo.OptimizationOnlyRefusal):
            _client().fetch_agent_head_generation_sync("Agent With Spaces")
        with pytest.raises(oo.MetadataContractViolation):
            _client().publish_best_config_sync(
                {**spec, "function_ref": "My Function Name"}, environment="default"
            )
    posts = [
        r
        for r in cap.requests
        if r.method == "POST" and r.path.endswith("/best-configs")
    ]
    assert len(posts) == 1
    sent = json.loads(posts[0].body)
    assert sent["spec"]["config"] == spec["config"]
    assert "note" not in sent["spec"]["provenance"]
    assert sent["spec"]["provenance"]["source"] == "run-1"
    assert [
        r for r in cap.requests if "best-configs" in r.path and r.method == "GET"
    ] == []


def test_runtime_requirements_economics_analytics_plan_refuse_free_text(
    monkeypatch, tmp_path
) -> None:
    _env(monkeypatch, tmp_path)
    monkeypatch.setenv(oo.ENV_VAR, "1")
    with capture_egress() as cap:
        from traigent.runtime.client import RuntimeClient
        from traigent.runtime.model import RuntimeRequirementError

        rt = RuntimeClient("proj-1", base_url=BACKEND_URL, api_key=FAKE_API_KEY)
        with pytest.raises(RuntimeRequirementError):
            rt.import_tvl(
                {"tvl": "free text\nwith lines", "name": "Alice's requirement"}
            )

        from traigent.economics.client import EconomicsTelemetryClient

        econ = EconomicsTelemetryClient(BACKEND_URL, api_key=FAKE_API_KEY)
        with pytest.raises(oo.OptimizationOnlyRefusal):
            asyncio.run(econ._post_with_retry(b'{"events":[{"note":"free text"}]}', {}))

        from traigent.cloud.analytics_client import BackendAnalyticsClient

        analytics = BackendAnalyticsClient(BACKEND_URL, api_key=FAKE_API_KEY)
        with pytest.raises(oo.OptimizationOnlyRefusal):
            asyncio.run(
                analytics._post_json(
                    "/api/v1/x", what="x", json_body={"q": "free text"}
                )
            )
    assert cap.requests == []


def test_content_less_labels_are_not_sent_in_mode(monkeypatch) -> None:
    from traigent.labels.transport import send_session_labels

    monkeypatch.setenv(oo.ENV_VAR, "1")
    with capture_egress() as cap:
        out = asyncio.run(
            send_session_labels(
                {"agent": "Alice's agent"},
                {"agent": "agentref_" + "a1" * 16},
                BACKEND_URL,
                lambda: {},
            )
        )
    assert out == {} and cap.requests == []


def test_closed_values_are_not_refused() -> None:
    assert oo.is_closed_value({"a": 1, "b": ["x", "sha256:" + "a" * 64], "c": None})
    assert not oo.is_closed_value({"a": "free text"})
    assert not oo.is_closed_value({"free key": 1})


async def _cloud_client_calls() -> None:
    from traigent.cloud.client import TraigentCloudClient

    cloud = TraigentCloudClient(api_key=FAKE_API_KEY, base_url=BACKEND_URL)
    try:
        await cloud.submit_trial_result(
            "sess-oo-1",
            "trial-oo-1",
            {"accuracy": 0.5, "undeclared_metric": 1.0},
            0.1,
            "failed",
            error_message="boom with Alice's text",
            metadata={"free_key": "free value"},
        )
        await cloud.get_next_trial("sess-oo-1")
        await cloud.finalize_optimization("sess-oo-1")
    finally:
        if hasattr(cloud, "close"):
            await cloud.close()


def _cloud_posts(cap) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in cap.requests:
        if r.method == "POST":
            out[r.path.rsplit("/", 1)[-1]] = json.loads(r.body or b"{}")
    return out


def test_cloud_client_routes_send_projected_bodies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _env(monkeypatch, tmp_path)
    with capture_egress() as cap:
        asyncio.run(_cloud_client_calls())
    legacy = _cloud_posts(cap)
    assert legacy["results"]["metrics"] == {"accuracy": 0.5, "undeclared_metric": 1.0}

    monkeypatch.setenv(oo.ENV_VAR, "1")
    with capture_egress() as cap:
        asyncio.run(_cloud_client_calls())
    sent = _cloud_posts(cap)
    assert set(sent) >= {"results", "next-trial", "finalize"}
    assert sent["results"]["metrics"] == {"accuracy": 0.5}
    assert sent["results"]["error_message"] == "trial_failed"
    assert "Alice" not in json.dumps(sent) and "free value" not in json.dumps(sent)
    assert sent["results"] == oo.project_for_phase(
        oo.PHASE_RESULTS, legacy["results"], session_id="sess-oo-1"
    )
    assert sent["finalize"] == oo.project_for_phase(
        oo.PHASE_FINALIZE, legacy["finalize"]
    )
