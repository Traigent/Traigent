"""optimization_only: EVERY guarded site honours a credential-scoped refusal latch.

The mode is otherwise off (no env, no keyword). A closed ``403 optimization_only`` is latched
for the credential the client holds; each guarded site must then refuse (zero captured
requests) or send the projected body. Recording transports only: no real network.
"""

from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from traigent.utils import optimization_only_egress as oo

from .harness import capture_egress
from .scenario import BACKEND_URL, FAKE_API_KEY
from .test_optimization_only_egress import DIRTY_TYPED_SESSION, _client, _env

pytestmark = pytest.mark.backend_online

CLOSED = json.dumps({"error": "optimization_only", "error_code": "optimization_only"})
KEY_HEADERS = {"X-API-Key": FAKE_API_KEY}


@pytest.fixture(autouse=True)
def _latched(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _env(monkeypatch, tmp_path)
    monkeypatch.delenv(oo.ENV_VAR, raising=False)
    oo.reset_latch()
    assert oo.note_response(403, CLOSED, oo.credential_scope(BACKEND_URL, KEY_HEADERS))
    yield
    oo.reset_latch()


def _cloud() -> Any:
    from traigent.cloud.client import TraigentCloudClient

    return TraigentCloudClient(api_key=FAKE_API_KEY, base_url=BACKEND_URL)


async def _hosted_optimization() -> None:
    await _cloud()._submit_optimization({"function_name": "f"})


async def _hosted_agent_optimization() -> None:
    await _cloud().optimize_agent(None, None, {})


async def _hosted_agent_execution() -> None:
    await _cloud().execute_agent({"name": "a"}, {}, {}, {})


async def _hybrid_create() -> None:
    await _client().create_hybrid_session(
        problem_statement="free text",
        search_space={"m": ["a"]},
        optimization_config={"objectives": ["accuracy"]},
    )


async def _dataset_upload() -> None:
    from traigent.cloud.dataset_converter import DatasetConverter
    from traigent.evaluators.base import Dataset, EvaluationExample

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


async def _economics() -> None:
    from traigent.economics.client import EconomicsTelemetryClient

    econ = EconomicsTelemetryClient(BACKEND_URL, api_key=FAKE_API_KEY)
    await econ._post_with_retry(b'{"events":[{"note":"free text"}]}', {})


async def _plan() -> None:
    from traigent.analytics.optimization_plan import OptimizationPlanClient

    plan = OptimizationPlanClient(BACKEND_URL, api_key=FAKE_API_KEY)
    await plan.get_optimization_plan(
        task_description="Alice has diabetes, help her",
        dataset_size=3,
        dataset_has_holdout=False,
        objectives=["accuracy"],
        max_trials=2,
        cost_limit_usd=1.0,
    )


async def _runtime_requirements() -> None:
    from traigent.runtime.client import RuntimeClient
    from traigent.runtime.model import RuntimeRequirementError

    rt = RuntimeClient("proj-1", base_url=BACKEND_URL, api_key=FAKE_API_KEY)
    try:
        rt.import_tvl({"tvl": "free text\nwith lines"})
    except RuntimeRequirementError as exc:
        raise oo.OptimizationOnlyRefusal("runtime") from exc


async def _analytics() -> None:
    from traigent.cloud.analytics_client import BackendAnalyticsClient

    analytics = BackendAnalyticsClient(BACKEND_URL, api_key=FAKE_API_KEY)
    await analytics._post_json("/api/v1/x", what="x", json_body={"q": "free text"})


async def _example_features() -> None:
    _client().upload_example_features("run-1", "difficulty", [{"example_id": "e1"}])


async def _best_config_fetch() -> None:
    _client().fetch_best_config_sync("cfg-1", function_ref="My Function Name")


async def _labels() -> None:
    from unittest import mock

    from traigent.labels import transport

    with mock.patch.object(transport, "sender_from_config") as sender:
        out = await transport.send_session_labels(
            {"agent": "Alice's agent"},
            {"agent": "agentref_" + "a1" * 16},
            BACKEND_URL,
            lambda: {},
            policy=oo.policy_of(_cloud()),
        )
    assert out == {} and sender.call_count == 0
    raise oo.OptimizationOnlyRefusal("labels")  # reached only if nothing was sent


async def _workflow_traces() -> None:
    from traigent.integrations.observability.workflow_traces import WorkflowTracesClient

    client = WorkflowTracesClient(BACKEND_URL, auth_token=FAKE_API_KEY)
    response = client.ingest_traces(spans=[])
    assert response.success is False
    raise oo.OptimizationOnlyRefusal("workflow_traces")


REFUSING_SITES = {
    "hosted_optimization": _hosted_optimization,
    "hosted_agent_optimization": _hosted_agent_optimization,
    "hosted_agent_execution": _hosted_agent_execution,
    "hybrid_create": _hybrid_create,
    "dataset_upload": _dataset_upload,
    "economics_telemetry": _economics,
    "optimization_plan": _plan,
    "runtime_requirements": _runtime_requirements,
    "analytics_post": _analytics,
    "example_feature_upload": _example_features,
    "best_config_fetch": _best_config_fetch,
    "label_dispatch": _labels,
    "workflow_traces": _workflow_traces,
}


@pytest.mark.parametrize("site", sorted(REFUSING_SITES))
def test_latched_credential_makes_every_refusing_site_send_nothing(site: str) -> None:
    with capture_egress() as cap:
        with pytest.raises(oo.OptimizationOnlyRefusal):
            asyncio.run(REFUSING_SITES[site]())
    assert cap.requests == [], [r.route for r in cap.requests]


def test_latch_for_another_credential_leaves_these_sites_alone() -> None:
    """Counter-check: with only a DIFFERENT credential latched the same sites do not refuse."""
    oo.reset_latch()
    oo.note_response(
        403, CLOSED, oo.credential_scope(BACKEND_URL, {"X-API-Key": "uk_" + "z" * 43})
    )
    with capture_egress() as cap:
        asyncio.run(_runtime_requirements_ok())
    assert [r for r in cap.requests if "production-requirements" in r.path]


async def _runtime_requirements_ok() -> None:
    from traigent.runtime.client import RuntimeClient

    rt = RuntimeClient("proj-1", base_url=BACKEND_URL, api_key=FAKE_API_KEY)
    try:
        rt.import_tvl({"tvl": "free text\nwith lines"})
    except Exception:  # noqa: BLE001 - the fake backend's answer is irrelevant
        pass


# ------------------------------------------------------------ projection sites


def _sync_manager() -> Any:
    from traigent.cloud.sync_manager import SyncManager
    from traigent.config.types import TraigentConfig

    return SyncManager(TraigentConfig(local_storage_path="."), api_key=FAKE_API_KEY)


def test_latched_credential_projects_the_sync_manager_bodies() -> None:
    with capture_egress() as cap:
        _sync_manager()._sync_create_session(
            json.loads(json.dumps(DIRTY_TYPED_SESSION))
        )
    (post,) = [
        r for r in cap.requests if r.method == "POST" and r.path.endswith("/sessions")
    ]
    assert "Alice" not in post.body.decode()
    assert json.loads(post.body) == oo.project_for_phase(
        oo.PHASE_SESSION_CREATE_AUTO, DIRTY_TYPED_SESSION
    )


def test_latched_credential_projects_optimizer_metric_bodies() -> None:
    from traigent.cloud.optimizer_client import OptimizerDirectClient

    token = "tok_" + "t" * 40
    oo.note_response(
        403,
        CLOSED,
        oo.credential_scope(
            f"{BACKEND_URL}/api/v1/hybrid/optimizer",
            {"Authorization": f"Bearer {token}"},
        ),
    )

    async def go() -> None:
        async with OptimizerDirectClient(
            f"{BACKEND_URL}/api/v1/hybrid/optimizer", token
        ) as direct:
            direct.set_batch_size(1)
            await direct.submit_metrics(
                "hybrid-oo",
                "trial-oo-1",
                {"accuracy": 0.5, "patient_diabetes": 1.0},
                0.1,
            )

    with capture_egress() as cap:
        asyncio.run(go())
    (post,) = [
        r for r in cap.requests if r.method == "POST" and "/hybrid/optimizer/" in r.path
    ]
    assert "patient_diabetes" not in post.body.decode()


def test_latched_credential_projects_best_config_publish() -> None:
    spec = {
        "schema_version": "traigent.best_config.v1",
        "config_id": "cfg-1",
        "config": {"model": "gpt-4o"},
        "provenance": {"source": "run-1", "note": "free text note"},
    }
    with capture_egress() as cap:
        _client().publish_best_config_sync(spec, environment="default")
    (post,) = [r for r in cap.requests if r.path.endswith("/best-configs")]
    assert "note" not in json.loads(post.body)["spec"]["provenance"]


# --------------------------------------------- enforcement: no guard without a scope


GUARDS = frozenset(
    {
        "is_active",
        "refuse_if_active",
        "refuse_embedded_examples",
        "assert_closed_body",
        "assert_closed_json_bytes",
        "assert_closed_ids",
        "assert_director_request",
        "project_for_send",
    }
)


_POLICY_FACTORIES = frozenset(
    {
        "policy_of",
        "policy_from_headers",
        "policy_from_environment",
        "Policy",
        "_optimization_only_policy",
    }
)


def _is_policy_expr(node: ast.AST, policy_names: set[str]) -> bool:
    """A real Policy-typed argument: a factory call, or a variable bound from one."""
    if isinstance(node, ast.Call):
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        return name in _POLICY_FACTORIES
    return isinstance(node, ast.Name) and node.id in policy_names


def _policy_names(fn: ast.AST) -> set[str]:
    names: set[str] = set()
    if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for arg in [*fn.args.args, *fn.args.kwonlyargs]:
            if arg.annotation is not None and "Policy" in ast.unparse(arg.annotation):
                names.add(arg.arg)
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and _is_policy_expr(node.value, names):
                names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


class _GuardVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.missing: list[int] = []
        self._names: list[set[str]] = [set()]

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        names = set(self._names[-1])
        for arg in [*node.args.args, *node.args.kwonlyargs]:
            if arg.annotation is not None and "Policy" in ast.unparse(arg.annotation):
                names.add(arg.arg)
        self._names.append(names)
        self.generic_visit(node)
        self._names.pop()

    visit_FunctionDef = _visit_function  # noqa: N815
    visit_AsyncFunctionDef = _visit_function  # noqa: N815

    def visit_Assign(self, node: ast.Assign) -> None:
        if _is_policy_expr(node.value, self._names[-1]):
            self._names[-1].update(
                t.id for t in node.targets if isinstance(t, ast.Name)
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        is_guard = name in GUARDS and (
            not isinstance(func, ast.Attribute)
            or (isinstance(func.value, ast.Name) and func.value.id == "oo_egress")
        )
        if is_guard:
            names = self._names[-1]
            ok = any(_is_policy_expr(a, names) for a in node.args) or any(
                (kw.arg == "scope" and not _is_none(kw.value))
                or (
                    kw.arg in ("explicit", "policy")
                    and _is_policy_expr(kw.value, names)
                )
                for kw in node.keywords
            )
            if not ok:
                self.missing.append(node.lineno)
        self.generic_visit(node)


def _guard_calls_without_scope(source: str) -> list[int]:
    """Guard calls with no real Policy argument: a policy factory call, a variable bound
    from one (or a parameter annotated ``Policy``), or a non-None ``scope=``. Aliased imports
    of the module are not followed. This is a lexical aid; the default-deny transport
    backstop is what actually guarantees coverage."""
    visitor = _GuardVisitor()
    visitor.visit(ast.parse(source))
    return sorted(set(visitor.missing))


def _is_none(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def test_no_guard_call_in_the_sdk_omits_the_credential_scope() -> None:
    root = Path(__file__).resolve().parents[2] / "traigent"
    offenders = {}
    for path in root.rglob("*.py"):
        if path.name == "optimization_only_egress.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "oo_egress" not in text:
            continue
        lines = _guard_calls_without_scope(text)
        if lines:
            offenders[str(path.relative_to(root))] = lines
    assert offenders == {}, offenders


def test_the_scope_checker_catches_an_unscoped_guard() -> None:
    assert _guard_calls_without_scope("oo_egress.refuse_if_active('x')") == [1]
    # a string that merely contains "scope" or "policy" no longer counts
    assert _guard_calls_without_scope("oo_egress.refuse_if_active('scope policy')") == [
        1
    ]
    assert _guard_calls_without_scope(
        "oo_egress.refuse_if_active('x', explicit=policy)"
    ) == [1]
    assert _guard_calls_without_scope("oo_egress.is_active(scope=None)") == [1]
    assert (
        _guard_calls_without_scope(
            "def f(policy: oo_egress.Policy):\n    oo_egress.is_active(policy)"
        )
        == []
    )
    assert (
        _guard_calls_without_scope(
            "def f(self):\n    p = oo_egress.policy_of(self)\n    oo_egress.refuse_if_active('x', explicit=p)"
        )
        == []
    )
    assert _guard_calls_without_scope(
        "oo_egress.refuse_if_active('x', explicit=p)"
    ) == [1]
    assert (
        _guard_calls_without_scope(
            "oo_egress.refuse_if_active('x', explicit=oo_egress.policy_of(self))"
        )
        == []
    )


# ------------------------------------- mode off: no extra header call, no new failure


def test_mode_off_label_dispatch_makes_no_header_call_and_cannot_fail_session_create(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review 2 / P3: a header-refresh failure used to fail a session that was created."""
    from types import SimpleNamespace
    from unittest import mock

    from traigent.cloud.api_operations import ApiOperations

    oo.reset_latch()
    header_calls = {"n": 0}

    async def refresh_fails(headers: dict[str, str]) -> dict[str, str]:
        header_calls["n"] += 1
        raise RuntimeError("token refresh failed")

    client = SimpleNamespace(
        optimization_only=None,
        backend_config=SimpleNamespace(api_base_url=f"{BACKEND_URL}/api/v1"),
        auth_manager=SimpleNamespace(augment_headers=refresh_fails),
    )
    sender_from_config = mock.MagicMock(return_value=None)  # no label writer configured
    monkeypatch.setattr(
        "traigent.labels.transport.sender_from_config", sender_from_config
    )
    monkeypatch.setattr(
        "traigent.cloud.content_less.label_texts", lambda _r: {"agent": "Alice's agent"}
    )
    result = SimpleNamespace(content_less_refs={"agent": "agentref_" + "a1" * 16})

    asyncio.run(ApiOperations(client)._send_content_less_labels(object(), result))

    assert header_calls["n"] == 0
    assert sender_from_config.call_count == 1  # the writer lookup still happens
