"""Agent identity labels: shared vectors (G1), behaviour (G4-narrow), broken propagation (G5).

The vendored vectors (``fixtures/identity_vectors_v1.json``) are replaced by the
TraigentSchema package once it is published; until then the sha256 pin below
keeps this SDK, the JS SDK and the backend on the same file. Every test drives
the real production functions: ``traigent.optimize``, ``observe`` and the real
``ObservabilityClient`` capturing what its transport would send.
"""

from __future__ import annotations

import asyncio
import contextvars
import copy
import functools
import hashlib
import json
import os
import threading
import time
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

import traigent
from traigent.identity import agent_scope as agent_scope_module
from traigent.identity.agent_scope import (
    AgentScope,
    ScopeLedger,
    current_scope,
    record_failed_boundary,
    reset_trace_ledger,
    set_trace_ledger,
)
from traigent.identity.workload import (
    AgentBinding,
    EnvironmentTooLongError,
    canonical_agent_key,
    reset_environment_warnings,
    resolve_environment,
)
from traigent.observability import ObservabilityClient, ObservabilityConfig, observe
from traigent.observability import client as client_module

VECTORS_PATH = Path(__file__).parent / "fixtures" / "identity_vectors_v1.json"
VECTORS_SHA256 = "294814bec6a947b3c32cc4ec1c438b16f2c3fdcca815adb4883258ac7f0ff4d4"
_VECTORS: dict[str, Any] = json.loads(VECTORS_PATH.read_text("utf-8"))

_ENV_VARS = (
    "TRAIGENT_ENVIRONMENT",
    "OTEL_RESOURCE_ATTRIBUTES",
    "ENVIRONMENT",
    "TRAIGENT_ENV",
    "TRAIGENT_EXPERIMENT_NAME",
    "TRAIGENT_AGENT_ID",
    "NODE_ENV",
    "FLASK_ENV",
    "AWS_EXECUTION_ENV",
    "KUBERNETES_SERVICE_HOST",
)


@pytest.fixture(autouse=True)
def _clean_identity_env(monkeypatch):
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    # The suite's local-backend URL check needs a non-production environment.
    monkeypatch.setenv("ENVIRONMENT", "development")
    reset_environment_warnings()


def _optimize(fn: Callable[..., Any], **kwargs: Any) -> Any:
    kwargs.setdefault("objectives", ["accuracy"])
    kwargs.setdefault("configuration_space", {"model": ["a", "b"]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return traigent.optimize(**kwargs)(fn)


class Harness:
    def __init__(self, **config: Any) -> None:
        self.batches: list[list[dict[str, Any]]] = []
        self._lock = threading.Lock()

        def sender(batch: list[dict[str, Any]]) -> None:
            with self._lock:
                self.batches.append(batch)

        self.client = ObservabilityClient(
            ObservabilityConfig(
                backend_origin="http://localhost:5000",
                enable_atexit_flush=False,
                **config,
            ),
            sender=sender,
        )
        # Every captured snapshot in capture order. The transport keeps only the
        # newest per trace, so the end_trace capture itself is observed here.
        self.captured: list[dict[str, Any]] = []
        submit = self.client._submit_trace_snapshot

        def record(trace_id: str, payload: dict[str, Any], **kwargs: Any) -> None:
            self.captured.append(copy.deepcopy(payload))
            return submit(trace_id, payload, **kwargs)

        self.client._submit_trace_snapshot = record  # type: ignore[method-assign]

    def snapshots(self, name: str) -> list[dict[str, Any]]:
        with self._lock:
            return [t for b in self.batches for t in b if t["name"] == name]

    def agent_id(self, name: str) -> str | None:
        """The agent_id on the last (final) snapshot of the named trace."""
        self.client.flush()
        snaps = self.snapshots(name)
        assert snaps, f"no snapshot for {name}"
        return (snaps[-1].get("execution_context") or {}).get("agent_id")

    def final(self, name: str) -> dict[str, Any]:
        self.client.flush()
        return self.snapshots(name)[-1]


@pytest.fixture
def harness():
    h = Harness()
    yield h
    h.client.close()


# Real agents. Their __name__ is the registration name, so each key is a
# function_derived key.


async def _stream(n: int = 3):
    for i in range(n):
        await asyncio.sleep(0)
        yield i


async def _awaited_body() -> int:
    await asyncio.sleep(0)
    return 1


async def _iterated_body() -> int:
    return sum([i async for i in _stream()])


def _agent(name: str, body: Callable[..., Any] = _awaited_body, **opts: Any) -> Any:
    async def run(*args: Any) -> Any:
        return await body(*args)

    run.__name__ = name
    run.__qualname__ = name
    return _optimize(run, **opts)


def _key(agent: Any) -> str:
    key = agent.agent_binding().key
    assert key is not None
    return key


# --------------------------------------------------------------------------- G1


def test_vendored_vectors_are_the_pinned_file() -> None:
    assert hashlib.sha256(VECTORS_PATH.read_bytes()).hexdigest() == VECTORS_SHA256


def _python_provenance_vectors() -> list[dict[str, Any]]:
    return [v for v in _VECTORS["provenance"] if v["input"]["sdk"] == "python"]


@pytest.mark.parametrize("vector", _python_provenance_vectors(), ids=lambda v: v["id"])
def test_g1_python_provenance_vectors(vector, monkeypatch) -> None:
    inp, exp = vector["input"], vector["expected"]
    if exp.get("outcome") == "unsupported":
        with pytest.raises(AttributeError):
            _optimize(functools.partial(print), objectives=inp["objectives"])
        return

    def probe(*args: Any, **kwargs: Any) -> None:
        return None

    probe.__name__ = inp["function_name"]
    probe.__qualname__ = inp.get("qualname") or inp["function_name"]
    opts: dict[str, Any] = {}
    if inp.get("objectives"):
        opts["objectives"] = inp["objectives"]
    if inp.get("agent_name"):
        opts["agent_name"] = inp["agent_name"]
    if inp.get("experiment_name_env"):
        monkeypatch.setenv("TRAIGENT_EXPERIMENT_NAME", inp["experiment_name_env"])
    agent = _optimize(
        probe, configuration_space={k: [1, 2] for k in inp["knobs"]}, **opts
    )

    binding = agent.agent_binding()
    assert agent.experiment_name == exp["registration_identity"]
    assert binding.registration_identity == exp["registration_identity"]
    assert binding.provenance == exp["provenance"]
    assert binding.key == exp["trace_agent_id"]
    assert binding.local_warning == exp["local_warning"]
    assert (
        canonical_agent_key(binding.registration_identity) == exp["backend_agent_key"]
    )


@pytest.mark.parametrize(
    "vector",
    [
        v
        for v in _VECTORS["provenance"]
        if v["input"]["sdk"] == "js" and v["expected"].get("backend_agent_key")
    ],
    ids=lambda v: v["id"],
)
def test_g1_backend_key_for_js_registration_identities(vector) -> None:
    exp = vector["expected"]
    assert canonical_agent_key(exp["registration_identity"]) == exp["backend_agent_key"]


@pytest.mark.parametrize("vector", _VECTORS["environment"], ids=lambda v: v["id"])
def test_g1_environment_vectors(vector, monkeypatch) -> None:
    inp, exp = vector["input"], vector["expected"]
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in inp["env"].items():
        monkeypatch.setenv(name, value)
    if not exp["valid"]:
        assert exp["error"] == "too_long"
        with pytest.raises(EnvironmentTooLongError):
            resolve_environment(inp.get("explicit"))
        return
    got = resolve_environment(inp.get("explicit"))
    assert (got.environment, got.source, got.warning) == (
        exp["environment"],
        exp["source"],
        exp["warning"],
    )


def test_g1_known_environment_labels_match_env_config() -> None:
    from traigent.utils.env_config import _KNOWN_ENVIRONMENT_NAMES

    assert set(_VECTORS["closed_sets"]["known_environment_labels"]) == set(
        _KNOWN_ENVIRONMENT_NAMES
    )


def test_g1_environment_warnings_never_echo_values(monkeypatch, caplog) -> None:
    secret = "alice@example.com"
    monkeypatch.setenv("TRAIGENT_ENV", secret)
    monkeypatch.setenv("TRAIGENT_ENVIRONMENT", " " * 70)
    resolve_environment()
    assert secret not in caplog.text
    assert "TRAIGENT_ENVIRONMENT" in caplog.text


def _native_scope_vectors() -> list[dict[str, Any]]:
    return [
        v
        for v in _VECTORS["trace_attribution"]
        if v["input"]["transport"] == "native"
        and v["input"].get("scopes")
        and v["input"]["final_snapshot_delivered"]
    ]


@pytest.mark.parametrize("vector", _native_scope_vectors(), ids=lambda v: v["id"])
def test_g1_native_scope_attribution_vectors(vector) -> None:
    ledger = ScopeLedger()
    for spec in vector["input"]["scopes"]:
        scope = AgentScope(AgentBinding("x", spec["provenance"], spec["key"]))
        ledger.register_open(scope)
        if spec["state_at_finalization"] == "closed":
            ledger.record_close(scope)
    decision = ledger.decide()
    assert (decision.key, decision.reason) == (
        vector["expected"]["attributed_to"],
        vector["expected"]["reason"],
    )


def test_g1_native_no_scopes_is_ineligible() -> None:
    vector = next(
        v for v in _VECTORS["trace_attribution"] if v["id"] == "attr-native-no-scopes"
    )
    decision = ScopeLedger().decide()
    assert (decision.key, decision.reason) == (
        vector["expected"]["attributed_to"],
        vector["expected"]["reason"],
    )


def test_g1_legacy_trace_level_agent_id_is_untouched(harness) -> None:
    vector = next(
        v
        for v in _VECTORS["trace_attribution"]
        if v["id"] == "attr-legacy-trace-level-only"
    )
    explicit = vector["input"]["legacy_trace_level_agent_id"]
    trace_id = harness.client.start_trace(
        "legacy", execution_context={"agent_id": explicit}
    )
    harness.client.end_trace(trace_id)
    assert harness.agent_id("legacy") == vector["expected"]["attributed_to"]


def test_g1_early_flush_without_final_is_never_attributed(harness) -> None:
    vector = next(
        v
        for v in _VECTORS["trace_attribution"]
        if v["id"] == "attr-native-early-flush-no-final"
    )
    agent_ref: dict[str, Any] = {}

    async def body() -> None:
        harness.client.start_trace("never-finalized")  # inside the agent scope
        harness.client.flush()

    agent_ref["agent"] = _agent("agent_early", body)
    asyncio.run(agent_ref["agent"]())
    harness.client.flush()
    snaps = harness.snapshots("never-finalized")
    assert snaps and vector["expected"]["attributed_to"] is None
    assert all("agent_id" not in (s.get("execution_context") or {}) for s in snaps)


# --------------------------------------------------------------------------- G4


def _root(h: Harness, name: str, inner: Callable[..., Any]) -> Callable[..., Any]:
    @observe(name, client=h.client)
    async def root() -> Any:
        return await inner()

    return root


BODIES = {"awaited": _awaited_body, "iterated": _iterated_body}


@pytest.mark.parametrize("body", sorted(BODIES))
@pytest.mark.parametrize("order", ["optimize_over_observe", "observe_over_optimize"])
def test_g4_both_decorator_orders_attribute_the_agent(harness, order, body) -> None:
    name = f"trace-{order}-{body}"
    if order == "observe_over_optimize":
        agent = _agent("agent_ordered", BODIES[body])
        root = observe(name, client=harness.client)(agent)
    else:

        @observe(name, client=harness.client)
        async def observed() -> Any:
            return await BODIES[body]()

        observed.__name__ = "agent_ordered"
        agent = _optimize(observed)
        root = agent
    asyncio.run(root())
    assert harness.agent_id(name) == _key(agent)


def test_g4_concurrent_agents_in_tasks_keep_their_own_label(harness) -> None:
    a, b = _agent("agent_a", _iterated_body), _agent("agent_b", _iterated_body)
    root_a = observe("trace-a", client=harness.client)(a)
    root_b = observe("trace-b", client=harness.client)(b)

    async def both() -> None:
        await asyncio.gather(*[f() for _ in range(5) for f in (root_a, root_b)])

    asyncio.run(both())
    harness.client.flush()
    for name, agent in (("trace-a", a), ("trace-b", b)):
        finals = {}
        for snap in harness.snapshots(name):
            finals[snap["id"]] = snap
        assert len(finals) == 5
        assert {
            (s.get("execution_context") or {}).get("agent_id") for s in finals.values()
        } == {_key(agent)}


def test_g4_concurrent_agents_in_threads_keep_their_own_label(harness) -> None:
    def sync_agent(name: str) -> Any:
        def run() -> int:
            time.sleep(0.01)
            return 1

        run.__name__ = name
        return _optimize(run)

    a, b = sync_agent("thread_a"), sync_agent("thread_b")
    root_a = observe("thread-trace-a", client=harness.client)(a)
    root_b = observe("thread-trace-b", client=harness.client)(b)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(f) for _ in range(6) for f in (root_a, root_b)]
        for future in futures:
            future.result()
    harness.client.flush()
    for name, agent in (("thread-trace-a", a), ("thread-trace-b", b)):
        ids = {
            (s.get("execution_context") or {}).get("agent_id")
            for s in {x["id"]: x for x in harness.snapshots(name)}.values()
        }
        assert ids == {_key(agent)}


def test_g4_nested_different_agents_are_unset(harness) -> None:
    b = _agent("agent_inner_b")

    async def outer_body() -> Any:
        return await b()

    a = _agent("agent_outer_a", outer_body)
    asyncio.run(_root(harness, "nested", a)())
    assert harness.agent_id("nested") is None


def test_g4_nested_same_key_is_attributed(harness) -> None:
    inner = _agent("agent_same")

    async def body() -> Any:
        return await inner()

    outer = _agent("agent_same", body)
    asyncio.run(_root(harness, "same-key", outer)())
    assert harness.agent_id("same-key") == _key(outer) == _key(inner)


def test_g4_agent_then_user_named_child_is_unset(harness) -> None:
    named = _agent("agent_named_child", agent_name="support-bot")

    async def body() -> Any:
        return await named()

    a = _agent("agent_parent", body)
    asyncio.run(_root(harness, "user-named-child", a)())
    assert harness.agent_id("user-named-child") is None


def test_g4_agent_then_lambda_child_is_unset(harness) -> None:
    anon = _optimize(lambda: 1)
    assert anon.agent_binding().provenance == "ineligible"

    async def body() -> Any:
        return anon()

    a = _agent("agent_lambda_parent", body)
    asyncio.run(_root(harness, "lambda-child", a)())
    assert harness.agent_id("lambda-child") is None


def test_g4_two_lambdas_with_identical_knobs_are_unset(harness) -> None:
    first, second = _optimize(lambda: 1), _optimize(lambda: 2)

    async def body() -> Any:
        return first() + second()

    asyncio.run(_root(harness, "two-lambdas", body)())
    assert harness.agent_id("two-lambdas") is None


def test_g4_user_named_agent_alone_is_unset(harness, monkeypatch) -> None:
    named = _agent("agent_declared", agent_name="support-bot")
    asyncio.run(observe("declared", client=harness.client)(named)())
    assert harness.agent_id("declared") is None

    monkeypatch.setenv("TRAIGENT_EXPERIMENT_NAME", "team-experiment")
    env_named = _agent("agent_env_declared")
    asyncio.run(observe("env-declared", client=harness.client)(env_named)())
    assert harness.agent_id("env-declared") is None


def test_g4_outer_scope_open_when_inner_closes_is_unset(harness) -> None:
    inner = _agent("agent_inner_closed", _awaited_body)

    def streaming_outer() -> Any:
        yield 1
        yield 2

    outer = _optimize(streaming_outer)
    opened: list[Any] = []

    @observe("outer-open", client=harness.client)
    def root() -> None:
        stream = outer()
        next(stream)  # outer scope is now open (suspended generator)
        opened.append(stream)
        asyncio.run(inner())  # inner closes while outer is still open

    root()
    try:
        assert harness.agent_id("outer-open") is None
    finally:
        opened[0].close()


def test_g4_outer_ineligible_scope_open_is_unset(harness) -> None:
    inner = _agent("agent_inner_ok")

    def streaming_anon() -> Any:
        yield 1

    streaming_anon.__name__ = "<lambda>"
    anon = _optimize(streaming_anon)
    assert anon.agent_binding().provenance == "ineligible"
    opened: list[Any] = []

    @observe("outer-ineligible-open", client=harness.client)
    def root() -> None:
        stream = anon()
        next(stream)
        opened.append(stream)
        asyncio.run(inner())

    root()
    try:
        assert harness.agent_id("outer-ineligible-open") is None
    finally:
        opened[0].close()


def test_g4_unscoped_handler_root_with_one_eligible_scope_is_attributed(
    harness,
) -> None:
    agent = _agent("agent_leaf")

    @observe("unscoped-child", client=harness.client)
    async def unscoped_child() -> int:
        return await _awaited_body()

    @observe("handler-root", client=harness.client)
    async def handler() -> Any:
        await unscoped_child()
        return await agent()

    asyncio.run(handler())
    assert harness.agent_id("handler-root") == _key(agent)


def _enclosing(h: Harness, trace: str, name: str, inner: Callable[..., Any]) -> Any:
    async def body() -> Any:
        async with observe(trace, client=h.client):
            return await inner()

    body.__name__ = name
    return _optimize(body)


def test_g4_enclosing_agent_alone_is_attributed(harness) -> None:
    outer = _enclosing(harness, "enclosed-alone", "agent_enclosing", _awaited_body)
    asyncio.run(outer())
    assert harness.agent_id("enclosed-alone") == _key(outer)


def test_g4_enclosing_agent_plus_inner_different_agent_is_unset(harness) -> None:
    inner = _agent("agent_inner_other")
    outer = _enclosing(harness, "enclosed-conflict", "agent_enclosing", inner)
    asyncio.run(outer())
    assert harness.agent_id("enclosed-conflict") is None


def test_g4_enclosing_ineligible_scope_is_unset(harness) -> None:
    outer = _enclosing(harness, "enclosed-anon", "<lambda>", _awaited_body)
    assert outer.agent_binding().provenance == "ineligible"
    asyncio.run(outer())
    assert harness.agent_id("enclosed-anon") is None


def test_g4_forced_early_flush_never_carries_the_scope_agent_id() -> None:
    h = Harness(batch_size=1, max_buffer_age=0.01)
    try:

        async def slow_body() -> int:
            await asyncio.sleep(0.05)
            h.client.flush()  # forced early flush while the scope is open
            await asyncio.sleep(0.05)  # lets the timer flush as well
            return 1

        agent = _agent("agent_flushy", slow_body)
        asyncio.run(observe("early-flush", client=h.client)(agent)())
        h.client.flush()
        snaps = h.snapshots("early-flush")
        early = [s for s in snaps if not s.get("ended_at")]
        final = [s for s in snaps if s.get("ended_at")]
        assert early, "the test must produce snapshots before end_trace"
        assert all("agent_id" not in (s.get("execution_context") or {}) for s in early)
        assert (final[-1].get("execution_context") or {}).get("agent_id") == _key(agent)
    finally:
        h.client.close()


def test_g4_explicit_agent_id_is_never_replaced(harness) -> None:
    agent = _agent("agent_not_used")
    asyncio.run(
        observe(
            "explicit-id",
            client=harness.client,
            execution_context={"agent_id": "explicit-agent"},
        )(agent)()
    )
    assert harness.agent_id("explicit-id") == "explicit-agent"
    assert all(
        s["execution_context"]["agent_id"] == "explicit-agent"
        for s in harness.snapshots("explicit-id")
    )


def test_g4_scope_is_restored_after_exception_and_cancellation() -> None:
    boom = _agent("agent_boom", _raiser)
    inner_failed = _agent("agent_inner_fail", _raiser)
    seen: dict[str, Any] = {}

    async def outer_body() -> None:
        outer_scope = current_scope()
        # Without this the restore asserts below pass trivially when scope
        # installation is disabled (both sides would be None).
        assert outer_scope is not None
        assert outer_scope.binding.key == _key(outer)
        with pytest.raises(RuntimeError):
            await inner_failed()
        assert current_scope() is outer_scope
        task = asyncio.ensure_future(_sleeper_agent()())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert current_scope() is outer_scope
        assert current_scope() is not None
        seen["ok"] = True

    outer = _agent("agent_restores", outer_body)
    assert current_scope() is None

    async def run() -> None:
        await outer()
        with pytest.raises(RuntimeError):
            await boom()
        assert current_scope() is None

    asyncio.run(run())
    assert seen == {"ok": True}
    assert current_scope() is None


async def _raiser() -> None:
    await asyncio.sleep(0)
    raise RuntimeError("boom")


def _sleeper_agent() -> Any:
    async def long_body() -> None:
        await asyncio.sleep(30)

    return _agent("agent_sleeper", long_body)


def test_g4_scope_follows_sync_and_async_generator_steps() -> None:
    observed: list[str | None] = []

    def gen() -> Any:
        for _ in range(2):
            scope = current_scope()
            observed.append(scope.binding.key if scope else None)
            yield 1

    async def agen() -> Any:
        for _ in range(2):
            scope = current_scope()
            observed.append(scope.binding.key if scope else None)
            await asyncio.sleep(0)
            yield 1

    sync_agent, async_agent = _optimize(gen), _optimize(agen)
    assert list(sync_agent()) == [1, 1]
    assert current_scope() is None

    async def drain() -> None:
        assert [x async for x in async_agent()] == [1, 1]
        assert current_scope() is None

    asyncio.run(drain())
    assert observed == [_key(sync_agent)] * 2 + [_key(async_agent)] * 2


# --------------------------------------------------------------------------- G5


def test_g5_work_in_a_thread_pool_without_context_copy_is_unset(harness) -> None:
    sync_agent = _optimize(_make_named_sync("agent_pool"))

    async def handler() -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, sync_agent)

    asyncio.run(observe("pool-no-copy", client=harness.client)(handler)())
    assert harness.agent_id("pool-no-copy") is None


def test_g5_work_with_context_copy_is_attributed_control(harness) -> None:
    sync_agent = _optimize(_make_named_sync("agent_pool_ctx"))

    async def handler() -> None:
        loop = asyncio.get_running_loop()
        ctx = contextvars.copy_context()
        await loop.run_in_executor(None, ctx.run, sync_agent)

    asyncio.run(observe("pool-with-copy", client=harness.client)(handler)())
    assert harness.agent_id("pool-with-copy") == _key(sync_agent)


def test_g5_fresh_event_loop_in_a_thread_is_unset(harness) -> None:
    agent = _agent("agent_fresh_loop")

    @observe("fresh-loop", client=harness.client)
    def handler() -> None:
        worker = threading.Thread(target=lambda: asyncio.run(agent()))
        worker.start()
        worker.join()

    handler()
    assert harness.agent_id("fresh-loop") is None


def _make_named_sync(name: str) -> Callable[..., int]:
    def run() -> int:
        return 1

    run.__name__ = name
    return run


# ---------------------------------------------- sealed ledger, atomic finalization


def _final_snapshots(h: Harness, name: str) -> list[dict[str, Any]]:
    h.client.flush()
    return [s for s in h.snapshots(name) if s.get("ended_at")]


def _snapshot_agent_id(snapshot: dict[str, Any]) -> str | None:
    return (snapshot.get("execution_context") or {}).get("agent_id")


def test_g4_negative_attribution_cases_have_positive_controls(harness) -> None:
    """Every "is unset" shape is paired with a control that differs only in the
    poisoning element, so a suite that attributes nothing cannot pass."""

    def nested(trace: str, child: Any, parent_name: str = "agent_ctl_parent") -> Any:
        async def body() -> Any:
            result = child()
            return await result if asyncio.iscoroutine(result) else result

        parent = _agent(parent_name, body)
        asyncio.run(_root(harness, trace, parent)())
        return parent

    # different agent child: suppressed; same-key child: attributed.
    nested("ctl-diff", _agent("agent_ctl_other"))
    assert harness.agent_id("ctl-diff") is None
    same = nested("ctl-same", _agent("agent_ctl_parent"))
    assert harness.agent_id("ctl-same") == _key(same)

    # user-named child: suppressed; function-derived child: attributed.
    nested("ctl-named", _agent("agent_ctl_named", agent_name="support-bot"))
    assert harness.agent_id("ctl-named") is None
    same = nested("ctl-derived", _agent("agent_ctl_parent"))
    assert harness.agent_id("ctl-derived") == _key(same)

    # lambda child: suppressed; named-function child: attributed.
    nested("ctl-lambda", _optimize(lambda: 1))
    assert harness.agent_id("ctl-lambda") is None
    same = nested("ctl-def", _optimize(_make_named_sync("agent_ctl_parent")))
    assert harness.agent_id("ctl-def") == _key(same)

    # user-named agent alone: suppressed; function-derived alone: attributed.
    named = _agent("agent_ctl_alone", agent_name="support-bot")
    asyncio.run(observe("ctl-named-alone", client=harness.client)(named)())
    assert harness.agent_id("ctl-named-alone") is None
    plain = _agent("agent_ctl_alone")
    asyncio.run(observe("ctl-plain-alone", client=harness.client)(plain)())
    assert harness.agent_id("ctl-plain-alone") == _key(plain)

    # outer scope still open: suppressed; the same outer closed first: attributed.
    def two_step_outer() -> Any:
        yield 1
        yield 2

    two_step_outer.__name__ = "agent_ctl_outer"
    outer = _optimize(two_step_outer)
    inner = _agent("agent_ctl_outer")  # same key as outer
    held: list[Any] = []

    @observe("ctl-open", client=harness.client)
    def open_root() -> None:
        stream = outer()
        next(stream)
        held.append(stream)
        asyncio.run(inner())

    @observe("ctl-closed", client=harness.client)
    def closed_root() -> None:
        assert list(outer()) == [1, 2]
        asyncio.run(inner())

    open_root()
    closed_root()
    try:
        assert harness.agent_id("ctl-open") is None
        assert harness.agent_id("ctl-closed") == _key(outer)
    finally:
        held[0].close()


def test_wire_sync_generator_agent_finalization(harness) -> None:
    def stream() -> Any:
        yield 1
        yield 2

    stream.__name__ = "agent_sync_gen"
    agent = _optimize(stream)

    @observe("gen-drained", client=harness.client)
    def drained() -> list[int]:
        return list(agent())

    assert drained() == [1, 2]
    finals = _final_snapshots(harness, "gen-drained")
    assert finals and _snapshot_agent_id(finals[-1]) == _key(agent)
    assert all(
        _snapshot_agent_id(s) is None
        for s in harness.snapshots("gen-drained")
        if not s.get("ended_at")
    )

    held: list[Any] = []

    @observe("gen-undrained", client=harness.client)
    def undrained() -> None:
        stream_ = agent()
        next(stream_)
        held.append(stream_)

    undrained()
    assert all(
        _snapshot_agent_id(s) is None
        for s in _final_snapshots(harness, "gen-undrained")
    )
    held[0].close()  # the scope closes after the trace was finalized
    harness.client.flush()
    assert all(
        _snapshot_agent_id(s) is None
        for s in _final_snapshots(harness, "gen-undrained")
    )


def test_wire_async_generator_agent_finalization(harness) -> None:
    async def stream() -> Any:
        for i in range(2):
            await asyncio.sleep(0)
            yield i

    stream.__name__ = "agent_async_gen"
    agent = _optimize(stream)

    @observe("agen-drained", client=harness.client)
    async def drained() -> list[int]:
        return [x async for x in agent()]

    assert asyncio.run(drained()) == [0, 1]
    finals = _final_snapshots(harness, "agen-drained")
    assert finals and _snapshot_agent_id(finals[-1]) == _key(agent)
    assert all(
        _snapshot_agent_id(s) is None
        for s in harness.snapshots("agen-drained")
        if not s.get("ended_at")
    )

    held: list[Any] = []

    @observe("agen-undrained", client=harness.client)
    async def undrained() -> None:
        stream_ = agent()
        await stream_.__anext__()
        held.append(stream_)

    async def run() -> None:
        await undrained()
        harness.client.flush()
        assert all(
            _snapshot_agent_id(s) is None
            for s in _final_snapshots(harness, "agen-undrained")
        )
        await held[0].aclose()  # scope closes after the trace was finalized

    asyncio.run(run())
    harness.client.flush()
    undrained_finals = _final_snapshots(harness, "agen-undrained")
    assert undrained_finals
    assert all(_snapshot_agent_id(s) is None for s in undrained_finals)


def test_item2_late_registration_withdraws_the_generated_agent_id(harness) -> None:
    a = _agent("agent_early_a")
    b = _agent("agent_late_b", agent_name="late-named")

    async def main() -> None:
        gate = asyncio.Event()
        box: dict[str, Any] = {}

        async def deferred() -> None:
            await gate.wait()
            await b()  # registers in the already-finalized trace
            harness.client.record_observation(box["trace_id"], name="late-obs")

        @observe("late-registration", client=harness.client)
        async def root() -> None:
            await a()
            box["task"] = asyncio.ensure_future(deferred())

        await root()
        harness.client.flush()
        snap = harness.snapshots("late-registration")[-1]
        box["trace_id"] = snap["id"]
        # control: right after finalization the generated id IS on the wire.
        assert snap.get("ended_at") and _snapshot_agent_id(snap) == _key(a)
        gate.set()
        await box["task"]

    asyncio.run(main())
    harness.client.flush()
    last = harness.snapshots("late-registration")[-1]
    assert "late-obs" in json.dumps(last)  # a later snapshot, after the finalize
    assert last.get("ended_at")
    assert _snapshot_agent_id(last) is None
    # another end_trace cannot bring it back either
    harness.client.end_trace(last["id"])
    harness.client.flush()
    assert _snapshot_agent_id(harness.snapshots("late-registration")[-1]) is None


def test_item2_explicit_agent_id_survives_late_registration(harness) -> None:
    a = _agent("agent_explicit_a")
    trace_id = harness.client.start_trace(
        "late-explicit", execution_context={"agent_id": "explicit-agent"}
    )
    ledger = harness.client.scope_ledger(trace_id)
    token = set_trace_ledger(ledger)
    try:
        scope = AgentScope(binding=a.agent_binding())
        with scope.active():
            pass
        scope.finish()
        harness.client.end_trace(trace_id)
        late = AgentScope(binding=AgentBinding("late", "user_named", None))
        with late.active():
            pass
        late.finish()
    finally:
        reset_trace_ledger(token)
    harness.client.record_observation(trace_id, name="after")
    harness.client.flush()
    assert all(
        _snapshot_agent_id(s) == "explicit-agent"
        for s in harness.snapshots("late-explicit")
    )


def _sync_agent(name: str) -> Any:
    def run() -> int:
        return 1

    run.__name__ = name
    run.__qualname__ = name
    return _optimize(run)


def _register_scope_in(ledger: ScopeLedger, binding: AgentBinding) -> None:
    token = set_trace_ledger(ledger)
    try:
        scope = AgentScope(binding=binding)
        with scope.active():
            pass
        scope.finish()
    finally:
        reset_trace_ledger(token)


def _first_final(h: Harness, name: str) -> dict[str, Any]:
    """The first ended snapshot captured for the trace (end_trace's own)."""
    finals = [s for s in h.captured if s["name"] == name and s.get("ended_at")]
    assert finals
    return finals[0]


def test_item3_registration_during_capture_lands_before_the_decision(
    harness, monkeypatch
) -> None:
    """Handshake at the capture seam: a registration made between serialization
    and the identity decision is counted, so the end_trace snapshot has no id.

    The hook runs after the trace is serialized and before the ledger decides;
    it starts a registering worker and waits for it. Waiting only terminates
    because no ledger lock is held across the seam (the join timeout is a hang
    guard, not the assertion)."""
    a = _agent("agent_atomic_a")
    trace_id = harness.client.start_trace("atomic-final")
    ledger = harness.client.scope_ledger(trace_id)
    _register_scope_in(ledger, a.agent_binding())

    at_seam = threading.Event()
    worker_done: list[bool] = []

    def hook() -> None:
        at_seam.set()
        worker = threading.Thread(
            target=_register_scope_in,
            args=(ledger, AgentBinding("late", "user_named", None)),
        )
        worker.start()
        worker.join(timeout=10)
        worker_done.append(not worker.is_alive())

    monkeypatch.setattr(client_module, "_after_serialize_hook", hook)
    harness.client.end_trace(trace_id)
    monkeypatch.setattr(client_module, "_after_serialize_hook", None)

    assert at_seam.is_set() and worker_done == [True]
    assert _snapshot_agent_id(_first_final(harness, "atomic-final")) is None


def test_item3_control_without_the_racer_the_agent_is_attributed(harness) -> None:
    a = _agent("agent_atomic_a")
    trace_id = harness.client.start_trace("atomic-ctl")
    _register_scope_in(harness.client.scope_ledger(trace_id), a.agent_binding())
    harness.client.end_trace(trace_id)
    assert _snapshot_agent_id(_first_final(harness, "atomic-ctl")) == _key(a)


class _CallsAgent:
    """Metadata object whose serialization runs caller code."""

    # The client deep-copies metadata at intake, so the switch and the action
    # live in a shared dict (functions and dicts-by-reference survive the copy
    # only when held in class state).
    switch: dict[str, Any] = {}

    def to_dict(self) -> dict[str, Any]:
        if self.switch.get("armed"):
            self.switch["act"]()
        return {"k": "v"}


def _end_with_serializer(harness, name: str, act_factory: Any) -> dict[str, Any]:
    a = _agent("agent_serial_a")
    meta = _CallsAgent()
    trace_id = harness.client.start_trace(name, metadata={"obj": meta})
    ledger = harness.client.scope_ledger(trace_id)
    _register_scope_in(ledger, a.agent_binding())
    _CallsAgent.switch.update(act=act_factory(ledger), armed=True)
    token = set_trace_ledger(ledger)  # the serializing thread's ambient trace
    try:
        harness.client.end_trace(trace_id)
    finally:
        reset_trace_ledger(token)
    _CallsAgent.switch.clear()
    return _first_final(harness, name)


def test_item2_reentrant_registration_in_a_serializer_withholds_the_id(
    harness,
) -> None:
    b = _sync_agent("agent_user_named_b")
    user_named = AgentBinding("support-bot", "user_named", None)

    def factory(_ledger: ScopeLedger) -> Callable[[], None]:
        def act() -> None:
            scope = AgentScope(binding=user_named)
            with scope.active():  # same thread: ledger is the ambient one
                b()
            scope.finish()

        return act

    snap = _end_with_serializer(harness, "serial-reentrant", factory)
    assert _snapshot_agent_id(snap) is None

    ctl = _end_with_serializer(harness, "serial-ctl", lambda _l: lambda: None)
    assert _snapshot_agent_id(ctl) == _key(_agent("agent_serial_a"))


def test_item2_serializer_waiting_for_copied_context_work_does_not_deadlock(
    harness,
) -> None:
    b = _sync_agent("agent_copied_ctx_b")
    outcome: list[bool] = []

    def factory(_ledger: ScopeLedger) -> Callable[[], None]:
        def act() -> None:
            ctx = contextvars.copy_context()

            def work() -> None:
                scope = AgentScope(binding=b.agent_binding())
                with scope.active():
                    b()
                scope.finish()

            worker = threading.Thread(target=lambda: ctx.run(work))
            worker.start()
            worker.join(timeout=10)  # hang guard only
            outcome.append(not worker.is_alive())

        return act

    snap = _end_with_serializer(harness, "serial-copied", factory)
    assert outcome == [True]  # completed: no deadlock
    assert _snapshot_agent_id(snap) is None  # agent B joined the trace


def test_late_setup_failure_after_seal_withdraws_generated_id_keeps_explicit(
    harness, monkeypatch
) -> None:
    a = _agent("agent_late_setup_a")

    def run_case(name: str, **ctx: Any) -> tuple[str, ScopeLedger]:
        trace_id = harness.client.start_trace(name, **ctx)
        ledger = harness.client.scope_ledger(trace_id)
        _register_scope_in(ledger, a.agent_binding())
        harness.client.end_trace(trace_id)
        return trace_id, ledger

    def fail_setup(ledger: ScopeLedger) -> None:
        token = set_trace_ledger(ledger)
        try:
            record_failed_boundary()
        finally:
            reset_trace_ledger(token)

    monkeypatch.setattr(
        ScopeLedger,
        "register_ineligible_boundary",
        lambda self: (_ for _ in ()).throw(RuntimeError("ledger unavailable")),
    )
    generated_id, generated = run_case("late-poison-generated")
    explicit_id, explicit = run_case(
        "late-poison-explicit", execution_context={"agent_id": "explicit-agent"}
    )
    # control: right after the seal the generated id is on the wire.
    assert _snapshot_agent_id(_first_final(harness, "late-poison-generated")) == _key(a)
    assert _snapshot_agent_id(_first_final(harness, "late-poison-explicit")) == (
        "explicit-agent"
    )

    fail_setup(generated)  # recording fails too -> the ledger is poisoned
    fail_setup(explicit)
    harness.client.record_observation(generated_id, name="after")
    harness.client.record_observation(explicit_id, name="after")
    harness.client.flush()
    assert _snapshot_agent_id(harness.snapshots("late-poison-generated")[-1]) is None
    assert (
        _snapshot_agent_id(harness.snapshots("late-poison-explicit")[-1])
        == "explicit-agent"
    )


def _boom_warning(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError("host logging handler failed")


def test_item4_scope_setup_failure_records_an_ineligible_boundary(
    harness, monkeypatch
) -> None:
    outer_only = _agent("agent_setup_ok")
    anon = _optimize(lambda: 7)
    assert anon.agent_binding().provenance == "ineligible"
    monkeypatch.setattr(agent_scope_module.logger, "warning", _boom_warning)

    # control: the failing logger alone does not disturb an eligible agent.
    monkeypatch.setattr(agent_scope_module, "_anonymous_warned", False)
    asyncio.run(observe("setup-ctl", client=harness.client)(outer_only)())
    assert harness.agent_id("setup-ctl") == _key(outer_only)

    results: list[int] = []

    async def body() -> int:
        results.append(anon())  # open_scope raises on the first anonymous warning
        return results[-1]

    a = _agent("agent_setup_ok", body)
    monkeypatch.setattr(agent_scope_module, "_anonymous_warned", False)
    asyncio.run(_root(harness, "setup-failure", a)())
    assert results == [7]  # the user's result is preserved
    assert harness.agent_id("setup-failure") is None


def test_item4_failure_to_record_the_boundary_poisons_the_ledger(
    harness, monkeypatch
) -> None:
    anon = _optimize(lambda: 7)
    monkeypatch.setattr(agent_scope_module.logger, "warning", _boom_warning)
    monkeypatch.setattr(agent_scope_module, "_anonymous_warned", False)

    def cannot_record(self: ScopeLedger) -> None:
        raise RuntimeError("ledger unavailable")

    monkeypatch.setattr(ScopeLedger, "register_ineligible_boundary", cannot_record)

    async def body() -> int:
        return anon()

    a = _agent("agent_setup_ok", body)
    asyncio.run(_root(harness, "setup-poisoned", a)())
    assert harness.agent_id("setup-poisoned") is None


class _FlippingEnviron:
    """TRAIGENT_EXPERIMENT_NAME is set on the first read and gone afterwards."""

    def __init__(self, real: Any) -> None:
        self._real = real
        self.reads = 0

    def get(self, key: str, default: Any = None) -> Any:
        if key == "TRAIGENT_EXPERIMENT_NAME":
            self.reads += 1
            return "customer-chosen-name" if self.reads == 1 else None
        return self._real.get(key, default)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def test_item5_identity_and_provenance_come_from_one_environment_read(
    monkeypatch,
) -> None:
    agent = _agent("agent_env_race")
    # control: stable environment, declared via the env var -> user_named, no key.
    with monkeypatch.context() as ctl:
        ctl.setenv("TRAIGENT_EXPERIMENT_NAME", "customer-chosen-name")
        stable = agent.agent_binding()
    assert (stable.provenance, stable.key) == ("user_named", None)
    # control: nothing declared -> function_derived with a key.
    derived = agent.agent_binding()
    assert derived.provenance == "function_derived" and derived.key

    flipping = _FlippingEnviron(os.environ)
    with monkeypatch.context() as racing:
        racing.setattr(os, "environ", flipping)
        binding = agent.agent_binding()
    assert binding.registration_identity == "customer-chosen-name"
    assert (binding.provenance, binding.key) == ("user_named", None)
