"""Lineage e2e: real optimizer trial -> instrumented span -> local OTLP stub.

The receiver-side lineage read is exercised in the Backend branch; here the
wire is decoded with the official opentelemetry-proto and compared with the
ids the optimizer itself reports.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

import traigent
import traigent.observability.otel as otel
from traigent.config.context import TrialContext, WorkflowTraceContext
from traigent.observability.otel import contract as C

DATASET = [
    {"input": {"q": "a"}, "expected_output": "a"},
    {"input": {"q": "b"}, "expected_output": "b"},
]


# Connected-mode semantics (mocked local collector), declared per file (#2033).
pytestmark = pytest.mark.backend_online


def _init(collector, **kw):
    return otel.init(
        api_key="k",
        endpoint=collector.base_url,
        exit_flush=False,
        schedule_delay_s=0.05,
        **kw,
    )


@pytest.fixture
def optimizer_env(monkeypatch):
    monkeypatch.setenv("TRAIGENT_COST_APPROVED", "true")
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")


def _real_optimizer_run(collector, *, max_trials=4, concurrency=1):
    h = _init(collector, service_name="lineage-e2e")

    @traigent.optimize(
        evaluation={"eval_dataset": DATASET},
        objectives=["accuracy"],
        scoring_function=lambda output, expected_output=None, **_: 1.0,
        configuration_space={"temperature": [0.0, 0.5, 1.0]},
        injection_mode="context",
        offline=True,
        mock_mode_config={"base_accuracy": 0.9, "variance": 0.0, "random_seed": 1},
        **(
            {"parallel_config": {"trial_concurrency": concurrency}}
            if concurrency > 1
            else {}
        ),
    )
    def agent(q: str) -> str:
        traigent.get_config()  # CONTEXT injection: read the per-trial config
        with otel.observe("llm-step", as_type="generation"):
            with otel.observe("inner-tool", as_type="tool"):
                pass
        return q

    results = asyncio.run(
        agent.optimize(algorithm="random", max_trials=max_trials, random_seed=1)
    )
    # a run OUTSIDE any trial, on the same process, must carry no lineage
    with otel.observe("after-optimization"):
        pass
    assert otel.flush(10).flushed
    return h, results


def _by_name(collector):
    out: dict[str, list[dict]] = {}
    for _rs, _ss, span in collector.spans():
        out.setdefault(span.name, []).append(collector.attrs(span))
    return out


def test_real_optimizer_trials_stamp_matching_ids_on_every_span(
    collector, optimizer_env
):
    _h, results = _real_optimizer_run(collector)
    reported = {str(t.trial_id) for t in results.trials}
    assert reported, "optimizer produced no trials"
    spans = _by_name(collector)
    llm = spans["llm-step"]
    assert llm, "no instrumented spans reached the receiver"
    seen = {a[C.ATTR_TRIAL_ID] for a in llm}
    assert len(llm) >= len(results.trials) * len(DATASET)  # every call exported
    assert seen <= reported and seen  # every span maps to a trial the optimizer reports
    for attrs in llm + spans["inner-tool"]:  # children stamped too
        assert attrs[C.ATTR_TRIAL_ID] in reported
        assert attrs[C.ATTR_OPTIMIZATION_SESSION_ID]
    # one optimization run id across the whole run
    assert len({a[C.ATTR_OPTIMIZATION_SESSION_ID] for a in llm}) == 1
    # context cleanup: nothing leaks to spans created after the run
    after = spans["after-optimization"][0]
    assert not {C.ATTR_TRIAL_ID, C.ATTR_OPTIMIZATION_SESSION_ID} & set(after)
    # ids only: no optimizer configs/metrics/payloads ever ride on spans
    for _rs, _ss, span in collector.spans():
        assert set(collector.attrs(span)) <= set(C.ATTRIBUTE_ALLOWLIST)


def test_concurrent_trials_keep_their_own_ids(collector, optimizer_env):
    _h, results = _real_optimizer_run(collector, max_trials=4, concurrency=3)
    reported = {str(t.trial_id) for t in results.trials}
    seen = {a[C.ATTR_TRIAL_ID] for a in _by_name(collector)["llm-step"]}
    assert seen <= reported and len(seen) > 1


def test_concurrent_asyncio_trials_do_not_bleed(collector):
    """Deterministic isolation check independent of the optimizer's scheduling."""
    _init(collector)

    async def trial(i: int):
        async with (
            TrialContext(trial_id=f"t{i}"),
            WorkflowTraceContext(
                {"configuration_run_id": f"t{i}", "workflow_trace_id": "run-1"}
            ),
        ):
            await asyncio.sleep(0.01 * (5 - i))
            with otel.observe(f"span-{i}"):
                await asyncio.sleep(0)
                with otel.observe(f"child-{i}"):
                    pass

    async def main():
        await asyncio.gather(*(trial(i) for i in range(5)))

    asyncio.run(main())
    with otel.observe("outside"):
        pass
    assert otel.flush(5).flushed
    spans = _by_name(collector)
    for i in range(5):
        for name in (f"span-{i}", f"child-{i}"):
            assert spans[name][0][C.ATTR_TRIAL_ID] == f"t{i}"
            assert spans[name][0][C.ATTR_OPTIMIZATION_SESSION_ID] == "run-1"
    assert C.ATTR_TRIAL_ID not in spans["outside"][0]


def test_concurrent_threads_and_tasks_stamp_only_their_own_trial(collector):
    """Threads and asyncio tasks running at once never see each other's trial id,
    and a span started after a trial ends carries no trial id."""
    _init(collector)
    barrier = threading.Barrier(4)
    errors: list[BaseException] = []

    def thread_trial(i: int):
        try:
            with TrialContext(trial_id=f"th{i}"):
                with WorkflowTraceContext(
                    {"configuration_run_id": f"th{i}", "workflow_trace_id": "run-9"}
                ):
                    barrier.wait(timeout=5)  # all trials are live simultaneously
                    with otel.observe(f"thread-span-{i}"):
                        with otel.observe(f"thread-child-{i}"):
                            pass
            with otel.observe(f"thread-after-{i}"):  # trial ended in this thread
                pass
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    async def task_trial(i: int):
        async with (
            TrialContext(trial_id=f"as{i}"),
            WorkflowTraceContext(
                {"configuration_run_id": f"as{i}", "workflow_trace_id": "run-9"}
            ),
        ):
            await asyncio.sleep(0.02)
            with otel.observe(f"task-span-{i}"):
                await asyncio.sleep(0)
        with otel.observe(f"task-after-{i}"):
            pass

    async def run_tasks():
        await asyncio.gather(*(task_trial(i) for i in range(3)))

    threads = [threading.Thread(target=thread_trial, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    asyncio.run(run_tasks())
    for t in threads:
        t.join()
    assert not errors, errors
    assert otel.flush(5).flushed
    spans = _by_name(collector)
    for i in range(4):
        for name in (f"thread-span-{i}", f"thread-child-{i}"):
            assert len(spans[name]) == 1
            assert spans[name][0][C.ATTR_TRIAL_ID] == f"th{i}"
        assert C.ATTR_TRIAL_ID not in spans[f"thread-after-{i}"][0]
    for i in range(3):
        assert spans[f"task-span-{i}"][0][C.ATTR_TRIAL_ID] == f"as{i}"
        assert C.ATTR_TRIAL_ID not in spans[f"task-after-{i}"][0]
        assert C.ATTR_OPTIMIZATION_SESSION_ID not in spans[f"task-after-{i}"][0]


def test_negative_control_shared_lineage_would_leak_across_concurrent_trials(
    collector, monkeypatch
):
    """Mutation: a process-global lineage (instead of per-context) leaks ids."""
    from traigent.observability.otel import lineage

    shared: dict[str, str] = {}
    real = lineage.current_lineage

    def leaky():
        cur = real()
        if cur:
            shared.update(cur)  # last writer wins, like a global
        return dict(shared)

    monkeypatch.setattr(lineage, "current_lineage", leaky)
    _init(collector)

    async def trial(i: int, gate: asyncio.Event, done: list):
        async with TrialContext(trial_id=f"n{i}"):
            if i == 0:
                with otel.observe("first"):
                    pass
                gate.set()
                await done[0].wait()
            else:
                await gate.wait()
                with otel.observe("second"):
                    pass
                done[0].set()

    async def main():
        gate, fin = asyncio.Event(), asyncio.Event()
        await asyncio.gather(trial(0, gate, [fin]), trial(1, gate, [fin]))
        with otel.observe("outside"):
            pass

    asyncio.run(main())
    assert otel.flush(5).flushed
    spans = _by_name(collector)
    # the leaky variant stamps the finished trial's id on a span outside any trial
    assert spans["outside"][0].get(C.ATTR_TRIAL_ID) == "n1"


def test_threads_only_see_lineage_when_context_is_propagated(collector):
    """Documented limit: a bare thread does not inherit trial context."""
    _init(collector)
    import contextvars

    def work(name):
        with otel.observe(name):
            pass

    with TrialContext(trial_id="tx"):
        bare = threading.Thread(target=work, args=("bare",))
        ctx = contextvars.copy_context()
        propagated = threading.Thread(target=ctx.run, args=(work, "propagated"))
        for t in (bare, propagated):
            t.start()
            t.join()
    assert otel.flush(5).flushed
    spans = _by_name(collector)
    assert C.ATTR_TRIAL_ID not in spans["bare"][0]
    assert spans["propagated"][0][C.ATTR_TRIAL_ID] == "tx"


def test_third_party_spans_are_stamped_too(collector):
    """Instrumentor-created spans (foreign scope) get lineage without our code on the path."""
    h = _init(collector)
    with TrialContext(trial_id="t9"):
        with h.provider.get_tracer("some.vendor.lib").start_as_current_span(
            "vendor-span"
        ):
            pass
    assert otel.flush(5).flushed
    assert _by_name(collector)["vendor-span"][0][C.ATTR_TRIAL_ID] == "t9"


def test_negative_control_without_stamping_the_ids_disappear(collector, monkeypatch):
    """Mutation: disabling stamping makes the e2e assertion fail."""
    monkeypatch.setattr(
        "traigent.observability.otel.processor.stamp_span", lambda *a, **k: None
    )
    _init(collector)
    with TrialContext(trial_id="t1"):
        with otel.observe("s"):
            pass
    assert otel.flush(5).flushed
    assert C.ATTR_TRIAL_ID not in _by_name(collector)["s"][0]


def test_parallel_grid_trials_have_unique_ids(optimizer_env):
    """4 distinct grid configs at trial_concurrency=2 must yield 4 distinct ids.

    Trial ids are a content hash of (session, config, dataset), so distinct
    configs must never share one; lineage joins on this id.
    """

    @traigent.optimize(
        evaluation={"eval_dataset": DATASET},
        objectives=["accuracy"],
        scoring_function=lambda output, expected_output=None, **_: 1.0,
        configuration_space={"temperature": [0.0, 1.0], "model": ["a", "b"]},
        injection_mode="context",
        offline=True,
        mock_mode_config={"base_accuracy": 0.9, "variance": 0.0, "random_seed": 1},
        parallel_config={"trial_concurrency": 2},
    )
    def agent(q: str) -> str:
        traigent.get_config()
        return q

    results = asyncio.run(agent.optimize(algorithm="grid", max_trials=4))
    ids = [str(t.trial_id) for t in results.trials]
    configs = {tuple(sorted(t.config.items())) for t in results.trials}
    assert len(ids) == 4 and len(configs) == 4
    assert len(set(ids)) == 4, ids
