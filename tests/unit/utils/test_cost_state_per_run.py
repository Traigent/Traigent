"""Per-run cost accounting state (identity/concurrency decision, item 1).

The unpriced-at-runtime model registry (#1407/#1597) and the usage-capture
counter used to be process-global and were reset at the start of every
``optimize()``. Two agents optimizing concurrently in one process therefore
interfered: agent B's run start wiped what agent A had recorded, and each
run's result could carry the other's models.

Both are now held per run (a ``ContextVar``-scoped state object opened by
``optimize()``), inherited by the asyncio tasks and worker threads the run
dispatches, and never shared between runs.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

import traigent
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.invokers.local import LocalInvoker
from traigent.utils import cost_calculator
from traigent.utils.cost_calculator import (
    captured_usage_count,
    cost_run_scope,
    get_unpriced_runtime_occurrences,
    record_captured_usage,
    record_unpriced_runtime_model,
    reset_captured_usage,
    reset_unpriced_runtime_models,
)


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("TRAIGENT_OFFLINE", "1")
    monkeypatch.delenv("TRAIGENT_OFFLINE_MODE", raising=False)
    monkeypatch.delenv("TRAIGENT_REQUIRE_CLOUD", raising=False)
    # Private results folder: never scan the developer's ~/.traigent.
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "results"))


# ---------------------------------------------------------------------------
# Mechanism: one state per scope, inherited by tasks and threads
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_scopes_do_not_see_or_reset_each_other():
    a_recorded = asyncio.Event()
    b_reset = asyncio.Event()

    async def run_a() -> tuple[dict[str, int], int]:
        with cost_run_scope():
            record_unpriced_runtime_model("model-a")
            record_captured_usage()
            a_recorded.set()
            await asyncio.wait_for(b_reset.wait(), timeout=10)
            return get_unpriced_runtime_occurrences(), captured_usage_count()

    async def run_b() -> tuple[dict[str, int], int]:
        with cost_run_scope():
            await asyncio.wait_for(a_recorded.wait(), timeout=10)
            # B's run start: previously wiped A's global registry.
            reset_unpriced_runtime_models()
            reset_captured_usage()
            record_unpriced_runtime_model("model-b")
            b_reset.set()
            return get_unpriced_runtime_occurrences(), captured_usage_count()

    (a_models, a_usage), (b_models, b_usage) = await asyncio.gather(run_a(), run_b())

    assert a_models == {"model-a": 1}
    assert a_usage == 1
    assert b_models == {"model-b": 1}
    assert b_usage == 0


def test_scope_is_inherited_by_the_sync_invoker_worker_thread():
    def agent_body() -> str:
        # Runs on LocalInvoker's shared thread pool, like a sync agent does.
        record_unpriced_runtime_model("model-in-thread")
        return "ok"

    async def main() -> dict[str, int]:
        with cost_run_scope():
            await LocalInvoker()._invoke_sync(agent_body, {}, 0.0)
            return get_unpriced_runtime_occurrences()

    assert asyncio.run(main()) == {"model-in-thread": 1}


def test_records_outside_any_run_scope_still_work():
    # Backward compatibility: direct callers outside optimize() keep the
    # module-level behaviour.
    reset_unpriced_runtime_models()
    try:
        record_unpriced_runtime_model("loose")
        assert cost_calculator.get_unpriced_runtime_models() == ["loose"]
    finally:
        reset_unpriced_runtime_models()


# ---------------------------------------------------------------------------
# Behaviour: two agents optimizing concurrently in one process
# ---------------------------------------------------------------------------


def _dataset() -> Dataset:
    return Dataset([EvaluationExample({"text": "q"}, "ok")], name="cost-iso")


def _scorer(prediction: str, expected: str) -> float:
    return 1.0 if prediction == expected else 0.0


def _decorate(func):
    return traigent.optimize(
        eval_dataset=_dataset(),
        objectives=["accuracy"],
        configuration_space={"temperature": [0.1]},
        scoring_function=_scorer,
        algorithm="grid",
    )(func)


@pytest.mark.asyncio
async def test_two_async_agents_optimizing_concurrently_keep_their_own_cost_state():
    a_recorded = asyncio.Event()
    b_done = asyncio.Event()

    @_decorate
    async def agent_a(text: str) -> str:
        # Stands in for the runtime cost path pricing a hard-coded model to $0.
        record_unpriced_runtime_model("model-a")
        a_recorded.set()
        # Hold agent A mid-run until agent B's whole run (start reset
        # included) has happened.
        await asyncio.wait_for(b_done.wait(), timeout=10)
        return "ok"

    @_decorate
    async def agent_b(text: str) -> str:
        record_unpriced_runtime_model("model-b")
        return "ok"

    async def run_b():
        await asyncio.wait_for(a_recorded.wait(), timeout=10)
        try:
            return await agent_b.optimize(max_trials=1)
        finally:
            b_done.set()

    result_a, result_b = await asyncio.wait_for(
        asyncio.gather(agent_a.optimize(max_trials=1), run_b()), timeout=60
    )

    assert result_a.metadata.get("unpriced_models_runtime") == ["model-a"]
    assert result_b.metadata.get("unpriced_models_runtime") == ["model-b"]


@pytest.mark.asyncio
async def test_two_sync_agents_in_worker_threads_keep_their_own_cost_state():
    a_recorded = threading.Event()
    b_done = threading.Event()

    @_decorate
    def agent_a(text: str) -> str:
        record_unpriced_runtime_model("model-a")
        a_recorded.set()
        assert b_done.wait(timeout=10)
        return "ok"

    @_decorate
    def agent_b(text: str) -> str:
        record_unpriced_runtime_model("model-b")
        return "ok"

    async def run_b():
        while not a_recorded.is_set():
            await asyncio.sleep(0.01)
        try:
            return await agent_b.optimize(max_trials=1)
        finally:
            b_done.set()

    result_a, result_b = await asyncio.wait_for(
        asyncio.gather(agent_a.optimize(max_trials=1), run_b()), timeout=60
    )

    assert result_a.metadata.get("unpriced_models_runtime") == ["model-a"]
    assert result_b.metadata.get("unpriced_models_runtime") == ["model-b"]


def test_scope_is_inherited_by_parallel_batch_trial_threads():
    from traigent.optimizers.batch_optimizers import (
        BatchOptimizationConfig,
        ParallelBatchOptimizer,
    )
    from traigent.optimizers.results import BatchTrial

    optimizer = ParallelBatchOptimizer(
        config_space={"temperature": [0.1, 0.9]},
        objectives=["accuracy"],
        batch_config=BatchOptimizationConfig(max_parallel_trials=2),
    )

    def trial_in_pool_thread(config, func, dataset, invoker, evaluator):
        # Runs on the optimizer's own ThreadPoolExecutor.
        record_unpriced_runtime_model("model-in-batch-thread")
        return BatchTrial(configuration=config, score=1.0, duration=0.0)

    optimizer._run_single_trial_sync = trial_in_pool_thread  # type: ignore[method-assign]

    async def main() -> dict[str, int]:
        with cost_run_scope():
            await optimizer.optimize(
                func=lambda **_: "ok",
                dataset=_dataset(),
                invoker=LocalInvoker(),
                evaluator=None,  # type: ignore[arg-type]
                max_trials=2,
            )
            return get_unpriced_runtime_occurrences()

    assert asyncio.run(main()) == {"model-in-batch-thread": 2}
