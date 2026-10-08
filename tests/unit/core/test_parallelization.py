import asyncio
import threading
import time

import pytest

import traigent
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator
from traigent.utils.exceptions import ConfigurationError


def _make_dataset(n: int) -> Dataset:
    examples = []
    for i in range(n):
        examples.append(
            EvaluationExample(
                input_data={"x": i},
                expected_output=f"val-{i}",
                metadata={"example_id": f"ex_{i}"},
            )
        )
    return Dataset(examples=examples, name="unit_test")


def _slow_sync_func(x: int) -> str:
    # Simulate ~100ms of work per call
    import time as _t

    _t.sleep(0.1)
    return f"val-{x}"


@pytest.mark.asyncio
async def test_local_evaluator_respects_max_workers():
    ds = _make_dataset(8)

    # Sequential (max_workers=1)
    seq_eval = LocalEvaluator(
        metrics=["accuracy"], timeout=5.0, max_workers=1, detailed=False
    )
    t0 = time.time()
    res_seq = await seq_eval.evaluate(_slow_sync_func, {"model": "dummy"}, ds)
    t1 = time.time()
    dur_seq = t1 - t0

    # Parallel (max_workers=4)
    par_eval = LocalEvaluator(
        metrics=["accuracy"], timeout=5.0, max_workers=4, detailed=False
    )
    t2 = time.time()
    res_par = await par_eval.evaluate(_slow_sync_func, {"model": "dummy"}, ds)
    t3 = time.time()
    dur_par = t3 - t2

    assert res_seq.metrics is not None
    assert res_par.metrics is not None
    # Parallel run should be significantly faster (rough heuristic)
    _failure_detail = (
        f"Expected parallel < 75% of sequential, got {dur_par:.3f} vs {dur_seq:.3f}"
    )
    assert dur_par < dur_seq * 0.75, _failure_detail


@pytest.mark.asyncio
async def test_orchestrator_parallel_trials(monkeypatch):
    """trial_concurrency=2 runs trials at the same time; =1 never does.

    Counts overlapping executions of the decorated function directly instead
    of comparing two wall-clock runs: each ``optimize()`` carries ~14 s of
    fixed overhead around ~0.4 s of work, so a duration ratio measured xdist
    load, not parallelism (#2478).

    The function is async so overlap reflects the orchestrator's trial
    scheduling alone. (A sync function's calls go through the shared
    evaluator's worker lane, sized by example_concurrency, which serialises
    them across trials; that is tracked separately from this test.)
    """
    ds = _make_dataset(4)

    lock = threading.Lock()
    active = 0
    max_active = 0

    @traigent.optimize(
        eval_dataset=ds,
        configuration_space={"p": [1, 2, 3, 4]},
        objectives=["accuracy"],
        execution_mode="local",
    )
    async def fn(x: int) -> str:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            await asyncio.sleep(0.2)
        finally:
            with lock:
                active -= 1
        return f"val-{x}"

    # Monkeypatch BackendIntegratedClient to avoid network/backends in orchestrator
    class _DummyBackend:
        def create_session(self, *a, **k):
            return "dummy-session"

        def submit_result(self, *a, **k):
            return True

        def finalize_session_sync(self, *a, **k):
            return {"status": "completed"}

    import traigent.cloud.backend_client as backend_mod

    monkeypatch.setattr(
        backend_mod, "BackendIntegratedClient", lambda *a, **k: _DummyBackend()
    )

    async def _peak_concurrency(trial_concurrency: int) -> int:
        nonlocal max_active
        max_active = 0
        await fn.optimize(
            algorithm="random",
            configuration_space={"p": [1, 2, 3, 4]},
            max_trials=4,
            parallel_config={
                "example_concurrency": 1,
                "trial_concurrency": trial_concurrency,
            },
            timeout=10.0,
            callbacks=[],
        )
        assert active == 0
        return max_active

    # Control: sequential trials with one example at a time never overlap,
    # which shows the counter is not trivially high.
    assert await _peak_concurrency(1) == 1
    # Two trials in flight: their example calls must overlap.
    assert await _peak_concurrency(2) == 2


@pytest.mark.asyncio
async def test_privacy_alias_fails_closed():
    ds_small = _make_dataset(1)

    with pytest.raises(ConfigurationError, match="fails closed"):

        @traigent.optimize(
            eval_dataset=ds_small,
            configuration_space={"p": [0]},
            objectives=["accuracy"],
            execution_mode="privacy",
        )
        def fn_priv(x: int) -> str:
            return f"val-{x}"
