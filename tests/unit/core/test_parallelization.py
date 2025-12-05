import time

import pytest

import traigent
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator


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
    assert (
        dur_par < dur_seq * 0.75
    ), f"Expected parallel < 75% of sequential, got {dur_par:.3f} vs {dur_seq:.3f}"


@pytest.mark.asyncio
async def test_orchestrator_parallel_trials(monkeypatch):
    # Create a simple function and decorate with optimize
    # Build dataset of 4 examples
    ds = _make_dataset(4)

    @traigent.optimize(
        eval_dataset=ds,
        configuration_space={"p": [1, 2, 3, 4]},
        objectives=["accuracy"],
        execution_mode="edge_analytics",
    )
    def fn(x: int) -> str:
        # Each call ~0.1s; with trial_concurrency=2 total ~ half of sequential
        import time as _t

        _t.sleep(0.1)
        return f"val-{x}"

    # Monkeypatch BackendIntegratedClient to avoid network/backends in orchestrator
    class _DummyBackend:
        def create_session(self, *a, **k):
            return "dummy-session"

        def submit_result(self, *a, **k):
            return True

        def finalize_session_sync(self, *a, **k):
            return {"status": "completed"}

    import traigent.core.orchestrator as orch_mod

    monkeypatch.setattr(
        orch_mod, "BackendIntegratedClient", lambda *a, **k: _DummyBackend()
    )

    # Run sequential trials (trial_concurrency=1)
    t0 = time.time()
    await fn.optimize(
        algorithm="random",
        configuration_space={"p": [1, 2, 3, 4]},
        max_trials=4,
        parallel_config={"example_concurrency": 1, "trial_concurrency": 1},
        timeout=10.0,
        callbacks=[],
    )
    t1 = time.time()
    dur_seq = t1 - t0

    # Run parallel trials (trial_concurrency=2)
    t2 = time.time()
    await fn.optimize(
        algorithm="random",
        configuration_space={"p": [1, 2, 3, 4]},
        max_trials=4,
        parallel_config={"example_concurrency": 1, "trial_concurrency": 2},
        timeout=10.0,
        callbacks=[],
    )
    t3 = time.time()
    dur_par = t3 - t2

    # Heuristic check: parallel trials should not be slower than sequential run
    assert (
        dur_par <= dur_seq
    ), f"Expected parallel trials no slower than sequential: {dur_par:.3f} vs {dur_seq:.3f}"


@pytest.mark.asyncio
async def test_privacy_alias_maps_to_hybrid():
    ds_small = _make_dataset(1)

    @traigent.optimize(
        eval_dataset=ds_small,
        configuration_space={"p": [0]},
        objectives=["accuracy"],
        execution_mode="privacy",
    )
    def fn_priv(x: int) -> str:
        return f"val-{x}"

    # Run a minimal optimization to build TraigentConfig
    await fn_priv.optimize(
        max_trials=1,
        parallel_config={"example_concurrency": 1},
        timeout=5.0,
    )

    # Access the attached config
    cfg = getattr(fn_priv, "traigent_config", None)
    assert cfg is not None
    assert cfg.execution_mode == "hybrid"
    assert cfg.privacy_enabled is True
