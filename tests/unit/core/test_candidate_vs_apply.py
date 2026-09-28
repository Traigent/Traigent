"""The advancing (apply=True) run vs promotion on one wrapper, and the loud
refusal of the withdrawn candidate mode (apply=False).

* ``optimize()`` (``apply=True``, the default) auto-applies the winner to the
  wrapper, so two of them on one wrapper race: the last finisher silently
  replaces the first winner. A second advancing run on a wrapper that already
  has one in flight is refused with :class:`OverlappingOptimizationError`.
  Promotion (``apply_best_config``) takes the same exclusive slot and holds it
  across the commit, so it is refused the same way while a run is in flight,
  and vice versa.
* ``optimize(apply=False)`` (candidate runs) is temporarily withdrawn pending a
  redesign of its config isolation under concurrent config changes. It fails
  loudly and immediately with ``ConfigurationError``, before any work starts,
  leaving the wrapper's state untouched and usable with ``apply=True``
  afterwards.
"""

from __future__ import annotations

import asyncio

import pytest

import traigent
from traigent.core.config_state_manager import OptimizationState
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.utils.exceptions import (
    ConfigurationError,
    OptimizationStateError,
    OverlappingOptimizationError,
)


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    # Fully local: no backend session, no cloud trial suggestions.
    monkeypatch.setenv("TRAIGENT_OFFLINE", "1")
    monkeypatch.delenv("TRAIGENT_OFFLINE_MODE", raising=False)
    monkeypatch.delenv("TRAIGENT_REQUIRE_CLOUD", raising=False)
    # Private results folder: never scan the developer's ~/.traigent.
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "results"))


def _dataset() -> Dataset:
    return Dataset([EvaluationExample({"text": "q"}, "HOT")], name="w2")


def _scorer(prediction: str, expected: str) -> float:
    return 1.0 if prediction == expected else 0.0


class _Gate:
    """Hold the first N calls of the agent inside its body until released."""

    def __init__(self, hold_calls: int = 1) -> None:
        self.hold_calls = hold_calls
        self.calls = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def maybe_hold(self) -> None:
        self.calls += 1
        if self.calls <= self.hold_calls:
            self.entered.set()
            await self.release.wait()


def _make_agent(gate: _Gate | None = None):
    @traigent.optimize(
        eval_dataset=_dataset(),
        objectives=["accuracy"],
        configuration_space={"temperature": [0.1, 0.9]},
        default_config={"temperature": 0.1},
        scoring_function=_scorer,
        algorithm="grid",
    )
    async def agent(text: str) -> str:
        if gate is not None:
            await gate.maybe_hold()
        cfg = traigent.get_config()
        return "HOT" if cfg.get("temperature") == 0.9 else "COLD"

    return agent


async def _wait(event: asyncio.Event) -> None:
    await asyncio.wait_for(event.wait(), timeout=10)


# ---------------------------------------------------------------------------
# 1. Overlapping advancing runs on one wrapper are refused
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_second_apply_run_on_one_wrapper_is_refused_while_first_in_flight():
    gate = _Gate(hold_calls=1)
    agent = _make_agent(gate)

    first = asyncio.ensure_future(agent.optimize(max_trials=3))
    await _wait(gate.entered)

    with pytest.raises(OverlappingOptimizationError) as excinfo:
        await agent.optimize(max_trials=3)
    # Catchable as the lifecycle error users already handle.
    assert isinstance(excinfo.value, OptimizationStateError)

    gate.release.set()
    result = await asyncio.wait_for(first, timeout=30)

    # The refused call left the in-flight run alone: it finished and applied.
    assert result.best_config == {"temperature": 0.9}
    assert agent.current_config["temperature"] == 0.9
    assert agent.state == OptimizationState.OPTIMIZED


@pytest.mark.asyncio
async def test_apply_run_guard_is_released_after_success_and_failure():
    agent = _make_agent()
    await agent.optimize(max_trials=3)
    # A second sequential run is fine: the guard was released.
    await agent.optimize(max_trials=3)

    with pytest.raises(TypeError, match="not_a_real_kwarg"):
        await agent.optimize(max_trials=3, not_a_real_kwarg=1)
    # A run that failed must not leave the wrapper locked forever.
    result = await agent.optimize(max_trials=3)
    assert result.best_config == {"temperature": 0.9}


# ---------------------------------------------------------------------------
# 2. apply=False (candidate runs) is temporarily withdrawn: loud, immediate
#    refusal, no work done, no state change, wrapper still usable afterwards.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_false_is_refused_loudly_before_any_work():
    agent = _make_agent()

    with pytest.raises(ConfigurationError, match="apply=False"):
        await agent.optimize(max_trials=3, apply=False)

    # Nothing happened: no state change from the refused call.
    assert agent.state == OptimizationState.UNOPTIMIZED
    assert agent.current_config == {"temperature": 0.1}
    assert agent.get_optimization_results() is None
    assert agent.get_optimization_history() == []
    assert agent.best_config is None

    # The wrapper remains usable with apply=True afterwards.
    result = await agent.optimize(max_trials=3)
    assert result.best_config == {"temperature": 0.9}
    assert agent.current_config["temperature"] == 0.9


def test_apply_false_is_refused_loudly_by_optimize_sync():
    agent = _make_agent()

    with pytest.raises(ConfigurationError, match="apply=False"):
        agent.optimize_sync(max_trials=3, apply=False)

    assert agent.state == OptimizationState.UNOPTIMIZED
    assert agent.current_config == {"temperature": 0.1}

    result = agent.optimize_sync(max_trials=3)
    assert result.best_config == {"temperature": 0.9}


@pytest.mark.asyncio
async def test_apply_false_never_invokes_the_agent_or_the_evaluator():
    agent_calls: list[str] = []
    scorer_calls: list[tuple[str, str]] = []

    def scorer(prediction: str, expected: str) -> float:
        scorer_calls.append((prediction, expected))
        return _scorer(prediction, expected)

    @traigent.optimize(
        eval_dataset=_dataset(),
        objectives=["accuracy"],
        configuration_space={"temperature": [0.1, 0.9]},
        default_config={"temperature": 0.1},
        scoring_function=scorer,
        algorithm="grid",
    )
    async def agent(text: str) -> str:
        agent_calls.append(text)
        return "HOT"

    with pytest.raises(ConfigurationError):
        await agent.optimize(max_trials=3, apply=False)
    assert agent_calls == []
    assert scorer_calls == []

    # A normal apply=True run afterwards does invoke both the agent and the
    # evaluator.
    await agent.optimize(max_trials=3)
    assert agent_calls
    assert scorer_calls


def test_optimize_sync_apply_false_is_refused_before_any_loop_or_thread():
    """No running loop: the guard must be optimize_sync's first statement --
    before it even checks for a running loop, before it calls optimize() to
    produce a coroutine, before asyncio.run(), and before any thread pool.

    Patching only ThreadPoolExecutor is not enough to prove this: the
    no-running-loop branch never creates one anyway (that only happens when
    optimize_sync is called from inside a running loop), so a version of the
    guard placed AFTER asyncio.get_running_loop()/self.optimize()/asyncio.run()
    -- i.e. inside optimize()'s own body, reached only once the coroutine is
    actually driven -- would still pass a ThreadPoolExecutor-only check here.
    Patch every step in between instead, so any of them firing fails the
    test with something other than ConfigurationError.
    """
    import concurrent.futures

    from traigent.core import optimized_function as of_module

    pool_created: list[bool] = []
    loop_checked: list[bool] = []
    run_called: list[bool] = []
    optimize_called: list[bool] = []

    def _boom(tag: str, sink: list[bool]):
        def _raise(*args, **kwargs):
            sink.append(True)
            raise AssertionError(
                f"{tag} must not be reached for a refused apply=False call"
            )

        return _raise

    agent = _make_agent()
    with pytest.MonkeyPatch.context() as patch:
        # Scoped to the refused call only: a real apply=True run below
        # legitimately reaches every one of these.
        patch.setattr(
            concurrent.futures,
            "ThreadPoolExecutor",
            _boom("ThreadPoolExecutor", pool_created),
        )
        patch.setattr(
            of_module.asyncio,
            "get_running_loop",
            _boom("asyncio.get_running_loop()", loop_checked),
        )
        patch.setattr(of_module.asyncio, "run", _boom("asyncio.run()", run_called))
        patch.setattr(
            agent, "optimize", _boom("optimize() (coroutine creation)", optimize_called)
        )
        with pytest.raises(ConfigurationError, match="apply=False"):
            agent.optimize_sync(max_trials=3, apply=False)
    assert pool_created == []
    assert loop_checked == []
    assert run_called == []
    assert optimize_called == []

    result = agent.optimize_sync(max_trials=3)
    assert result.best_config == {"temperature": 0.9}


@pytest.mark.asyncio
async def test_optimize_sync_apply_false_is_refused_before_any_thread_inside_a_running_loop(
    monkeypatch: pytest.MonkeyPatch,
):
    """Inside a running loop, optimize_sync used to create a ThreadPoolExecutor
    and submit asyncio.run(coro) to it before the coroutine ever ran far
    enough to hit the guard. The guard must now fire first, so no pool is
    ever created."""
    import concurrent.futures

    created: list[bool] = []

    def _boom(*args, **kwargs):
        created.append(True)
        raise AssertionError(
            "ThreadPoolExecutor must not be created for a refused apply=False call"
        )

    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", _boom)

    agent = _make_agent()
    # This test body runs inside pytest-asyncio's event loop, so
    # asyncio.get_running_loop() inside optimize_sync succeeds here --
    # exactly the branch that used to spin up a thread pool.
    assert asyncio.get_running_loop() is not None
    with pytest.raises(ConfigurationError, match="apply=False"):
        agent.optimize_sync(max_trials=3, apply=False)
    assert created == []


def test_optimize_with_guidance_apply_false_is_refused_before_touching_any_override():
    """optimize_with_guidance forwards **optimize_kwargs (including `apply`)
    to optimize_sync only deep inside a generation round, after resolving
    rewrite_llm, loading the dataset, and setting a dataset override whose
    `finally` unconditionally clears it back to None. That finally must not
    run at all: the guard has to fire before any of that setup, so an
    existing override survives untouched and no rewrite_llm is required."""
    agent = _make_agent()
    existing = _dataset()
    agent.set_eval_dataset_override(existing)

    with pytest.raises(ConfigurationError, match="apply=False"):
        # No rewrite_llm passed: reaching resolve_rewrite_llm(None) would
        # raise GenerationProviderError instead, which is exactly the bug
        # this guards against -- the withdrawal error must win.
        agent.optimize_with_guidance(provider=object(), apply=False)

    assert agent._dataset_override is existing


@pytest.mark.asyncio
async def test_promotion_is_refused_while_an_advancing_run_is_in_flight():
    """Promotion and run admission share one exclusive slot per wrapper."""
    source = _make_agent()
    result_to_promote = await source.optimize(max_trials=3)

    gate = _Gate(hold_calls=1)
    held = _make_agent(gate)
    advancing = asyncio.ensure_future(held.optimize(max_trials=3))
    await _wait(gate.entered)

    # Applying a result mid-run would be overwritten by the in-flight run's
    # own apply: promotion is serialized with it.
    with pytest.raises(OverlappingOptimizationError):
        held.apply_best_config(result_to_promote)

    gate.release.set()
    await asyncio.wait_for(advancing, timeout=30)
    assert held.apply_best_config(result_to_promote) is True


# ---------------------------------------------------------------------------
# 3. Review fixes (PR #2406 @ 22604bb2)
# ---------------------------------------------------------------------------


def test_promotion_holds_the_admission_slot_across_the_commit(
    monkeypatch: pytest.MonkeyPatch,
):
    """Blocker 1: promotion and run admission are one exclusive slot.

    Deterministic two-thread schedule: thread P promotes and is parked inside
    the commit; the main thread then tries to admit an applying run and a
    second promotion. Both must be refused until P's commit returns.
    """
    import threading

    from traigent.core.config_state_manager import ConfigStateManager

    agent = _make_agent()
    source = _make_agent()
    candidate = source.optimize_sync(max_trials=3)

    in_commit = threading.Event()
    release = threading.Event()
    real_apply = ConfigStateManager.apply_best_config
    parked: list[bool] = []

    def parking_apply(self, *args, **kwargs):
        if not parked:
            parked.append(True)
            in_commit.set()
            assert release.wait(timeout=30)
        return real_apply(self, *args, **kwargs)

    monkeypatch.setattr(ConfigStateManager, "apply_best_config", parking_apply)

    outcome: list[object] = []

    def promote() -> None:
        try:
            outcome.append(agent.apply_best_config(candidate))
        except BaseException as exc:  # surfaced by the assertion below
            outcome.append(exc)

    promoter = threading.Thread(target=promote)
    promoter.start()
    assert in_commit.wait(timeout=30)
    try:
        # Promotion is mid-commit: an applying run must not be admitted...
        with pytest.raises(OverlappingOptimizationError):
            asyncio.run(agent.optimize(max_trials=3))
        # ...and neither may a second promotion.
        with pytest.raises(OverlappingOptimizationError):
            agent.apply_best_config(candidate)
    finally:
        release.set()
        promoter.join()

    assert outcome == [True]
    assert agent.current_config["temperature"] == 0.9
    # The slot was released: an applying run is admitted again.
    result = asyncio.run(agent.optimize(max_trials=3))
    assert result.best_config == {"temperature": 0.9}


def _hold_and_fail(gate: _Gate, exc: BaseException):
    """Patch target: an orchestrator run that is in flight, then fails."""

    async def optimize(self, *args, **kwargs):
        await gate.maybe_hold()
        raise exc

    return optimize


@pytest.mark.asyncio
async def test_cancelled_apply_run_releases_the_slot():
    """Follow-up 4: cancelling the caller's task mid-run frees the slot.

    The orchestrator turns a cancellation inside the trial loop into a
    user-cancelled result by design, so either outcome is legitimate here:
    a returned (cancelled) result or a propagated CancelledError. What must
    hold in both: no lingering OPTIMIZING state, readable config, and the
    next applying run is admitted.
    """
    gate = _Gate(hold_calls=1)
    agent = _make_agent(gate)

    run = asyncio.ensure_future(agent.optimize(max_trials=3))
    await _wait(gate.entered)
    assert agent.state == OptimizationState.OPTIMIZING

    run.cancel()
    try:
        await run
    except asyncio.CancelledError:
        pass

    assert agent.state != OptimizationState.OPTIMIZING
    assert "temperature" in agent.current_config  # readable
    gate.release.set()
    result = await agent.optimize(max_trials=3)  # admitted again
    assert result.best_config == {"temperature": 0.9}


@pytest.mark.asyncio
async def test_cancellation_escaping_the_orchestrator_does_not_strand_optimizing(
    monkeypatch: pytest.MonkeyPatch,
):
    """Follow-up 4: a CancelledError that escapes past the orchestrator.

    Pre-existing defect: ``_run_and_finalize_optimization`` only moved the
    lifecycle to ERROR on ``Exception``, so a ``CancelledError`` (a
    ``BaseException``) left the wrapper in OPTIMIZING, where
    ``current_config`` raises forever.
    """
    from traigent.core.orchestrator import OptimizationOrchestrator

    gate = _Gate(hold_calls=1)
    agent = _make_agent()
    # Scoped patch: monkeypatch.undo() would also drop the env set by the
    # autouse fixtures (e.g. CI run approval).
    with monkeypatch.context() as patch:
        patch.setattr(
            OptimizationOrchestrator,
            "optimize",
            _hold_and_fail(gate, asyncio.CancelledError()),
        )

        run = asyncio.ensure_future(agent.optimize(max_trials=3))
        await _wait(gate.entered)
        gate.release.set()
        with pytest.raises(asyncio.CancelledError):
            await run

    assert agent.state == OptimizationState.ERROR
    assert agent.current_config == {"temperature": 0.1}  # readable, unchanged
    result = await agent.optimize(max_trials=3)  # admitted again
    assert result.best_config == {"temperature": 0.9}


@pytest.mark.asyncio
async def test_failed_in_flight_apply_run_releases_the_slot(
    monkeypatch: pytest.MonkeyPatch,
):
    """Follow-up 4: a failure raised mid-run frees the slot."""
    from traigent.core.orchestrator import OptimizationOrchestrator

    gate = _Gate(hold_calls=1)
    agent = _make_agent()
    with monkeypatch.context() as patch:
        patch.setattr(
            OptimizationOrchestrator,
            "optimize",
            _hold_and_fail(gate, RuntimeError("boom mid-run")),
        )

        run = asyncio.ensure_future(agent.optimize(max_trials=3))
        await _wait(gate.entered)
        with pytest.raises(OverlappingOptimizationError):
            await agent.optimize(max_trials=3)
        gate.release.set()
        with pytest.raises(Exception, match="boom mid-run"):
            await run

    assert agent.state == OptimizationState.ERROR
    assert agent.current_config == {"temperature": 0.1}
    result = await agent.optimize(max_trials=3)
    assert result.best_config == {"temperature": 0.9}
