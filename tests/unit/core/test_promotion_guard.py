"""One exclusive slot per wrapper: applying runs and promotion are serialized.

Identity/concurrency decision, owner ruling "option A" (only promotion is
serialized), item 1 minus candidate mode:

* ``optimize()`` applies its winner to the wrapper when it finishes, so two
  runs on one wrapper race: the last finisher silently replaces the first
  winner. A second run while the first is in flight is refused with
  :class:`OverlappingOptimizationError`.
* ``apply_best_config()`` claims the same slot and holds it across the commit,
  so a promotion is atomic with run admission.
* The slot is released on success, failure and cancellation, and a
  cancellation escaping the orchestrator no longer strands the lifecycle in
  OPTIMIZING.
"""

from __future__ import annotations

import asyncio

import pytest

import traigent
from traigent.core.config_state_manager import OptimizationState
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.utils.exceptions import (
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
    assert "already in flight" in str(excinfo.value)

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


@pytest.mark.asyncio
async def test_promotion_is_refused_while_an_advancing_run_is_in_flight():
    agent = _make_agent()
    candidate = await agent.optimize(max_trials=3)

    gate = _Gate(hold_calls=1)
    held = _make_agent(gate)
    advancing = asyncio.ensure_future(held.optimize(max_trials=3))
    await _wait(gate.entered)

    # Applying a candidate mid-run would be overwritten by the in-flight
    # run's own apply: promotion is serialized with it.
    with pytest.raises(OverlappingOptimizationError):
        held.apply_best_config(candidate)

    gate.release.set()
    await asyncio.wait_for(advancing, timeout=30)
    assert held.apply_best_config(candidate) is True


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

    other = _make_agent()
    candidate = other.optimize_sync(max_trials=3)  # a result from another wrapper
    agent = _make_agent()

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
