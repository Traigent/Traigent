"""Candidate runs vs the advancing (apply) run on one wrapper.

Owner ruling "option A" (identity/concurrency decision, item 1): parallel
candidate runs are allowed, and only the run that changes what the wrapper
serves is serialized.

* ``optimize()`` (``apply=True``, the default) auto-applies the winner to the
  wrapper, so two of them on one wrapper race: the last finisher silently
  replaces the first winner. A second advancing run on a wrapper that already
  has one in flight is refused with :class:`OverlappingOptimizationError`.
* ``optimize(apply=False)`` returns the result -- ``result.best_config`` is the
  candidate -- without applying it. It runs on an isolated per-run copy of the
  wrapper, so it may run beside other candidate runs and beside an advancing
  run, and none of its run state (lifecycle state, results history, transient
  runtime overrides) reaches the wrapper. ``apply_best_config(result)``
  promotes a candidate later, explicitly.
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
    assert "apply=False" in str(excinfo.value)

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
# 2. apply=False returns a candidate without applying it
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_false_returns_candidate_and_leaves_wrapper_untouched():
    agent = _make_agent()

    result = await agent.optimize(max_trials=3, apply=False)

    assert result.best_config == {"temperature": 0.9}
    # Nothing leaked into the wrapper: not applied, not recorded, state intact.
    assert agent.current_config == {"temperature": 0.1}
    assert agent.state == OptimizationState.UNOPTIMIZED
    assert agent.get_optimization_results() is None
    assert agent.get_optimization_history() == []
    assert agent.best_config is None

    # Promotion is a separate, explicit step.
    assert agent.apply_best_config(result) is True
    assert agent.current_config["temperature"] == 0.9


@pytest.mark.asyncio
async def test_apply_false_never_calls_apply(monkeypatch: pytest.MonkeyPatch):
    # "Without applying it" is the contract itself, not only its visible effect
    # on the wrapper: no state manager -- the wrapper's or the run copy's --
    # applies a configuration during a candidate run.
    from traigent.core.config_state_manager import ConfigStateManager

    applied: list[object] = []
    real_apply = ConfigStateManager.apply_best_config

    def spy(self, *args, **kwargs):
        applied.append(self)
        return real_apply(self, *args, **kwargs)

    monkeypatch.setattr(ConfigStateManager, "apply_best_config", spy)
    agent = _make_agent()

    await agent.optimize(max_trials=3, apply=False)
    assert applied == []

    # Control: the advancing run does apply, through the same method.
    await agent.optimize(max_trials=3)
    assert applied == [agent._csm]


def test_apply_false_is_accepted_by_optimize_sync():
    agent = _make_agent()
    result = agent.optimize_sync(max_trials=3, apply=False)
    assert result.best_config == {"temperature": 0.9}
    assert agent.current_config == {"temperature": 0.1}


@pytest.mark.asyncio
async def test_apply_false_run_does_not_mark_wrapper_as_optimizing():
    gate = _Gate(hold_calls=1)
    agent = _make_agent(gate)

    candidate = asyncio.ensure_future(agent.optimize(max_trials=3, apply=False))
    await _wait(gate.entered)

    # The served wrapper stays readable while a candidate is being produced
    # (current_config raises OptimizationStateError while OPTIMIZING).
    assert agent.state == OptimizationState.UNOPTIMIZED
    assert agent.current_config == {"temperature": 0.1}

    gate.release.set()
    result = await asyncio.wait_for(candidate, timeout=30)
    assert result.best_config == {"temperature": 0.9}


# ---------------------------------------------------------------------------
# 3. Candidate runs are parallel with each other and with the advancing run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_candidate_runs_run_in_parallel_beside_an_in_flight_apply_run():
    gate = _Gate(hold_calls=1)
    agent = _make_agent(gate)

    # The advancing run carries a transient runtime objective override that
    # lives on the wrapper only for the duration of that run.
    advancing = asyncio.ensure_future(
        agent.optimize(max_trials=3, objectives=["accuracy", "latency"])
    )
    await _wait(gate.entered)
    assert agent.state == OptimizationState.OPTIMIZING

    # Two candidate runs start and finish while the advancing run is held.
    c1, c2 = await asyncio.wait_for(
        asyncio.gather(
            agent.optimize(max_trials=3, apply=False),
            agent.optimize(max_trials=3, apply=False),
        ),
        timeout=30,
    )
    assert c1.best_config == {"temperature": 0.9}
    assert c2.best_config == {"temperature": 0.9}
    # They ran against the wrapper at rest, not the advancing run's transient
    # override.
    assert list(c1.objectives) == ["accuracy"]
    assert list(c2.objectives) == ["accuracy"]
    assert not advancing.done()

    gate.release.set()
    result = await asyncio.wait_for(advancing, timeout=30)
    assert result.best_config == {"temperature": 0.9}
    # Only the advancing run reached the wrapper.
    assert agent.get_optimization_history() == [result]
    assert agent.state == OptimizationState.OPTIMIZED


@pytest.mark.asyncio
async def test_promotion_is_refused_while_an_advancing_run_is_in_flight():
    agent = _make_agent()
    candidate = await agent.optimize(max_trials=3, apply=False)

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
