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


# ---------------------------------------------------------------------------
# 4. Review fixes (Astra, PR #2406 @ 22604bb2)
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
    candidate = agent.optimize_sync(max_trials=3, apply=False)

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


@pytest.mark.asyncio
async def test_candidate_cannot_mutate_served_nested_config_defaults():
    """Blocker 2: nested mutable defaults are detached for a candidate run."""
    import json

    mutate = {"on": True}
    served_seen: list[dict] = []

    @traigent.optimize(
        eval_dataset=_dataset(),
        objectives=["accuracy"],
        configuration_space={"temperature": [0.1, 0.9]},
        default_config={
            "temperature": 0.1,
            "stop_sequences": ["END"],
            "extra": {"k": "v"},
        },
        scoring_function=_scorer,
        algorithm="grid",
    )
    async def agent(text: str) -> str:
        cfg = traigent.get_config()
        if mutate["on"]:
            # A candidate body scribbling on nested defaults it can reach
            # through the actual injection path.
            cfg["stop_sequences"].append("LEAK")
            cfg["extra"]["leak"] = True
        else:
            served_seen.append(json.loads(json.dumps(cfg, sort_keys=True)))
        return "HOT" if cfg.get("temperature") == 0.9 else "COLD"

    before_current = json.dumps(agent.current_config, sort_keys=True)
    before_default = json.dumps(agent.default_config, sort_keys=True)

    result = await agent.optimize(max_trials=3, apply=False)
    assert result.best_config is not None

    # The served wrapper's config is byte-identical afterwards...
    assert json.dumps(agent.current_config, sort_keys=True) == before_current
    assert json.dumps(agent.default_config, sort_keys=True) == before_default
    # ...and so is what the served callable actually injects.
    mutate["on"] = False
    await agent("q")
    assert served_seen[-1]["stop_sequences"] == ["END"]
    assert served_seen[-1]["extra"] == {"k": "v"}


def test_candidate_fork_shares_no_mutable_config_value_with_the_wrapper():
    """Blocker 2, structurally: every config dict a candidate can reach is detached.

    The injection-path test above covers current_config; this pins the rest
    (default_config on the wrapper and on its state manager, best_config),
    which are reachable through runtime TVL/discovery code paths.
    """
    agent = _make_agent()
    agent.default_config["stop_sequences"] = ["END"]
    agent.default_config["extra"] = {"k": "v"}
    agent._csm._best_config = {"stop_sequences": ["B"]}

    fork = agent._fork_for_candidate_run()

    def mutable_ids(value, acc):
        if isinstance(value, (dict, list)):
            acc.add(id(value))
            for item in value.values() if isinstance(value, dict) else value:
                mutable_ids(item, acc)
        return acc

    served = set()
    for cfg in (
        agent.default_config,
        agent._csm.default_config,
        agent._csm._current_config,
        agent._csm._best_config,
    ):
        mutable_ids(cfg, served)
    for cfg in (
        fork.default_config,
        fork._csm.default_config,
        fork._csm._current_config,
        fork._csm._best_config,
    ):
        assert not (mutable_ids(cfg, set()) & served)
    assert fork.default_config == agent.default_config
    assert fork._csm._best_config == agent._csm._best_config


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
    monkeypatch.setattr(
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
    monkeypatch.undo()
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
    monkeypatch.setattr(
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
    monkeypatch.undo()
    result = await agent.optimize(max_trials=3)
    assert result.best_config == {"temperature": 0.9}
