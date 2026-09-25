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


# ---------------------------------------------------------------------------
# 5. Candidate isolation by a type-preserving structural detacher (Astra
#    delta review @ db91ce0c). Exact dict/list/tuple are rebuilt (tuples stay
#    tuples), JSON scalars are copied by value, functions/builtins/classes/Enum
#    members pass by reference; anything else fails closed with
#    CandidateIsolationError. No user hook ever runs, and detaching happens
#    outside the process-wide guard lock.
# ---------------------------------------------------------------------------


def _isolation_error():
    from traigent.utils.exceptions import CandidateIsolationError

    return CandidateIsolationError


@pytest.mark.asyncio
async def test_live_config_holding_a_lock_is_refused_and_served_config_untouched():
    """Astra's reproduction: a lock nested in live config used to make the
    fallback share the surrounding dict, so a candidate appended to the
    served list. Now the candidate is refused and nothing is shared."""
    import threading

    agent = _make_agent()
    lock = threading.Lock()
    agent._csm._current_config["extra"] = {"stops": ["END"], "lock": lock}

    with pytest.raises(_isolation_error()) as excinfo:
        await agent.optimize(max_trials=3, apply=False)
    assert "extra" in str(excinfo.value) and "lock" in str(excinfo.value)

    served = agent._csm._current_config["extra"]
    assert served["stops"] == ["END"]
    assert served["lock"] is lock
    assert agent.state == OptimizationState.UNOPTIMIZED
    # The refused candidate held nothing: an applying run is admitted.
    del agent._csm._current_config["extra"]
    result = await agent.optimize(max_trials=3)
    assert result.best_config == {"temperature": 0.9}


@pytest.mark.asyncio
async def test_shared_list_default_and_choice_mutated_by_candidate_trial_stays_put():
    """The same list object is a default AND a categorical choice; a candidate
    trial mutating the value it was handed must not reach the wrapper."""
    import json

    shared = ["END"]

    @traigent.optimize(
        eval_dataset=_dataset(),
        objectives=["accuracy"],
        configuration_space={"stops": [shared, ["STOP"]]},
        default_config={"stops": shared},
        scoring_function=_scorer,
        algorithm="grid",
    )
    async def agent(text: str) -> str:
        traigent.get_config()["stops"].append("LEAK")
        return "HOT"

    assert agent.default_config["stops"] is agent.configuration_space["stops"][0]
    before_default = json.dumps(agent.default_config, sort_keys=True)
    before_space = json.dumps(agent.configuration_space, sort_keys=True)
    before_current = json.dumps(agent.current_config, sort_keys=True)

    await agent.optimize(max_trials=3, apply=False)

    assert json.dumps(agent.default_config, sort_keys=True) == before_default
    assert json.dumps(agent.configuration_space, sort_keys=True) == before_space
    assert json.dumps(agent.current_config, sort_keys=True) == before_current
    assert shared == ["END"]


def test_deepcopy_hook_is_never_invoked_and_cannot_deadlock_admission():
    """A list subclass whose __deepcopy__ calls another wrapper's
    apply_best_config() used to deadlock (copying ran under the global guard
    lock). Now the subclass is refused without running any hook."""
    import threading

    other = _make_agent()
    other_candidate = other.optimize_sync(max_trials=3, apply=False)
    invoked: list[bool] = []

    class HookList(list):
        def __deepcopy__(self, memo):
            invoked.append(True)
            other.apply_best_config(other_candidate)
            return HookList(self)

        def __copy__(self):
            invoked.append(True)
            return HookList(self)

    agent = _make_agent()
    agent._csm._current_config["hook"] = HookList(["x"])

    outcome: list[BaseException | object] = []

    def run_candidate() -> None:
        try:
            outcome.append(asyncio.run(agent.optimize(max_trials=3, apply=False)))
        except BaseException as exc:
            outcome.append(exc)

    worker = threading.Thread(target=run_candidate, daemon=True)
    worker.start()
    worker.join(timeout=60)
    assert not worker.is_alive(), "candidate fork deadlocked"
    assert len(outcome) == 1 and isinstance(outcome[0], _isolation_error())
    assert "hook" in str(outcome[0])
    assert invoked == []
    # Admission is not wedged: the other wrapper can still promote.
    assert other.apply_best_config(other_candidate) is True


def test_candidate_snapshot_is_taken_under_the_guard_lock(
    monkeypatch: pytest.MonkeyPatch,
):
    """Owner contract (round 4): the detacher runs no user code (identity-typed
    builtins and NumPy scalars only -- see the hook/spoof tests), so the
    candidate snapshot is a real copy taken atomically under _RUN_GUARD_LOCK,
    consistent with run admission and promotion."""
    from traigent.core import config_state_manager, optimized_function

    held: list[bool] = []
    real = config_state_manager.detach_candidate_value

    def spy(value, path="config"):
        held.append(optimized_function._RUN_GUARD_LOCK.locked())
        return real(value, path)

    monkeypatch.setattr(config_state_manager, "detach_candidate_value", spy)
    monkeypatch.setattr(optimized_function, "detach_candidate_value", spy)

    agent = _make_agent()
    agent._fork_for_candidate_run()
    assert held and all(held)


def _mutable_ids(value, acc):
    if isinstance(value, (dict, list, tuple)):
        if isinstance(value, (dict, list)):
            acc.add(id(value))
        items = value.values() if isinstance(value, dict) else value
        for item in items:
            _mutable_ids(item, acc)
    return acc


def test_candidate_fork_search_space_shares_no_mutable_value():
    """The structural no-shared-value check, extended to the search space."""
    agent = _make_agent()
    agent.configuration_space = {
        "temperature": [0.1, 0.9],
        "stops": [["END"], ["STOP"]],
        "nested": [{"k": ["v"]}],
    }
    agent._csm.configuration_space = agent.configuration_space

    fork = agent._fork_for_candidate_run()

    served = _mutable_ids(agent.configuration_space, set())
    _mutable_ids(agent._csm.configuration_space, served)
    assert not (_mutable_ids(fork.configuration_space, set()) & served)
    assert not (_mutable_ids(fork._csm.configuration_space, set()) & served)
    assert fork.configuration_space == agent.configuration_space


@pytest.mark.asyncio
async def test_call_time_search_space_override_is_detached(
    monkeypatch: pytest.MonkeyPatch,
):
    """A call-time configuration_space override is detached for a candidate."""
    import json

    override = {"temperature": [0.1, 0.9], "stops": [["END"]]}
    before = json.dumps(override, sort_keys=True)

    @traigent.optimize(
        eval_dataset=_dataset(),
        objectives=["accuracy"],
        configuration_space={"temperature": [0.1, 0.9], "stops": [["END"]]},
        scoring_function=_scorer,
        algorithm="grid",
    )
    async def agent(text: str) -> str:
        traigent.get_config()["stops"].append("LEAK")
        return "HOT"

    await agent.optimize(max_trials=2, apply=False, configuration_space=override)
    assert json.dumps(override, sort_keys=True) == before


@pytest.mark.asyncio
async def test_candidate_on_range_and_intrange_searches_the_continuous_range():
    """Tuples mean continuous ranges; detaching must keep them tuples (a JSON
    round-trip would turn Range(0, 1) into the two-value categorical [0, 1])."""
    from traigent.api.parameter_ranges import IntRange, Range

    @traigent.optimize(
        eval_dataset=_dataset(),
        objectives=["accuracy"],
        configuration_space={"temperature": Range(0.0, 1.0), "k": IntRange(1, 8)},
        scoring_function=_scorer,
        algorithm="random",
    )
    async def agent(text: str) -> str:
        return "HOT"

    fork = agent._fork_for_candidate_run()
    assert fork.configuration_space["temperature"] == (0.0, 1.0)
    assert isinstance(fork.configuration_space["temperature"], tuple)
    assert isinstance(fork.configuration_space["k"], tuple)

    result = await agent.optimize(max_trials=8, apply=False)
    temps = {t.config["temperature"] for t in result.trials}
    ks = {t.config["k"] for t in result.trials}
    assert temps - {0.0, 1.0}, temps
    assert ks - {1, 8}, ks


@pytest.mark.parametrize("kind", ["function", "builtin", "enum", "class"])
def test_function_builtin_class_and_enum_choices_are_refused_never_invoked(kind):
    """No by-reference allowlist: functions, builtins, classes and Enum members
    can carry mutable state (defaults, attributes, class attributes), so they
    are refused with the key path -- and never invoked while classifying.
    (At base bad5953b such choices already fail end to end offline with "not
    JSON serializable", for apply=True too, so nothing that worked is lost.)"""
    import enum

    calls: list[str] = []

    def fmt_a() -> str:
        calls.append("a")
        return "a"

    def fmt_b() -> str:
        calls.append("b")
        return "b"

    class Strategy(enum.Enum):
        FAST = "fast"
        SLOW = "slow"

    class Planner:
        pass

    choices = {
        "function": [fmt_a, fmt_b],
        "builtin": [len, max],
        "enum": [Strategy.FAST, Strategy.SLOW],
        "class": [Planner, dict],
    }[kind]

    agent = _make_agent()
    agent.configuration_space = {"temperature": [0.1, 0.9], "choice": choices}
    agent._csm.configuration_space = agent.configuration_space

    with pytest.raises(_isolation_error()) as excinfo:
        agent._fork_for_candidate_run()
    assert "configuration_space['choice'][0]" in str(excinfo.value)
    assert calls == []


# ---------------------------------------------------------------------------
# 6. Identity-only classification, real snapshot inside the lock, retry on
#    concurrent served mutation (Astra review @ cfc8f4b6; owner contract).
# ---------------------------------------------------------------------------


def _fork_with_served(**served_config):
    agent = _make_agent()
    agent._csm._current_config.update(served_config)
    return agent


def test_intenum_with_mutable_attribute_is_refused():
    """Astra: an IntEnum member carrying a mutable attribute leaked mutation."""
    import enum

    class Level(enum.IntEnum):
        LOW = 1
        HIGH = 2

    Level.LOW.tags = ["served"]  # type: ignore[attr-defined]
    agent = _fork_with_served(level=Level.LOW)
    with pytest.raises(_isolation_error()) as excinfo:
        agent._fork_for_candidate_run()
    assert "current_config['level']" in str(excinfo.value)
    assert Level.LOW.tags == ["served"]  # type: ignore[attr-defined]


def test_function_with_mutable_default_is_refused():
    """A function's mutable default (or attribute) is shared state."""

    def pick(stops=["END"]):  # noqa: B006 - the shared default IS the hazard
        return stops

    agent = _fork_with_served(picker=pick)
    with pytest.raises(_isolation_error()) as excinfo:
        agent._fork_for_candidate_run()
    assert "current_config['picker']" in str(excinfo.value)


def test_bound_method_choice_is_refused():
    """``some_list.append`` is a builtin_function_or_method bound to a mutable
    object: passing it by reference would let a candidate mutate that list."""
    served_list = ["END"]
    agent = _make_agent()
    agent.configuration_space = {
        "temperature": [0.1, 0.9],
        "hook": [served_list.append],
    }
    agent._csm.configuration_space = agent.configuration_space
    with pytest.raises(_isolation_error()) as excinfo:
        agent._fork_for_candidate_run()
    assert "configuration_space['hook'][0]" in str(excinfo.value)
    assert served_list == ["END"]


def test_metaclass_eq_spoof_is_refused():
    """A class whose metaclass __eq__ claims equality with str must not be
    classified as a scalar (tuple/set membership would consult it)."""
    eq_calls: list[object] = []

    class Liar(type):
        def __eq__(cls, other):
            eq_calls.append(other)
            return True

        __hash__ = type.__hash__

    class Sneaky(metaclass=Liar):
        def __init__(self) -> None:
            self.payload = ["served"]

    value = Sneaky()
    agent = _fork_with_served(sneaky=value)
    with pytest.raises(_isolation_error()):
        agent._fork_for_candidate_run()
    assert eq_calls == []  # classification never consulted the hook


def test_class_property_spoof_is_refused():
    """``isinstance`` trusts a ``__class__`` property; ``type()`` does not."""

    class Pretender:
        def __init__(self) -> None:
            self.payload = ["served"]

        @property  # type: ignore[misc]
        def __class__(self):  # noqa: D401
            return int

    value = Pretender()
    assert isinstance(value, int)  # the spoof works against isinstance
    agent = _fork_with_served(pretender=value)
    with pytest.raises(_isolation_error()):
        agent._fork_for_candidate_run()


@pytest.mark.asyncio
async def test_numpy_int64_default_and_choice_are_accepted_and_normalized():
    """NumPy scalars go through the SDK's existing JSON normalizer
    (best_config_runtime._normalize_json_value) to plain Python scalars."""
    np = pytest.importorskip("numpy")

    @traigent.optimize(
        eval_dataset=_dataset(),
        objectives=["accuracy"],
        configuration_space={"k": [np.int64(1), np.int64(3)]},
        default_config={"k": np.int64(3)},
        scoring_function=_scorer,
        algorithm="grid",
    )
    async def agent(text: str) -> str:
        return "HOT"

    fork = agent._fork_for_candidate_run()
    assert fork.default_config == {"k": 3}
    assert type(fork.default_config["k"]) is int
    assert fork.configuration_space["k"] == [1, 3]
    assert all(type(v) is int for v in fork.configuration_space["k"])
    # The served wrapper still holds its own numpy values.
    assert type(agent.default_config["k"]) is np.int64

    result = await agent.optimize(max_trials=2, apply=False)
    assert result.best_config["k"] in (1, 3)


def test_concurrent_served_mutation_during_copy_retries_to_a_consistent_copy(
    monkeypatch: pytest.MonkeyPatch,
):
    """Deterministic interleaving: the served dict grows while the copy is
    iterating it (as a get_config() writer in another thread would). The copy
    retries and yields a consistent snapshot, never a RuntimeError."""
    from traigent.core import config_state_manager

    marker = ["m"]
    agent = _fork_with_served(a=marker, b=1)
    served = agent._csm._current_config
    real = config_state_manager._detach
    hits: list[int] = []

    def racing(value, path, on_path):
        if value is marker and not hits:
            hits.append(1)
            served["late"] = 2  # mutation lands mid-iteration of `served`
        return real(value, path, on_path)

    monkeypatch.setattr(config_state_manager, "_detach", racing)
    fork = agent._fork_for_candidate_run()
    assert fork._csm._current_config == served
    assert fork._csm._current_config["late"] == 2
    assert fork._csm._current_config["a"] is not marker


def test_persistent_served_mutation_during_copy_raises_the_clean_error(
    monkeypatch: pytest.MonkeyPatch,
):
    from traigent.core import config_state_manager

    marker = ["m"]
    agent = _fork_with_served(a=marker)
    served = agent._csm._current_config
    real = config_state_manager._detach
    counter = iter(range(10_000))

    def always_racing(value, path, on_path):
        if value is marker:
            served[f"late{next(counter)}"] = 1
        return real(value, path, on_path)

    monkeypatch.setattr(config_state_manager, "_detach", always_racing)
    with pytest.raises(_isolation_error(), match="changed during candidate snapshot"):
        agent._fork_for_candidate_run()


def test_threaded_served_mutation_never_crashes_the_candidate_snapshot():
    """Real threads: a writer mutates the served config through get_config()-
    style dict writes while candidates are forked. Every fork is either a
    consistent copy or the clean CandidateIsolationError -- never a crash."""
    import threading

    agent = _fork_with_served(**{f"k{i}": [i] for i in range(200)})
    served = agent._csm._current_config
    stop = threading.Event()

    def writer() -> None:
        n = 0
        while not stop.is_set():
            served[f"w{n % 50}"] = [n]
            served.pop(f"w{(n + 25) % 50}", None)
            n += 1

    thread = threading.Thread(target=writer, daemon=True)
    thread.start()
    try:
        for _ in range(200):
            try:
                fork = agent._fork_for_candidate_run()
            except _isolation_error() as exc:
                assert "changed during candidate snapshot" in str(exc)
                continue
            assert type(fork._csm._current_config) is dict
    finally:
        stop.set()
        thread.join(timeout=10)
    assert not thread.is_alive()


@pytest.mark.asyncio
async def test_at_rest_snapshot_is_unaffected_by_mutation_after_admission():
    """The at-rest snapshot is a real copy taken at admission: in-place
    mutations of the served config/search space during the applying run do
    not reach candidates forked afterwards."""
    gate = _Gate(hold_calls=1)
    agent = _make_agent(gate)
    agent._csm._current_config["extra"] = {"stops": ["END"]}

    run = asyncio.ensure_future(agent.optimize(max_trials=3))
    await _wait(gate.entered)
    # Mutate the served state in place after admission.
    agent._csm._current_config["extra"]["stops"].append("MID-RUN")
    agent.configuration_space["temperature"].append(0.5)

    fork = agent._fork_for_candidate_run()
    assert fork._csm._current_config["extra"]["stops"] == ["END"]
    assert fork.configuration_space["temperature"] == [0.1, 0.9]

    gate.release.set()
    await asyncio.wait_for(run, timeout=30)
    agent.configuration_space["temperature"].remove(0.5)
