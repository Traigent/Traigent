"""Constraint-rejected configs must not consume max_trials slots (Traigent#3210)."""

from __future__ import annotations

import asyncio

import pytest

import traigent

SPACE = {"n": [1, 2, 3], "x": [0.0, 0.5, 1.0]}


def _run(algorithm: str, constraint, max_trials: int, space=None, **kwargs):
    calls: list[dict] = []

    @traigent.optimize(
        configuration_space=space or SPACE,
        objectives=["accuracy"],
        eval_dataset=[{"input_data": {"r": 1}, "expected_output": 1}],
        offline=True,
        constraints=[constraint],
        scoring_function=lambda output, expected_output=None, example=None: 1.0,
    )
    def f(r):
        calls.append(dict(traigent.get_config()))
        return 1

    result = asyncio.run(
        f.optimize(algorithm=algorithm, max_trials=max_trials, **kwargs)
    )
    return result, calls


@pytest.mark.parametrize("algorithm", ["grid", "random"])
def test_rejected_configs_do_not_consume_max_trials(algorithm):
    # n>=2 rejects a third of the space; 6 feasible configs remain.
    result, calls = _run(algorithm, lambda c: c["n"] >= 2, 4, random_seed=3)

    assert len(calls) == 4
    assert all(c["n"] >= 2 for c in calls)
    assert result.stop_reason == "max_trials_reached"


@pytest.mark.parametrize("algorithm", ["grid", "random"])
def test_constraint_rejecting_everything_evaluates_nothing_and_terminates(algorithm):
    result, calls = _run(algorithm, lambda c: False, 4, random_seed=3)

    assert calls == []
    assert result.stop_reason != "max_trials_reached"


def test_unsatisfiable_constraint_on_continuous_space_hits_rejection_cap():
    calls: list[dict] = []

    @traigent.optimize(
        configuration_space={"t": (0.0, 1.0)},
        objectives=["accuracy"],
        eval_dataset=[{"input_data": {"r": 1}, "expected_output": 1}],
        offline=True,
        constraints=[lambda c: False],
        scoring_function=lambda output, expected_output=None, example=None: 1.0,
    )
    def f(r):
        calls.append(dict(traigent.get_config()))
        return 1

    result = asyncio.run(f.optimize(algorithm="random", max_trials=2, random_seed=1))

    assert calls == []
    assert result.stop_reason == "optimizer"


def test_grid_reaches_feasible_point_after_many_rejections():
    # 290 leading infeasible points exceed 100 x max_trials; the finite grid
    # must still reach the feasible block.
    space = {"a": list(range(30)), "b": list(range(10))}
    result, calls = _run("grid", lambda c: c["a"] >= 29, 1, space=space)

    assert len(calls) == 1
    assert calls[0]["a"] == 29


@pytest.mark.parametrize("algorithm", ["grid", "random"])
def test_parallel_batches_do_not_lose_slots_to_rejections(algorithm):
    result, calls = _run(
        algorithm,
        lambda c: c["n"] >= 2,
        4,
        random_seed=3,
        parallel_config={"trial_concurrency": 2},
    )

    assert len(calls) == 4
    assert all(c["n"] >= 2 for c in calls)


def test_parallel_grid_reaches_feasible_point_after_long_infeasible_prefix():
    # 290 leading infeasible points exceed the 100 x slots attempt cap of the
    # parallel batch generator; the finite grid must end by exhaustion only.
    space = {"a": list(range(30)), "b": list(range(10))}
    result, calls = _run(
        "grid",
        lambda c: c["a"] >= 26,
        4,
        space=space,
        parallel_config={"trial_concurrency": 2},
    )

    assert len(calls) == 4
    assert all(c["a"] >= 26 for c in calls)
    assert result.stop_reason == "max_trials_reached"


def test_finite_random_search_is_not_capped_before_exhaustion():
    # 300 points, one feasible: far more than 100 x max_trials draws are
    # rejected before the feasible point is found.
    space = {"a": list(range(300))}
    result, calls = _run(
        "random", lambda c: c["a"] == 137, 1, space=space, random_seed=5
    )

    assert calls == [{"a": 137}]


class _FixedKnobResolver:
    """Resolves every sample to a config that hashes differently."""

    def resolve(self, config):
        class _Resolved:
            used_fallbacks = ()

        resolved = _Resolved()
        resolved.config = {**config, "fixed_knob": "pinned"}
        return resolved


@pytest.mark.asyncio
async def test_rejected_sample_identity_is_the_optimizers_not_the_resolved_config():
    from tests.unit.core.test_trial_lifecycle import (
        MockOrchestrator,
        create_mock_dataset,
    )
    from traigent.core.trial_lifecycle import TrialLifecycle
    from traigent.optimizers.random import RandomSearchOptimizer

    def reject_all(config, metrics=None):
        return False

    reject_all.__tvl_constraint__ = {"id": "r", "message": "reject"}

    orchestrator = MockOrchestrator()
    orchestrator.optimizer = RandomSearchOptimizer(
        {"a": [1, 2, 3]}, ["accuracy"], max_trials=1, random_seed=0
    )
    orchestrator.max_trials = 1
    orchestrator.knob_resolver = _FixedKnobResolver()
    orchestrator._constraints_pre_eval = [reject_all]
    lifecycle = TrialLifecycle(orchestrator)

    async def func(input_data):
        return "x"

    seen: list[int] = []
    for _ in range(3):
        _, action = await lifecycle.run_sequential_trial(
            func=func,
            dataset=create_mock_dataset(),
            session_id=None,
            function_name="f",
            trial_count=0,
        )
        seen.append(len(orchestrator.optimizer._tried_config_hashes))
        assert action == "continue"

    # Each rejected sample is registered once under its sampled identity, so
    # three rejections exhaust the 3-point space (no resolved-hash inflation,
    # and no redraw of an already rejected sample).
    assert seen == [1, 2, 3]
    assert all(
        "fixed_knob" not in h for h in orchestrator.optimizer._tried_config_hashes
    )
    assert orchestrator.optimizer.is_config_space_exhausted()


def test_progress_does_not_count_constraint_rejected_records():
    from types import SimpleNamespace

    from traigent.core.orchestrator import OptimizationOrchestrator

    rejected = SimpleNamespace(metadata={"constraint_rejected": True})
    executed = SimpleNamespace(metadata={})
    fake = SimpleNamespace(
        _start_time=None,
        _successful_trials=1,
        _failed_trials=0,
        best_result=None,
        _trials=[rejected, rejected, executed],
        max_trials=4,
        optimizer=SimpleNamespace(objectives=["accuracy"]),
    )

    info = OptimizationOrchestrator._get_progress_info(fake, 1)

    assert info.completed_trials == 1
