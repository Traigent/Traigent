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
