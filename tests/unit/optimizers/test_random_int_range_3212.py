"""Regression tests for #3212: IntRange without a step is discrete (step 1)."""

import asyncio

import traigent
from traigent.api.parameter_ranges import IntRange, Range
from traigent.optimizers.random import RandomSearchOptimizer
from traigent.utils.exceptions import OptimizationError


def test_int_range_without_step_is_finite_for_random_search():
    space = {"n": IntRange(1, 3).to_config_value()}
    optimizer = RandomSearchOptimizer(space, ["accuracy"], max_trials=10, random_seed=1)

    assert optimizer.config_space_cardinality == 3

    seen = []
    while True:
        try:
            seen.append(optimizer.suggest_next_trial([])["n"])
        except OptimizationError as exc:
            assert "Config space exhausted" in str(exc)
            break

    assert sorted(seen) == [1, 2, 3]


def test_float_range_mixed_with_int_range_stays_continuous():
    space = {
        "n": IntRange(1, 3).to_config_value(),
        "t": Range(0.0, 1.0).to_config_value(),
    }
    optimizer = RandomSearchOptimizer(space, ["accuracy"], max_trials=10, random_seed=1)

    assert optimizer.config_space_cardinality is None
    configs = [optimizer.suggest_next_trial([]) for _ in range(10)]
    assert len({c["t"] for c in configs}) == 10
    assert all(isinstance(c["n"], int) and 1 <= c["n"] <= 3 for c in configs)


def test_decorated_run_stops_after_three_distinct_configs():
    @traigent.optimize(
        configuration_space={"n": IntRange(1, 3)},
        objectives=["accuracy"],
        offline=True,
        eval_dataset=[{"input_data": {"r": 1}, "expected_output": 1}],
        scoring_function=lambda output, expected_output=None, example=None: 1.0,
    )
    def f(r):
        return 1

    res = asyncio.run(f.optimize(algorithm="random", max_trials=10, random_seed=1))

    assert sorted(t.config["n"] for t in res.trials) == [1, 2, 3]
    assert res.stop_reason != "max_trials_reached"


def test_wide_int_range_samples_without_materializing_the_grid():
    space = {"n": IntRange(1, 10**12).to_config_value()}
    optimizer = RandomSearchOptimizer(space, ["accuracy"], max_trials=5, random_seed=3)

    for _ in range(5):
        assert 1 <= optimizer.suggest_next_trial([])["n"] <= 10**12


def test_int_range_above_float_precision_stays_inside_bounds():
    low = 2**53 + 1
    space = {"n": IntRange(low, low + 2).to_config_value()}
    for seed in range(20):
        optimizer = RandomSearchOptimizer(
            space, ["accuracy"], max_trials=3, random_seed=seed
        )
        assert low <= optimizer.suggest_next_trial([])["n"] <= low + 2


def test_stepped_int_range_keeps_off_grid_high_endpoint():
    space = {"n": IntRange(1, 10, step=4).to_config_value()}
    optimizer = RandomSearchOptimizer(space, ["accuracy"], max_trials=10, random_seed=1)

    seen = set()
    while True:
        try:
            seen.add(optimizer.suggest_next_trial([])["n"])
        except OptimizationError:
            break
    assert seen == {1, 5, 9, 10}


def test_wandb_sweep_exports_int_range_as_searchable_domain():
    from traigent.integrations.observability.wandb import create_wandb_sweep_config

    config = create_wandb_sweep_config(
        "f", {"n": IntRange(1, 3).to_config_value()}, ["accuracy"]
    )
    assert config["parameters"]["n"] == {
        "min": 1,
        "max": 3,
        "distribution": "int_uniform",
    }


def test_wandb_sweep_exports_stepped_int_range_as_exact_grid():
    from traigent.integrations.observability.wandb import create_wandb_sweep_config

    config = create_wandb_sweep_config(
        "f", {"n": IntRange(1, 10, step=4).to_config_value()}, ["accuracy"]
    )
    assert config["parameters"]["n"] == {"values": [1, 5, 9, 10]}
