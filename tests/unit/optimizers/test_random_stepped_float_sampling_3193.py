"""Regression tests for #3193: stepped float sampling must not build the grid."""

from __future__ import annotations

import math
import random
import time

import pytest

import traigent.utils.discrete_domains as dd
from traigent.optimizers.random import RandomSearchOptimizer
from traigent.utils.discrete_domains import stepped_float_values
from traigent.utils.exceptions import OptimizationError


def _optimizer(step: float, seed: int = 3, high: float = 10.0):
    return RandomSearchOptimizer(
        {"x": {"type": "float", "low": 0.0, "high": high, "step": step}},
        ["accuracy"],
        max_trials=5,
        random_seed=seed,
    )


@pytest.mark.parametrize("step", [1e-6, 1e-7, 1e-300])
def test_tiny_step_samples_fast_and_in_range(step: float):
    opt = _optimizer(step)
    start = time.perf_counter()
    values = [opt._sample_range_float("x", opt.config_space["x"]) for _ in range(3)]
    assert time.perf_counter() - start < 2.0
    for v in values:
        assert 0.0 <= v <= 10.0 and math.isfinite(v)


def test_large_grid_values_are_grid_members():
    low, high, step = 0.0, 10.0, 1e-5  # 1,000,001 points, above the list cap
    rng = random.Random(11)
    # Membership against the exact grid built once (the old behaviour).
    grid = set(stepped_float_values(low, high, step))
    for _ in range(200):
        assert dd.sample_stepped_float(rng, low, high, step) in grid


class _FixedIndexRng:
    """Stand-in RNG whose randrange returns a chosen index."""

    def __init__(self, index: int):
        self.index = index

    def randrange(self, count: int) -> int:
        return self.index if self.index >= 0 else count + self.index


def test_large_grid_samples_off_grid_high_endpoint():
    low, high, step = 0.0, 10.00003, 0.001  # 10,001 regular points, high off-grid
    regular = 10_001
    last_regular = dd.sample_stepped_float(_FixedIndexRng(regular - 1), low, high, step)
    endpoint = dd.sample_stepped_float(_FixedIndexRng(regular), low, high, step)
    assert last_regular == 10.0
    assert endpoint == round(high, 6)
    assert endpoint in set(stepped_float_values(low, high, step)[-2:])


def test_clamped_points_near_high_are_not_overweighted():
    """A span below the tolerance must not make ``high`` half of all draws."""
    low, high, step = 0.0, 1e-9, 1e-13
    rng = random.Random(1)
    draws = [dd.sample_stepped_float(rng, low, high, step) for _ in range(2000)]
    assert draws.count(high) < 20  # ~1 expected from 10,001 grid values


def test_interval_count_overflow_is_rejected_with_value_error():
    with pytest.raises(ValueError, match="too large"):
        dd.sample_stepped_float(random.Random(1), 0.0, 10.0, 1e-309)


def test_same_seed_same_sequence_large_grid():
    a = [dd.sample_stepped_float(random.Random(5), 0.0, 10.0, 1e-6) for _ in range(1)]
    b = [dd.sample_stepped_float(random.Random(5), 0.0, 10.0, 1e-6) for _ in range(1)]
    assert a == b
    r1, r2 = random.Random(9), random.Random(9)
    assert [dd.sample_stepped_float(r1, 0.0, 10.0, 1e-6) for _ in range(20)] == [
        dd.sample_stepped_float(r2, 0.0, 10.0, 1e-6) for _ in range(20)
    ]


def test_small_grid_sequence_unchanged():
    """Below the cap the draw is rng.choice over the materialized grid."""
    grid = stepped_float_values(0.0, 1.0, 0.01)
    r1, r2 = random.Random(42), random.Random(42)
    assert [dd.sample_stepped_float(r1, 0.0, 1.0, 0.01) for _ in range(30)] == [
        r2.choice(grid) for _ in range(30)
    ]


@pytest.mark.parametrize(
    "low,high,step",
    [
        (0.0, 1.0, 0.0),
        (0.0, 1.0, -0.1),
        (0.0, 1.0, math.inf),
        (0.0, 1.0, math.nan),
        (0.0, math.inf, 0.1),
        (1.0, 0.0, 0.1),
        (-1e308, 1e308, 1e-300),
    ],
)
def test_degenerate_inputs_raise_value_error(low, high, step):
    with pytest.raises(ValueError):
        dd.sample_stepped_float(random.Random(1), low, high, step)


@pytest.mark.parametrize("step", [0, -1.0, "abc"])
def test_optimizer_rejects_bad_step(step):
    opt = _optimizer(1.0)
    with pytest.raises(OptimizationError):
        opt._sample_range_float(
            "x", {"type": "float", "low": 0.0, "high": 1.0, "step": step}
        )


def test_optimizer_wraps_overflow_in_optimization_error():
    opt = _optimizer(1.0)
    with pytest.raises(OptimizationError):
        opt._sample_range_float(
            "x", {"type": "float", "low": -1e308, "high": 1e308, "step": 1e-300}
        )
