"""Random search samples log-scale ranges log-uniformly (#3214).

``LogRange`` / ``Range(log=True)`` / ``IntRange(log=True)`` emit ``log: True``
in their range dict, but random search drew ``uniform(low, high)``, so a range
over several decades spent ~90% of trials in the top decade.
"""

from __future__ import annotations

import math
import statistics

import pytest

from traigent.api.parameter_ranges import IntRange, LogRange, Range
from traigent.optimizers.random import RandomSearchOptimizer
from traigent.utils.exceptions import OptimizationError

N = 2000


def _samples(definition: object, seed: int = 7, n: int = N) -> list:
    opt = RandomSearchOptimizer(
        {"p": definition}, ["accuracy"], max_trials=n, random_seed=seed
    )
    return [opt.suggest_next_trial([])["p"] for _ in range(n)]


@pytest.mark.parametrize(
    "definition",
    [
        LogRange(0.001, 0.1).to_config_value(),
        Range(0.001, 0.1, log=True).to_config_value(),
    ],
)
def test_log_float_range_is_log_uniform(definition: object) -> None:
    values = _samples(definition)
    geometric_mean = math.sqrt(0.001 * 0.1)  # 0.01
    below = sum(v < geometric_mean for v in values) / N
    assert 0.45 <= below <= 0.55
    # Each decade gets about half of the draws.
    assert 0.45 <= sum(0.001 <= v < 0.01 for v in values) / N <= 0.55
    assert all(0.001 <= v <= 0.1 for v in values)


def test_log_int_range_is_log_uniform() -> None:
    # A wide range keeps duplicate rejection (discrete spaces sample unique
    # configs) from distorting the distribution of a small sample.
    values = _samples(IntRange(1, 1_000_000, log=True).to_config_value(), n=200)
    assert all(isinstance(v, int) and 1 <= v <= 1_000_000 for v in values)
    below = sum(v <= 1000 for v in values) / len(values)
    # log-uniform puts ~half the mass at or below the geometric mean (1000);
    # uniform sampling would put ~0.1% there.
    assert 0.4 <= below <= 0.6


def test_linear_range_still_uniform() -> None:
    values = _samples({"type": "float", "low": 0.0, "high": 1.0})
    assert 0.45 <= statistics.median(values) <= 0.55


def test_seeded_log_sampling_is_reproducible() -> None:
    definition = LogRange(1e-5, 1e-1).to_config_value()
    assert _samples(definition, seed=3, n=20) == _samples(definition, seed=3, n=20)


@pytest.mark.parametrize("low", [0.0, -1.0])
def test_log_range_with_non_positive_low_fails_loudly(low: float) -> None:
    opt = RandomSearchOptimizer(
        {"p": {"type": "float", "low": low, "high": 1.0, "log": True}},
        ["accuracy"],
        max_trials=1,
        random_seed=0,
    )
    with pytest.raises(OptimizationError, match="positive"):
        opt.suggest_next_trial([])
