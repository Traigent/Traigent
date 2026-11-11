"""Random range-dict samples keep their precision and stay inside the range (#3216).

Continuous range dicts (the form LogRange / Range(log=True) take) were rounded
to 6 decimals, so bounds below 1e-6 collapsed onto a 1e-6 lattice and even
produced 0.0, below the declared low.
"""

from __future__ import annotations

import pytest

from traigent.api.parameter_ranges import LogRange
from traigent.optimizers.random import RandomSearchOptimizer


def _samples(definition: object, n: int = 20, seed: int = 1) -> list[float]:
    opt = RandomSearchOptimizer(
        {"x": definition}, ["accuracy"], max_trials=n, random_seed=seed
    )
    return [opt.suggest_next_trial([])["x"] for _ in range(n)]


@pytest.mark.parametrize(
    ("definition", "low", "high"),
    [
        (LogRange(1e-9, 1e-5).to_config_value(), 1e-9, 1e-5),
        ({"type": "float", "low": 1e-9, "high": 5e-7}, 1e-9, 5e-7),
    ],
)
def test_small_bounds_stay_in_range_and_distinct(
    definition: object, low: float, high: float
) -> None:
    values = _samples(definition)
    assert all(low <= v <= high for v in values), values
    assert len(set(values)) == len(values)


def test_values_keep_twelve_significant_digits() -> None:
    values = _samples({"type": "float", "low": 1e-9, "high": 5e-7})
    for value in values:
        assert value == float(f"{value:.12g}")
        assert value != round(value, 6)
    # More precision than a 6-significant-digit rounding would keep.
    assert any(value != float(f"{value:.6g}") for value in values)


def test_degenerate_range_returns_its_bound() -> None:
    assert _samples({"type": "float", "low": 2.5e-8, "high": 2.5e-8}, n=1) == [2.5e-8]
