"""Non-finite grid bounds have zero cardinality, matching the value generator."""

import math

import pytest

from traigent.utils.discrete_domains import (
    stepped_float_cardinality,
    stepped_float_values,
)


@pytest.mark.parametrize(
    "args", [(0.0, 1.0, math.inf), (0.0, math.inf, 0.1), (math.nan, 1.0, 0.1)]
)
def test_non_finite_grid_has_zero_cardinality(args) -> None:
    assert stepped_float_values(*args) == ()
    assert stepped_float_cardinality(*args) == 0
