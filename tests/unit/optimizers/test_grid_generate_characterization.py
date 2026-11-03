"""Golden characterization for GridSearchOptimizer._generate_grid."""

from __future__ import annotations

import pytest

from traigent.optimizers.grid import GridSearchOptimizer
from traigent.utils.exceptions import OptimizationError


def _grid(space):
    return GridSearchOptimizer(space, ["accuracy"])._grid_points


def test_list_fixed_and_discrete_dict_product_order():
    pts = _grid({"a": [1, 2], "b": "fixed", "c": {"low": 0, "high": 2, "step": 1}})
    assert pts == [
        {"a": 1, "b": "fixed", "c": 0},
        {"a": 1, "b": "fixed", "c": 1},
        {"a": 1, "b": "fixed", "c": 2},
        {"a": 2, "b": "fixed", "c": 0},
        {"a": 2, "b": "fixed", "c": 1},
        {"a": 2, "b": "fixed", "c": 2},
    ]


def test_only_fixed_values_single_point():
    assert _grid({"a": 5, "b": None}) == [{"a": 5, "b": None}]


def test_three_tuple_is_fixed_value():
    assert _grid({"a": (1, 2, 3)}) == [{"a": (1, 2, 3)}]


def test_continuous_dict_error_message():
    with pytest.raises(OptimizationError) as exc:
        _grid({"a": [1], "t": {"low": 0.0, "high": 1.0}})
    msg = str(exc.value)
    assert msg.startswith(
        "Grid search cannot optimize continuous parameter 't' with range"
    )
    assert "(1) provide a step for the range" in msg
    assert msg.endswith("instead for continuous ranges")


def test_int_tuple_suggestion():
    with pytest.raises(OptimizationError) as exc:
        _grid({"t": (0, 10)})
    assert str(exc.value) == (
        "Grid search cannot optimize continuous parameter 't' "
        "with range (0, 10). Grid search requires discrete values to enumerate. "
        "Options: (1) Use a list of discrete values: [0, 5, 10], "
        "(2) Use 'random' optimizer instead for continuous ranges"
    )


def test_float_tuple_suggestion():
    with pytest.raises(OptimizationError) as exc:
        _grid({"t": (0.0, 1.0)})
    assert "[0.0, 0.50, 1.0]" in str(exc.value)
    with pytest.raises(OptimizationError) as exc:
        _grid({"t": (0, 1.0)})
    assert "[0, 0.50, 1.0]" in str(exc.value)


def test_first_offending_parameter_in_order_is_reported():
    with pytest.raises(OptimizationError) as exc:
        _grid({"a": [1], "t1": (0, 2), "t2": {"low": 0.0, "high": 1.0}})
    assert "'t1'" in str(exc.value)


def test_empty_list_param_yields_no_combinations():
    with pytest.raises(OptimizationError):
        GridSearchOptimizer({"a": []}, ["accuracy"])
