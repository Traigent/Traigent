"""ParetoFront equality compares objective values element-wise (#3113).

The generated dataclass ``__eq__`` compared the ndarray field inside a tuple,
which raised "truth value of an array is ambiguous" for any front with more
than one objective value.
"""

from __future__ import annotations

import numpy as np
import pytest

from traigent.api.types import ParetoFront


def _front(values: list[float] | list[list[float]], **overrides: object) -> ParetoFront:
    fields: dict[str, object] = {
        "configurations": [{"a": 1}, {"a": 2}],
        "objective_values": np.array(values),
        "objectives": ["x"],
        "is_maximized": [True],
    }
    fields.update(overrides)
    return ParetoFront(**fields)  # type: ignore[arg-type]


def test_identical_multi_value_fronts_are_equal() -> None:
    assert _front([1.0, 2.0]) == _front([1.0, 2.0])
    assert _front([[1.0, 0.1], [2.0, 0.2]]) == _front([[1.0, 0.1], [2.0, 0.2]])


def test_fronts_with_different_values_or_shape_are_unequal() -> None:
    assert _front([1.0, 2.0]) != _front([1.0, 3.0])
    assert _front([1.0, 2.0]) != _front([[1.0, 2.0]])


@pytest.mark.parametrize(
    "override",
    [
        {"configurations": [{"a": 1}, {"a": 3}]},
        {"objectives": ["y"]},
        {"is_maximized": [False]},
    ],
)
def test_other_fields_still_compared(override: dict[str, object]) -> None:
    assert _front([1.0, 2.0]) != _front([1.0, 2.0], **override)


def test_container_operations_work() -> None:
    fronts = [_front([1.0, 2.0]), _front([3.0, 4.0])]
    assert _front([3.0, 4.0]) in fronts
    assert fronts.index(_front([3.0, 4.0])) == 1


def test_not_equal_to_other_types_and_unhashable() -> None:
    assert _front([1.0, 2.0]) != "front"
    with pytest.raises(TypeError):
        hash(_front([1.0, 2.0]))
