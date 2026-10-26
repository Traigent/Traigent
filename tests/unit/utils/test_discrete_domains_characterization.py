"""Characterization tests for config-space finite-domain helpers.

Pin the exact behaviour of ``discrete_cardinality_for_config_param`` and
``discrete_values_for_config_param`` across list, tuple, categorical, fixed,
integer and float definitions, including degenerate and malformed inputs.
"""

from __future__ import annotations

import pytest

from traigent.utils.discrete_domains import (
    discrete_cardinality_for_config_param,
    discrete_values_for_config_param,
)

_RAISES = "raises"


CARDINALITY_CASES = [
    ([1, 2, 3], 3),
    ((), 1),
    ((0.1, 0.9), None),
    ((1, 2, 3), 1),
    (5, 1),
    ("x", 1),
    (None, 1),
    ({"type": "categorical", "choices": ["a", "b"]}, 2),
    ({"type": "choice", "values": [1, 2, 3]}, 3),
    ({"choices": ["a"]}, 1),
    ({"values": ("p", "q")}, 2),
    ({"type": "CHOICE", "choices": 5}, None),
    ({"choices": [1], "values": [1, 2]}, 1),
    ({"type": "categorical"}, 0),
    ({"low": 0, "high": 1, "choices": [1, 2]}, None),
    ({"type": "fixed", "value": 3}, 1),
    ({"type": "constant", "default": 4}, 1),
    ({"type": "fixed"}, 1),
    ({"type": "fixed", "value": 1, "default": 2}, 1),
    ({"type": "int", "low": 0, "high": 10}, 11),
    ({"type": "integer", "low": 0, "high": 10, "step": 3}, 5),
    ({"type": "int", "low": 0, "high": 10, "step": None}, 11),
    ({"type": "int", "low": 5, "high": 1}, None),
    ({"type": "int", "low": 0, "high": 10, "step": 0}, None),
    ({"type": "int", "low": "a", "high": 3}, None),
    ({"type": "int", "low": None, "high": 3}, None),
    ({"type": "int", "low": 0, "high": 2.9, "step": "1"}, 3),
    ({"type": "float", "low": 0.0, "high": 1.0, "step": 0.25}, 5),
    ({"type": "float", "low": 0.0, "high": 1.0}, None),
    ({"type": "float", "low": 0.0, "high": 1.0, "step": 0.3}, 5),
    ({"type": "float", "low": 1.0, "high": 0.0, "step": 0.1}, None),
    ({"type": "float", "low": 0, "high": 1, "step": -1}, None),
    ({"type": "float", "low": "x", "high": 1, "step": 0.1}, None),
    ({"low": 0, "high": 1, "step": 0.5}, 3),
    ({"low": 0, "high": 1}, None),
    ({"type": "float", "low": 0, "high": 1e300, "step": 1e300}, 2),
    ({"type": "unknown"}, 1),
    ({"type": "unknown", "low": 0, "high": 2, "step": 1}, 3),
    ({}, 1),
    ({"type": "int", "low": 0, "high": 0}, 1),
    ({"low": 0}, 1),
    ({"type": "categorical", "choices": []}, 0),
]


VALUES_CASES = [
    ([1, 2, 3], (1, 2, 3)),
    ((), ((),)),
    ((0.1, 0.9), None),
    ((1, 2, 3), ((1, 2, 3),)),
    (5, (5,)),
    ("x", ("x",)),
    (None, (None,)),
    ({"type": "categorical", "choices": ["a", "b"]}, ("a", "b")),
    ({"type": "choice", "values": [1, 2, 3]}, (1, 2, 3)),
    ({"choices": ["a"]}, ("a",)),
    ({"values": ("p", "q")}, ("p", "q")),
    ({"type": "CHOICE", "choices": 5}, _RAISES),
    ({"choices": [1], "values": [1, 2]}, (1,)),
    ({"type": "categorical"}, ()),
    ({"low": 0, "high": 1, "choices": [1, 2]}, None),
    ({"type": "fixed", "value": 3}, (3,)),
    ({"type": "constant", "default": 4}, (4,)),
    ({"type": "fixed"}, ({"type": "fixed"},)),
    ({"type": "fixed", "value": 1, "default": 2}, (1,)),
    ({"type": "int", "low": 0, "high": 10}, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)),
    ({"type": "integer", "low": 0, "high": 10, "step": 3}, (0, 3, 6, 9, 10)),
    (
        {"type": "int", "low": 0, "high": 10, "step": None},
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    ),
    ({"type": "int", "low": 5, "high": 1}, ()),
    ({"type": "int", "low": 0, "high": 10, "step": 0}, ()),
    ({"type": "int", "low": "a", "high": 3}, None),
    ({"type": "int", "low": None, "high": 3}, None),
    ({"type": "int", "low": 0, "high": 2.9, "step": "1"}, (0, 1, 2)),
    (
        {"type": "float", "low": 0.0, "high": 1.0, "step": 0.25},
        (0.0, 0.25, 0.5, 0.75, 1.0),
    ),
    ({"type": "float", "low": 0.0, "high": 1.0}, None),
    (
        {"type": "float", "low": 0.0, "high": 1.0, "step": 0.3},
        (0.0, 0.3, 0.6, 0.9, 1.0),
    ),
    ({"type": "float", "low": 1.0, "high": 0.0, "step": 0.1}, ()),
    ({"type": "float", "low": 0, "high": 1, "step": -1}, ()),
    ({"type": "float", "low": "x", "high": 1, "step": 0.1}, None),
    ({"low": 0, "high": 1, "step": 0.5}, (0.0, 0.5, 1.0)),
    ({"low": 0, "high": 1}, None),
    ({"type": "float", "low": 0, "high": 1e300, "step": 1e300}, (0.0, 1e300)),
    ({"type": "unknown"}, ({"type": "unknown"},)),
    ({"type": "unknown", "low": 0, "high": 2, "step": 1}, (0.0, 1.0, 2.0)),
    ({}, ({},)),
    ({"type": "int", "low": 0, "high": 0}, (0,)),
    ({"low": 0}, ({"low": 0},)),
    ({"type": "categorical", "choices": []}, ()),
]


@pytest.mark.parametrize(("definition", "expected"), CARDINALITY_CASES)
def test_discrete_cardinality_for_config_param(definition, expected):
    if expected == _RAISES:
        with pytest.raises(TypeError):
            discrete_cardinality_for_config_param(definition)
    else:
        assert discrete_cardinality_for_config_param(definition) == expected


@pytest.mark.parametrize(("definition", "expected"), VALUES_CASES)
def test_discrete_values_for_config_param(definition, expected):
    if expected == _RAISES:
        with pytest.raises(TypeError):
            discrete_values_for_config_param(definition)
    else:
        result = discrete_values_for_config_param(definition)
        assert result == expected
        assert type(result) is type(expected)
