"""Characterization tests for TVL tuple-component domain resolution.

Pin the values produced and the exact error messages for every branch of
``_tuple_component_values`` (values/set/components/registry/range forms).
"""

from __future__ import annotations

import math

import pytest

from traigent.tvl.models import _tuple_component_values

NAME = "T"
IDX = 1

OK_CASES = [
    ([1, 2], [1, 2]),
    ([], []),
    ({"values": [1, 2], "set": [9]}, [1, 2]),
    ({"values": (3, 4)}, [3, 4]),
    ({"set": ["a", "b"]}, ["a", "b"]),
    ({"values": [], "range": [0, 5]}, []),
    ({"components": [[1, 2], [3]]}, [(1, 3), (2, 3)]),
    (
        {"components": [{"values": [1, 2]}, {"range": [0, 2]}]},
        [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
    ),
    ({"range": [0, 3]}, [0, 1, 2, 3]),
    ({"range": [2, 2]}, [2]),
    ({"range": [0.0, 1.0], "resolution": 0.25}, [0.0, 0.25, 0.5, 0.75, 1.0]),
    ({"range": [0, 4], "resolution": 2}, [0, 2, 4]),
    ({"range": [0, 4], "resolution": 2.0}, [0, 2, 4]),
    ({"range": [0, 1], "resolution": 0.5}, [0.0, 0.5, 1.0]),
    (
        {"range": [0.0, 1.0], "resolution": 0.1},
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    ),
    ({"range": [0, 4], "resolution": None}, [0, 1, 2, 3, 4]),
    ({"range": [0.0, 2.0], "resolution": 1}, [0.0, 1.0, 2.0]),
]

ERROR_CASES = [
    (
        "str",
        ValueError,
        "TVAR 'T' tuple component 1 must be a list or a domain mapping",
    ),
    (5, ValueError, "TVAR 'T' tuple component 1 must be a list or a domain mapping"),
    (None, ValueError, "TVAR 'T' tuple component 1 must be a list or a domain mapping"),
    (
        (1, 2),
        ValueError,
        "TVAR 'T' tuple component 1 must be a list or a domain mapping",
    ),
    (
        {"components": []},
        ValueError,
        "TVAR 'T' tuple domain must define a non-empty 'components' list",
    ),
    (
        {"components": "x"},
        ValueError,
        "TVAR 'T' tuple domain must define a non-empty 'components' list",
    ),
    (
        {"components": [[1], []]},
        ValueError,
        "TVAR 'T' tuple component 1 has empty domain",
    ),
    (
        {"registry": "models"},
        ValueError,
        "TVAR 'T' tuple component 1 uses a registry domain, which is not supported inside tuple components",
    ),
    (
        {"registry": "m", "range": [0, 3]},
        ValueError,
        "TVAR 'T' tuple component 1 uses a registry domain, which is not supported inside tuple components",
    ),
    (
        {"range": [3, 2]},
        ValueError,
        "TVAR 'T' tuple component 1 range min must not exceed max",
    ),
    ({"range": [0]}, ValueError, "TVAR 'T' tuple component 1 range must be [min, max]"),
    (
        {"range": [0, 1, 2]},
        ValueError,
        "TVAR 'T' tuple component 1 range must be [min, max]",
    ),
    (
        {"range": "ab"},
        ValueError,
        "TVAR 'T' tuple component 1 range must be [min, max]",
    ),
    (
        {"range": (0, 2)},
        ValueError,
        "TVAR 'T' tuple component 1 range must be [min, max]",
    ),
    (
        {"range": [True, 3]},
        ValueError,
        "TVAR 'T' tuple component 1 range bounds must be numbers",
    ),
    (
        {"range": ["a", 3]},
        ValueError,
        "TVAR 'T' tuple component 1 range bounds must be numbers",
    ),
    (
        {"range": [0, math.inf]},
        ValueError,
        "TVAR 'T' tuple component 1 range bounds must be finite",
    ),
    (
        {"range": [math.nan, 1]},
        ValueError,
        "TVAR 'T' tuple component 1 range bounds must be finite",
    ),
    (
        {"range": [0.0, 1.0]},
        ValueError,
        "TVAR 'T' tuple component 1 has a float range but no 'resolution'; add a resolution to enumerate grid points",
    ),
    (
        {"range": [0, 1.0]},
        ValueError,
        "TVAR 'T' tuple component 1 has a float range but no 'resolution'; add a resolution to enumerate grid points",
    ),
    (
        {"range": [0, 10001]},
        ValueError,
        "TVAR 'T' tuple component 1 range enumerates too many grid points",
    ),
    (
        {"range": [1, 0], "resolution": 1},
        ValueError,
        "TVAR 'T' tuple component 1 range min must not exceed max",
    ),
    (
        {"range": [0, 4], "resolution": 0},
        ValueError,
        "TVAR 'T' tuple component 1 resolution must be a positive number",
    ),
    (
        {"range": [0, 4], "resolution": -1},
        ValueError,
        "TVAR 'T' tuple component 1 resolution must be a positive number",
    ),
    (
        {"range": [0, 4], "resolution": True},
        ValueError,
        "TVAR 'T' tuple component 1 resolution must be a positive number",
    ),
    (
        {"range": [0, 4], "resolution": "1"},
        ValueError,
        "TVAR 'T' tuple component 1 resolution must be a positive number",
    ),
    (
        {"range": [0, 4], "resolution": math.inf},
        ValueError,
        "TVAR 'T' tuple component 1 resolution must be a positive number",
    ),
    (
        {"range": [0, 1000000], "resolution": 1},
        ValueError,
        "TVAR 'T' tuple component 1 range enumerates too many grid points",
    ),
    (
        {"range": [1e16, 1.0000000000000016e16], "resolution": 0.001},
        ValueError,
        "TVAR 'T' tuple component 1 range enumerates too many grid points",
    ),
    (
        {},
        ValueError,
        "TVAR 'T' tuple component 1 must define one of values, set, range, or components",
    ),
    (
        {"other": 1},
        ValueError,
        "TVAR 'T' tuple component 1 must define one of values, set, range, or components",
    ),
    ({"values": 5}, TypeError, "'int' object is not iterable"),
    (
        {"range": [1e16, 1.0000000000000002e16], "resolution": 0.5},
        ValueError,
        "TVAR 'T' tuple component 1 resolution is finer than float precision allows",
    ),
]


@pytest.mark.parametrize(("component", "expected"), OK_CASES)
def test_resolves_values(component, expected):
    result = _tuple_component_values(NAME, IDX, component)
    assert result == expected
    assert [type(v) for v in result] == [type(v) for v in expected]


@pytest.mark.parametrize(("component", "exc_type", "message"), ERROR_CASES)
def test_rejects_with_exact_message(component, exc_type, message):
    with pytest.raises(exc_type) as err:
        _tuple_component_values(NAME, IDX, component)
    assert str(err.value) == message


def test_integer_range_at_the_enumeration_limit_is_accepted():
    values = _tuple_component_values(NAME, IDX, {"range": [0, 9999]})
    assert len(values) == 10_000
    assert values[0] == 0 and values[-1] == 9999


def test_grid_values_stay_inside_the_bounds():
    values = _tuple_component_values(
        NAME, IDX, {"range": [0.0, 1.0], "resolution": 0.3}
    )
    assert values[0] == 0.0
    assert all(0.0 <= v <= 1.0 and math.isfinite(v) for v in values)
    assert len(values) == 4
