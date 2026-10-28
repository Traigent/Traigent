"""Characterization tests pinning evaluators.base._coerce_string_to_expected_type."""

from __future__ import annotations

import pytest

from traigent.evaluators.base import _coerce_string_to_expected_type as coerce


@pytest.mark.parametrize(
    "actual,expected,result",
    [
        (" True ", True, (True, True)),
        ("false", False, (False, True)),
        ("FALSE", True, (False, True)),
        ("yes", True, ("yes", False)),
        ("", False, ("", False)),
        (" 5 ", 5, (5.0, True)),
        ("-2.5", 3, (-2.5, True)),
        ("nan", 3, None),  # float('nan'); checked separately below
        ("five", 3, ("five", False)),
        ("1e3", 2.0, (1000.0, True)),
        ("x", 2.0, ("x", False)),
        (" 7", 0, (7.0, True)),
        ('{"a": 1}', {"k": 1}, ({"a": 1}, True)),
        ("[1, 2]", [9], ([1, 2], True)),
        ("[1, 2]", (9,), ([1, 2], True)),
        ('{"a": 1}', [9], ('{"a": 1}', False)),
        ("[1]", {"k": 1}, ("[1]", False)),
        ("not json", {"k": 1}, ("not json", False)),
        ("5", [1], ("5", False)),
        ("null", [1], ("null", False)),
        ("abc", "abc", ("abc", False)),
        ("abc", None, ("abc", False)),
    ],
)
def test_coerce(actual, expected, result):
    if result is None:
        value, ok = coerce(actual, expected)
        assert ok is True and value != value
        return
    assert coerce(actual, expected) == result
    got = coerce(actual, expected)
    assert type(got[0]) is type(result[0])
