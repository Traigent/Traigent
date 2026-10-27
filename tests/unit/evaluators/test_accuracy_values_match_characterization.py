"""Characterization tests pinning evaluators.base._accuracy_values_match."""

from __future__ import annotations

import pytest

from traigent.evaluators import base


@pytest.fixture
def reports(monkeypatch):
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(base, "_report_coercion", lambda k, n: calls.append((k, n)))
    return calls


@pytest.mark.parametrize(
    "actual,expected,result",
    [
        # strings
        ("Hello", " hello ", True),
        ("a", "b", False),
        # numeric strings
        ("1.0", "1", True),
        (".5", "0.5", True),
        ("1e5", "100000", True),
        ("007", "7", False),
        ("7", "007", False),
        ("007", "007", True),
        ("abc", "1", False),
        ("1", "abc", False),
        # typed coercion from actual string
        ("5", 5, True),
        ("5", 6, False),
        ("true", True, True),
        ("[1, 2]", [1, 2], True),
        ('{"a": 1}', {"a": 1}, True),
        # floats
        (0.1 + 0.2, 0.3, True),
        (1, 1.0, True),
        (1.0, 2, False),
        (float("nan"), float("nan"), False),
        (True, 1.0, False),
        (1.0, True, False),
        (None, 1.0, False),
        ([1], 1.0, False),
        # ints and others
        (3, 3, True),
        (3, 4, False),
        (None, None, True),
        # dicts
        ({"a": 1}, {"a": 1}, True),
        ({"a": 1}, {"a": 2}, False),
        ({"a": 1}, {"b": 1}, False),
        ({"a": 1, "b": 2}, {"a": 1}, False),
        ({"a": "X"}, {"a": "x"}, True),
        ({"a": "1.0"}, {"a": 1}, True),
        # sequences
        ([1, 2], [1, 2], True),
        ([1, 2], (1, 2), True),
        ([1, 2], [1], False),
        ([1, 2], [1, 3], False),
        (["A", ["b"]], ["a", ["B"]], True),
        ([1.0], [1], True),
        # shape mismatches
        ({"a": 1}, [1], False),
        ([1], {"a": 1}, False),
    ],
)
def test_values_match(actual, expected, result, reports):
    assert base._accuracy_values_match(actual, expected) is result


def test_string_to_typed_coercion_is_reported(reports):
    assert base._accuracy_values_match("5", 5) is True
    assert reports == [("int", 1)]


def test_numeric_string_pair_is_reported(reports):
    assert base._accuracy_values_match("1.0", "1") is True
    assert reports == [("numeric", 3)]


def test_failed_numeric_parse_not_reported(reports):
    assert base._accuracy_values_match("x", "1") is False
    assert base._accuracy_values_match("007", "7") is False
    assert reports == []


def test_exact_string_match_not_reported(reports):
    assert base._accuracy_values_match("A", "a") is True
    assert reports == []


def test_nested_coercion_reported_per_element(reports):
    assert base._accuracy_values_match(["1", "2"], [1, 2]) is True
    assert reports == [("int", 1), ("int", 1)]
