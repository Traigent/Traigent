"""Characterization tests pinning TVLRegistry._split_at_depth_zero."""

from __future__ import annotations

import pytest

from traigent.tvl import registry as registry_module

_Cls = next(
    v
    for v in vars(registry_module).values()
    if isinstance(v, type) and hasattr(v, "_split_at_depth_zero")
)


def _split(expr: str, op: str) -> list[str]:
    return _Cls._split_at_depth_zero(object.__new__(_Cls), expr, op)


@pytest.mark.parametrize(
    "expr,op,expected",
    [
        ("a = 1", "AND", ["a = 1"]),
        ("a = 1 AND b = 2", "AND", ["a = 1", "b = 2"]),
        ("a = 1 and b = 2", "AND", ["a = 1", "b = 2"]),
        ("a = 1 AND b = 2 AND c = 3", "AND", ["a = 1", "b = 2", "c = 3"]),
        ("a = 1 OR b = 2 AND c = 3", "OR", ["a = 1", "b = 2 AND c = 3"]),
        ("a = 1 OR b = 2 AND c = 3", "AND", ["a = 1 OR b = 2", "c = 3"]),
        ("(a = 1 AND b = 2)", "AND", ["(a = 1 AND b = 2)"]),
        ("(a = 1 AND b = 2) AND c = 3", "AND", ["(a = 1 AND b = 2)", "c = 3"]),
        ("((a AND b) AND (c AND d))", "AND", ["((a AND b) AND (c AND d))"]),
        ("x = 'a AND b' AND y = 2", "AND", ["x = 'a AND b'", "y = 2"]),
        ('x = "a AND b"', "AND", ['x = "a AND b"']),
        ("x = 'it(' AND y = 2", "AND", ["x = 'it('", "y = 2"]),
        ("x = \"a'b\" AND y = 2", "AND", ["x = \"a'b\"", "y = 2"]),
        ("BRAND = 1", "AND", ["BRAND = 1"]),
        ("a ANDROID b", "AND", ["a ANDROID b"]),
        ("a AND", "AND", ["a AND"]),  # trailing operator: single part -> original
        ("AND a", "AND", ["", "a"]),  # leading operator yields an empty first part
        ("a AND  b", "AND", ["a", "b"]),
        ("a AND\tb", "AND", ["a", "b"]),
        ("a)  AND b", "AND", ["a)  AND b"]),  # unbalanced close: depth -1
        ("(a AND b", "AND", ["(a AND b"]),  # unbalanced open: depth stays 1
        ("", "AND", [""]),
        ("   ", "OR", ["   "]),
        ("a OR b", "AND", ["a OR b"]),
    ],
)
def test_split_at_depth_zero(expr, op, expected):
    assert _split(expr, op) == expected
