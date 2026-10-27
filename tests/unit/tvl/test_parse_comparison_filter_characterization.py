"""Golden characterization for FileRegistryResolver._parse_comparison_filter."""

from __future__ import annotations

import pytest

from traigent.tvl.registry import FileRegistryResolver

parse = FileRegistryResolver._parse_comparison_filter


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("", None),
        ("   ", None),
        ("version >= '1.0'", ("version", ">=", "1.0")),
        ('version<="2"', ("version", "<=", "2")),
        ("a > 3", ("a", ">", "3")),
        ("a<3", ("a", "<", "3")),
        ("  a_b1   >=   x y  ", ("a_b1", ">=", "x y")),
        ("a >= 1.2", ("a", ">=", "1.2")),
        ("a >= ''", ("a", ">=", "")),
        ("a >= '", None),
        ("a >= 'abc", None),
        ("a >= abc'", ("a", ">=", "abc'")),
        ("a >= \"abc'", None),
        ("a >=", None),
        ("a >=   ", None),
        (">= 1", None),
        ("a == 1", None),
        ("a = 1", None),
        ("a ! 1", None),
        ("a", None),
        ("a-b >= 1", None),
        ("é >= 1", ("é", ">=", "1")),
        # '>' wins over '>=' only when '>=' does not match first
        ("a >> 1", ("a", ">", "> 1")),
        ("a <> 1", ("a", "<", "> 1")),
    ],
)
def test_parse_comparison_filter_golden(expr, expected):
    assert parse(expr) == expected
