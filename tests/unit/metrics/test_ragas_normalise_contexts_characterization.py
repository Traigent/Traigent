"""Characterization tests pinning ragas_metrics._normalise_contexts."""

from __future__ import annotations

import pytest

from traigent.metrics.ragas_metrics import _normalise_contexts


class _Weird:
    def __str__(self) -> str:
        return "weird"


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ("abc", ["abc"]),
        ("", [""]),
        ([], None),
        ((), None),
        ([None, None], None),
        (["a", None, "b"], ["a", "b"]),
        (("a", "b"), ["a", "b"]),
        ([1, 2.5, True], ["1", "2.5", "True"]),
        ([{"text": "t", "content": "c"}], ["t"]),
        ([{"text": "", "content": "c"}], ["c"]),
        ([{"text": None, "content": None}], ["{'text': None, 'content': None}"]),
        ([{"content": "c"}], ["c"]),
        ([{"text": 5}], ["5"]),
        ([{"other": 1}], ["{'other': 1}"]),
        ([{}], ["{}"]),
        ({"a": "x", "b": "y"}, ["x", "y"]),
        ({}, None),
        ({"a": None}, None),
        ({"a": {"text": "deep"}}, ["deep"]),
        (5, ["5"]),
        (_Weird(), ["weird"]),
        ([_Weird()], ["weird"]),
        ([["nested", "list"]], ["['nested', 'list']"]),
        (b"bytes", ["b'bytes'"]),
    ],
)
def test_normalise_contexts(value, expected):
    assert _normalise_contexts(value) == expected
