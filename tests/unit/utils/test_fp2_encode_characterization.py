"""Characterization of the fp2 iterative encoder (exact output and errors)."""

from __future__ import annotations

import pytest

from traigent.utils import fp2
from traigent.utils.fp2 import MAX_DEPTH, Fp2UnsupportedValue, canonicalize


def _nested_list(levels: int) -> list:
    value: list = []
    for _ in range(levels - 1):
        value = [value]
    return value


class _Str(str):
    pass


class _Int(int):
    pass


class _Float(float):
    pass


class _Dict(dict):
    pass


class _List(list):
    pass


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "null"),
        (True, "true"),
        (False, "false"),
        ('a"b', '"a\\"b"'),
        (0, "0"),
        (-7, "-7"),
        (2**53 - 1, "9007199254740991"),
        (0.5, "0.5"),
        (-0.0, "0"),
        (1e21, "1e+21"),
        (1.5e-7, "1.5e-7"),
        ([], "[]"),
        ({}, "{}"),
        ([1, [2, None], True], "[1,[2,null],true]"),
        ({"b": 1, "a": [1, 2]}, '{"a":[1,2],"b":1}'),
        ({"a": {"y": 1, "x": 2}}, '{"a":{"x":2,"y":1}}'),
        ([[], {}, [[]]], "[[],{},[[]]]"),
        ({"\U0001f600": 1, "￿": 2}, '{"\U0001f600":1,"￿":2}'),
    ],
)
def test_exact_outputs(value, expected):
    assert canonicalize(value) == expected


def test_same_object_twice_is_not_circular():
    shared = [1]
    assert canonicalize([shared, shared]) == "[[1],[1]]"
    assert canonicalize({"a": shared, "b": shared}) == '{"a":[1],"b":[1]}'


def test_depth_limit_boundary():
    assert canonicalize(_nested_list(MAX_DEPTH)) == "[" * MAX_DEPTH + "]" * MAX_DEPTH
    with pytest.raises(Fp2UnsupportedValue) as info:
        canonicalize(_nested_list(MAX_DEPTH + 1))
    assert str(info.value) == f"manifest nests deeper than the fp2 limit of {MAX_DEPTH}"


def test_depth_limit_for_dicts():
    value: dict = {}
    for _ in range(MAX_DEPTH):
        value = {"k": value}
    with pytest.raises(Fp2UnsupportedValue, match="nests deeper"):
        canonicalize(value)


@pytest.mark.parametrize("kind", ["list", "dict"])
def test_circular_reference(kind):
    if kind == "list":
        value: list = []
        value.append(value)
    else:
        value = {}
        value["k"] = value
    with pytest.raises(Fp2UnsupportedValue) as info:
        canonicalize(value)
    assert str(info.value) == "circular reference"


def test_circular_reference_via_nesting():
    inner: list = []
    outer = [inner]
    inner.append(outer)
    with pytest.raises(Fp2UnsupportedValue, match="^circular reference$"):
        canonicalize(outer)


def test_non_string_key_rejected_with_type_name():
    with pytest.raises(Fp2UnsupportedValue) as info:
        canonicalize({1: "x"})
    assert str(info.value) == "object key is not a plain string: int"


def test_str_subclass_key_rejected():
    with pytest.raises(Fp2UnsupportedValue) as info:
        canonicalize({_Str("a"): 1})
    assert str(info.value) == "object key is not a plain string: _Str"


@pytest.mark.parametrize(
    ("value", "name"),
    [
        (_Str("a"), "_Str"),
        (_Int(1), "_Int"),
        (_Float(1.0), "_Float"),
        (_Dict(), "_Dict"),
        (_List(), "_List"),
        ((1, 2), "tuple"),
        (b"x", "bytes"),
        (object(), "object"),
        ({1}, "set"),
    ],
)
def test_unsupported_type(value, name):
    with pytest.raises(Fp2UnsupportedValue) as info:
        canonicalize([value])
    assert str(info.value) == f"unsupported type: {name}"


def test_unsupported_root_type():
    with pytest.raises(Fp2UnsupportedValue, match="^unsupported type: tuple$"):
        canonicalize(())


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite(value):
    with pytest.raises(Fp2UnsupportedValue) as info:
        canonicalize([value])
    assert str(info.value) == "non-finite number (NaN or Infinity)"


def test_unsafe_integer():
    with pytest.raises(Fp2UnsupportedValue, match="safe range"):
        canonicalize({"a": 2**53})
    assert canonicalize(-(2**53 - 1)) == "-9007199254740991"


def test_first_error_in_emission_order_wins():
    # Sorted key order is "a" then "b": the unsafe int under "a" is hit first.
    with pytest.raises(Fp2UnsupportedValue, match="safe range"):
        canonicalize({"b": float("nan"), "a": 2**60})


def test_bad_key_checked_before_any_child():
    with pytest.raises(Fp2UnsupportedValue, match="object key is not a plain"):
        canonicalize({"a": float("nan"), 2: 1})


def test_depth_checked_before_circularity():
    value: list = _nested_list(MAX_DEPTH)
    deepest = value
    while deepest:
        deepest = deepest[0]
    deepest.append(value)
    with pytest.raises(Fp2UnsupportedValue, match="nests deeper"):
        canonicalize(value)


def test_module_tags_unchanged():
    assert (fp2._EMIT, fp2._VALUE, fp2._CLOSE) == (0, 1, 2)
