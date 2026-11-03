"""Characterization tests pinning the fp2 iterative encoder's exact behaviour.

Covers output text, the order in which errors are raised, and the exact error
messages for each rejection path, so the encoder can be restructured safely.
"""

from __future__ import annotations

import pytest

from traigent.utils import fp2
from traigent.utils.fp2 import MAX_DEPTH, Fp2UnsupportedValue, canonicalize


class _Str(str):
    pass


class _Int(int):
    pass


class _List(list):
    pass


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "null"),
        (True, "true"),
        (False, "false"),
        ("a", '"a"'),
        (0, "0"),
        (-7, "-7"),
        (1.5, "1.5"),
        ([], "[]"),
        ({}, "{}"),
        ([1, [2, [3]], None], "[1,[2,[3]],null]"),
        ({"b": 1, "a": [True, None]}, '{"a":[true,null],"b":1}'),
        ({"a": {"c": 1, "b": 2}}, '{"a":{"b":2,"c":1}}'),
        ([{"z": 1}, {"y": 2}], '[{"z":1},{"y":2}]'),
        ({"\U0001f600": 1, "\uffff": 2}, '{"\U0001f600":1,"\uffff":2}'),
    ],
)
def test_encoded_text(value, expected):
    assert canonicalize(value) == expected


def test_shared_non_circular_reference_is_allowed_twice():
    shared = [1]
    assert canonicalize([shared, shared]) == "[[1],[1]]"
    assert canonicalize({"a": shared, "b": shared}) == '{"a":[1],"b":[1]}'


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (_Str("x"), "unsupported type: _Str"),
        (_Int(1), "unsupported type: _Int"),
        (_List([1]), "unsupported type: _List"),
        ((1, 2), "unsupported type: tuple"),
        (b"x", "unsupported type: bytes"),
        (object(), "unsupported type: object"),
        ({1}, "unsupported type: set"),
        ([1, (2,)], "unsupported type: tuple"),
        ({"a": _Str("x")}, "unsupported type: _Str"),
        ({1: "a"}, "object key is not a plain string: int"),
        ({_Str("k"): 1}, "object key is not a plain string: _Str"),
        ({"a": 1, 2: 3}, "object key is not a plain string: int"),
    ],
)
def test_unsupported_values_have_exact_messages(value, message):
    with pytest.raises(Fp2UnsupportedValue) as err:
        canonicalize(value)
    assert str(err.value) == message


def test_key_check_runs_before_any_value_is_inspected():
    # A bad key is reported even though an earlier-sorted value is also bad.
    with pytest.raises(Fp2UnsupportedValue) as err:
        canonicalize({"a": object(), 1: 2})
    assert str(err.value) == "object key is not a plain string: int"


def test_first_unsupported_value_in_emission_order_wins():
    with pytest.raises(Fp2UnsupportedValue) as err:
        canonicalize({"b": object(), "a": (1,)})
    assert str(err.value) == "unsupported type: tuple"


def test_integer_range_limit():
    limit = fp2._MAX_SAFE_INTEGER
    assert canonicalize(limit) == str(limit)
    assert canonicalize(-limit) == str(-limit)
    with pytest.raises(Fp2UnsupportedValue, match="safe range"):
        canonicalize(limit + 1)
    with pytest.raises(Fp2UnsupportedValue, match="safe range"):
        canonicalize([-limit - 1])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_floats_are_rejected(bad):
    with pytest.raises(Fp2UnsupportedValue):
        canonicalize([bad])


def _nest(levels: int):
    value: list = []
    for _ in range(levels - 1):
        value = [value]
    return value


def test_depth_limit_boundary():
    assert canonicalize(_nest(MAX_DEPTH)) == "[" * MAX_DEPTH + "]" * MAX_DEPTH
    with pytest.raises(Fp2UnsupportedValue) as err:
        canonicalize(_nest(MAX_DEPTH + 1))
    assert str(err.value) == f"manifest nests deeper than the fp2 limit of {MAX_DEPTH}"


def test_depth_limit_applies_through_dicts():
    value: dict = {}
    for _ in range(MAX_DEPTH):
        value = {"k": value}
    with pytest.raises(Fp2UnsupportedValue, match="nests deeper"):
        canonicalize(value)


def test_circular_list_and_dict_are_rejected():
    loop: list = []
    loop.append(loop)
    with pytest.raises(Fp2UnsupportedValue) as err:
        canonicalize(loop)
    assert str(err.value) == "circular reference"

    d: dict = {}
    d["self"] = {"inner": d}
    with pytest.raises(Fp2UnsupportedValue) as err2:
        canonicalize(d)
    assert str(err2.value) == "circular reference"


def test_container_is_closed_so_sibling_reuse_after_it_is_not_circular():
    shared: list = []
    assert canonicalize([[shared], shared]) == "[[[]],[]]"


def test_foreign_exception_is_translated_with_type_only():
    class Boom(dict):
        pass

    class Weird:
        pass

    with pytest.raises(Fp2UnsupportedValue):
        canonicalize(Boom())
    with pytest.raises(Fp2UnsupportedValue):
        canonicalize(Weird())
