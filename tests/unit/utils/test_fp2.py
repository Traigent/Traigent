"""Direct unit tests for `traigent/utils/fp2.py`'s load-bearing rules.

The vendored corpus (`test_fp2_agent_lifecycle_parity.py`,
`data/fp2/agent_lifecycle_cases.json`) proves the ported module agrees with
TraigentSchema's reference implementation on 18 pinned manifests. It does
not exercise every rule ALR-1301 calls load-bearing, because those 18 cases
were built for a different story's scope (agent-lifecycle-record parity).
This file closes that gap directly against the module, independent of any
vendored corpus:

* astral-vs-BMP UTF-16 key ordering (fp2.md's own worked example: 😀
  U+1F600 sorts *before* Ａ U+FF21 under code-unit order, the reverse of
  code-point order -- Python's bare `sorted()` would get this backwards),
* ECMAScript `Number::toString` notation thresholds (values taken verbatim
  from fp2.md's own table, not from this implementation, so this is an
  independent check),
* exact-type dispatch (`type(x) is dict`, never `isinstance`) for a `dict`
  subclass, a `bool` masquerading through `int`-like contexts, and an
  `IntEnum`,
* tuples rejected (never silently treated as arrays),
* the `2**53 - 1` safe-integer boundary,
* nesting beyond `MAX_DEPTH`,
* circular references.
"""

from __future__ import annotations

import enum
from typing import Any

import pytest

from traigent.utils.fp2 import MAX_DEPTH, Fp2UnsupportedValue, canonicalize, digest


# ---------------------------------------------------------------------------
# UTF-16 code-unit key ordering (fp2.md "Key ordering").
# ---------------------------------------------------------------------------


def test_astral_key_sorts_before_bmp_key_under_utf16_code_unit_order() -> None:
    """😀 (U+1F600) encodes as the surrogate pair D83D DE00; Ａ (U+FF21) is a
    single BMP unit FF21. D83D < FF21, so 😀 sorts FIRST under UTF-16
    code-unit order even though its code point (0x1F600) is numerically
    larger than Ａ's (0xFF21) -- code point and code unit order invert here,
    which is exactly why bare Python `sorted()` (code-point order) is
    forbidden by the spec."""
    canonical = canonicalize({"Ａ": 2, "😀": 1})
    assert canonical == '{"😀":1,"Ａ":2}'


def test_astral_key_ordering_after_common_prefix() -> None:
    canonical = canonicalize({"aＡ": 2, "a😀": 1})
    assert canonical == '{"a😀":1,"aＡ":2}'


def test_ascii_uppercase_sorts_before_lowercase() -> None:
    canonical = canonicalize({"a": 2, "Z": 1})
    assert canonical == '{"Z":1,"a":2}'


def test_lone_surrogate_key_fails_closed() -> None:
    with pytest.raises(Fp2UnsupportedValue):
        canonicalize({"a\ud800b": 1})


# ---------------------------------------------------------------------------
# ECMAScript Number::toString (fp2.md "Numbers" table, values verbatim).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected_text",
    [
        (1e16, "10000000000000000"),
        (1e20, "100000000000000000000"),
        (1e-5, "0.00001"),
        (1e-7, "1e-7"),
        (1e21, "1e+21"),
        (0.1, "0.1"),
        (3.0, "3"),
        (-0.0, "0"),
    ],
    ids=["1e16", "1e20", "1e-5", "1e-7", "1e21", "0.1", "3.0", "neg-zero"],
)
def test_ecmascript_number_notation_thresholds(
    value: float, expected_text: str
) -> None:
    assert canonicalize(value) == expected_text


def test_nan_and_infinity_fail_closed() -> None:
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(Fp2UnsupportedValue):
            canonicalize(value)


# ---------------------------------------------------------------------------
# Exact-type dispatch (fp2.md "Types are matched exactly, never by subclass").
# ---------------------------------------------------------------------------


class _DictSubclass(dict):
    """Overrides nothing, but its mere type identity must be rejected."""


class _IntSubclass(int):
    pass


class _Color(enum.IntEnum):
    RED = 1


def test_dict_subclass_is_rejected_even_though_isinstance_dict_is_true() -> None:
    value = _DictSubclass(a=1)
    assert isinstance(value, dict)  # sanity: isinstance() would admit it
    with pytest.raises(Fp2UnsupportedValue):
        canonicalize(value)


def test_int_subclass_is_rejected() -> None:
    with pytest.raises(Fp2UnsupportedValue):
        canonicalize(_IntSubclass(3))


def test_intenum_is_rejected_despite_isinstance_int_is_true() -> None:
    assert isinstance(_Color.RED, int)
    with pytest.raises(Fp2UnsupportedValue):
        canonicalize(_Color.RED)


def test_bool_is_still_accepted_as_its_own_literal_not_via_int_dispatch() -> None:
    """True/False are handled by identity check before the int branch, so
    they must still canonicalize as JSON booleans, not as 1/0."""
    assert canonicalize(True) == "true"
    assert canonicalize(False) == "false"


def test_ordereddict_is_rejected() -> None:
    import collections

    with pytest.raises(Fp2UnsupportedValue):
        canonicalize(collections.OrderedDict(a=1))


def test_str_subclass_key_is_rejected() -> None:
    class _StrSubclass(str):
        pass

    with pytest.raises(Fp2UnsupportedValue):
        canonicalize({_StrSubclass("a"): 1})


# ---------------------------------------------------------------------------
# Tuples rejected (fp2.md "Tuples are rejected, and this was a close call").
# ---------------------------------------------------------------------------


def test_bare_tuple_is_rejected() -> None:
    with pytest.raises(Fp2UnsupportedValue):
        canonicalize((1, 2, 3))


def test_tuple_nested_in_a_list_is_rejected() -> None:
    with pytest.raises(Fp2UnsupportedValue):
        canonicalize([1, (2, 3)])


def test_namedtuple_is_rejected() -> None:
    import collections

    Point = collections.namedtuple("Point", ["x", "y"])
    with pytest.raises(Fp2UnsupportedValue):
        canonicalize(Point(1, 2))


# ---------------------------------------------------------------------------
# Safe-integer boundary (fp2.md "an integer outside the IEEE-754 safe
# integer range ... is an unsupported value").
# ---------------------------------------------------------------------------

_MAX_SAFE_INTEGER = 2**53 - 1


def test_max_safe_integer_is_accepted() -> None:
    assert canonicalize(_MAX_SAFE_INTEGER) == str(_MAX_SAFE_INTEGER)
    assert canonicalize(-_MAX_SAFE_INTEGER) == str(-_MAX_SAFE_INTEGER)


def test_one_past_max_safe_integer_fails_closed() -> None:
    with pytest.raises(Fp2UnsupportedValue):
        canonicalize(_MAX_SAFE_INTEGER + 1)
    with pytest.raises(Fp2UnsupportedValue):
        canonicalize(-(_MAX_SAFE_INTEGER + 1))


def test_float_of_the_same_magnitude_is_still_fine() -> None:
    """A *float* beyond 2**53-1 round-trips through a JS Number exactly
    (it's already lost precision at the bit level, consistently in both
    languages), so only the int type is restricted."""
    value = float(_MAX_SAFE_INTEGER) * 1000
    assert canonicalize(value)  # does not raise


# ---------------------------------------------------------------------------
# Nesting depth (fp2.md "Nesting depth").
# ---------------------------------------------------------------------------


def _nest(depth: int) -> Any:
    value: Any = "leaf"
    for _ in range(depth):
        value = [value]
    return value


def test_manifest_at_the_depth_limit_is_accepted() -> None:
    # The outermost container is level 1, so MAX_DEPTH nested lists is
    # exactly at the limit.
    assert canonicalize(_nest(MAX_DEPTH)) is not None


def test_manifest_one_level_past_the_depth_limit_fails_closed() -> None:
    with pytest.raises(Fp2UnsupportedValue):
        canonicalize(_nest(MAX_DEPTH + 1))


# ---------------------------------------------------------------------------
# Circular references (fp2.md unsupported-values list).
# ---------------------------------------------------------------------------


def test_circular_list_reference_fails_closed() -> None:
    value: list[Any] = []
    value.append(value)
    with pytest.raises(Fp2UnsupportedValue):
        canonicalize(value)


def test_circular_dict_reference_fails_closed() -> None:
    value: dict[str, Any] = {}
    value["self"] = value
    with pytest.raises(Fp2UnsupportedValue):
        canonicalize(value)


# ---------------------------------------------------------------------------
# digest() format and end-to-end sanity.
# ---------------------------------------------------------------------------


def test_digest_is_algorithm_prefixed_sha256() -> None:
    result = digest({"a": 1})
    assert result.startswith("sha256:")
    assert len(result) == len("sha256:") + 64
    int(result.removeprefix("sha256:"), 16)  # raises ValueError if not hex


def test_digest_raises_the_same_error_type_as_canonicalize_never_a_different_one() -> (
    None
):
    with pytest.raises(Fp2UnsupportedValue):
        digest(float("nan"))
