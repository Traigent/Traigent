"""#3196 (constructors) and #3197 (condition builders): typed, named errors."""

from __future__ import annotations

import math

import pytest

from traigent.api.parameter_ranges import Choices, IntRange, LogRange, Range

NUMERIC_BUILDERS = ["equals", "not_equals", "gt", "gte", "lt", "lte"]
BAD_TYPES = [None, [1], {}, "x", "5", True]


# ---------------------------------------------------------------- #3197
@pytest.mark.parametrize("builder", NUMERIC_BUILDERS)
@pytest.mark.parametrize("bad", BAD_TYPES)
@pytest.mark.parametrize(
    "make",
    [
        lambda: Range(0.0, 10.0, name="t"),
        lambda: IntRange(0, 10, name="t"),
        lambda: LogRange(0.1, 10.0, name="t"),
    ],
)
def test_builders_reject_non_numbers_naming_parameter(make, builder, bad):
    with pytest.raises(
        TypeError, match=rf"t\.{builder}\(\) needs a number, got "
    ) as ei:
        getattr(make(), builder)(bad)
    assert "'5'" not in str(ei.value)


@pytest.mark.parametrize("builder", ["is_in", "not_in"])
@pytest.mark.parametrize("bad", [None, "5", True, [1]])
def test_membership_builders_reject_non_numbers(builder, bad):
    with pytest.raises(TypeError, match=rf"t\.{builder}\(\)"):
        getattr(Range(0.0, 10.0, name="t"), builder)([bad])


def test_membership_builders_reject_non_sequence():
    with pytest.raises(TypeError, match=r"t\.is_in\(\) needs a sequence"):
        Range(0.0, 10.0, name="t").is_in("123")


@pytest.mark.parametrize("builder", NUMERIC_BUILDERS)
def test_builders_overflow_is_a_named_error(builder):
    with pytest.raises(ValueError, match=rf"t\.{builder}\(\) value is too large"):
        getattr(Range(0.0, 10.0, name="t"), builder)(10**400)


def test_in_range_rejections():
    r = Range(0.0, 1.0, name="t")
    with pytest.raises(ValueError, match="NaN"):
        r.in_range(math.nan, math.nan)
    for bad in (None, "5", True, [1]):
        with pytest.raises(TypeError, match=r"t\.in_range\(\) needs a number"):
            r.in_range(bad, bad)
    with pytest.raises(ValueError, match="too large"):
        r.in_range(0, 10**400)


@pytest.mark.parametrize(
    "call",
    [
        lambda k: k.equals(2.5),
        lambda k: k.not_equals(2.5),
        lambda k: k.is_in([2.5]),
        lambda k: k.not_in([2.5]),
        lambda k: k.gt("5"),
        lambda k: k.equals("5"),
    ],
)
def test_int_range_builders_reject_non_integers(call):
    with pytest.raises(TypeError):
        call(IntRange(0, 10, name="k"))


def test_valid_builders_still_work():
    r = Range(0.0, 1.0, name="t")
    assert r.gt(0.5).operator == ">"
    assert r.gte(0).operator == ">="  # int into a float range
    assert r.in_range(0.2, 0.4).value == (0.2, 0.4)
    assert r.is_in([0.0, 0.5]).value == (0.0, 0.5)
    k = IntRange(0, 10, name="k")
    assert k.equals(3).value == 3
    assert k.equals(3.0).value == 3.0  # integral float is fine
    assert k.gt(2.5).operator == ">"  # ordering ops accept any real
    assert k.is_in([1, 2]).value == (1, 2)
    assert LogRange(0.1, 10.0, name="l").lte(5).operator == "<="
    with pytest.raises(ValueError, match="outside the parameter domain"):
        r.gt(2)  # existing domain error preserved
    with pytest.raises(ValueError, match="does not overlap"):
        r.in_range(5, 6)


# ---------------------------------------------------------------- #3196
@pytest.mark.parametrize("bad", [None, "5", [1], True])
@pytest.mark.parametrize("cls,args", [(Range, (0, 10)), (LogRange, (1e-3, 10))])
def test_float_range_bounds_step_default_reject_bad_types(cls, args, bad):
    with pytest.raises(TypeError, match=r"bounds must be real numbers"):
        cls(bad, args[1])
    with pytest.raises(TypeError, match=r"bounds must be real numbers"):
        cls(args[0], bad)
    if bad is not None:  # None means "unset" for default
        with pytest.raises(TypeError, match=r"default must be a real number"):
            cls(*args, default=bad)


@pytest.mark.parametrize("bad", ["5", [1], True])
def test_range_step_rejects_bad_types(bad):
    with pytest.raises(TypeError, match=r"Range step must be a real number"):
        Range(0, 10, step=bad)


def test_bounds_errors_do_not_echo_values():
    with pytest.raises(TypeError) as ei:
        Range("secret-lo", 10)
    assert "secret-lo" not in str(ei.value)


def test_bool_bounds_rejected():
    with pytest.raises(TypeError):
        Range(False, True)
    with pytest.raises(TypeError, match="IntRange bounds must be integers"):
        IntRange(False, 10)


@pytest.mark.parametrize("bad", [2.5, 1e-300, True, "3", [1]])
def test_int_range_default_must_be_int(bad):
    with pytest.raises(TypeError, match="IntRange default must be an integer"):
        IntRange(0, 10, default=bad)


def test_step_larger_than_span_is_an_error():
    with pytest.raises(ValueError, match="larger than the span"):
        Range(0, 1, step=5)
    with pytest.raises(ValueError, match="larger than the span"):
        IntRange(0, 10, step=20)


@pytest.mark.parametrize("bad", [None, 5, 1.5, object()])
def test_choices_values_must_be_a_sequence(bad):
    with pytest.raises(TypeError, match="Choices values must be a list or tuple"):
        Choices(bad)
    with pytest.raises(TypeError, match="Choices values must be a list or tuple"):
        Choices({"a": 1})


def test_valid_constructors_still_work():
    assert Range(0, 10).to_tuple() == (0, 10)
    assert Range(0.0, 1.0, step=0.5, default=0.5).step == 0.5
    assert IntRange(0, 10, step=2, default=4).get_default() == 4
    assert LogRange(1e-5, 1e-1, default=1e-3).get_default() == 1e-3
    assert Choices(["a", "b"], default="a").get_default() == "a"
    assert Choices((1, 2)).to_list() == [1, 2]


# ---------------------------------------------------------------- review round 1
def test_int_range_equality_rejects_fraction_that_rounds_to_an_int():
    from fractions import Fraction

    k = IntRange(0, 10, name="k")
    with pytest.raises(TypeError, match="needs an integer"):
        k.equals(Fraction(20000000000000001, 10000000000000000))


def test_int_range_subclass_keeps_integer_checks():
    class Child(IntRange):
        pass

    with pytest.raises(TypeError, match="needs an integer"):
        Child(0, 10, name="k").equals(2.5)


def test_int_range_step_list_is_a_type_error_and_bool_keeps_value_error():
    with pytest.raises(TypeError, match="IntRange step must be an integer"):
        IntRange(0, 10, step=[1])
    with pytest.raises(ValueError, match="step must be a positive integer"):
        IntRange(0, 10, step=True)


def test_in_range_rejects_infinite_endpoints():
    with pytest.raises(ValueError, match="must be finite"):
        Range(0.0, 1.0, name="t").in_range(-math.inf, math.inf)


def test_int_range_accepts_numpy_integer_default():
    np = pytest.importorskip("numpy")
    assert IntRange(0, 10, default=np.int64(2)).get_default() == 2


def test_step_equal_to_span_with_float_error_is_accepted():
    assert Range(0.1, 0.3, step=0.2).step == 0.2


def test_in_range_reversed_fractions_that_round_equal_are_rejected():
    from fractions import Fraction as F

    with pytest.raises(ValueError, match="lower bound must be <= upper bound"):
        Range(0, 1, name="t").in_range(
            F(50000000000000001, 100000000000000000), F(1, 2)
        )
