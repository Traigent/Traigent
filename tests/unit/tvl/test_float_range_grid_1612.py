"""#1612 axis 2: a float range that declares ``resolution`` is a discrete grid."""

import pytest

from traigent.tvl.models import parse_domain_spec


def _entry(lo, hi, **extra):
    domain = {"range": [lo, hi], **extra}
    return parse_domain_spec("x", "float", domain).to_configuration_space_entry()


def test_declared_resolution_is_a_grid():
    assert _entry(0.0, 1.0, resolution=0.25) == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_grid_values_are_not_float_noise():
    assert _entry(0.0, 0.3, resolution=0.1) == [0.0, 0.1, 0.2, 0.3]


def test_off_grid_upper_bound_is_never_exceeded():
    assert _entry(0.0, 1.0, resolution=0.3) == [0.0, 0.3, 0.6, 0.9]


def test_undeclared_resolution_stays_continuous():
    assert _entry(0.0, 1.0) == (0.0, 1.0)


def test_int_range_with_resolution_unchanged():
    entry = parse_domain_spec(
        "x", "int", {"range": [1, 5], "resolution": 1}
    ).to_configuration_space_entry()
    assert entry == (1, 5)


def test_huge_grid_rejected():
    with pytest.raises(ValueError, match="too many grid points"):
        _entry(0.0, 1.0, resolution=1e-6)


def test_tiny_resolution_keeps_first_value_and_bounds():
    """sol r9: a fixed 12-place rounding emitted 0.0 below the minimum."""
    values = _entry(4e-13, 2.4e-12, resolution=1e-12)
    assert values[0] == 4e-13
    assert all(4e-13 <= v <= 2.4e-12 for v in values)
    assert values == pytest.approx([4e-13, 1.4e-12, 2.4e-12], rel=1e-9)
    assert len(set(values)) == len(values)


def test_sub_picoscale_steps_stay_distinct_and_even():
    values = _entry(0.0, 5e-15, resolution=1e-15)
    assert len(values) == 6
    assert len(set(values)) == 6
    assert values[0] == 0.0
    assert values == pytest.approx([i * 1e-15 for i in range(6)], rel=1e-9)
    assert max(values) <= 5e-15


def test_step_finer_than_float_precision_fails_closed():
    with pytest.raises(ValueError, match="finer than"):
        _entry(1.0, 1.0 + 1e-15, resolution=1e-17)
