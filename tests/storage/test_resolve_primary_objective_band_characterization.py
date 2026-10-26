"""Characterization tests pinning LocalStorageManager._resolve_primary_objective_band."""

from __future__ import annotations

import pytest

from traigent.storage.local_storage import LocalStorageManager

resolve = LocalStorageManager._resolve_primary_objective_band


def _cfg(*definitions):
    return {"objective_schema": {"objectives": list(definitions)}}


def _obj(name="acc", target=(0.2, 0.8)):
    return {"name": name, "band": {"target": list(target)}}


def test_returns_float_band():
    out = resolve(_cfg(_obj(target=(1, 3))), "acc")
    assert out == (1.0, 3.0)
    assert all(isinstance(v, float) for v in out)


def test_picks_matching_name_among_definitions():
    cfg = _cfg(_obj("a", (0, 1)), _obj("b", (5, 6)))
    assert resolve(cfg, "b") == (5.0, 6.0)


def test_equal_bounds_allowed():
    assert resolve(_cfg(_obj(target=(2, 2))), "acc") == (2.0, 2.0)


def test_duplicate_name_skips_invalid_first_and_uses_later_valid():
    cfg = _cfg(_obj("acc", (9, 1)), _obj("acc", (1, 2)))
    assert resolve(cfg, "acc") == (1.0, 2.0)


@pytest.mark.parametrize(
    "target",
    [
        [1],
        [1, 2, 3],
        (1, 2),  # tuple, not list
        "ab",
        None,
        [True, 2],
        [1, False],
        [float("nan"), 2],
        [1, float("inf")],
        [3, 1],
        ["1", "2"],
    ],
)
def test_invalid_target_raises(target):
    cfg = _cfg({"name": "acc", "band": {"target": target}})
    with pytest.raises(ValueError, match="complete target band"):
        resolve(cfg, "acc")


@pytest.mark.parametrize(
    "cfg",
    [
        None,
        {},
        [],
        {"objective_schema": None},
        {"objective_schema": {"objectives": "x"}},
        {"objective_schema": {"objectives": [None, 3, {"name": "other"}]}},
        _cfg({"name": "acc"}),
        _cfg({"name": "acc", "band": "x"}),
    ],
)
def test_missing_or_malformed_raises(cfg):
    with pytest.raises(ValueError, match="complete target band"):
        resolve(cfg, "acc")


def test_huge_int_low_with_non_numeric_high_raises_value_error_not_overflow():
    # The non-numeric upper bound must be rejected before any finiteness
    # check runs on the (unconvertible) huge lower bound.
    cfg = _cfg({"name": "acc", "band": {"target": [10**400, None]}})
    with pytest.raises(ValueError, match="complete target band"):
        resolve(cfg, "acc")


def test_huge_int_low_with_non_numeric_high_falls_through_to_later_definition():
    cfg = _cfg(
        {"name": "acc", "band": {"target": [10**400, None]}},
        _obj("acc", (1, 2)),
    )
    assert resolve(cfg, "acc") == (1.0, 2.0)
