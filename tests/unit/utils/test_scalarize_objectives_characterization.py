"""Characterization tests pinning scalarize_objectives behaviour."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from traigent.utils.multi_objective import scalarize_objectives


def _obj(name, orientation="maximize", weight=1.0, band=None):
    return SimpleNamespace(name=name, orientation=orientation, weight=weight, band=band)


def _schema(*objs):
    return SimpleNamespace(objectives=list(objs))


def _band(low, high):
    return SimpleNamespace(low=low, high=high)


def test_empty_objectives_returns_zero():
    assert scalarize_objectives({}, {"a": 1.0}) == 0.0


def test_weighted_average_explicit_minimize():
    got = scalarize_objectives(
        {"a": 0.5, "b": 0.25}, {"a": 2.0, "b": 1.0}, minimize_objectives=["b"]
    )
    assert got == pytest.approx((2.0 * 0.5 - 0.25) / 3.0)


def test_empty_minimize_list_means_all_maximize():
    got = scalarize_objectives({"cost": 1.0, "accuracy": 1.0}, {}, [])
    assert got == pytest.approx(1.0)


def test_none_minimize_resolves_defaults_for_known_names():
    got = scalarize_objectives({"cost": 1.0, "accuracy": 1.0}, {}, None)
    assert got == pytest.approx(0.0)


def test_none_minimize_unknown_name_raises():
    with pytest.raises(ValueError) as exc:
        scalarize_objectives({"zzz_unknown_metric": 1.0}, {}, None)
    assert "zzz_unknown_metric" in str(exc.value)


def test_none_values_skipped():
    got = scalarize_objectives({"a": 1.0, "b": None}, {"a": 1.0, "b": 5.0}, [])
    assert got == pytest.approx(1.0)


def test_all_none_returns_zero():
    assert scalarize_objectives({"a": None}, {"a": 1.0}, []) == 0.0


def test_zero_total_weight_uses_equal_weights_with_minimize():
    got = scalarize_objectives(
        {"a": 1.0, "b": 0.5, "c": None},
        {"a": 0.0, "b": 0.0, "c": 0.0},
        minimize_objectives=["b"],
    )
    assert got == pytest.approx((1.0 - 0.5) / 2)


def test_zero_total_weight_maximize_only():
    got = scalarize_objectives({"a": 1.0, "b": 0.5}, {"a": 0.0, "b": 0.0}, [])
    assert got == pytest.approx(0.75)


def test_missing_weight_defaults_to_one():
    assert scalarize_objectives({"a": 0.4, "b": 0.8}, {}, []) == pytest.approx(0.6)


def test_zero_weights_all_none_in_zero_branch():
    # weight lookups give 0 but values are None -> total_weight 0, no valid values
    assert scalarize_objectives({"a": None}, {"a": 0.0}, []) == 0.0


def test_schema_overrides_weights_and_minimize():
    schema = _schema(
        _obj("a", "maximize", 3.0), _obj("b", "minimize", 1.0), _obj("z", "maximize", 9)
    )
    got = scalarize_objectives(
        {"a": 1.0, "b": 0.5, "extra": 100.0},
        {"a": 100.0},
        minimize_objectives=["a"],
        objective_schema=schema,
    )
    # extra is not declared -> dropped; a maximize w3, b minimize w1
    assert got == pytest.approx((3.0 * 1.0 - 0.5) / 4.0)


def test_schema_with_no_matching_objectives_returns_zero():
    schema = _schema(_obj("a"))
    assert scalarize_objectives({"q": 1.0}, {}, objective_schema=schema) == 0.0


def test_schema_band_inside_is_zero():
    schema = _schema(_obj("lat", "band", 2.0, _band(1.0, 3.0)), _obj("a", weight=2.0))
    got = scalarize_objectives({"lat": 2.0, "a": 1.0}, {}, objective_schema=schema)
    assert got == pytest.approx((0.0 + 2.0 * 1.0) / 4.0)


def test_schema_band_outside_is_negative_distance():
    schema = _schema(_obj("lat", "band", 1.0, _band(1.0, 3.0)))
    below = scalarize_objectives({"lat": 0.5}, {}, objective_schema=schema)
    above = scalarize_objectives({"lat": 4.5}, {}, objective_schema=schema)
    assert below == pytest.approx(-0.5)
    assert above == pytest.approx(-1.5)


@pytest.mark.parametrize(
    "band", [None, _band(None, 3.0), _band(1.0, None)], ids=["none", "nolow", "nohigh"]
)
def test_schema_incomplete_band_raises(band):
    schema = _schema(_obj("lat", "band", 1.0, band))
    with pytest.raises(ValueError) as exc:
        scalarize_objectives({"lat": 2.0}, {}, objective_schema=schema)
    assert str(exc.value) == "Banded objective 'lat' requires a complete target band"


def test_schema_incomplete_band_not_checked_when_objective_absent():
    schema = _schema(_obj("lat", "band", 1.0, None), _obj("a"))
    assert scalarize_objectives(
        {"a": 0.7}, {}, objective_schema=schema
    ) == pytest.approx(0.7)


def test_caller_dicts_not_mutated():
    weights = {"a": 1.0}
    objs = {"a": 0.5}
    minimize = ["a"]
    scalarize_objectives(objs, weights, minimize)
    assert (weights, objs, minimize) == ({"a": 1.0}, {"a": 0.5}, ["a"])
