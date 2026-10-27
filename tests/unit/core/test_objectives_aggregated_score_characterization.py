"""Characterization tests for ObjectiveSchema.compute_aggregated_score."""

from __future__ import annotations

import math

import pytest

from traigent.core.objectives import (
    AggregationMode,
    ObjectiveDefinition,
    ObjectiveSchema,
)


def _schema(*defs: ObjectiveDefinition) -> ObjectiveSchema:
    return ObjectiveSchema.from_objectives(list(defs))


@pytest.fixture
def schema() -> ObjectiveSchema:
    return _schema(
        ObjectiveDefinition(name="accuracy", orientation="maximize", weight=3.0),
        ObjectiveDefinition(name="cost", orientation="minimize", weight=1.0),
    )


@pytest.mark.parametrize("mode", list(AggregationMode))
def test_empty_metrics_returns_none(schema, mode):
    assert schema.compute_aggregated_score({}, mode) is None
    assert schema.compute_aggregated_score(None, mode) is None


def test_weighted_sum_default(schema):
    got = schema.compute_aggregated_score({"accuracy": 0.8, "cost": 1.0})
    assert got == pytest.approx(0.75 * 0.8 + 0.25 * 0.5)


def test_reference_point_for_minimize(schema):
    got = schema.compute_aggregated_score(
        {"accuracy": 0.8, "cost": 1.0}, reference_point={"cost": 1.0}
    )
    assert got == pytest.approx(0.75 * 0.8 + 0.25 * 0.5)
    got = schema.compute_aggregated_score(
        {"accuracy": 0.8, "cost": 1.0}, reference_point={"cost": 3.0}
    )
    assert got == pytest.approx(0.75 * 0.8 + 0.25 * 0.75)


def test_missing_none_nonfinite_and_zero_weight_values_are_skipped(schema):
    assert schema.compute_aggregated_score({"accuracy": 0.8}) == pytest.approx(0.6)
    assert schema.compute_aggregated_score(
        {"accuracy": None, "cost": 1.0}
    ) == pytest.approx(0.125)
    assert (
        schema.compute_aggregated_score({"accuracy": math.nan, "cost": math.inf})
        is None
    )
    assert schema.compute_aggregated_score({"other": 1.0}) is None


def test_non_numeric_value_raises_type_error(schema):
    with pytest.raises(TypeError):
        schema.compute_aggregated_score({"accuracy": "high"})


def test_negative_maximize_value_clamped(schema):
    assert schema.compute_aggregated_score({"accuracy": -2.0}) == pytest.approx(0.0)


def test_harmonic_and_chebyshev(schema):
    metrics = {"accuracy": 0.8, "cost": 1.0}
    harmonic = schema.compute_aggregated_score(metrics, AggregationMode.HARMONIC)
    assert harmonic == pytest.approx(1.0 / (0.75 / 0.8 + 0.25 / 0.5))
    cheb = schema.compute_aggregated_score(metrics, AggregationMode.CHEBYSHEV)
    assert cheb == pytest.approx(-0.15)


def test_unknown_mode_raises_only_when_values_present(schema):
    assert schema.compute_aggregated_score({}, "bogus") is None
    assert schema.compute_aggregated_score({"other": 1.0}, "bogus") is None
    with pytest.raises(ValueError, match="Unknown aggregation mode: bogus"):
        schema.compute_aggregated_score({"accuracy": 0.5}, "bogus")


def test_ranges_mode_uses_normalize_metrics(schema):
    ranges = {"accuracy": (0.0, 1.0), "cost": (0.0, 4.0)}
    metrics = {"accuracy": 0.5, "cost": 1.0}
    normalized = schema.normalize_metrics(metrics, ranges)
    expected = 0.75 * normalized["accuracy"] + 0.25 * normalized["cost"]
    assert schema.compute_aggregated_score(metrics, ranges=ranges) == pytest.approx(
        expected
    )


def test_ranges_mode_skips_objectives_absent_from_normalized_map(schema):
    ranges = {"accuracy": (0.0, 1.0), "cost": (0.0, 4.0)}
    got = schema.compute_aggregated_score({"accuracy": 0.5}, ranges=ranges)
    assert got == pytest.approx(
        0.75 * schema.normalize_metrics({"accuracy": 0.5}, ranges)["accuracy"]
    )


def test_ranges_mode_empty_dict_still_uses_ranges_branch(schema):
    # ranges={} is not None: the range-based path is taken, nothing normalizes.
    metrics = {"accuracy": 0.5}
    expected = schema.normalize_metrics(metrics, {})
    got = schema.compute_aggregated_score(metrics, ranges={})
    if expected.get("accuracy") is None:
        assert got is None
    else:
        assert got == pytest.approx(0.75 * expected["accuracy"])


def test_ranges_mode_ignores_nonfinite_filter_and_reference_point(schema):
    ranges = {"accuracy": (0.0, 1.0), "cost": (0.0, 4.0)}
    a = schema.compute_aggregated_score(
        {"accuracy": 0.5, "cost": 1.0}, ranges=ranges, reference_point={"cost": 9.0}
    )
    b = schema.compute_aggregated_score({"accuracy": 0.5, "cost": 1.0}, ranges=ranges)
    assert a == pytest.approx(b)


def test_zero_weight_objective_skipped():
    sch = _schema(
        ObjectiveDefinition(name="a", orientation="maximize", weight=1.0),
        ObjectiveDefinition(name="b", orientation="maximize", weight=1.0),
    )
    sch.weights_normalized["b"] = 0.0
    assert sch.compute_aggregated_score({"a": 0.4, "b": 0.9}) == pytest.approx(
        sch.weights_normalized["a"] * 0.4
    )
    assert sch.compute_aggregated_score({"b": 0.9}) is None
