"""Objective bounds: explicit zero is honoured; NaN/inf/bool bounds rejected."""

from __future__ import annotations

import pytest

from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema


def _schema(bounds=(1.0, 10.0), orientation="minimize") -> ObjectiveSchema:
    return ObjectiveSchema.from_objectives(
        [
            ObjectiveDefinition(
                name="cost", orientation=orientation, weight=1.0, bounds=bounds
            )
        ]
    )


def test_explicit_zero_min_val_not_replaced_by_schema_bound() -> None:
    s = _schema()
    assert s.normalize_value("cost", 5.0, 0.0, 10.0) == pytest.approx(0.5)
    assert s.normalize_value("cost", 5.0, min_val=0.0) == pytest.approx(0.5)


def test_explicit_zero_max_val_not_replaced_by_schema_bound() -> None:
    s = _schema(bounds=(-10.0, -1.0), orientation="maximize")
    # max_val=0.0 explicit: (-5 - -10) / (0 - -10) = 0.5
    assert s.normalize_value("cost", -5.0, max_val=0.0) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "bounds",
    [
        (float("nan"), 1.0),
        (0.0, float("nan")),
        (0.0, float("inf")),
        (float("-inf"), 1.0),
        (float("-inf"), float("inf")),
        (True, 2.0),
    ],
)
def test_non_finite_or_bool_bounds_rejected(bounds) -> None:
    with pytest.raises(ValueError, match="[Bb]ounds"):
        ObjectiveDefinition(
            name="acc", orientation="maximize", weight=1.0, bounds=bounds
        )


def test_from_dict_rejects_nan_bounds() -> None:
    with pytest.raises(ValueError, match="[Bb]ounds"):
        ObjectiveDefinition.from_dict(
            {
                "name": "acc",
                "orientation": "maximize",
                "weight": 1.0,
                "bounds": [float("nan"), 1.0],
            }
        )
