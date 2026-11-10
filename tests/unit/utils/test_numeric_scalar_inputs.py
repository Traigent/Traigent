"""Numpy scalars are accepted wherever plain numbers are; bool/NaN/inf stay rejected.

Regression tests for the review of the numeric-input cluster (#2894, #2917,
#3135): the validators used ``isinstance(value, int | float)``, which rejected
``numpy.int64`` and ``numpy.float32`` values that the rest of the SDK accepts.
"""

from __future__ import annotations

import numpy as np
import pytest

from traigent.core.objectives import ObjectiveDefinition
from traigent.tvl.statistics import bonferroni_adjust
from traigent.utils.validation import validate_max_trials_value, validate_timeout_value


@pytest.mark.parametrize("value", [np.int64(5), np.int32(1)])
def test_max_trials_accepts_numpy_integers(value: object) -> None:
    validate_max_trials_value(value)


def test_max_trials_still_rejects_bool_and_float() -> None:
    with pytest.raises(TypeError):
        validate_max_trials_value(True)
    with pytest.raises(TypeError):
        validate_max_trials_value(2.5)


def test_max_trials_numpy_zero_is_rejected_as_non_positive() -> None:
    with pytest.raises(ValueError, match="max_trials"):
        validate_max_trials_value(np.int64(0))


@pytest.mark.parametrize("value", [np.int64(3), np.float32(1.5), np.float64(0.0)])
def test_timeout_accepts_numpy_real_numbers(value: object) -> None:
    validate_timeout_value(value)


@pytest.mark.parametrize("value", [np.float64("nan"), np.float32("inf"), True])
def test_timeout_still_rejects_nan_inf_and_bool(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        validate_timeout_value(value)


def test_objective_bounds_accept_numpy_scalars() -> None:
    objective = ObjectiveDefinition(
        name="accuracy", orientation="maximize", weight=1.0, bounds=(np.float32(0.0), np.int64(1))
    )
    assert objective.bounds is not None


@pytest.mark.parametrize("bad", [np.float32("nan"), np.float64("inf"), True])
def test_objective_bounds_still_reject_non_finite_and_bool(bad: object) -> None:
    with pytest.raises(ValueError, match="[Bb]ounds"):
        ObjectiveDefinition(
            name="accuracy", orientation="maximize", weight=1.0, bounds=(bad, 1.0)
        )


def test_bonferroni_accepts_numpy_float32_p_values() -> None:
    adjusted = bonferroni_adjust([np.float32(0.01), 0.02])
    assert adjusted == pytest.approx([0.02, 0.04], rel=1e-6)


def test_bonferroni_still_rejects_bool_p_value() -> None:
    with pytest.raises(ValueError, match="Invalid p-value"):
        bonferroni_adjust([True, 0.02])
