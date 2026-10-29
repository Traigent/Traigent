"""#2822: configuration-space constructors reject non-finite and non-integral
inputs at construction (matching traigent-JS), and the stepped-float grid keeps
every point for tiny steps."""

from __future__ import annotations

import math

import pytest

from traigent.api.parameter_ranges import Choices, IntRange, LogRange, Range
from traigent.optimizers.random import RandomSearchOptimizer
from traigent.utils.discrete_domains import (
    stepped_float_cardinality,
    stepped_float_values,
)
from traigent.utils.validation import Validators

NAN = float("nan")
INF = float("inf")


@pytest.mark.parametrize(
    ("build", "field_name"),
    [
        (lambda: Range(NAN, 1), "low"),
        (lambda: Range(0, NAN), "high"),
        (lambda: Range(0, INF), "high"),
        (lambda: Range(-INF, 1), "low"),
        (lambda: Range(0, 1, step=INF), "step"),
        (lambda: Range(0, 1, step=NAN), "step"),
        (lambda: Range(0, 1, default=NAN), "default"),
        (lambda: LogRange(1e-5, INF), "high"),
        (lambda: LogRange(NAN, 1.0), "low"),
        (lambda: LogRange(1e-5, 1.0, default=NAN), "default"),
        (lambda: Choices([NAN, 0.2]), "value"),
        (lambda: Choices([INF, 1.0]), "value"),
        (lambda: Choices([NAN, 1.0], default=NAN), "value"),
    ],
)
def test_non_finite_construction_raises_naming_the_field(build, field_name) -> None:
    with pytest.raises(ValueError, match=f"{field_name} must be finite"):
        build()


def test_finite_values_are_unaffected() -> None:
    Range(0.0, 1.0, step=0.25, default=0.5)
    LogRange(1e-5, 1.0, default=1e-3)
    Choices([0.0, 0.5, "x", None, True], enforce_type=False)


@pytest.mark.parametrize("step", [2.5, 0.5, True, NAN, INF])
def test_int_range_rejects_non_integer_step(step) -> None:
    with pytest.raises(ValueError, match="step must be a positive integer"):
        IntRange(1, 8, step=step)


def test_int_range_accepts_integer_steps() -> None:
    IntRange(1, 8, step=2)
    IntRange(1, 8, step=2.0)


def test_infinite_range_never_yields_a_space_for_an_optimizer() -> None:
    with pytest.raises(ValueError):
        RandomSearchOptimizer({"t": Range(0.0, INF).to_config_value()}, ["accuracy"])


def test_raw_tuple_and_dict_forms_are_validated_as_non_finite() -> None:
    result = Validators.validate_configuration_space({"t": (0.0, INF)})
    assert not result.is_valid
    assert "finite" in result.errors[0].message
    typed = Validators.validate_configuration_space(
        {"t": {"type": "float", "low": 0.0, "high": NAN}}
    )
    assert not typed.is_valid
    choice = Validators.validate_configuration_space({"t": [0.1, NAN]})
    assert not choice.is_valid
    step = Validators.validate_configuration_space(
        {"t": {"type": "float", "low": 0.0, "high": 1.0, "step": INF}}
    )
    assert not step.is_valid


def test_tiny_step_grid_keeps_every_point_inside_the_range() -> None:
    values = stepped_float_values(1e-7, 1e-6, 1e-7)
    assert len(values) == 10
    assert all(1e-7 <= v <= 1e-6 for v in values)
    assert stepped_float_cardinality(1e-7, 1e-6, 1e-7) == len(values)
    assert len(set(values)) == len(values)


@pytest.mark.parametrize(
    ("low", "high", "step"),
    [(0.0, 1.0, 0.25), (0.0, 1.0, 0.1), (0.5, 2.0, 0.5), (0.0, 0.002, 0.0005)],
)
def test_ordinary_grids_match_cardinality(low, high, step) -> None:
    values = stepped_float_values(low, high, step)
    assert stepped_float_cardinality(low, high, step) == len(values)
    assert values[0] == low and math.isclose(values[-1], high)


def test_non_finite_grid_inputs_yield_an_empty_grid_not_a_hang() -> None:
    assert stepped_float_values(0.0, INF, 0.1) == ()
    assert stepped_float_values(0.0, 1.0, INF) == ()
    assert stepped_float_values(NAN, 1.0, 0.1) == ()


@pytest.mark.parametrize(
    "space",
    [
        {"t": (0.0, INF)},
        {"t": {"low": 0.0, "high": INF}},
        {"t": {"low": NAN, "high": 1.0}},
        {"t": {"low": 0.0, "high": 1.0, "step": INF}},
        {"t": [NAN, 0.2]},
        {"t": {"values": [0.1, INF]}},
    ],
)
def test_optimizer_rejects_raw_non_finite_space(space) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        RandomSearchOptimizer(space, ["accuracy"])


def test_optimizer_accepts_finite_raw_spaces() -> None:
    RandomSearchOptimizer(
        {"t": (0.0, 1.0), "m": ["a", "b"], "n": {"low": 1, "high": 4}}, ["accuracy"]
    )


def test_numpy_float_nan_is_rejected() -> None:
    import numpy as np
    import pytest

    from traigent.api.parameter_ranges import Range

    with pytest.raises(ValueError, match="finite"):
        Range(np.float32("nan"), 1.0)
