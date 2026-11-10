"""NaN, inf, bool and fractional values are rejected by numeric validators."""

from __future__ import annotations

import pytest

from traigent.api.types import StrategyConfig
from traigent.observability.config import ObservabilityConfig
from traigent.observability.dtos import (
    ObservationDTO,
    ObservationType,
    PromptReferenceDTO,
)
from traigent.utils.constraints import (
    DependencyConstraint,
    MutuallyExclusiveConstraint,
    ParameterRangeConstraint,
)

BAD_NUMBERS = [float("nan"), float("inf"), float("-inf"), True]


@pytest.mark.parametrize("field", ["latency_ms", "cost_usd"])
@pytest.mark.parametrize("bad", BAD_NUMBERS)
def test_observation_dto_rejects_non_finite_and_bool(field: str, bad: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        ObservationDTO("a", ObservationType.SPAN, "n", **{field: bad})


@pytest.mark.parametrize("bad", [*BAD_NUMBERS, 1.5, "3"])
def test_observation_dto_tokens_must_be_integers(bad: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        ObservationDTO("a", ObservationType.SPAN, "n", input_tokens=bad)


@pytest.mark.parametrize("bad", [*BAD_NUMBERS, 1.5, "2"])
def test_prompt_reference_version_must_be_integer(bad: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        PromptReferenceDTO("n", version=bad)


@pytest.mark.parametrize("name", ["batch_size", "max_queue_size", "max_batch_bytes"])
@pytest.mark.parametrize("bad", [0.5, 2.5, True, "5", float("nan"), float("inf")])
def test_observability_config_integer_sizes_are_strict(name: str, bad: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        ObservabilityConfig(offline_mode=True, backend_origin="", **{name: bad})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 1.5, True, "2"])
def test_strategy_config_parallel_workers_is_strict(bad: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        StrategyConfig("grid", parallel_workers=bad).validate()


def test_valid_values_still_pass() -> None:
    ObservationDTO("a", ObservationType.SPAN, "n", latency_ms=12.5, input_tokens=3)
    PromptReferenceDTO("n", version=2)
    StrategyConfig("grid", parallel_workers=2).validate()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, False])
def test_range_constraint_rejects_non_finite_and_bool(bad: object) -> None:
    constraint = ParameterRangeConstraint("temperature", 0.0, 1.0)
    assert constraint.validate({"temperature": bad}) is False
    assert constraint.validate({"temperature": 0.5}) is True


def test_membership_constraints_do_not_raise_on_unhashable_values() -> None:
    exclusive = MutuallyExclusiveConstraint(["a"], ["x"])
    assert exclusive.validate({"a": ["x"]}) is True
    dependency = DependencyConstraint("a", "b", ["x"])
    assert dependency.validate({"a": 1, "b": ["x"]}) is False
    assert dependency.validate({"a": 1, "b": {"k": 1}}) is False
    assert dependency.validate({"a": 1, "b": "x"}) is True


@pytest.mark.parametrize(
    "name", ["max_buffer_age", "flush_timeout", "request_timeout", "per_trace_timeout"]
)
def test_observability_config_float_fields_reject_bool(name: str) -> None:
    with pytest.raises(TypeError):
        ObservabilityConfig(offline_mode=True, backend_origin="", **{name: True})
