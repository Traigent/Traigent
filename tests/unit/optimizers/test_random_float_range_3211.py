"""Regression tests for #3211: Range with integer-valued bounds is continuous."""

import pytest

from traigent.api.parameter_ranges import IntRange, Range
from traigent.integrations.observability.wandb import create_wandb_sweep_config
from traigent.optimizers.random import RandomSearchOptimizer


@pytest.mark.parametrize("high", [1, 2])
def test_float_range_with_integer_bounds_draws_floats(high):
    space = {"x": Range(0, high).to_config_value()}
    optimizer = RandomSearchOptimizer(space, ["accuracy"], max_trials=20, random_seed=1)

    assert optimizer.config_space_cardinality is None
    values = [optimizer.suggest_next_trial([])["x"] for _ in range(20)]
    assert all(isinstance(v, float) and 0 <= v <= high for v in values)
    assert any(v != int(v) for v in values)
    assert len(set(values)) > high + 1


def test_range_config_value_is_typed_float_dict():
    assert Range(0, 1).to_config_value() == {"type": "float", "low": 0, "high": 1}


def test_int_range_behaviour_unchanged():
    assert IntRange(1, 3).to_config_value()["step"] == 1


def test_wandb_exports_float_range_as_uniform_bounds():
    config = create_wandb_sweep_config(
        "f", {"x": Range(0, 1).to_config_value()}, ["accuracy"]
    )
    assert config["parameters"]["x"] == {
        "min": 0.0,
        "max": 1.0,
        "distribution": "uniform",
    }


def test_config_space_keeps_declared_float_type_with_integer_bounds():
    from traigent.api.config_space import ConfigSpace

    space = ConfigSpace.from_decorator_args(
        configuration_space={"x": Range(0, 1).to_config_value()}
    )
    assert isinstance(space.tvars["x"], Range)


def test_agent_spec_midpoint_for_typed_range_dict():
    from traigent.agents.specification_generator import SpecificationGenerator

    gen = SpecificationGenerator.__new__(SpecificationGenerator)
    params = gen._extract_model_parameters(
        {"configuration_space": {"temperature": Range(0, 1).to_config_value()}}
    )
    assert params["temperature"] == 0.5


def test_mock_remote_service_suggests_numbers_for_typed_ranges():
    import asyncio

    from traigent.optimizers.remote_services import MockRemoteService

    async def run():
        service = MockRemoteService()
        await service.connect()
        space = {
            "x": Range(0, 1).to_config_value(),
            "n": IntRange(1, 3).to_config_value(),
        }
        session = await service.create_session(space, ["accuracy"])
        return await service.suggest_configuration(session.session_id, [])

    config = asyncio.run(run())
    assert isinstance(config["x"], float) and 0 <= config["x"] <= 1
    assert config["n"] in (1, 2, 3)
