"""Typed range dict consumers after #3211 (Range / IntRange -> dict)."""

import asyncio
import random

import pytest

from traigent.api.parameter_ranges import IntRange, Range
from traigent.optimizers.random import RandomSearchOptimizer


def test_narrow_float_range_stays_within_bounds():
    space = {"x": Range(1e-8, 2e-8).to_config_value()}
    optimizer = RandomSearchOptimizer(space, ["accuracy"], max_trials=20, random_seed=1)
    for _ in range(20):
        x = optimizer.suggest_next_trial([])["x"]
        assert 1e-8 <= x <= 2e-8 and x != 0.0


def test_normal_width_float_range_draw_is_rounded_to_six_places():
    space = {"x": Range(0.0, 1.0).to_config_value()}
    optimizer = RandomSearchOptimizer(space, ["accuracy"], max_trials=5, random_seed=7)
    expected_rng = random.Random(7)
    got = optimizer.suggest_next_trial([])["x"]
    assert got == round(expected_rng.uniform(0.0, 1.0), 6)


def test_session_manager_samples_typed_ranges():
    from traigent.cloud.sessions import SessionManager

    class _S:
        configuration_space = {
            "x": Range(0, 1).to_config_value(),
            "n": IntRange(1, 3).to_config_value(),
        }

    config = SessionManager._suggest_configuration(
        SessionManager.__new__(SessionManager), _S(), []
    )
    assert isinstance(config["x"], float) and 0 <= config["x"] <= 1
    assert config["n"] in (1, 2, 3)


def test_mock_service_wide_int_range_does_not_build_a_list():
    from traigent.optimizers.remote_services import MockRemoteService

    async def run():
        service = MockRemoteService()
        await service.connect()
        session = await service.create_session(
            {"n": IntRange(0, 10**9).to_config_value()}, ["accuracy"]
        )
        return await service.suggest_configuration(session.session_id, [])

    config = asyncio.run(run())
    assert 0 <= config["n"] <= 10**9


def test_sample_stepped_int_wide_range_is_arithmetic():
    from traigent.utils.discrete_domains import sample_stepped_int

    class _Rng:
        def randrange(self, n):
            assert n == 10**9 + 1
            return n - 1

    assert sample_stepped_int(_Rng(), 0, 10**9, 1) == 10**9


@pytest.mark.parametrize(
    ("definition", "expected"),
    [
        (Range(0, 1, step=0.2).to_config_value(), {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}),
        (IntRange(0, 10, step=4).to_config_value(), {0, 4, 8, 10}),
    ],
)
def test_agent_spec_midpoint_snaps_to_step_grid(definition, expected):
    from traigent.agents.specification_generator import SpecificationGenerator

    gen = SpecificationGenerator.__new__(SpecificationGenerator)
    params = gen._extract_model_parameters(
        {"configuration_space": {"temperature": definition}}
    )
    assert params["temperature"] in expected


def test_spec_validator_recognises_typed_range_dict():
    from traigent.tvl.spec_validator import _is_numeric_config

    assert _is_numeric_config(Range(0, 1).to_config_value())
    assert _is_numeric_config(IntRange(1, 3).to_config_value())
    assert not _is_numeric_config({"choices": [1, 2]})


def test_meta_learning_classifies_typed_ranges_by_type():
    from traigent.analytics.meta_learning import (
        AlgorithmSelector,
        PerformancePredictor,
    )

    space = {
        "x": Range(0, 1).to_config_value(),
        "n": IntRange(1, 3).to_config_value(),
    }
    selector = AlgorithmSelector.__new__(AlgorithmSelector)
    chars = selector._extract_problem_characteristics("f", space, ["accuracy"])
    assert chars.param_types == {"x": "continuous", "n": "integer"}

    predictor = PerformancePredictor.__new__(PerformancePredictor)
    assert predictor._get_problem_characteristics(space).param_types == {
        "x": "continuous",
        "n": "integer",
    }
