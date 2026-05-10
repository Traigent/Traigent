"""Coverage for security-hardened random sampling paths."""

import importlib
from datetime import UTC, datetime

import pytest

from traigent.cloud import dataset_converter as dataset_converter_module
from traigent.cloud import sessions as sessions_module
from traigent.cloud.models import OptimizationSession, OptimizationSessionStatus
from traigent.core import types as types_module
from traigent.evaluators.base import Dataset, EvaluationExample

retry_module = importlib.import_module("traigent.utils.retry")


class DeterministicRandom:
    """Small deterministic stand-in for SystemRandom in unit tests."""

    def choice(self, values):
        return list(values)[0]

    def randint(self, lower, upper):
        return lower

    def uniform(self, lower, upper):
        return (lower + upper) / 2

    def random(self):
        return 0.5

    def sample(self, population, size):
        return list(population)[-size:]


def test_dataset_converter_random_sampling_uses_secure_random(monkeypatch):
    monkeypatch.setattr(
        dataset_converter_module, "_SECURE_RANDOM", DeterministicRandom()
    )
    dataset = Dataset(
        examples=[
            EvaluationExample(input_data={"index": index}, expected_output=str(index))
            for index in range(5)
        ],
        name="secure-random-sample",
    )

    indices = dataset_converter_module.converter.create_dataset_subset_indices(
        dataset, subset_size=3, strategy="random_sampling"
    )

    assert indices == [2, 3, 4]


@pytest.mark.asyncio
async def test_session_manager_secure_random_suggestions(monkeypatch):
    monkeypatch.setattr(sessions_module, "_SECURE_RANDOM", DeterministicRandom())
    manager = sessions_module.SessionManager(
        storage=sessions_module.InMemorySessionStorage()
    )
    session = OptimizationSession(
        session_id="secure-random-session",
        function_name="optimize",
        configuration_space={
            "model": ["gpt-4o", "gpt-4o-mini"],
            "batch_size": (1, 8),
            "temperature": (0.0, 1.0),
            "fixed": "value",
        },
        objectives=["accuracy"],
        max_trials=20,
        status=OptimizationSessionStatus.ACTIVE,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        optimization_strategy={"min_examples_per_trial": 3},
    )

    config = await manager._suggest_configuration(session, [])
    subset = await manager._suggest_subset_indices(session, [], dataset_size=5)

    assert config == {
        "model": "gpt-4o",
        "batch_size": 1,
        "temperature": 0.5,
        "fixed": "value",
    }
    assert subset.indices == [2, 3, 4]


def test_configuration_space_sample_config_uses_secure_random(monkeypatch):
    monkeypatch.setattr(types_module, "_SECURE_RANDOM", DeterministicRandom())
    space = types_module.ConfigurationSpace(
        parameters=[
            types_module.Parameter(
                "temperature", types_module.ParameterType.FLOAT, (0.0, 1.0)
            ),
            types_module.Parameter(
                "batch_size", types_module.ParameterType.INTEGER, (1, 8)
            ),
            types_module.Parameter(
                "model",
                types_module.ParameterType.CATEGORICAL,
                ["gpt-4o", "gpt-4o-mini"],
            ),
            types_module.Parameter("stream", types_module.ParameterType.BOOLEAN, []),
        ]
    )

    assert space.sample_config() == {
        "temperature": 0.5,
        "batch_size": 1,
        "model": "gpt-4o",
        "stream": True,
    }


def test_retry_delay_jitter_uses_secure_random(monkeypatch):
    monkeypatch.setattr(retry_module, "_SECURE_RANDOM", DeterministicRandom())

    fixed_config = retry_module.RetryConfig(
        initial_delay=10.0,
        jitter=True,
        strategy=retry_module.RetryStrategy.FIXED,
    )
    jitter_config = retry_module.RetryConfig(
        initial_delay=10.0,
        jitter=False,
        strategy=retry_module.RetryStrategy.JITTER,
    )

    assert fixed_config.calculate_delay(attempt=1) == 10.5
    assert jitter_config.calculate_delay(attempt=1) == 5.0
