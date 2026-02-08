"""Tests for Optuna-based optimizers."""

from __future__ import annotations

import math

import optuna
import pytest

from traigent.optimizers.optuna_adapter import OptunaAdapter
from traigent.optimizers.optuna_coordinator import OptunaCoordinator
from traigent.optimizers.optuna_optimizer import OptunaGridOptimizer, OptunaTPEOptimizer
from traigent.optimizers.optuna_utils import (
    config_space_to_distributions,
    suggest_from_definition,
)
from traigent.utils.exceptions import OptimizationError

CONFIG_SPACE = {
    "model": ["gpt-4", "gpt-3.5"],
    "temperature": (0.0, 1.0),
    "max_tokens": (256, 1024),
}


def test_optuna_tpe_optimizer_suggest_and_tell():
    optimizer = OptunaTPEOptimizer(CONFIG_SPACE, ["accuracy", "latency"], max_trials=2)

    config = optimizer.suggest_next_trial([])
    trial_id = config["_optuna_trial_id"]

    # Intermediate reporting should not raise and returns a boolean flag
    should_prune = optimizer.report_intermediate_value(trial_id, step=0, value=0.1)
    assert should_prune in {True, False}

    optimizer.report_trial_result(trial_id, [0.8, 0.2])

    assert optimizer.best_config is not None
    assert optimizer.best_score is not None

    # Second suggestion should succeed; third should raise due to max_trials
    optimizer.suggest_next_trial([])
    with pytest.raises((RuntimeError, OptimizationError)):
        optimizer.suggest_next_trial([])


def test_optuna_coordinator_handles_lifecycle():
    coordinator = OptunaCoordinator(config_space=CONFIG_SPACE, objectives=["accuracy"])

    configs, _ = coordinator.ask_batch(2)
    assert len(configs) == 2

    first = configs[0]
    trial_id = first["_trial_id"]

    prune = coordinator.report_intermediate(trial_id, step=0, value=0.05)
    assert prune in {True, False}

    coordinator.tell_result(trial_id, 0.9)

    second = configs[1]
    coordinator.tell_failure(second["_trial_id"], "timeout")

    assert len(coordinator.study.trials) >= 2


def test_optuna_adapter_basic_flow():
    def quadratic(model: str, temperature: float, max_tokens: int) -> float:
        _ = model  # model choice does not impact the simple function
        return -((temperature - 0.5) ** 2) - abs(max_tokens - 512) / 1024.0

    result = OptunaAdapter.optimize(
        quadratic,
        CONFIG_SPACE,
        ["score"],
        algorithm="tpe",
        n_trials=10,
    )

    assert "best_params" in result
    assert "trials" in result
    assert result["n_trials"] == len(result["trials"])

    first_success = next(t for t in result["trials"] if t["values"] is not None)
    assert math.isfinite(first_success["values"][0])


def test_optuna_conditional_parameter_support():
    definition = {
        "type": "int",
        "low": 100,
        "high": 200,
        "conditions": {"model": "gpt-4"},
        "default": 128,
    }

    good_trial = optuna.trial.FixedTrial({"model": "gpt-4", "max_tokens": 156})
    good_config = {
        "model": good_trial.suggest_categorical("model", ["gpt-4", "gpt-3.5"])
    }
    assert (
        suggest_from_definition(good_trial, "max_tokens", definition, good_config)
        == 156
    )

    bad_trial = optuna.trial.FixedTrial({"model": "gpt-3.5"})
    bad_config = {"model": bad_trial.suggest_categorical("model", ["gpt-4", "gpt-3.5"])}
    assert (
        suggest_from_definition(bad_trial, "max_tokens", definition, bad_config) == 128
    )


def test_optuna_infers_float_range_dict_without_type_in_distributions():
    config_space = {
        "temperature": {"low": 0.0, "high": 1.0},
        "model": ["gpt-4", "gpt-3.5"],
    }
    distributions = config_space_to_distributions(config_space)

    temp_distribution = distributions["temperature"]
    assert isinstance(temp_distribution, optuna.distributions.FloatDistribution)
    assert temp_distribution.low == 0.0
    assert temp_distribution.high == 1.0


def test_suggest_from_definition_infers_float_range_dict_without_type():
    trial = optuna.trial.FixedTrial({"temperature": 0.42})
    definition = {"low": 0.0, "high": 1.0}

    value = suggest_from_definition(trial, "temperature", definition, current_config={})
    assert value == 0.42


def test_optuna_optimizer_accepts_hybrid_float_range_dict():
    sampler = optuna.samplers.RandomSampler(seed=42)
    config_space = {
        "model": ["gpt-4", "gpt-3.5"],
        "temperature": {"low": 0.0, "high": 1.0},
    }

    optimizer = OptunaTPEOptimizer(
        config_space,
        ["accuracy"],
        max_trials=1,
        sampler=sampler,
    )

    config = optimizer.suggest_next_trial([])
    assert config["model"] in {"gpt-4", "gpt-3.5"}
    assert 0.0 <= config["temperature"] <= 1.0


def test_optuna_grid_optimizer_accepts_hybrid_float_range_dict():
    config_space = {
        "model": ["gpt-4", "gpt-3.5"],
        "temperature": {"low": 0.0, "high": 1.0},
    }
    optimizer = OptunaGridOptimizer(
        config_space,
        ["accuracy"],
        max_trials=1,
        n_bins=3,
    )

    config = optimizer.suggest_next_trial([])
    assert config["model"] in {"gpt-4", "gpt-3.5"}
    assert 0.0 <= config["temperature"] <= 1.0


def test_optuna_optimizer_applies_conditional_defaults():
    sampler = optuna.samplers.RandomSampler(seed=42)
    config_space = {
        "model": ["gpt-4", "gpt-3.5"],
        "max_tokens": {
            "type": "int",
            "low": 100,
            "high": 300,
            "conditions": {"model": "gpt-4"},
            "default": 120,
        },
    }

    optimizer = OptunaTPEOptimizer(
        config_space,
        ["accuracy"],
        max_trials=2,
        sampler=sampler,
    )

    seen_defaults = False
    seen_conditionals = False
    for _ in range(2):
        config = optimizer.suggest_next_trial([])
        if config["model"] == "gpt-4":
            assert 100 <= config["max_tokens"] <= 300
            seen_conditionals = True
        else:
            assert config["max_tokens"] == 120
            seen_defaults = True

    assert seen_defaults
    assert seen_conditionals


def test_optuna_optimizer_filters_provider_specific_parameters():
    sampler = optuna.samplers.RandomSampler(seed=5)
    config_space = {
        "provider": ["openai", "anthropic"],
        "openai.temperature": (0.1, 1.0),
        "anthropic.temperature": (0.0, 0.5),
    }

    optimizer = OptunaTPEOptimizer(
        config_space,
        ["score"],
        max_trials=3,
        sampler=sampler,
    )

    for _ in range(3):
        cfg = optimizer.suggest_next_trial([])
        assert "provider" in cfg
        assert "temperature" in cfg
        assert not any(
            key.startswith("openai.") or key.startswith("anthropic.") for key in cfg
        )

        if cfg["provider"] == "openai":
            assert 0.1 <= cfg["temperature"] <= 1.0
        else:
            assert 0.0 <= cfg["temperature"] <= 0.5


def test_optuna_mock_mode_returns_deterministic_config():
    optimizer = OptunaTPEOptimizer(
        {"temperature": (0.0, 1.0), "model": ["alpha", "beta"]},
        ["score"],
        max_trials=2,
        mock_mode=True,
    )

    config1 = optimizer.suggest_next_trial([])
    config2 = optimizer.suggest_next_trial([])

    assert config1["model"] == config2["model"] == "alpha"
    assert config1["temperature"] == config2["temperature"] == 0.5
