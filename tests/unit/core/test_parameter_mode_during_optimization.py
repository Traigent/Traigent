"""Tests for parameter mode injection during optimization trials.

This tests the fix for the issue where parameter injection mode wasn't working
during optimization. The root cause was:
1. _traigent_injection_mode attribute was never set on functions
2. _prepare_call_arguments() in base.py spread config values as individual kwargs
   instead of passing them as a single dict to the config parameter

The fix:
1. _setup_function_wrapper() now sets _traigent_injection_mode and _traigent_config_param
   attributes on the function
2. _prepare_call_arguments() now checks injection mode and passes config as a single
   dict when mode is "parameter"
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from unittest.mock import patch

import pytest

import traigent
from traigent.core.optimized_function import OptimizedFunction
from traigent.evaluators.base import Dataset, EvaluationExample


@pytest.fixture
def simple_dataset() -> Dataset:
    """Create a simple dataset for testing."""
    return Dataset(
        examples=[
            EvaluationExample(input_data={"prompt": "test 1"}, expected_output="result1"),
            EvaluationExample(input_data={"prompt": "test 2"}, expected_output="result2"),
        ],
        name="test_dataset",
    )


class TestParameterModeInjectionDuringOptimization:
    """Test that parameter injection mode works during optimization trials."""

    def test_injection_mode_attribute_set_on_function(self) -> None:
        """Verify _traigent_injection_mode is set on function after decoration."""

        @traigent.optimize(
            configuration_space={"model": ["gpt-3.5", "gpt-4"]},
            injection_mode="parameter",
            config_param="config",
        )
        def my_function(prompt: str, config: dict) -> str:
            return config.get("model", "default")

        opt_fn: OptimizedFunction = my_function  # type: ignore[assignment]
        original_func = opt_fn.func

        # The injection mode attribute should be set
        assert hasattr(original_func, "_traigent_injection_mode")
        assert original_func._traigent_injection_mode == "parameter"  # type: ignore[attr-defined]

    def test_config_param_attribute_set_on_function(self) -> None:
        """Verify _traigent_config_param is set on function after decoration."""

        @traigent.optimize(
            configuration_space={"model": ["gpt-3.5", "gpt-4"]},
            injection_mode="parameter",
            config_param="cfg",
        )
        def my_function(prompt: str, cfg: dict) -> str:
            return cfg.get("model", "default")

        opt_fn: OptimizedFunction = my_function  # type: ignore[assignment]
        original_func = opt_fn.func

        # The config param attribute should be set
        assert hasattr(original_func, "_traigent_config_param")
        assert original_func._traigent_config_param == "cfg"  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_parameter_mode_receives_config_dict_during_optimization(
        self, simple_dataset: Dataset
    ) -> None:
        """Function receives config as dict during optimization trials."""
        received_configs: list[dict[str, Any]] = []

        @traigent.optimize(
            configuration_space={"model": ["gpt-3.5", "gpt-4"], "temperature": [0.1, 0.5]},
            injection_mode="parameter",
            config_param="config",
            objectives=["accuracy"],
        )
        def my_function(prompt: str, config: dict) -> str:
            # Capture what config we received
            received_configs.append(dict(config) if isinstance(config, Mapping) else {"raw": config})
            model = config.get("model", "unset") if isinstance(config, Mapping) else "not-a-dict"
            return f"model={model}"

        opt_fn: OptimizedFunction = my_function  # type: ignore[assignment]
        opt_fn.eval_dataset = simple_dataset

        # Run optimization with mock mode
        with patch.dict("os.environ", {"TRAIGENT_MOCK_LLM": "true", "TRAIGENT_OFFLINE_MODE": "true"}):
            await opt_fn.optimize(max_trials=3)

        # Verify we received configs during optimization
        assert len(received_configs) > 0, "Should have received configs during optimization"

        # Each config should be a dict with our configuration values
        for config in received_configs:
            assert isinstance(config, dict), f"Config should be dict, got {type(config)}"
            # Should contain actual model values, not "unset"
            if "model" in config:
                assert config["model"] in ["gpt-3.5", "gpt-4"], f"Unexpected model: {config['model']}"

    @pytest.mark.asyncio
    async def test_parameter_mode_config_accessible_via_get(
        self, simple_dataset: Dataset
    ) -> None:
        """Config should be accessible via .get() method during optimization."""
        get_results: list[str] = []

        @traigent.optimize(
            configuration_space={"model": ["gpt-3.5", "gpt-4"]},
            injection_mode="parameter",
            config_param="config",
            objectives=["accuracy"],
        )
        def my_function(prompt: str, config: dict) -> str:
            # Use .get() to access config value
            model = config.get("model", "fallback")
            get_results.append(model)
            return model

        opt_fn: OptimizedFunction = my_function  # type: ignore[assignment]
        opt_fn.eval_dataset = simple_dataset

        with patch.dict("os.environ", {"TRAIGENT_MOCK_LLM": "true", "TRAIGENT_OFFLINE_MODE": "true"}):
            await opt_fn.optimize(max_trials=2)

        # Should have actual model values, not "fallback"
        assert len(get_results) > 0
        for result in get_results:
            assert result != "fallback", "config.get() should return actual values, not fallback"
            assert result in ["gpt-3.5", "gpt-4"]

    @pytest.mark.asyncio
    async def test_parameter_mode_config_accessible_via_bracket_notation(
        self, simple_dataset: Dataset
    ) -> None:
        """Config should be accessible via bracket notation during optimization."""
        bracket_results: list[str] = []

        @traigent.optimize(
            configuration_space={"model": ["gpt-3.5", "gpt-4"]},
            injection_mode="parameter",
            config_param="config",
            objectives=["accuracy"],
        )
        def my_function(prompt: str, config: dict) -> str:
            # Use bracket notation to access config value
            model = config["model"]
            bracket_results.append(model)
            return model

        opt_fn: OptimizedFunction = my_function  # type: ignore[assignment]
        opt_fn.eval_dataset = simple_dataset

        with patch.dict("os.environ", {"TRAIGENT_MOCK_LLM": "true", "TRAIGENT_OFFLINE_MODE": "true"}):
            await opt_fn.optimize(max_trials=2)

        # Should have actual model values
        assert len(bracket_results) > 0
        for result in bracket_results:
            assert result in ["gpt-3.5", "gpt-4"]

    @pytest.mark.asyncio
    async def test_parameter_mode_with_custom_config_param_name(
        self, simple_dataset: Dataset
    ) -> None:
        """Custom config_param name should work during optimization."""
        received_configs: list[dict[str, Any]] = []

        @traigent.optimize(
            configuration_space={"model": ["gpt-3.5", "gpt-4"]},
            injection_mode="parameter",
            config_param="traigent_config",  # Custom name
            objectives=["accuracy"],
        )
        def my_function(prompt: str, traigent_config: dict) -> str:
            received_configs.append(dict(traigent_config))
            return traigent_config.get("model", "unset")

        opt_fn: OptimizedFunction = my_function  # type: ignore[assignment]
        opt_fn.eval_dataset = simple_dataset

        with patch.dict("os.environ", {"TRAIGENT_MOCK_LLM": "true", "TRAIGENT_OFFLINE_MODE": "true"}):
            await opt_fn.optimize(max_trials=2)

        assert len(received_configs) > 0
        for config in received_configs:
            assert "model" in config
            assert config["model"] in ["gpt-3.5", "gpt-4"]


class TestParameterModeVsOtherModes:
    """Compare parameter mode behavior with other injection modes."""

    @pytest.mark.asyncio
    async def test_seamless_mode_injects_as_kwargs(self, simple_dataset: Dataset) -> None:
        """Seamless mode should inject config values as individual kwargs."""
        received_values: list[tuple[str, float]] = []

        @traigent.optimize(
            configuration_space={"model": ["gpt-3.5", "gpt-4"], "temperature": [0.1, 0.5]},
            injection_mode="seamless",
            objectives=["accuracy"],
        )
        def my_function(prompt: str, model: str = "default", temperature: float = 0.0) -> str:
            received_values.append((model, temperature))
            return f"{model}:{temperature}"

        opt_fn: OptimizedFunction = my_function  # type: ignore[assignment]
        opt_fn.eval_dataset = simple_dataset

        with patch.dict("os.environ", {"TRAIGENT_MOCK_LLM": "true", "TRAIGENT_OFFLINE_MODE": "true"}):
            await opt_fn.optimize(max_trials=2)

        # Seamless mode: values should be passed as individual kwargs
        assert len(received_values) > 0
        for model, temp in received_values:
            assert model in ["gpt-3.5", "gpt-4", "default"]
            assert temp in [0.1, 0.5, 0.0]


@pytest.mark.parametrize("injection_mode", ["parameter", "seamless", "context"])
def test_injection_mode_attribute_matches_configured_mode(injection_mode: str) -> None:
    """All injection modes should set the correct attribute on the function."""

    if injection_mode == "parameter":
        @traigent.optimize(
            configuration_space={"model": ["gpt-3.5"]},
            injection_mode=injection_mode,
            config_param="config",
        )
        def fn(prompt: str, config: dict) -> str:
            return prompt
    else:
        @traigent.optimize(
            configuration_space={"model": ["gpt-3.5"]},
            injection_mode=injection_mode,
        )
        def fn(prompt: str, model: str = "default") -> str:
            return prompt

    opt_fn: OptimizedFunction = fn  # type: ignore[assignment]
    original_func = opt_fn.func

    assert hasattr(original_func, "_traigent_injection_mode")
    assert original_func._traigent_injection_mode == injection_mode  # type: ignore[attr-defined]
