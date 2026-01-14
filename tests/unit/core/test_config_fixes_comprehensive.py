"""Comprehensive validation tests for config behavior fixes.

This file validates both fixes together:
1. Parameter mode injection during optimization
2. Second optimization using previous best config as baseline

Run with:
    TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true pytest tests/unit/core/test_config_fixes_comprehensive.py -v

These tests demonstrate:
- Functions receive config dict during optimization trials
- Config values are accessible via .get() and [] notation
- Second optimization starts with previous best config
- Continuous optimization builds on previous results
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

import traigent
from traigent.core.optimized_function import OptimizedFunction
from traigent.evaluators.base import Dataset, EvaluationExample

# Internal fields added by TraigentConfig that aren't part of the config space
_INTERNAL_CONFIG_FIELDS = {
    "execution_mode",
    "local_storage_path",
    "minimal_logging",
    "auto_sync",
    "privacy_enabled",
    "strict_metrics_nulls",
    "enable_usage_analytics",
    "analytics_endpoint",
    "anonymous_user_id",
}


def _extract_config_space_values(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Extract only configuration space values, excluding internal fields."""
    return {k: v for k, v in config_dict.items() if k not in _INTERNAL_CONFIG_FIELDS}


@pytest.fixture
def simple_dataset() -> Dataset:
    """Create a simple dataset for testing."""
    return Dataset(
        examples=[
            EvaluationExample(
                input_data={"prompt": "test 1"}, expected_output="result1"
            ),
            EvaluationExample(
                input_data={"prompt": "test 2"}, expected_output="result2"
            ),
            EvaluationExample(
                input_data={"prompt": "test 3"}, expected_output="result3"
            ),
        ],
        name="test_dataset",
    )


class TestParameterModeAndSecondOptimizationIntegration:
    """Integration tests validating both fixes work together."""

    @pytest.mark.asyncio
    async def test_full_optimization_workflow_with_parameter_mode(
        self, simple_dataset: Dataset
    ) -> None:
        """
        Complete workflow test:
        1. Run optimization with parameter mode
        2. Verify configs are received correctly
        3. Run second optimization
        4. Verify it starts with best config from run 1
        """
        run1_configs: list[dict[str, Any]] = []
        run2_configs: list[dict[str, Any]] = []
        current_run = 1

        @traigent.optimize(
            configuration_space={
                "model": ["gpt-3.5", "gpt-4", "gpt-4-turbo"],
                "temperature": [0.1, 0.3, 0.5, 0.7],
            },
            injection_mode="parameter",
            config_param="config",
            objectives=["accuracy"],
        )
        def my_llm_function(prompt: str, config: dict) -> str:
            """Function that uses config dict during optimization."""
            # Capture the config for verification (only config space values)
            config_copy = _extract_config_space_values(config.to_dict())

            if current_run == 1:
                run1_configs.append(config_copy)
            else:
                run2_configs.append(config_copy)

            # Access config values using .get() - this is the key test
            model = config.get("model", "MISSING")
            temperature = config.get("temperature", -1.0)

            return f"model={model}, temp={temperature}"

        opt_fn: OptimizedFunction = my_llm_function  # type: ignore[assignment]
        opt_fn.eval_dataset = simple_dataset

        with patch.dict(
            "os.environ", {"TRAIGENT_MOCK_LLM": "true", "TRAIGENT_OFFLINE_MODE": "true"}
        ):
            # === RUN 1 ===
            print("\n=== Running First Optimization ===")
            result1 = await opt_fn.optimize(max_trials=4)

            # === VERIFY RUN 1 ===
            print(f"\nRun 1 completed with {len(run1_configs)} trials")
            print(f"Run 1 best config: {result1.best_config}")

            # Fix 1 verification: Configs should have actual values
            assert len(run1_configs) > 0, "Run 1 should have captured configs"
            for i, config in enumerate(run1_configs):
                assert "model" in config, f"Trial {i}: Config should have 'model'"
                assert config["model"] in [
                    "gpt-3.5",
                    "gpt-4",
                    "gpt-4-turbo",
                ], f"Trial {i}: Model should be valid, got {config['model']}"
                assert (
                    "temperature" in config
                ), f"Trial {i}: Config should have 'temperature'"
                assert config["temperature"] in [
                    0.1,
                    0.3,
                    0.5,
                    0.7,
                ], f"Trial {i}: Temperature should be valid, got {config['temperature']}"

            # === RUN 2 ===
            current_run = 2
            print("\n=== Running Second Optimization ===")
            result2 = await opt_fn.optimize(max_trials=4)

            # === VERIFY RUN 2 ===
            print(f"\nRun 2 completed with {len(run2_configs)} trials")
            print(f"Run 2 first config: {run2_configs[0] if run2_configs else 'None'}")
            print(f"Run 2 best config: {result2.best_config}")

            # Fix 2 verification: First config of run 2 should be best from run 1
            assert len(run2_configs) > 0, "Run 2 should have captured configs"
            if result1.best_config:
                assert run2_configs[0] == result1.best_config, (
                    f"Run 2's first config should be Run 1's best.\n"
                    f"Expected: {result1.best_config}\n"
                    f"Got: {run2_configs[0]}"
                )

    @pytest.mark.asyncio
    async def test_parameter_mode_config_access_methods(
        self, simple_dataset: Dataset
    ) -> None:
        """Verify all config access methods work during optimization."""
        access_results: list[dict[str, Any]] = []

        @traigent.optimize(
            configuration_space={
                "model": ["gpt-3.5", "gpt-4"],
                "temperature": [0.1, 0.9],
                "max_tokens": [100, 500],
            },
            injection_mode="parameter",
            config_param="config",
            objectives=["accuracy"],
        )
        def test_access_methods(prompt: str, config: dict) -> str:
            result = {
                # Method 1: .get() with default
                "model_get": config.get("model", "FALLBACK"),
                # Method 2: .get() without default
                "temp_get": config.get("temperature"),
                # Method 3: bracket notation
                "tokens_bracket": config["max_tokens"],
                # Method 4: 'in' operator
                "has_model": "model" in config,
                # Method 5: keys()
                "keys": list(config.keys()),
            }
            access_results.append(result)
            return str(result)

        opt_fn: OptimizedFunction = test_access_methods  # type: ignore[assignment]
        opt_fn.eval_dataset = simple_dataset

        with patch.dict(
            "os.environ", {"TRAIGENT_MOCK_LLM": "true", "TRAIGENT_OFFLINE_MODE": "true"}
        ):
            await opt_fn.optimize(max_trials=2)

        # Verify all access methods worked
        assert len(access_results) > 0
        for result in access_results:
            # .get() should return actual values, not fallbacks
            assert result["model_get"] in [
                "gpt-3.5",
                "gpt-4",
            ], f"config.get() failed: {result['model_get']}"
            assert result["temp_get"] in [
                0.1,
                0.9,
            ], f"config.get() without default failed: {result['temp_get']}"
            assert result["tokens_bracket"] in [
                100,
                500,
            ], f"config[] bracket access failed: {result['tokens_bracket']}"
            assert (
                result["has_model"] is True
            ), f"'in' operator failed: {result['has_model']}"
            # .keys() should contain the config space keys (may also include internal fields)
            config_space_keys = {"model", "temperature", "max_tokens"}
            assert config_space_keys.issubset(
                set(result["keys"])
            ), f".keys() should contain config space keys: {result['keys']}"

    @pytest.mark.asyncio
    async def test_continuous_optimization_progression(
        self, simple_dataset: Dataset
    ) -> None:
        """Verify that optimization builds on previous results across runs."""
        run_first_configs: list[dict[str, Any]] = []
        run_best_configs: list[dict[str, Any] | None] = []

        @traigent.optimize(
            configuration_space={
                "model": ["gpt-3.5", "gpt-4", "gpt-4-turbo", "claude-3"],
                "temperature": [0.1, 0.3, 0.5, 0.7, 0.9],
            },
            injection_mode="parameter",
            config_param="config",
            objectives=["accuracy"],
        )
        def progressive_function(prompt: str, config: dict) -> str:
            return config.get("model", "none")

        opt_fn: OptimizedFunction = progressive_function  # type: ignore[assignment]
        opt_fn.eval_dataset = simple_dataset

        with patch.dict(
            "os.environ", {"TRAIGENT_MOCK_LLM": "true", "TRAIGENT_OFFLINE_MODE": "true"}
        ):
            num_runs = 4

            for run_num in range(num_runs):
                # Capture _current_config before optimization (baseline for this run)
                first_config = (
                    opt_fn._current_config.copy() if opt_fn._current_config else None
                )

                result = await opt_fn.optimize(max_trials=2)

                run_first_configs.append(first_config or {})
                run_best_configs.append(result.best_config)

                print(f"Run {run_num + 1}:")
                print(f"  Started with: {first_config}")
                print(f"  Found best: {result.best_config}")

        # Verify progression: _current_config should be updated to last best
        if run_best_configs[-1]:
            assert (
                opt_fn._current_config == run_best_configs[-1]
            ), "Final _current_config should be last best_config"


@pytest.mark.parametrize("num_optimization_runs", [2, 3, 5])
@pytest.mark.asyncio
async def test_multiple_sequential_optimizations(
    num_optimization_runs: int, simple_dataset: Dataset
) -> None:
    """Parameterized test for varying number of sequential optimization runs."""
    all_first_configs: list[dict[str, Any] | None] = []
    all_best_configs: list[dict[str, Any] | None] = []
    is_first_trial: bool = True
    current_first_config: dict[str, Any] | None = None

    @traigent.optimize(
        configuration_space={
            "model": ["a", "b", "c"],
            "param": [1, 2, 3],
        },
        injection_mode="parameter",
        config_param="config",
        objectives=["accuracy"],
    )
    def test_fn(prompt: str, config: dict) -> str:
        nonlocal is_first_trial, current_first_config
        if is_first_trial:
            current_first_config = _extract_config_space_values(config.to_dict())
            is_first_trial = False
        return config.get("model", "x")

    opt_fn: OptimizedFunction = test_fn  # type: ignore[assignment]
    opt_fn.eval_dataset = simple_dataset

    with patch.dict(
        "os.environ", {"TRAIGENT_MOCK_LLM": "true", "TRAIGENT_OFFLINE_MODE": "true"}
    ):
        for _ in range(num_optimization_runs):
            is_first_trial = True
            current_first_config = None

            result = await opt_fn.optimize(max_trials=2)

            all_first_configs.append(current_first_config)
            all_best_configs.append(result.best_config)

    # Verify chain: run N+1 starts with run N's best
    for i in range(1, num_optimization_runs):
        prev_best = all_best_configs[i - 1]
        current_first = all_first_configs[i]

        if prev_best:
            assert current_first == prev_best, (
                f"Run {i + 1} should start with run {i}'s best config.\n"
                f"Expected: {prev_best}\n"
                f"Got: {current_first}"
            )


class TestEdgeCases:
    """Edge case tests for both fixes."""

    @pytest.mark.asyncio
    async def test_empty_default_config_first_run(
        self, simple_dataset: Dataset
    ) -> None:
        """First run with no default_config should work correctly."""
        received_configs: list[dict[str, Any]] = []

        @traigent.optimize(
            configuration_space={"model": ["a", "b"]},
            injection_mode="parameter",
            config_param="config",
            objectives=["accuracy"],
            # No default_config specified
        )
        def fn(prompt: str, config: dict) -> str:
            received_configs.append(_extract_config_space_values(config.to_dict()))
            return config.get("model", "none")

        opt_fn: OptimizedFunction = fn  # type: ignore[assignment]
        opt_fn.eval_dataset = simple_dataset

        with patch.dict(
            "os.environ", {"TRAIGENT_MOCK_LLM": "true", "TRAIGENT_OFFLINE_MODE": "true"}
        ):
            result = await opt_fn.optimize(max_trials=2)

        # Should have received valid configs
        assert len(received_configs) > 0
        assert result.best_config is not None
        # _current_config should be updated to best_config
        assert opt_fn._current_config == result.best_config

    @pytest.mark.asyncio
    async def test_custom_config_param_name(self, simple_dataset: Dataset) -> None:
        """Custom config_param name should work correctly."""
        received: list[dict[str, Any]] = []

        @traigent.optimize(
            configuration_space={"model": ["x", "y"]},
            injection_mode="parameter",
            config_param="traigent_cfg",  # Non-standard name
            objectives=["accuracy"],
        )
        def fn(prompt: str, traigent_cfg: dict) -> str:
            received.append(_extract_config_space_values(traigent_cfg.to_dict()))
            return traigent_cfg.get("model", "none")

        opt_fn: OptimizedFunction = fn  # type: ignore[assignment]
        opt_fn.eval_dataset = simple_dataset

        with patch.dict(
            "os.environ", {"TRAIGENT_MOCK_LLM": "true", "TRAIGENT_OFFLINE_MODE": "true"}
        ):
            await opt_fn.optimize(max_trials=2)

        assert len(received) > 0
        for cfg in received:
            assert "model" in cfg
            assert cfg["model"] in ["x", "y"]
