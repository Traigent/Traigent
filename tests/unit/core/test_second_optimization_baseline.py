"""Tests for second optimization using previous best config as baseline.

This tests the fix for the issue where the second optimization run would start
fresh from the search space instead of using the best config from the previous
run as the baseline (first trial).

The root cause was:
- default_config was not updated after optimization completed
- Each optimization run creates a new orchestrator, which uses default_config
  for the first trial

The fix:
- After optimization completes, update self.default_config = result.best_config.copy()
- This ensures subsequent optimization runs start with the previous best config
"""

from __future__ import annotations

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


class TestSecondOptimizationBaseline:
    """Test that second optimization uses previous best config as baseline."""

    @pytest.mark.asyncio
    async def test_default_config_updated_after_optimization(
        self, simple_dataset: Dataset
    ) -> None:
        """default_config should be updated to best_config after optimization."""

        @traigent.optimize(
            configuration_space={
                "model": ["gpt-3.5", "gpt-4", "gpt-4-turbo"],
                "temperature": [0.1, 0.3, 0.5, 0.7],
            },
            injection_mode="parameter",
            config_param="config",
            objectives=["accuracy"],
        )
        def my_function(prompt: str, config: dict) -> str:
            return config.get("model", "unset")

        opt_fn: OptimizedFunction = my_function  # type: ignore[assignment]
        opt_fn.eval_dataset = simple_dataset

        # Store initial default_config
        initial_default = opt_fn.default_config.copy() if opt_fn.default_config else {}

        with patch.dict("os.environ", {"TRAIGENT_MOCK_LLM": "true", "TRAIGENT_OFFLINE_MODE": "true"}):
            result = await opt_fn.optimize(max_trials=3)

        # After optimization, default_config should equal best_config
        if result.best_config:
            assert opt_fn.default_config is not None
            assert opt_fn.default_config == result.best_config, (
                f"default_config should be updated to best_config. "
                f"Got {opt_fn.default_config}, expected {result.best_config}"
            )

    @pytest.mark.asyncio
    async def test_second_optimization_first_trial_uses_best_config(
        self, simple_dataset: Dataset
    ) -> None:
        """Second optimization's first trial should use best config from run 1."""
        all_configs: list[list[dict[str, Any]]] = []
        current_run_configs: list[dict[str, Any]] = []

        @traigent.optimize(
            configuration_space={
                "model": ["gpt-3.5", "gpt-4", "gpt-4-turbo"],
                "temperature": [0.1, 0.3, 0.5],
            },
            injection_mode="parameter",
            config_param="config",
            objectives=["accuracy"],
        )
        def my_function(prompt: str, config: dict) -> str:
            current_run_configs.append(dict(config))
            return config.get("model", "unset")

        opt_fn: OptimizedFunction = my_function  # type: ignore[assignment]
        opt_fn.eval_dataset = simple_dataset

        with patch.dict("os.environ", {"TRAIGENT_MOCK_LLM": "true", "TRAIGENT_OFFLINE_MODE": "true"}):
            # First optimization run
            result1 = await opt_fn.optimize(max_trials=3)
            all_configs.append(current_run_configs.copy())
            current_run_configs.clear()

            # Second optimization run
            result2 = await opt_fn.optimize(max_trials=3)
            all_configs.append(current_run_configs.copy())

        run1_configs = all_configs[0]
        run2_configs = all_configs[1]

        # First trial of run 2 should use best config from run 1
        if result1.best_config and len(run2_configs) > 0:
            first_config_run2 = run2_configs[0]
            assert first_config_run2 == result1.best_config, (
                f"Run 2's first config should be run 1's best config.\n"
                f"Run 2 first: {first_config_run2}\n"
                f"Run 1 best: {result1.best_config}"
            )

    @pytest.mark.asyncio
    async def test_third_optimization_uses_second_best(
        self, simple_dataset: Dataset
    ) -> None:
        """Third optimization should use best config from second optimization."""
        run_first_configs: list[dict[str, Any]] = []
        current_first_config: dict[str, Any] | None = None
        is_first_of_run = True

        @traigent.optimize(
            configuration_space={
                "model": ["gpt-3.5", "gpt-4"],
                "temperature": [0.1, 0.5],
            },
            injection_mode="parameter",
            config_param="config",
            objectives=["accuracy"],
        )
        def my_function(prompt: str, config: dict) -> str:
            nonlocal current_first_config, is_first_of_run
            if is_first_of_run:
                current_first_config = dict(config)
                is_first_of_run = False
            return config.get("model", "unset")

        opt_fn: OptimizedFunction = my_function  # type: ignore[assignment]
        opt_fn.eval_dataset = simple_dataset
        best_configs: list[dict[str, Any] | None] = []

        with patch.dict("os.environ", {"TRAIGENT_MOCK_LLM": "true", "TRAIGENT_OFFLINE_MODE": "true"}):
            for run_num in range(3):
                is_first_of_run = True
                current_first_config = None

                result = await opt_fn.optimize(max_trials=2)

                run_first_configs.append(current_first_config or {})
                best_configs.append(result.best_config)

        # Run 2's first config should match Run 1's best
        if best_configs[0]:
            assert run_first_configs[1] == best_configs[0], (
                f"Run 2 should start with run 1's best config"
            )

        # Run 3's first config should match Run 2's best
        if best_configs[1]:
            assert run_first_configs[2] == best_configs[1], (
                f"Run 3 should start with run 2's best config"
            )

    @pytest.mark.asyncio
    async def test_best_config_none_on_first_optimization(
        self, simple_dataset: Dataset
    ) -> None:
        """First optimization should handle case where no prior best_config exists."""

        @traigent.optimize(
            configuration_space={"model": ["gpt-3.5", "gpt-4"]},
            injection_mode="parameter",
            config_param="config",
            objectives=["accuracy"],
        )
        def my_function(prompt: str, config: dict) -> str:
            return config.get("model", "unset")

        opt_fn: OptimizedFunction = my_function  # type: ignore[assignment]
        opt_fn.eval_dataset = simple_dataset

        # Before any optimization, best_config should be None
        assert opt_fn._best_config is None

        with patch.dict("os.environ", {"TRAIGENT_MOCK_LLM": "true", "TRAIGENT_OFFLINE_MODE": "true"}):
            # First optimization should complete without errors
            result = await opt_fn.optimize(max_trials=2)

        # After optimization, we should have a best_config
        assert result.best_config is not None
        assert opt_fn._best_config is not None

    @pytest.mark.asyncio
    async def test_default_config_preserved_when_no_best_found(
        self, simple_dataset: Dataset
    ) -> None:
        """If optimization finds no best config, default_config should be preserved."""

        @traigent.optimize(
            configuration_space={"model": ["gpt-3.5", "gpt-4"]},
            injection_mode="parameter",
            config_param="config",
            default_config={"model": "gpt-3.5"},
            objectives=["accuracy"],
        )
        def my_function(prompt: str, config: dict) -> str:
            return config.get("model", "unset")

        opt_fn: OptimizedFunction = my_function  # type: ignore[assignment]
        opt_fn.eval_dataset = simple_dataset

        initial_default = opt_fn.default_config.copy() if opt_fn.default_config else {}

        with patch.dict("os.environ", {"TRAIGENT_MOCK_LLM": "true", "TRAIGENT_OFFLINE_MODE": "true"}):
            result = await opt_fn.optimize(max_trials=2)

        # If best_config was found, default should be updated
        # If not, default should be preserved
        if result.best_config:
            assert opt_fn.default_config == result.best_config
        else:
            assert opt_fn.default_config == initial_default


class TestCurrentConfigVsDefaultConfig:
    """Test interaction between current_config and default_config."""

    @pytest.mark.asyncio
    async def test_current_config_updated_after_optimization(
        self, simple_dataset: Dataset
    ) -> None:
        """_current_config should be updated to best_config after optimization."""

        @traigent.optimize(
            configuration_space={"model": ["gpt-3.5", "gpt-4"]},
            injection_mode="parameter",
            config_param="config",
            objectives=["accuracy"],
        )
        def my_function(prompt: str, config: dict) -> str:
            return config.get("model", "unset")

        opt_fn: OptimizedFunction = my_function  # type: ignore[assignment]
        opt_fn.eval_dataset = simple_dataset

        with patch.dict("os.environ", {"TRAIGENT_MOCK_LLM": "true", "TRAIGENT_OFFLINE_MODE": "true"}):
            result = await opt_fn.optimize(max_trials=2)

        if result.best_config:
            assert opt_fn._current_config == result.best_config

    @pytest.mark.asyncio
    async def test_both_current_and_default_updated(
        self, simple_dataset: Dataset
    ) -> None:
        """Both _current_config and default_config should be updated."""

        @traigent.optimize(
            configuration_space={"model": ["gpt-3.5", "gpt-4"]},
            injection_mode="parameter",
            config_param="config",
            objectives=["accuracy"],
        )
        def my_function(prompt: str, config: dict) -> str:
            return config.get("model", "unset")

        opt_fn: OptimizedFunction = my_function  # type: ignore[assignment]
        opt_fn.eval_dataset = simple_dataset

        with patch.dict("os.environ", {"TRAIGENT_MOCK_LLM": "true", "TRAIGENT_OFFLINE_MODE": "true"}):
            result = await opt_fn.optimize(max_trials=2)

        if result.best_config:
            # Both should equal best_config
            assert opt_fn._current_config == result.best_config
            assert opt_fn.default_config == result.best_config
            # And they should equal each other
            assert opt_fn._current_config == opt_fn.default_config


class TestContinuousImprovement:
    """Test that continuous optimization leads to improvement."""

    @pytest.mark.asyncio
    async def test_multiple_optimizations_preserve_best(
        self, simple_dataset: Dataset
    ) -> None:
        """Multiple optimization runs should preserve and build on best configs."""
        call_count = 0
        best_configs_history: list[dict[str, Any] | None] = []

        @traigent.optimize(
            configuration_space={
                "model": ["gpt-3.5", "gpt-4", "gpt-4-turbo"],
                "temperature": [0.1, 0.3, 0.5, 0.7, 0.9],
            },
            injection_mode="parameter",
            config_param="config",
            objectives=["accuracy"],
        )
        def my_function(prompt: str, config: dict) -> str:
            nonlocal call_count
            call_count += 1
            return config.get("model", "unset")

        opt_fn: OptimizedFunction = my_function  # type: ignore[assignment]
        opt_fn.eval_dataset = simple_dataset

        with patch.dict("os.environ", {"TRAIGENT_MOCK_LLM": "true", "TRAIGENT_OFFLINE_MODE": "true"}):
            for _ in range(3):
                result = await opt_fn.optimize(max_trials=2)
                best_configs_history.append(result.best_config)

        # All optimizations should have found a best config
        for i, best in enumerate(best_configs_history):
            assert best is not None, f"Run {i+1} should have a best config"

        # Final default_config should be the last best config
        assert opt_fn.default_config == best_configs_history[-1]
