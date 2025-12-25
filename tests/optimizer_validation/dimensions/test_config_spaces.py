"""Tests for configuration space types.

Tests categorical, continuous (range), and mixed configuration spaces.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.optimizer_validation.specs import (
    ExpectedResult,
    TestScenario,
    config_space_scenario,
)


class TestCategoricalConfigSpace:
    """Tests for categorical-only configuration spaces."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_simple_categorical(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test simple categorical config space with string values."""
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
        }

        scenario = config_space_scenario(
            name="simple_categorical",
            config_space=config_space,
            max_trials=2,
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_multiple_categorical_params(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test multiple categorical parameters."""
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
            "safety_filter": ["strict", "moderate", "lenient"],
            "response_format": ["text", "json"],
        }

        scenario = config_space_scenario(
            name="multi_categorical",
            config_space=config_space,
            max_trials=3,
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_numeric_categorical_values(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test categorical parameters with numeric values."""
        config_space = {
            "max_tokens": [100, 500, 1000, 2000],
            "retry_count": [1, 2, 3],
        }

        scenario = config_space_scenario(
            name="numeric_categorical",
            config_space=config_space,
            max_trials=2,
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestContinuousConfigSpace:
    """Tests for continuous (range-based) configuration spaces."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_simple_continuous(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test simple continuous config space with float range."""
        config_space = {
            "temperature": (0.0, 1.0),
        }

        scenario = config_space_scenario(
            name="simple_continuous",
            config_space=config_space,
            max_trials=2,
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_multiple_continuous_params(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test multiple continuous parameters."""
        config_space = {
            "temperature": (0.0, 2.0),
            "top_p": (0.1, 1.0),
            "frequency_penalty": (-2.0, 2.0),
        }

        scenario = config_space_scenario(
            name="multi_continuous",
            config_space=config_space,
            max_trials=3,
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_narrow_range(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test continuous parameter with narrow range."""
        config_space = {
            "temperature": (0.5, 0.7),
        }

        scenario = config_space_scenario(
            name="narrow_range",
            config_space=config_space,
            max_trials=2,
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestMixedConfigSpace:
    """Tests for mixed (categorical + continuous) configuration spaces."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_simple_mixed(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test mixed config space with both types."""
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": (0.0, 1.0),
        }

        scenario = config_space_scenario(
            name="simple_mixed",
            config_space=config_space,
            max_trials=2,
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_complex_mixed(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test complex mixed config space."""
        config_space = {
            # Categorical
            "model": ["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
            "response_format": ["text", "json"],
            # Continuous
            "temperature": (0.0, 2.0),
            "top_p": (0.1, 1.0),
            # Numeric categorical
            "max_tokens": [100, 500, 1000],
        }

        scenario = config_space_scenario(
            name="complex_mixed",
            config_space=config_space,
            max_trials=4,
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestConfigSpaceEdgeCases:
    """Tests for edge cases in configuration spaces."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_single_value_categorical(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test categorical with single value (no optimization possible)."""
        config_space = {
            "model": ["gpt-4"],  # Single value
            "temperature": [0.5, 0.7],  # Multiple values
        }

        scenario = config_space_scenario(
            name="single_value",
            config_space=config_space,
            max_trials=2,
        )

        func, result = await scenario_runner(scenario)

        # Should still work, just limited optimization space
        assert not isinstance(result, Exception)
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_large_categorical_space(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test categorical with many values."""
        config_space = {
            "model": [f"model-{i}" for i in range(10)],
            "temperature": [0.1 * i for i in range(10)],
        }

        scenario = config_space_scenario(
            name="large_categorical",
            config_space=config_space,
            max_trials=5,
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()
