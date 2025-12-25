"""Tests for injection mode configurations.

Tests all injection modes: CONTEXT, PARAMETER, ATTRIBUTE, SEAMLESS
"""

from __future__ import annotations

import pytest

from tests.optimizer_validation.specs import (
    ExpectedOutcome,
    ExpectedResult,
    TestScenario,
    basic_scenario,
)

# All supported injection modes
INJECTION_MODES = ["context", "parameter", "attribute", "seamless"]


class TestInjectionModeMatrix:
    """Test matrix for all injection modes."""

    @pytest.mark.parametrize("injection_mode", INJECTION_MODES)
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_injection_mode_basic(
        self,
        injection_mode: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test each injection mode works with basic configuration."""
        scenario = basic_scenario(
            name=f"basic_{injection_mode}",
            injection_mode=injection_mode,
            max_trials=2,
        )

        func, result = await scenario_runner(scenario)

        # Should not raise exception
        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Validate result
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.parametrize("injection_mode", INJECTION_MODES)
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_injection_mode_with_default_config(
        self,
        injection_mode: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test injection modes respect default configuration."""
        default_config = {"model": "gpt-3.5-turbo", "temperature": 0.5}

        scenario = basic_scenario(
            name=f"default_config_{injection_mode}",
            injection_mode=injection_mode,
            default_config=default_config,
            max_trials=2,
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestContextInjection:
    """Tests specific to CONTEXT injection mode."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_context_injection_is_thread_safe(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test context injection maintains thread safety."""
        scenario = basic_scenario(
            name="context_thread_safe",
            injection_mode="context",
            max_trials=3,
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestParameterInjection:
    """Tests specific to PARAMETER injection mode."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_parameter_injection_adds_config_param(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test parameter injection adds traigent_config parameter."""
        scenario = basic_scenario(
            name="parameter_config_param",
            injection_mode="parameter",
            max_trials=2,
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestAttributeInjection:
    """Tests specific to ATTRIBUTE injection mode."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_attribute_injection_stores_config(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test attribute injection stores config as function attribute."""
        scenario = basic_scenario(
            name="attribute_storage",
            injection_mode="attribute",
            max_trials=2,
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestSeamlessInjection:
    """Tests specific to SEAMLESS injection mode."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_seamless_injection_modifies_source(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test seamless injection works via AST transformation."""
        scenario = basic_scenario(
            name="seamless_ast",
            injection_mode="seamless",
            max_trials=2,
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()
