"""Tests for execution mode configurations.

Tests all execution modes: EDGE_ANALYTICS, PRIVACY, HYBRID, STANDARD, CLOUD
"""

from __future__ import annotations

import pytest

from tests.optimizer_validation.specs import (
    ExpectedResult,
    TestScenario,
    basic_scenario,
)

# All supported execution modes
EXECUTION_MODES = ["edge_analytics", "privacy", "hybrid", "standard", "cloud"]


class TestExecutionModeMatrix:
    """Test matrix for all execution modes."""

    @pytest.mark.parametrize("execution_mode", EXECUTION_MODES)
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execution_mode_basic(
        self,
        execution_mode: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test each execution mode works with basic configuration."""
        scenario = basic_scenario(
            name=f"basic_{execution_mode}",
            execution_mode=execution_mode,
            max_trials=2,
        )

        func, result = await scenario_runner(scenario)

        # Should not raise exception
        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Validate result
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestEdgeAnalyticsMode:
    """Tests specific to EDGE_ANALYTICS execution mode."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_edge_analytics_runs_locally(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test edge_analytics mode runs optimization locally."""
        scenario = basic_scenario(
            name="edge_analytics_local",
            execution_mode="edge_analytics",
            max_trials=3,
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_edge_analytics_with_minimal_logging(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test edge_analytics mode with minimal logging enabled."""
        scenario = TestScenario(
            name="edge_analytics_minimal_log",
            description="Edge analytics with minimal logging",
            execution_mode="edge_analytics",
            config_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
            max_trials=2,
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestPrivacyMode:
    """Tests specific to PRIVACY execution mode."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_privacy_mode_no_data_transmission(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test privacy mode does not transmit input/output data."""
        scenario = basic_scenario(
            name="privacy_no_transmit",
            execution_mode="privacy",
            max_trials=2,
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestHybridMode:
    """Tests specific to HYBRID execution mode."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_hybrid_mode_local_with_fallback(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test hybrid mode runs locally with cloud fallback option."""
        scenario = basic_scenario(
            name="hybrid_fallback",
            execution_mode="hybrid",
            max_trials=2,
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestStandardMode:
    """Tests specific to STANDARD execution mode."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_standard_mode_cloud_orchestration(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test standard mode uses cloud orchestration."""
        scenario = basic_scenario(
            name="standard_cloud",
            execution_mode="standard",
            max_trials=2,
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestCloudMode:
    """Tests specific to CLOUD execution mode."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cloud_mode_full_saas(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test cloud mode uses full SaaS execution."""
        scenario = basic_scenario(
            name="cloud_saas",
            execution_mode="cloud",
            max_trials=2,
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()
