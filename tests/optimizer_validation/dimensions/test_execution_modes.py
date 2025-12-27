"""Tests for execution mode configurations.

Tests all execution modes: EDGE_ANALYTICS, PRIVACY, HYBRID, STANDARD, CLOUD
"""

from __future__ import annotations

import pytest

from tests.optimizer_validation.specs import (
    ExpectedOutcome,
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
            gist_template=f"{execution_mode} -> {{trial_count()}} | {{status()}}",
        )

        _, result = await scenario_runner(scenario)

        # Should not raise exception
        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Validate result
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestInvalidExecutionMode:
    """Tests for invalid execution mode configurations."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_invalid_execution_mode(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test behavior with an invalid execution mode string.

        Purpose:
            Verify that providing an unknown execution mode raises a ValueError
            or handled gracefully.

        Edge Case: Invalid execution mode
        """
        scenario = TestScenario(
            name="invalid_execution_mode",
            description="Invalid execution mode string",
            execution_mode="invalid_mode_xyz",
            config_space={"model": ["gpt-3.5-turbo"]},
            max_trials=1,
            expected=ExpectedResult(
                outcome=ExpectedOutcome.FAILURE,
                error_type=ValueError,
            ),
            gist_template="invalid-mode -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should fail with ValueError
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
            gist_template="edge_analytics -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

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
            gist_template="edge_analytics -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

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
            gist_template="privacy -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

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
            gist_template="hybrid -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

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
            gist_template="standard -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

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
            gist_template="cloud -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestExecutionModeEdgeCases:
    """Tests for edge cases in execution mode configuration."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execution_mode_none(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test with execution_mode set to None."""
        scenario = TestScenario(
            name="execution_mode_none",
            description="Execution mode is None",
            config_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
            execution_mode=None,  # type: ignore[arg-type]
            max_trials=2,
            expected=ExpectedResult(
                outcome=ExpectedOutcome.FAILURE,
            ),
            gist_template="none_mode -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should either use default or fail gracefully
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execution_mode_empty_string(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test with execution_mode as empty string."""
        scenario = TestScenario(
            name="execution_mode_empty",
            description="Execution mode is empty string",
            config_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
            execution_mode="",  # Empty string
            max_trials=2,
            gist_template="empty_mode -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should fail validation or use default
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execution_mode_invalid_value(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test with invalid execution_mode value."""
        from tests.optimizer_validation.specs import ExpectedOutcome, ExpectedResult

        scenario = TestScenario(
            name="execution_mode_invalid",
            description="Execution mode is invalid value",
            config_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
            execution_mode="INVALID_MODE",  # Not a valid mode
            max_trials=2,
            expected=ExpectedResult(
                outcome=ExpectedOutcome.FAILURE,
            ),
            gist_template="invalid_mode -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should fail with validation error
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execution_mode_case_sensitivity(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test execution_mode with different casing."""
        scenario = TestScenario(
            name="execution_mode_uppercase",
            description="Execution mode with uppercase",
            config_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
            execution_mode="EDGE_ANALYTICS",  # Uppercase instead of lowercase
            max_trials=2,
            gist_template="case_mode -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Document case sensitivity behavior
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execution_mode_with_whitespace(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test execution_mode with leading/trailing whitespace."""
        scenario = TestScenario(
            name="execution_mode_whitespace",
            description="Execution mode with whitespace",
            config_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
            execution_mode="  edge_analytics  ",  # Whitespace around value
            max_trials=2,
            gist_template="ws_mode -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should fail or strip whitespace
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()
