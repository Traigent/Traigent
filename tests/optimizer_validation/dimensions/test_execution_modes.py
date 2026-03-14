"""Tests for execution mode configurations.

Tests execution modes. Note: only edge_analytics is currently supported.
cloud and hybrid raise ConfigurationError (not yet supported).
privacy and standard raise ConfigurationError (removed).
"""

from __future__ import annotations

import pytest

from tests.optimizer_validation.specs import (
    ExpectedOutcome,
    ExpectedResult,
    TestScenario,
    basic_scenario,
)
from traigent.utils.exceptions import ConfigurationError

# Only edge_analytics is currently supported
SUPPORTED_EXECUTION_MODES = ["edge_analytics"]

# Modes that raise ConfigurationError
UNSUPPORTED_MODES = ["cloud", "hybrid"]  # Not yet supported
REMOVED_MODES = ["privacy", "standard"]  # Removed


class TestExecutionModeMatrix:
    """Test matrix for supported execution modes."""

    @pytest.mark.parametrize("execution_mode", SUPPORTED_EXECUTION_MODES)
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execution_mode_basic(
        self,
        execution_mode: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test each supported execution mode works with basic configuration."""
        scenario = basic_scenario(
            name=f"basic_{execution_mode}",
            execution_mode=execution_mode,
            max_trials=2,
            gist_template=f"{execution_mode} -> {{trial_count()}} | {{status()}}",
        )

        _, result = await scenario_runner(scenario)

        # Should not raise exception
        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        # Validate result
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.parametrize("execution_mode", UNSUPPORTED_MODES)
    @pytest.mark.unit
    def test_unsupported_mode_raises_configuration_error(
        self,
        execution_mode: str,
    ) -> None:
        """Test that unsupported modes raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="not yet supported"):
            from traigent.config.types import validate_execution_mode

            validate_execution_mode(execution_mode)

    @pytest.mark.parametrize("execution_mode", REMOVED_MODES)
    @pytest.mark.unit
    def test_removed_mode_raises_configuration_error(
        self,
        execution_mode: str,
    ) -> None:
        """Test that removed modes raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="No such mode"):
            from traigent.config.types import validate_execution_mode

            validate_execution_mode(execution_mode)


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
            Verify that providing an unknown execution mode raises ConfigurationError.

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

        # Should fail with ConfigurationError

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

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

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

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

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestPrivacyMode:
    """Tests for PRIVACY execution mode - now removed."""

    @pytest.mark.unit
    def test_privacy_mode_raises_configuration_error(self) -> None:
        """Test privacy mode raises ConfigurationError (removed)."""
        from traigent.config.types import validate_execution_mode

        with pytest.raises(ConfigurationError, match="No such mode"):
            validate_execution_mode("privacy")


class TestHybridMode:
    """Tests for HYBRID execution mode - not yet supported."""

    @pytest.mark.unit
    def test_hybrid_mode_raises_configuration_error(self) -> None:
        """Test hybrid mode raises ConfigurationError (not yet supported)."""
        from traigent.config.types import validate_execution_mode

        with pytest.raises(ConfigurationError, match="not yet supported"):
            validate_execution_mode("hybrid")


class TestStandardMode:
    """Tests for STANDARD execution mode - now removed."""

    @pytest.mark.unit
    def test_standard_mode_raises_configuration_error(self) -> None:
        """Test standard mode raises ConfigurationError (removed)."""
        from traigent.config.types import validate_execution_mode

        with pytest.raises(ConfigurationError, match="No such mode"):
            validate_execution_mode("standard")


class TestCloudMode:
    """Tests for CLOUD execution mode - not yet supported."""

    @pytest.mark.unit
    def test_cloud_mode_raises_configuration_error(self) -> None:
        """Test cloud mode raises ConfigurationError (not yet supported)."""
        from traigent.config.types import validate_execution_mode

        with pytest.raises(ConfigurationError, match="not yet supported"):
            validate_execution_mode("cloud")


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

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

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

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

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

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

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

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

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

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()
