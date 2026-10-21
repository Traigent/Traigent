"""Tests for execution mode configurations.

Tests execution modes. The real modes are the members of ExecutionMode
(local, hybrid, hybrid_api). standard is a deprecated compatibility value that
warns and resolves. edge_analytics, privacy and cloud are removed and fail
closed instead of normalizing; they appear here only as negative cases.
"""

from __future__ import annotations

import pytest

from tests.optimizer_validation.specs import (
    ExpectedOutcome,
    ExpectedResult,
    TestScenario,
    basic_scenario,
)
from traigent.config.types import ExecutionMode
from traigent.utils.exceptions import ConfigurationError

# Real execution modes that can run in this suite, derived from the SDK enum.
# hybrid_api needs a live API endpoint, so it is covered by validation tests only.
SUPPORTED_EXECUTION_MODES = [
    mode.value for mode in ExecutionMode if mode is not ExecutionMode.HYBRID_API
]

# Removed selectors that must fail closed.
REMOVED_EXECUTION_MODES = ["edge_analytics", "privacy", "cloud"]


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

    @pytest.mark.unit
    def test_cloud_mode_fails_closed(self) -> None:
        """Deprecated cloud mode no longer normalizes to a runtime mode."""
        from traigent.config.types import validate_execution_mode

        with pytest.raises(ConfigurationError, match="fails closed"):
            validate_execution_mode("cloud")

    @pytest.mark.unit
    def test_standard_mode_warns_and_resolves_to_hybrid(self) -> None:
        """Deprecated standard mode preserves the legacy hybrid mapping."""
        from traigent.config.types import ExecutionMode, validate_execution_mode

        with pytest.warns(DeprecationWarning):
            assert validate_execution_mode("standard") is ExecutionMode.HYBRID


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
                error_type=ConfigurationError,
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


class TestLocalMode:
    """Tests specific to LOCAL execution mode."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_local_runs_locally(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test local mode runs optimization locally."""
        scenario = basic_scenario(
            name="local_mode",
            execution_mode="local",
            max_trials=3,
            gist_template="local -> {trial_count()} | {status()}",
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
    async def test_local_with_minimal_logging(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test local mode with minimal logging enabled."""
        scenario = TestScenario(
            name="local_minimal_log",
            description="Local mode with minimal logging",
            execution_mode="local",
            config_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
            max_trials=2,
            gist_template="local -> {trial_count()} | {status()}",
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
    """Tests for PRIVACY execution mode legacy alias."""

    @pytest.mark.unit
    def test_privacy_mode_fails_closed(self) -> None:
        """Privacy no longer normalizes to hybrid."""
        from traigent.config.types import validate_execution_mode

        with pytest.raises(ConfigurationError, match="fails closed"):
            validate_execution_mode("privacy")


class TestRemovedEdgeAnalyticsMode:
    """edge_analytics was removed and must fail closed, never silently remap."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "removed_mode", ["edge_analytics", "EDGE_ANALYTICS", "  edge_analytics  "]
    )
    def test_edge_analytics_fails_closed(self, removed_mode: str) -> None:
        """validate_execution_mode raises the removal error for edge_analytics."""
        from traigent.config.types import validate_execution_mode

        with pytest.raises(ConfigurationError, match="has been removed"):
            validate_execution_mode(removed_mode)

    @pytest.mark.unit
    def test_edge_analytics_enum_access_fails_closed(self) -> None:
        """The ExecutionMode enum no longer exposes or accepts edge_analytics."""
        with pytest.raises(ValueError, match="has been removed"):
            ExecutionMode("edge_analytics")
        with pytest.raises(KeyError, match="removed"):
            ExecutionMode["EDGE_ANALYTICS"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_edge_analytics_scenario_is_rejected(self, scenario_runner) -> None:
        """A scenario using edge_analytics surfaces the removal ConfigurationError."""
        scenario = basic_scenario(
            name="edge_analytics_removed",
            execution_mode="edge_analytics",
            max_trials=1,
            gist_template="edge_analytics -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert isinstance(result, ConfigurationError), f"Got: {result!r}"
        assert "has been removed" in str(result)


class TestRealExecutionModesAreTheEnum:
    """The suite's real-mode list must follow the SDK enum."""

    @pytest.mark.unit
    def test_real_modes_match_enum(self) -> None:
        assert {m.value for m in ExecutionMode} == {"local", "hybrid", "hybrid_api"}
        for removed in REMOVED_EXECUTION_MODES:
            assert removed not in {m.value for m in ExecutionMode}


class TestHybridMode:
    """Tests for HYBRID execution mode."""

    @pytest.mark.unit
    def test_hybrid_mode_is_supported(self) -> None:
        """Hybrid is the supported portal-tracked local execution mode."""
        from traigent.config.types import ExecutionMode, validate_execution_mode

        assert validate_execution_mode("hybrid") is ExecutionMode.HYBRID


class TestStandardMode:
    """Tests for deprecated STANDARD execution mode."""

    @pytest.mark.unit
    def test_standard_mode_warns_and_resolves_to_hybrid(self) -> None:
        """Test standard mode emits DeprecationWarning and resolves to hybrid."""
        from traigent.config.types import ExecutionMode, validate_execution_mode

        with pytest.warns(DeprecationWarning):
            assert validate_execution_mode("standard") is ExecutionMode.HYBRID


class TestCloudMode:
    """Tests for deprecated CLOUD execution mode."""

    @pytest.mark.unit
    def test_cloud_mode_fails_closed(self) -> None:
        """Test cloud mode fails closed instead of resolving."""
        from traigent.config.types import validate_execution_mode

        with pytest.raises(ConfigurationError, match="fails closed"):
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
                outcome=ExpectedOutcome.SUCCESS,
            ),
            gist_template="none_mode -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # None means the mode was omitted, so the new algorithm policy applies.

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
            execution_mode="LOCAL",  # Uppercase instead of lowercase
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
            execution_mode="  local  ",  # Whitespace around value
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
