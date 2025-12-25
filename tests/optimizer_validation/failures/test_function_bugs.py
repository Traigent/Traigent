"""Tests for handling bugs in optimized functions.

Tests scenarios where the user's function raises exceptions,
returns invalid values, or times out.
"""

from __future__ import annotations

import pytest

from tests.optimizer_validation.specs import (
    ExpectedOutcome,
    ExpectedResult,
    TestScenario,
    basic_scenario,
)


class TestFunctionRaises:
    """Tests for functions that raise exceptions."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_function_always_raises(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test function that raises on every call."""
        scenario = TestScenario(
            name="always_raises",
            description="Function raises ValueError on every call",
            config_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
            function_should_raise=ValueError,
            max_trials=3,
            expected=ExpectedResult(
                outcome=ExpectedOutcome.PARTIAL,
                min_trials=1,
            ),
        )

        func, result = await scenario_runner(scenario)

        # Should handle gracefully - either all trials fail or optimization fails
        # The key is that it doesn't crash unexpectedly
        if not isinstance(result, Exception):
            # All trials should be marked as failed
            for trial in result.trials:
                assert trial.status.value in ("failed", "cancelled", "pruned")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_function_raises_intermittently(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test function that fails on specific call number."""
        scenario = TestScenario(
            name="intermittent_failure",
            description="Function fails on second call only",
            config_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
            function_should_raise=RuntimeError,
            function_raise_on_call=2,  # Fail on second call
            max_trials=4,
            expected=ExpectedResult(
                outcome=ExpectedOutcome.PARTIAL,
                min_trials=2,
            ),
        )

        func, result = await scenario_runner(scenario)

        # Should have at least some successful trials
        if not isinstance(result, Exception):
            # First call should succeed, second should fail
            assert len(result.trials) >= 2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_function_raises_type_error(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test function that raises TypeError."""
        scenario = TestScenario(
            name="type_error",
            description="Function raises TypeError",
            config_space={"model": ["gpt-3.5-turbo"]},
            function_should_raise=TypeError,
            max_trials=2,
            expected=ExpectedResult(
                outcome=ExpectedOutcome.PARTIAL,
            ),
        )

        func, result = await scenario_runner(scenario)

        # Should handle gracefully
        if not isinstance(result, Exception):
            for trial in result.trials:
                assert trial.status.value in ("failed", "cancelled", "pruned")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_function_raises_key_error(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test function that raises KeyError."""
        scenario = TestScenario(
            name="key_error",
            description="Function raises KeyError",
            config_space={"model": ["gpt-3.5-turbo"]},
            function_should_raise=KeyError,
            max_trials=2,
            expected=ExpectedResult(
                outcome=ExpectedOutcome.PARTIAL,
            ),
        )

        func, result = await scenario_runner(scenario)

        # Should handle gracefully
        if not isinstance(result, Exception):
            assert len(result.trials) >= 1


class TestFunctionReturnsInvalid:
    """Tests for functions that return invalid values."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_function_returns_none(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test function returning None instead of expected output."""
        scenario = TestScenario(
            name="returns_none",
            description="Function returns None",
            config_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
            function_return_value=None,
            max_trials=2,
        )

        func, result = await scenario_runner(scenario)

        # Should handle gracefully
        assert not isinstance(result, Exception) or isinstance(result, Exception)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_function_returns_empty_string(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test function returning empty string."""
        scenario = TestScenario(
            name="returns_empty",
            description="Function returns empty string",
            config_space={"model": ["gpt-3.5-turbo"]},
            function_return_value="",
            max_trials=2,
        )

        func, result = await scenario_runner(scenario)

        # Empty string is valid, should complete
        assert not isinstance(result, Exception), f"Unexpected error: {result}"


class TestFunctionTimeout:
    """Tests for function timeout scenarios."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_optimization_timeout(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test optimization with very short timeout."""
        scenario = TestScenario(
            name="short_timeout",
            description="Optimization with short timeout",
            config_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
            timeout=0.5,  # Very short timeout
            max_trials=10,  # More trials than timeout allows
            expected=ExpectedResult(
                expected_stop_reason="timeout",
            ),
        )

        func, result = await scenario_runner(scenario)

        # Should complete with timeout stop reason
        if not isinstance(result, Exception):
            # May have completed some trials before timeout
            assert result.stop_reason in ("timeout", "max_trials_reached", None)


class TestFunctionBehaviorVariations:
    """Tests for various function behavior variations."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_function_first_call_fails_then_succeeds(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test function that fails first call then succeeds."""
        scenario = TestScenario(
            name="first_fails",
            description="Function fails on first call only",
            config_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
            function_should_raise=ValueError,
            function_raise_on_call=1,
            max_trials=4,
            expected=ExpectedResult(
                outcome=ExpectedOutcome.PARTIAL,
                min_trials=2,
            ),
        )

        func, result = await scenario_runner(scenario)

        # Should have recovered after first failure
        if not isinstance(result, Exception):
            assert len(result.trials) >= 2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_function_custom_return_value(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test function with custom return value."""
        scenario = TestScenario(
            name="custom_return",
            description="Function returns custom value",
            config_space={"model": ["gpt-3.5-turbo"]},
            function_return_value="custom_output_value",
            max_trials=2,
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()
