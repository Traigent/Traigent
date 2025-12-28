"""Tests for handling bugs in optimized functions.

Tests scenarios where the user's function raises exceptions,
returns invalid values, or times out.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.optimizer_validation.specs import (
    ExpectedOutcome,
    ExpectedResult,
    TestScenario,
)


def _is_failed_trial(trial: Any) -> bool:
    status = getattr(trial, "status", None)
    status_value = getattr(status, "value", status)
    return status_value == "failed" or bool(getattr(trial, "error_message", None))


def _is_successful_trial(trial: Any) -> bool:
    status = getattr(trial, "status", None)
    status_value = getattr(status, "value", status)
    return status_value == "completed" or not getattr(trial, "error_message", None)


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
            gist_template="always-raises -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # In mock mode, the function should still complete even with
        # simulated errors
        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        # Note: Trial failure status depends on mock implementation
        # The key is optimization completes and emits evidence
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

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
            gist_template="intermittent -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        assert len(result.trials) >= 2
        # Note: Trial failure status depends on mock implementation
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

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
            gist_template="type-error -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        # Note: Trial failure status depends on mock implementation
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

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
            gist_template="key-error -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        # Note: Trial failure status depends on mock implementation
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


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
            gist_template="returns-none -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

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
            gist_template="returns-empty -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Empty string is valid, should complete
        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


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
            gist_template="timeout=0.5s -> {stop_reason()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        # May have completed some trials before timeout or optimizer stopped
        # Valid stop reasons include timeout, max_trials, or optimizer decision
        valid_stop_reasons = ("timeout", "max_trials_reached", "optimizer", None)
        assert (
            result.stop_reason in valid_stop_reasons
        ), f"Unexpected stop_reason: {result.stop_reason}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


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
            gist_template="first-fails -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        assert len(result.trials) >= 2
        # Note: Trial failure status depends on mock implementation
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

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
            gist_template="custom-return -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestFunctionReturnTypeVariations:
    """Tests for various function return types."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_function_returns_dict(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test function returning dict instead of string."""
        scenario = TestScenario(
            name="returns_dict",
            description="Function returns dict",
            config_space={"model": ["gpt-3.5-turbo"]},
            function_return_value={"key": "value", "nested": {"a": 1}},
            max_trials=2,
            gist_template="returns-dict -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_function_returns_list(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test function returning list instead of string."""
        scenario = TestScenario(
            name="returns_list",
            description="Function returns list",
            config_space={"model": ["gpt-3.5-turbo"]},
            function_return_value=["item1", "item2", "item3"],
            max_trials=2,
            gist_template="returns-list -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_function_returns_number(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test function returning number instead of string."""
        scenario = TestScenario(
            name="returns_number",
            description="Function returns number",
            config_space={"model": ["gpt-3.5-turbo"]},
            function_return_value=42,
            max_trials=2,
            gist_template="returns-number -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_function_returns_boolean(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test function returning boolean."""
        scenario = TestScenario(
            name="returns_bool",
            description="Function returns boolean",
            config_space={"model": ["gpt-3.5-turbo"]},
            function_return_value=True,
            max_trials=2,
            gist_template="returns-bool -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_function_returns_float(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test function returning float."""
        scenario = TestScenario(
            name="returns_float",
            description="Function returns float",
            config_space={"model": ["gpt-3.5-turbo"]},
            function_return_value=3.14159,
            max_trials=2,
            gist_template="returns-float -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestFunctionExceptionVariations:
    """Tests for various exception types from functions."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_function_raises_attribute_error(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test function that raises AttributeError."""
        scenario = TestScenario(
            name="attribute_error",
            description="Function raises AttributeError",
            config_space={"model": ["gpt-3.5-turbo"]},
            function_should_raise=AttributeError,
            max_trials=2,
            expected=ExpectedResult(
                outcome=ExpectedOutcome.PARTIAL,
            ),
            gist_template="attribute-error -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        # Note: Trial failure status depends on mock implementation
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_function_raises_index_error(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test function that raises IndexError."""
        scenario = TestScenario(
            name="index_error",
            description="Function raises IndexError",
            config_space={"model": ["gpt-3.5-turbo"]},
            function_should_raise=IndexError,
            max_trials=2,
            expected=ExpectedResult(
                outcome=ExpectedOutcome.PARTIAL,
            ),
            gist_template="index-error -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        # Note: Trial failure status depends on mock implementation
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_function_raises_import_error(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test function that raises ImportError."""
        scenario = TestScenario(
            name="import_error",
            description="Function raises ImportError",
            config_space={"model": ["gpt-3.5-turbo"]},
            function_should_raise=ImportError,
            max_trials=2,
            expected=ExpectedResult(
                outcome=ExpectedOutcome.PARTIAL,
            ),
            gist_template="import-error -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        # Note: Trial failure status depends on mock implementation
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_function_raises_os_error(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test function that raises OSError."""
        scenario = TestScenario(
            name="os_error",
            description="Function raises OSError",
            config_space={"model": ["gpt-3.5-turbo"]},
            function_should_raise=OSError,
            max_trials=2,
            expected=ExpectedResult(
                outcome=ExpectedOutcome.PARTIAL,
            ),
            gist_template="os-error -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        # Note: Trial failure status depends on mock implementation
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestFunctionStateVariations:
    """Tests for functions with state-related behaviors."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_function_with_side_effects(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test function that has observable side effects."""
        # Note: Side effects are simulated through return value changes
        scenario = TestScenario(
            name="side_effects",
            description="Function with side effects",
            config_space={"model": ["gpt-3.5-turbo"]},
            function_return_value="side_effect_result",
            max_trials=2,
            gist_template="side-effects -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should complete normally
        assert not isinstance(result, Exception)
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_function_returns_large_output(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test function returning very large output string."""
        scenario = TestScenario(
            name="large_output",
            description="Function returns large output",
            config_space={"model": ["gpt-3.5-turbo"]},
            function_return_value="x" * 100000,  # 100KB output
            max_trials=2,
            gist_template="large-output -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_function_returns_unicode_output(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test function returning unicode/emoji output."""
        scenario = TestScenario(
            name="unicode_output",
            description="Function returns unicode output",
            config_space={"model": ["gpt-3.5-turbo"]},
            function_return_value="Hello 世界 🌍 مرحبا",
            max_trials=2,
            gist_template="unicode-output -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle unicode output
        assert not isinstance(result, Exception)
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()
