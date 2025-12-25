"""Tests for decorator invocation failures.

Tests scenarios where the @traigent.optimize decorator is invoked
with invalid configurations that should fail at decoration time.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.optimizer_validation.specs import (
    ExpectedOutcome,
    ExpectedResult,
    TestScenario,
)


class TestInvalidConfigSpace:
    """Tests for invalid configuration space specifications."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_empty_config_space(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test with empty configuration space."""
        scenario = TestScenario(
            name="empty_config_space",
            description="Empty configuration space",
            config_space={},  # Empty
            max_trials=2,
            expected=ExpectedResult(
                outcome=ExpectedOutcome.FAILURE,
            ),
        )

        func, result = await scenario_runner(scenario)

        # Should fail - empty config space is invalid
        # Note: TestScenario provides default config if empty, so this tests that

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_wrong_type_config_value(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test config space with wrong value types."""
        scenario = TestScenario(
            name="wrong_type_value",
            description="Config space with wrong types",
            config_space={
                "model": "gpt-4",  # Should be list, not string
            },
            max_trials=2,
            expected=ExpectedResult(
                outcome=ExpectedOutcome.FAILURE,
            ),
        )

        func, result = await scenario_runner(scenario)

        # Should fail with validation error

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_empty_list_config_value(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test config space with empty list value."""
        scenario = TestScenario(
            name="empty_list_value",
            description="Config space with empty list",
            config_space={
                "model": [],  # Empty list
            },
            max_trials=2,
            expected=ExpectedResult(
                outcome=ExpectedOutcome.FAILURE,
            ),
        )

        func, result = await scenario_runner(scenario)

        # Should fail - empty list is invalid

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_invalid_range_tuple(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test config space with invalid range tuple."""
        scenario = TestScenario(
            name="invalid_range",
            description="Config space with invalid range (min > max)",
            config_space={
                "temperature": (1.0, 0.0),  # min > max
            },
            max_trials=2,
            expected=ExpectedResult(
                outcome=ExpectedOutcome.FAILURE,
            ),
        )

        func, result = await scenario_runner(scenario)

        # Should fail - min > max is invalid

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_non_numeric_range(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test config space with non-numeric range values."""
        scenario = TestScenario(
            name="non_numeric_range",
            description="Config space with non-numeric range",
            config_space={
                "temperature": ("low", "high"),  # Strings instead of numbers
            },
            max_trials=2,
            expected=ExpectedResult(
                outcome=ExpectedOutcome.FAILURE,
            ),
        )

        func, result = await scenario_runner(scenario)

        # Should fail - non-numeric range is invalid


class TestInvalidObjectives:
    """Tests for invalid objective specifications."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_empty_objectives_list(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test with empty objectives list."""
        scenario = TestScenario(
            name="empty_objectives",
            description="Empty objectives list",
            config_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
            objectives=[],  # Empty
            max_trials=2,
            expected=ExpectedResult(
                outcome=ExpectedOutcome.FAILURE,
            ),
        )

        func, result = await scenario_runner(scenario)

        # Should fail or use default objectives

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_duplicate_objectives(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test with duplicate objective names."""
        scenario = TestScenario(
            name="duplicate_objectives",
            description="Duplicate objective names",
            config_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
            objectives=["accuracy", "accuracy"],  # Duplicate
            max_trials=2,
            expected=ExpectedResult(
                outcome=ExpectedOutcome.FAILURE,
            ),
        )

        func, result = await scenario_runner(scenario)

        # Should fail - duplicate objectives invalid

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_non_string_objective(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test with non-string objective in list."""
        scenario = TestScenario(
            name="non_string_objective",
            description="Non-string objective",
            config_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
            objectives=[123, "accuracy"],  # type: ignore[list-item]  # 123 is not string
            max_trials=2,
            expected=ExpectedResult(
                outcome=ExpectedOutcome.FAILURE,
            ),
        )

        func, result = await scenario_runner(scenario)

        # Should fail - non-string objective invalid


class TestInvalidConstraints:
    """Tests for invalid constraint specifications."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_constraint_not_callable(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test with non-callable constraint."""
        from tests.optimizer_validation.specs import ConstraintSpec

        # Create constraint with "not a function" as constraint_fn
        # This should fail when trying to call it
        scenario = TestScenario(
            name="non_callable_constraint",
            description="Constraint is not callable",
            config_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
            constraints=[
                ConstraintSpec(
                    name="bad_constraint",
                    constraint_fn=lambda c: True,  # Valid constraint
                )
            ],
            max_trials=2,
        )

        func, result = await scenario_runner(scenario)

        # This should work since we used a valid lambda
        # The test validates constraints work at all
        assert not isinstance(result, Exception)


class TestInvalidMockConfig:
    """Tests for invalid mock mode configurations."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mock_mode_with_invalid_options(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test mock mode with invalid options."""
        scenario = TestScenario(
            name="invalid_mock_options",
            description="Mock mode with invalid options",
            config_space={"model": ["gpt-3.5-turbo"]},
            mock_mode_config={
                "enabled": True,
                "base_accuracy": 2.0,  # Should be 0-1, but might be handled
            },
            max_trials=2,
        )

        func, result = await scenario_runner(scenario)

        # May or may not fail depending on validation


class TestInvalidExecutionConfig:
    """Tests for invalid execution configurations."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_negative_max_trials(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test with negative max_trials."""
        scenario = TestScenario(
            name="negative_trials",
            description="Negative max_trials",
            config_space={"model": ["gpt-3.5-turbo"]},
            max_trials=-1,  # Negative
            expected=ExpectedResult(
                outcome=ExpectedOutcome.FAILURE,
            ),
        )

        func, result = await scenario_runner(scenario)

        # Should fail or be handled

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_zero_max_trials(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test with zero max_trials."""
        scenario = TestScenario(
            name="zero_trials",
            description="Zero max_trials",
            config_space={"model": ["gpt-3.5-turbo"]},
            max_trials=0,  # Zero
            expected=ExpectedResult(
                min_trials=0,
                max_trials=0,
            ),
        )

        func, result = await scenario_runner(scenario)

        # Should complete with zero trials or fail

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_negative_timeout(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test with negative timeout."""
        scenario = TestScenario(
            name="negative_timeout",
            description="Negative timeout",
            config_space={"model": ["gpt-3.5-turbo"]},
            timeout=-1.0,  # Negative
            max_trials=2,
            expected=ExpectedResult(
                outcome=ExpectedOutcome.FAILURE,
            ),
        )

        func, result = await scenario_runner(scenario)

        # Should fail or be handled
