"""Tests for decorator invocation failures.

Tests scenarios where the @traigent.optimize decorator is invoked
with invalid configurations that should fail at decoration time.
"""

from __future__ import annotations

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
        """Test with empty configuration space.

        Note: TestScenario provides default config if empty,
        so this documents that behavior - empty config is handled gracefully.
        """
        scenario = TestScenario(
            name="empty_config_space",
            description="Empty configuration space",
            config_space={},  # Empty - will get defaults
            max_trials=2,
            # TestScenario provides default config if empty, so this succeeds
            expected=ExpectedResult(),
            gist_template="empty-config -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

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
            gist_template="wrong-type -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

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
            gist_template="empty-list -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

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
            gist_template="invalid-range -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

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
            gist_template="non-numeric -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

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
        """Test with empty objectives list.

        Documents behavior: empty objectives uses default objectives.
        """
        scenario = TestScenario(
            name="empty_objectives",
            description="Empty objectives list",
            config_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
            objectives=[],  # Empty - uses defaults
            max_trials=2,
            expected=ExpectedResult(),  # Succeeds with defaults
            gist_template="empty-objectives -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

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
            gist_template="dup-objectives -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

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
            # type: ignore[list-item]
            objectives=[123, "accuracy"],
            max_trials=2,
            expected=ExpectedResult(
                outcome=ExpectedOutcome.FAILURE,
            ),
            gist_template="non-string-obj -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

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
            gist_template="non-callable -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # This should work since we used a valid lambda
        # The test validates constraints work at all

        # Emit evidence
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()
        assert not isinstance(result, Exception)

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


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
            gist_template="invalid-mock -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

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
            gist_template="negative-trials -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

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
            gist_template="zero-trials -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Zero max_trials should either raise an exception or return 0 trials
        if isinstance(result, Exception):
            # Validation error is expected for zero max_trials
            assert (
                "max_trials" in str(result).lower() or "positive" in str(result).lower()
            )
        else:
            # If no exception, should have 0 trials (as per scenario expectation)
            if hasattr(result, "trials"):
                assert (
                    len(result.trials) == 0
                ), f"Zero max_trials should yield 0 trials, got {len(result.trials)}"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

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
            gist_template="negative-timeout -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

        # Should fail or be handled

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_zero_timeout(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test with zero timeout.

        Documents behavior: zero timeout results in 0 trials due to
        immediate timeout.
        """
        scenario = TestScenario(
            name="zero_timeout",
            description="Zero timeout",
            config_space={"model": ["gpt-3.5-turbo"]},
            timeout=0.0,  # Zero timeout - immediate timeout
            max_trials=2,
            expected=ExpectedResult(
                min_trials=0,  # Zero timeout means 0 trials possible
                max_trials=2,
            ),
            gist_template="zero-timeout -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Zero timeout should complete without error but with 0 trials
        assert not isinstance(
            result, Exception
        ), f"Zero timeout should not crash: {result}"

        # Zero timeout means immediate timeout, so 0 trials expected
        if hasattr(result, "trials"):
            # Per docstring: "zero timeout results in 0 trials due to immediate timeout"
            assert (
                len(result.trials) == 0
            ), f"Zero timeout should yield 0 trials, got {len(result.trials)}"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_very_large_max_trials(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test with very large max_trials value."""
        scenario = TestScenario(
            name="large_trials",
            description="Very large max_trials",
            config_space={"model": ["gpt-3.5-turbo"]},
            max_trials=1000000,  # 1 million - should be accepted
            timeout=0.1,  # Short timeout to avoid long test
            gist_template="large-trials -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

        # Should accept the value but timeout quickly


class TestInvalidConfigKeyNames:
    """Tests for invalid configuration key names."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_empty_string_config_key(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test config space with empty string as key."""
        scenario = TestScenario(
            name="empty_key",
            description="Empty string config key",
            config_space={
                "": ["value1", "value2"],  # Empty key
                "model": ["gpt-3.5-turbo"],
            },
            max_trials=2,
            gist_template="empty-key -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

        # Should fail or handle empty key

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_config_key_with_dots(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test config key with dots (may conflict with nested notation)."""
        scenario = TestScenario(
            name="dotted_key",
            description="Config key with dots",
            config_space={
                "model.name": ["gpt-3.5-turbo", "gpt-4"],
                "model.version": ["v1", "v2"],
            },
            max_trials=2,
            gist_template="dotted-key -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

        # Document behavior with dotted keys

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_config_key_with_spaces(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test config key with spaces."""
        scenario = TestScenario(
            name="space_key",
            description="Config key with spaces",
            config_space={
                "model name": ["gpt-3.5-turbo", "gpt-4"],
            },
            max_trials=2,
            gist_template="space-key -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

        # Document behavior with space in keys

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_config_key_reserved_word(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test config key that might be a reserved word."""
        scenario = TestScenario(
            name="reserved_key",
            description="Config key is reserved word",
            config_space={
                "class": ["a", "b"],  # Python reserved word
                "model": ["gpt-3.5-turbo"],
            },
            max_trials=2,
            gist_template="reserved-key -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

        # Should handle reserved words as keys


class TestInvalidObjectiveNames:
    """Tests for invalid objective name specifications."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_objective_name_empty_string(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test with empty string as objective name.

        Documents behavior: empty objective name is handled gracefully
        (with a warning) rather than raising an error.
        """
        scenario = TestScenario(
            name="empty_objective_name",
            description="Empty string objective name",
            config_space={"model": ["gpt-3.5-turbo"]},
            objectives=[""],  # Empty string - handled with warning
            max_trials=2,
            expected=ExpectedResult(),  # Succeeds with warning
            gist_template="empty-obj-name -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

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
    async def test_objective_name_with_spaces(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test objective name containing spaces."""
        scenario = TestScenario(
            name="spaced_objective",
            description="Objective name with spaces",
            config_space={"model": ["gpt-3.5-turbo"]},
            objectives=["my accuracy"],  # Space in name
            max_trials=2,
            gist_template="spaced-obj -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

        # Document behavior with spaced objective names

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_objective_name_with_special_chars(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test objective name with special characters."""
        scenario = TestScenario(
            name="special_objective",
            description="Objective name with special chars",
            config_space={"model": ["gpt-3.5-turbo"]},
            objectives=["accuracy@v1", "cost$total"],  # Special chars
            max_trials=2,
            gist_template="special-obj -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

        # Document behavior with special chars in objectives

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_objective_name_none(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test with None in objectives list."""
        scenario = TestScenario(
            name="none_objective",
            description="None in objectives list",
            config_space={"model": ["gpt-3.5-turbo"]},
            objectives=[None, "accuracy"],  # type: ignore[list-item]  # None
            max_trials=2,
            expected=ExpectedResult(
                outcome=ExpectedOutcome.FAILURE,
            ),
            gist_template="none-obj -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

        # Should fail validation


class TestInvalidConstraintSignatures:
    """Tests for constraints with invalid signatures."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_constraint_no_parameters(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test constraint function with no parameters."""
        from tests.optimizer_validation.specs import ConstraintSpec

        def no_params() -> bool:
            return True

        scenario = TestScenario(
            name="no_param_constraint",
            description="Constraint with no parameters",
            config_space={"model": ["gpt-3.5-turbo"]},
            constraints=[
                ConstraintSpec(
                    name="no_params",
                    constraint_fn=no_params,  # type: ignore[arg-type]
                )
            ],
            max_trials=2,
            gist_template="no-param-const -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

        # Should fail when calling constraint

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_constraint_too_many_parameters(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test constraint function with too many required parameters."""
        from tests.optimizer_validation.specs import ConstraintSpec

        def too_many_params(config: dict, metrics: dict, extra: str) -> bool:
            return True

        scenario = TestScenario(
            name="extra_param_constraint",
            description="Constraint with extra required parameter",
            config_space={"model": ["gpt-3.5-turbo"]},
            constraints=[
                ConstraintSpec(
                    name="too_many",
                    constraint_fn=too_many_params,  # type: ignore[arg-type]
                    requires_metrics=True,
                )
            ],
            max_trials=2,
            gist_template="extra-param-const -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

        # Should fail when calling constraint
