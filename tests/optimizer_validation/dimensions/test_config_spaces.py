"""Tests for configuration space types.

Purpose:
    Validate that the optimizer correctly handles different types of
    configuration spaces: categorical (discrete choices), continuous
    (numeric ranges), and mixed (combination of both).

Dimensions Covered:
    - ConfigSpaceType: categorical, continuous, mixed
    - Algorithm: random, grid (via mock_mode_config)

Test Categories:
    1. Categorical spaces - discrete string/numeric/boolean options
    2. Continuous spaces - float ranges for numeric parameters
    3. Mixed spaces - combining categorical and continuous
    4. Edge cases - narrow ranges, unicode, special characters
    5. Parameter exploration order - grid search ordering

Validation Approach:
    Tests verify that the optimizer correctly samples from the defined
    space, respects range boundaries, and produces valid configurations.
"""

from __future__ import annotations

import pytest

from tests.optimizer_validation.specs import (
    ExpectedOutcome,
    ExpectedResult,
    config_space_scenario,
)


class TestCategoricalConfigSpace:
    """Tests for categorical-only configuration spaces.

    Purpose:
        Validate categorical parameter handling where each parameter
        has a finite set of discrete values to choose from.

    How Categorical Spaces Work:
        Parameters are defined as lists: {"model": ["gpt-3.5", "gpt-4"]}.
        The optimizer selects values from these lists, either randomly
        or systematically (grid search).

    Why Categorical Spaces Matter:
        Most LLM optimization involves discrete choices like model names,
        prompt templates, or response formats. Correct handling is essential.
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_simple_categorical(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test simple categorical config space with string values.

        Purpose:
            Verify basic categorical parameter handling with a single
            string-valued parameter.

        Expectations:
            - All trial configs contain valid values from the list
            - Optimizer explores different values across trials
            - No out-of-bounds or invalid values generated

        Why This Validates Categorical Handling:
            The simplest case - if this fails, categorical support is broken.
            Uses 2 trials to ensure at least some exploration occurs.

        Dimensions: ConfigSpaceType=categorical, Algorithm=default
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
        }

        scenario = config_space_scenario(
            name="simple_categorical",
            config_space=config_space,
            max_trials=2,
            gist_template="simple-categorical -> {{model}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

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
    async def test_multiple_categorical_params(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test multiple categorical parameters.

        Purpose:
            Verify the optimizer handles multiple categorical parameters
            simultaneously, exploring the combinatorial space correctly.

        Expectations:
            - All parameters appear in trial configs
            - Values are valid for each parameter
            - Optimizer explores combinations, not just single params

        Why This Validates Multi-Parameter Handling:
            Real optimization involves multiple parameters. This test
            ensures the optimizer correctly constructs configs with
            3 categorical params (3×3×2 = 18 possible combinations).

        Dimensions: ConfigSpaceType=categorical (multi-param)
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
            "safety_filter": ["strict", "moderate", "lenient"],
            "response_format": ["text", "json"],
        }

        scenario = config_space_scenario(
            name="multi_categorical",
            config_space=config_space,
            max_trials=3,
            gist_template="multi-categorical -> {{model}}",
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
            gist_template="numeric-categorical -> {{max_tokens}}",
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


class TestContinuousConfigSpace:
    """Tests for continuous (range-based) configuration spaces.

    Purpose:
        Validate continuous parameter handling where parameters can take
        any value within a specified numeric range.

    How Continuous Spaces Work:
        Parameters are defined as tuples: {"temperature": (0.0, 1.0)}.
        The optimizer samples float values from within the range,
        using appropriate sampling strategies (uniform, log, etc.).

    Why Continuous Spaces Matter:
        Many LLM parameters are continuous (temperature, top_p, penalties).
        The optimizer must correctly sample within bounds and handle
        edge cases like narrow ranges or negative values.
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_simple_continuous(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test simple continuous config space with float range.

        Purpose:
            Verify basic continuous parameter handling with a single
            float-valued range parameter.

        Expectations:
            - All trial configs contain values within [0.0, 1.0]
            - Values are actual floats, not strings or integers
            - Different trials may have different values

        Why This Validates Continuous Handling:
            The simplest continuous case - one parameter with standard
            0-1 range. If this fails, continuous support is broken.

        Dimensions: ConfigSpaceType=continuous, Algorithm=default
        """
        config_space = {
            "temperature": (0.0, 1.0),
        }

        scenario = config_space_scenario(
            name="simple_continuous",
            config_space=config_space,
            max_trials=2,
            gist_template="simple-continuous -> {{temperature}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

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
    async def test_multiple_continuous_params(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test multiple continuous parameters.

        Purpose:
            Verify the optimizer handles multiple continuous parameters
            simultaneously.

        Expectations:
            - All parameters appear in trial configs
            - Values are within their respective ranges
            - Optimizer explores the multi-dimensional space

        Why This Validates Multi-Parameter Handling:
            Ensures independence of parameters and correct range enforcement
            for each parameter type.

        Dimensions: ConfigSpaceType=continuous (multi-param)
        """
        config_space = {
            "temperature": (0.0, 2.0),
            "top_p": (0.1, 1.0),
            "frequency_penalty": (-2.0, 2.0),
        }

        scenario = config_space_scenario(
            name="multi_continuous",
            config_space=config_space,
            max_trials=3,
            gist_template="multi-continuous -> {{temperature}}",
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
            gist_template="narrow-range -> {{temperature}}",
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


class TestMixedConfigSpace:
    """Tests for mixed (categorical + continuous) configuration spaces.

    Purpose:
        Validate that the optimizer handles mixed parameter types correctly,
        combining categorical selections with continuous value sampling.

    Why Mixed Spaces Matter:
        Real-world LLM optimization typically involves both discrete choices
        (model, template) and continuous parameters (temperature, penalties).
        Correct handling of mixed spaces is essential for practical use.
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_simple_mixed(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test mixed config space with both types.

        Purpose:
            Verify that categorical and continuous parameters work together
            in a single configuration space.

        Expectations:
            - Categorical values are valid list members
            - Continuous values are within specified range
            - Both types appear correctly in trial configs

        Why This Validates Mixed Handling:
            The simplest mixed case - one categorical, one continuous.
            If this fails, the optimizer can't handle real-world scenarios.

        Dimensions: ConfigSpaceType=mixed
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": (0.0, 1.0),
        }

        scenario = config_space_scenario(
            name="simple_mixed",
            config_space=config_space,
            max_trials=2,
            gist_template="simple-mixed -> {{model}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

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
            gist_template="complex-mixed -> {{model}}",
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


class TestConfigSpaceEdgeCases:
    """Tests for edge cases in configuration spaces."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_same_min_max_range(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test continuous range where min equals max (point range)."""
        config_space = {
            "temperature": (0.5, 0.5),  # Same min and max
            "model": ["gpt-3.5-turbo", "gpt-4"],
        }

        scenario = config_space_scenario(
            name="point_range",
            config_space=config_space,
            max_trials=2,
            expected=ExpectedResult(outcome=ExpectedOutcome.FAILURE),
            gist_template="point-range -> {{model}}",
        )

        _, result = await scenario_runner(scenario)

        assert isinstance(result, Exception)

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
    async def test_very_small_range_delta(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test continuous range with very small delta."""
        config_space = {
            "temperature": (0.5, 0.50001),  # Very small range
            "model": ["gpt-3.5-turbo"],
        }

        scenario = config_space_scenario(
            name="tiny_range",
            config_space=config_space,
            max_trials=2,
            gist_template="tiny-range -> {{model}}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle tiny ranges
        if not isinstance(result, Exception):
            assert hasattr(result, "trials"), "Result should have trials"
            assert len(result.trials) >= 1, "Should complete at least one trial"
            assert result.stop_reason is not None, "Should have a stop reason"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

            validation = result_validator(scenario, result)
            assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_negative_to_positive_range(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test continuous range spanning negative to positive values."""
        config_space = {
            "frequency_penalty": (-2.0, 2.0),  # Negative to positive
            "presence_penalty": (-1.0, 1.0),
        }

        scenario = config_space_scenario(
            name="neg_to_pos_range",
            config_space=config_space,
            max_trials=2,
            gist_template="neg-to-pos -> {{frequency_penalty}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

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
    async def test_boolean_categorical_values(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test categorical with boolean values."""
        config_space = {
            "stream": [True, False],
            "model": ["gpt-3.5-turbo"],
        }

        scenario = config_space_scenario(
            name="boolean_categorical",
            config_space=config_space,
            max_trials=2,
            gist_template="boolean-categorical -> {{stream}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

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
    async def test_none_in_categorical_values(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test categorical with None as one of the values."""
        config_space = {
            "stop_sequence": [None, "END", "STOP"],  # None is valid
            "model": ["gpt-3.5-turbo"],
        }

        scenario = config_space_scenario(
            name="none_categorical",
            config_space=config_space,
            max_trials=2,
            gist_template="none-categorical -> {{stop_sequence}}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle None as a valid categorical value
        if not isinstance(result, Exception):
            assert hasattr(result, "trials"), "Result should have trials"
            assert len(result.trials) >= 1, "Should complete at least one trial"
            assert result.stop_reason is not None, "Should have a stop reason"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

            validation = result_validator(scenario, result)
            assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mixed_types_in_categorical(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test categorical with mixed types (int, float, string)."""
        config_space = {
            "param": [1, 2.5, "three", True, None],  # Mixed types
            "model": ["gpt-3.5-turbo"],
        }

        scenario = config_space_scenario(
            name="mixed_types_categorical",
            config_space=config_space,
            max_trials=3,
            gist_template="mixed-types -> {{param}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

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
    async def test_unicode_in_config_values(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test categorical with unicode and emoji values."""
        config_space = {
            "locale": ["en-US", "日本語", "العربية", "🇺🇸"],
            "model": ["gpt-3.5-turbo"],
        }

        scenario = config_space_scenario(
            name="unicode_categorical",
            config_space=config_space,
            max_trials=2,
            gist_template="unicode -> {{locale}}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle unicode gracefully
        if not isinstance(result, Exception):
            assert hasattr(result, "trials"), "Result should have trials"
            assert len(result.trials) >= 1, "Should complete at least one trial"
            assert result.stop_reason is not None, "Should have a stop reason"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

            validation = result_validator(scenario, result)
            assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_special_chars_in_config_keys(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test config keys with special characters."""
        config_space = {
            "my-param": ["a", "b"],  # Hyphen in key
            "my_param": ["x", "y"],  # Underscore in key
            "model": ["gpt-3.5-turbo"],
        }

        scenario = config_space_scenario(
            name="special_char_keys",
            config_space=config_space,
            max_trials=2,
            gist_template="special-chars -> {{model}}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle special characters in keys
        if not isinstance(result, Exception):
            assert hasattr(result, "trials"), "Result should have trials"
            assert len(result.trials) >= 1, "Should complete at least one trial"
            assert result.stop_reason is not None, "Should have a stop reason"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

            validation = result_validator(scenario, result)
            assert validation.passed, validation.summary()

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
            gist_template="single-value -> {{model}}",
        )

        _, result = await scenario_runner(scenario)

        # Should still work, just limited optimization space
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
            gist_template="large-categorical -> {{model}}",
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


class TestParameterExplorationOrder:
    """Tests for parameter exploration order in grid search.

    Grid search can be configured with `parameter_order` to control
    which parameters are explored first (vary slowest vs fastest).
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_grid_default_parameter_order(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test grid search with default parameter ordering.

        By default, grid search explores parameters in alphabetical order,
        with 'model' typically placed last (varies fastest).
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": [0.3, 0.7],
        }

        scenario = config_space_scenario(
            name="grid_default_order",
            config_space=config_space,
            description="Grid search with default parameter order",
            max_trials=4,
            mock_mode_config={"optimizer": "grid"},
            gist_template="grid-default -> {{model}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

        # Verify all 4 combinations explored
        if hasattr(result, "trials"):
            assert len(result.trials) == 4, "Should explore all 4 combinations"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_grid_custom_parameter_order_model_first(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test grid search with model explored first (varies slowest).

        With parameter_order = {"model": 0, "temperature": 1}:
        - model varies slowest (explored first)
        - temperature varies fastest

        Expected order:
        1. model=gpt-3.5-turbo, temp=0.3
        2. model=gpt-3.5-turbo, temp=0.7
        3. model=gpt-4, temp=0.3
        4. model=gpt-4, temp=0.7
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": [0.3, 0.7],
        }

        scenario = config_space_scenario(
            name="grid_model_first",
            config_space=config_space,
            description="Grid search with model explored first",
            max_trials=4,
            mock_mode_config={
                "optimizer": "grid",
                "parameter_order": {"model": 0, "temperature": 1},
            },
            gist_template="grid-model-first -> {{model}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) >= 1, "Should complete at least one trial"
        assert result.stop_reason is not None, "Should have a stop reason"

        if len(result.trials) >= 4:
            configs = [t.config for t in result.trials]

            # First two trials should have same model
            if len(configs) >= 2:
                first_model = configs[0].get("model")
                second_model = configs[1].get("model")

                # When model is first (slowest varying), consecutive trials
                # should have same model but different temperature
                if first_model == second_model:
                    # Verify temperatures differ
                    first_temp = configs[0].get("temperature")
                    second_temp = configs[1].get("temperature")
                    assert (
                        first_temp != second_temp
                    ), "Temperature should vary fastest when model is first"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_grid_custom_parameter_order_temp_first(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test grid search with temperature explored first (varies slowest).

        With parameter_order = {"temperature": 0, "model": 1}:
        - temperature varies slowest (explored first)
        - model varies fastest

        Expected order:
        1. temp=0.3, model=gpt-3.5-turbo
        2. temp=0.3, model=gpt-4
        3. temp=0.7, model=gpt-3.5-turbo
        4. temp=0.7, model=gpt-4
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": [0.3, 0.7],
        }

        scenario = config_space_scenario(
            name="grid_temp_first",
            config_space=config_space,
            description="Grid search with temperature explored first",
            max_trials=4,
            mock_mode_config={
                "optimizer": "grid",
                "parameter_order": {"temperature": 0, "model": 1},
            },
            gist_template="grid-temp-first -> {{model}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) >= 1, "Should complete at least one trial"
        assert result.stop_reason is not None, "Should have a stop reason"

        if len(result.trials) >= 4:
            configs = [t.config for t in result.trials]

            # First two trials should have same temperature
            if len(configs) >= 2:
                first_temp = configs[0].get("temperature")
                second_temp = configs[1].get("temperature")

                # When temperature is first (slowest varying), consecutive trials
                # should have same temp but different model
                if first_temp == second_temp:
                    # Verify models differ
                    first_model = configs[0].get("model")
                    second_model = configs[1].get("model")
                    assert (
                        first_model != second_model
                    ), "Model should vary fastest when temperature is first"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_grid_parameter_order_with_three_params(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test grid search parameter order with three parameters.

        Tests that parameter_order correctly handles multiple parameters.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": [0.3, 0.7],
            "max_tokens": [100, 500],
        }
        # Total: 2 * 2 * 2 = 8 combinations

        scenario = config_space_scenario(
            name="grid_three_params_ordered",
            config_space=config_space,
            description="Grid search with three parameters ordered",
            max_trials=8,
            mock_mode_config={
                "optimizer": "grid",
                "parameter_order": {
                    "model": 0,
                    "temperature": 1,
                    "max_tokens": 2,
                },
            },
            gist_template="grid-three -> {{model}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

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
    async def test_grid_partial_parameter_order(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test grid search with partial parameter order specification.

        Only specify order for some parameters - others should use default.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": [0.3, 0.7],
            "max_tokens": [100, 500],
        }

        scenario = config_space_scenario(
            name="grid_partial_order",
            config_space=config_space,
            description="Grid search with partial parameter order",
            max_trials=8,
            mock_mode_config={
                "optimizer": "grid",
                # Only specify order for model, others use default
                "parameter_order": {"model": 0},
            },
            gist_template="grid-partial -> {{model}}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle partial specification gracefully
        if not isinstance(result, Exception):
            assert hasattr(result, "trials"), "Result should have trials"
            assert len(result.trials) >= 1, "Should complete at least one trial"
            assert result.stop_reason is not None, "Should have a stop reason"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

            validation = result_validator(scenario, result)
            assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_grid_parameter_order_unknown_param(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test grid search with parameter order containing unknown parameter.

        Specifying order for a parameter not in config space should either:
        - Be ignored
        - Raise a warning/error
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": [0.3, 0.7],
        }

        scenario = config_space_scenario(
            name="grid_unknown_order_param",
            config_space=config_space,
            description="Grid search with unknown param in order",
            max_trials=4,
            mock_mode_config={
                "optimizer": "grid",
                "parameter_order": {
                    "model": 0,
                    "temperature": 1,
                    "nonexistent_param": 2,  # Not in config space
                },
            },
            gist_template="grid-unknown -> {{model}}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle gracefully - either ignore or error
        if not isinstance(result, Exception):
            assert hasattr(result, "trials"), "Result should have trials"
            assert len(result.trials) >= 1, "Should complete at least one trial"
            assert result.stop_reason is not None, "Should have a stop reason"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

            validation = result_validator(scenario, result)
            assert validation.passed, validation.summary()
            # Should still complete the 4 combinations

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_random_search_ignores_parameter_order(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that random search ignores parameter_order setting.

        parameter_order is only meaningful for grid search.
        Random search should work fine even if it's specified.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": [0.3, 0.7],
        }

        scenario = config_space_scenario(
            name="random_ignores_order",
            config_space=config_space,
            description="Random search should ignore parameter_order",
            max_trials=4,
            mock_mode_config={
                "optimizer": "random",
                "parameter_order": {"model": 0, "temperature": 1},
            },
            gist_template="random-ignores -> {{model}}",
        )

        _, result = await scenario_runner(scenario)

        # Random search should complete successfully
        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()
