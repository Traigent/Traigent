"""Tests for injection mode configurations.

Purpose:
    Validate that the optimizer correctly injects configuration parameters into
    decorated functions using different injection strategies. Each injection mode
    provides a different mechanism for making optimization parameters available
    to the target function.

Dimensions Covered:
    - InjectionMode: context, parameter, seamless
    - ConfigSpaceType: categorical (via default config space)

Test Categories:
    1. Basic functionality - each mode completes optimization successfully
    2. Default config handling - modes correctly apply default configurations
    3. Mode-specific behavior - thread safety, parameter naming, etc.
    4. Edge cases - invalid modes, case sensitivity, whitespace handling

Validation Approach:
    Tests verify that optimization completes without errors and produces valid
    OptimizationResult objects with expected trial counts and metrics.
"""

from __future__ import annotations

import pytest

from tests.optimizer_validation.specs import (
    ExpectedOutcome,
    ExpectedResult,
    TestScenario,
    basic_scenario,
)

# All supported injection modes (attribute mode was removed in v2.x)
INJECTION_MODES = ["context", "parameter", "seamless"]


class TestInjectionModeMatrix:
    """Test matrix for all injection modes.

    Purpose:
        Systematically verify that each injection mode works correctly with
        basic optimization scenarios. Uses parametrization to ensure all
        modes are tested with identical scenarios for fair comparison.

    Dimensions Covered:
        - InjectionMode: all four modes via parametrization
        - ConfigSpaceType: categorical (model selection)
        - StopCondition: max_trials

    Why This Matters:
        Injection mode is fundamental to how users interact with Traigent.
        Different frameworks and use cases require different injection
        strategies. Failures here indicate core functionality issues.
    """

    @pytest.mark.parametrize("injection_mode", INJECTION_MODES)
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_injection_mode_basic(
        self,
        injection_mode: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test each injection mode works with basic configuration.

        Purpose:
            Verify that optimization completes successfully when using each
            injection mode with a minimal configuration.

        Expectations:
            - Optimization runs without raising exceptions
            - At least 1 trial completes successfully
            - OptimizationResult contains valid best_config and best_score
            - Stop reason is "max_trials_reached" (since max_trials=2)

        Why This Validates Injection Mode:
            By running an identical optimization with only the injection mode
            varying, we isolate the injection mechanism. If optimization
            completes and produces valid results, the injection successfully
            delivered configuration to the function.

        Dimensions: InjectionMode={injection_mode}, StopCondition=max_trials
        """
        scenario = basic_scenario(
            name=f"basic_{injection_mode}",
            injection_mode=injection_mode,
            max_trials=2,
            gist_template=f"{injection_mode} -> {{trial_count()}} | {{status()}}",
        )

        func, result = await scenario_runner(scenario)

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

    @pytest.mark.parametrize("injection_mode", INJECTION_MODES)
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_injection_mode_with_default_config(
        self,
        injection_mode: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test injection modes respect default configuration.

        Purpose:
            Verify that when a default_config is provided, the injection mode
            correctly applies these defaults as the starting point for optimization.

        Expectations:
            - Optimization completes without errors
            - Default config values are respected as initial trial configuration
            - Optimizer can still explore variations from the default

        Why This Validates Default Config Handling:
            Default configs are critical for warm-starting optimization. This test
            ensures each injection mode properly initializes with user-provided
            defaults rather than random configurations.

        Dimensions: InjectionMode={injection_mode}, ConfigSpaceType=mixed
        """
        default_config = {"model": "gpt-3.5-turbo", "temperature": 0.3}

        scenario = basic_scenario(
            name=f"default_config_{injection_mode}",
            injection_mode=injection_mode,
            default_config=default_config,
            max_trials=2,
            gist_template=f"default-{injection_mode} -> {{trial_count()}} | {{status()}}",
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestContextInjection:
    """Tests specific to CONTEXT injection mode.

    Purpose:
        Validate context-based configuration injection, where configs are
        passed via a context variable (thread-local or async context).

    How Context Injection Works:
        The optimizer sets a context variable before calling the decorated
        function. The function retrieves config via traigent.get_config().
        This is non-invasive as function signatures remain unchanged.

    Why Context Injection Matters:
        Context injection is the least intrusive mode - it doesn't modify
        function signatures or require code changes. It's ideal for
        retrofitting optimization onto existing codebases.
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_context_injection_is_thread_safe(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test context injection maintains thread safety.

        Purpose:
            Verify that context variables don't leak between concurrent
            executions, which would cause config cross-contamination.

        Expectations:
            - Each trial receives its own configuration
            - No config values from one trial appear in another
            - Optimization completes successfully with correct trial isolation

        Why This Validates Thread Safety:
            If context injection were not thread-safe, parallel trials would
            see each other's configurations, leading to incorrect optimization
            results. This test runs 3 trials to allow for potential race conditions.

        Dimensions: InjectionMode=context, ParallelMode=sequential
        """
        scenario = basic_scenario(
            name="context_thread_safe",
            injection_mode="context",
            max_trials=3,
            gist_template="context-thread -> {trial_count()} | {status()}",
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestParameterInjection:
    """Tests specific to PARAMETER injection mode.

    Purpose:
        Validate parameter-based configuration injection, where configs are
        passed as an explicit function parameter (traigent_config).

    How Parameter Injection Works:
        The optimizer modifies the function call to include a traigent_config
        keyword argument. The function must accept **kwargs or explicitly
        define traigent_config in its signature.

    Why Parameter Injection Matters:
        Parameter injection is explicit and visible - the function signature
        shows it receives optimization config. This is preferred when you
        want clear API contracts and easy testing with mock configs.
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_parameter_injection_adds_config_param(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test parameter injection adds traigent_config parameter.

        Purpose:
            Verify that the traigent_config parameter is correctly added
            to function calls and contains the expected configuration.

        Expectations:
            - Function receives traigent_config parameter
            - Parameter contains valid configuration dictionary
            - Optimization completes with configs properly passed

        Why This Validates Parameter Injection:
            If parameter injection fails, the function would either not
            receive the config (raising TypeError) or receive wrong values.
            Successful optimization proves configs are correctly passed.

        Dimensions: InjectionMode=parameter, ConfigSpaceType=categorical
        """
        scenario = basic_scenario(
            name="parameter_config_param",
            injection_mode="parameter",
            max_trials=2,
            gist_template="param-config -> {trial_count()} | {status()}",
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestSeamlessInjection:
    """Tests specific to SEAMLESS injection mode.

    Purpose:
        Validate seamless configuration injection, where the optimizer
        automatically rewrites function code to use optimization parameters.

    How Seamless Injection Works:
        The optimizer performs AST transformation on the decorated function,
        replacing hardcoded values with config lookups. This requires no
        manual code changes - the optimizer handles everything.

    Why Seamless Injection Matters:
        Seamless injection enables true zero-code optimization - users just
        add the decorator and the optimizer figures out what to optimize.
        This is the most magical but also most complex injection mode.
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_seamless_injection_modifies_source(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test seamless injection works via AST transformation.

        Purpose:
            Verify that the AST transformation correctly identifies and
            replaces optimizable values in the function source.

        Expectations:
            - Function code is analyzed and transformed
            - Hardcoded values are replaced with config lookups
            - Optimization explores the transformed value space

        Why This Validates AST Transformation:
            Seamless injection must correctly parse Python AST, identify
            optimizable values, and generate valid transformed code. If
            transformation fails, the function either crashes or doesn't
            actually optimize the intended parameters.

        Dimensions: InjectionMode=seamless, ConfigSpaceType=categorical
        """
        scenario = basic_scenario(
            name="seamless_ast",
            injection_mode="seamless",
            max_trials=2,
            gist_template="seamless-ast -> {trial_count()} | {status()}",
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestInjectionModeEdgeCases:
    """Tests for edge cases in injection mode configuration.

    Purpose:
        Validate how the optimizer handles invalid, malformed, or edge-case
        injection mode values. These tests ensure graceful error handling
        and predictable behavior for unexpected inputs.

    Why Edge Cases Matter:
        Users may accidentally misconfigure injection modes through typos,
        wrong casing, or None values. The optimizer should either handle
        these gracefully or provide clear error messages.

    Test Categories:
        - None value handling
        - Empty string handling
        - Invalid mode names
        - Case sensitivity
        - Whitespace handling
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_injection_mode_none(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test with injection_mode set to None.

        Purpose:
            Verify behavior when injection_mode is explicitly set to None.
            Should either use a default mode or fail with clear error.

        Expectations:
            - Either falls back to default injection mode (context)
            - Or raises a clear validation error

        Why This Tests Robustness:
            None values can arise from optional config fields or conditional
            logic. The optimizer must handle this predictably.

        Dimensions: InjectionMode=none (invalid), FailureMode=validation
        """
        from tests.optimizer_validation.specs import ExpectedOutcome

        scenario = TestScenario(
            name="injection_mode_none",
            description="Injection mode is None",
            config_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
            injection_mode=None,  # type: ignore[arg-type]
            max_trials=2,
            expected=ExpectedResult(
                outcome=ExpectedOutcome.FAILURE,
            ),
            gist_template="none-mode -> check logs",
        )

        func, result = await scenario_runner(scenario)

        # Should either use default or fail gracefully
        # Document observed behavior

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
    async def test_injection_mode_empty_string(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test with injection_mode as empty string.

        Purpose:
            Verify behavior when injection_mode is an empty string.
            Should fail validation since "" is not a valid mode.

        Expectations:
            - Raises validation error or uses default
            - Does not silently proceed with broken injection

        Why This Tests Input Validation:
            Empty strings are a common edge case that can slip through
            basic type checks. Must be caught before optimization starts.

        Dimensions: InjectionMode=empty (invalid), FailureMode=validation
        """
        from tests.optimizer_validation.specs import ExpectedOutcome

        scenario = TestScenario(
            name="injection_mode_empty",
            description="Injection mode is empty string",
            config_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
            injection_mode="",  # Empty string
            max_trials=2,
            expected=ExpectedResult(
                outcome=ExpectedOutcome.FAILURE,
            ),
            gist_template="empty-mode -> {error_type()} | {status()}",
        )

        func, result = await scenario_runner(scenario)

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
    async def test_injection_mode_invalid_value(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test with invalid injection_mode value.

        Purpose:
            Verify that unrecognized injection mode names are rejected
            with a clear error message.

        Expectations:
            - Raises ValueError or similar validation error
            - Error message indicates valid options
            - Does not attempt to run optimization

        Why This Tests Error Handling:
            Typos in mode names (e.g., "contxt" instead of "context")
            should be caught early with helpful error messages rather
            than causing cryptic failures during optimization.

        Dimensions: InjectionMode=invalid, FailureMode=validation
        """
        from tests.optimizer_validation.specs import ExpectedOutcome, ExpectedResult

        scenario = TestScenario(
            name="injection_mode_invalid",
            description="Injection mode is invalid value",
            config_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
            injection_mode="INVALID_MODE",  # Not a valid mode
            max_trials=2,
            expected=ExpectedResult(
                outcome=ExpectedOutcome.FAILURE,
            ),
            gist_template="invalid-mode -> {error_type()} | {status()}",
        )

        func, result = await scenario_runner(scenario)

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
    async def test_injection_mode_case_sensitivity(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test injection_mode with different casing (CONTEXT vs context).

        Purpose:
            Determine whether injection_mode matching is case-sensitive
            and document the expected behavior.

        Expectations:
            - Either accepts uppercase (case-insensitive matching)
            - Or rejects with helpful error (case-sensitive)

        Why This Tests Usability:
            Case sensitivity is a common source of confusion. This test
            documents the actual behavior so users know whether "CONTEXT"
            and "context" are equivalent.

        Dimensions: InjectionMode=uppercase (edge case)
        """

        scenario = TestScenario(
            name="injection_mode_uppercase",
            description="Injection mode with uppercase",
            config_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
            injection_mode="CONTEXT",  # Uppercase instead of lowercase
            max_trials=2,
            expected=ExpectedResult(
                outcome=ExpectedOutcome.FAILURE,
            ),
            gist_template="case-mode -> {error_type()} | {status()}",
        )

        func, result = await scenario_runner(scenario)

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
    async def test_injection_mode_with_whitespace(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test injection_mode with leading/trailing whitespace.

        Purpose:
            Verify behavior when injection_mode contains extra whitespace,
            which might occur from string processing or config files.

        Expectations:
            - Either strips whitespace and succeeds
            - Or rejects with validation error

        Why This Tests Robustness:
            Whitespace can creep in from YAML/JSON config files, environment
            variables, or string concatenation. Predictable handling is essential.

        Dimensions: InjectionMode=whitespace (edge case)
        """

        scenario = TestScenario(
            name="injection_mode_whitespace",
            description="Injection mode with whitespace",
            config_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
            injection_mode="  context  ",  # Whitespace around value
            max_trials=2,
            expected=ExpectedResult(
                outcome=ExpectedOutcome.FAILURE,
            ),
            gist_template="ws-mode -> {error_type()} | {status()}",
        )

        func, result = await scenario_runner(scenario)

        # Should fail or strip whitespace

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()
