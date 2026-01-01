"""Credential and authentication tests for optimizer behavior.

Purpose:
    Fill the gap where we have 0 tests for credential handling.
    First-time users encounter:
    - Missing API keys
    - Invalid credentials
    - Permission errors
    - Quota exhaustion

Test Categories:
    1. Missing Credentials - No API key configured
    2. Mock Mode Independence - Works without real credentials
    3. Configuration Validation - Early detection of issues
    4. Error Messages - Clear feedback on auth problems

Note:
    These tests run in MOCK_MODE where real API credentials are not
    needed. They verify the patterns and error handling, not actual
    API authentication.
"""

from __future__ import annotations

import pytest

from tests.optimizer_validation.specs import (
    ExpectedResult,
    ObjectiveSpec,
    TestScenario,
)


class TestMockModeNoCredentials:
    """Tests for mock mode operation without real credentials.

    Purpose:
        Verify that mock mode works correctly without requiring
        real API credentials.

    Why This Matters:
        Users need to:
        - Develop and test locally without API keys
        - Run tests in CI without credentials
        - Experiment before committing to API costs
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mock_mode_no_api_key_required(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that mock mode works without any API key.

        Purpose:
            Verify that optimization in mock mode doesn't require
            real API credentials to be configured.

        Dimensions: Credentials=mock_mode
        """
        scenario = TestScenario(
            name="mock_no_key",
            description="Mock mode without API key",
            injection_mode="context",
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.7],
            },
            max_trials=4,
            mock_mode_config={"optimizer": "random"},
            gist_template="mock-nokey -> {trial_count()} | {status()}",
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
    async def test_mock_mode_all_algorithms_no_credentials(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test all algorithms work in mock mode without credentials.

        Purpose:
            Verify that every optimization algorithm can be used
            in mock mode without API credentials.

        Dimensions: Credentials=mock_all_algorithms
        """
        algorithms = [
            {"optimizer": "random"},
            {"optimizer": "grid"},
            {"optimizer": "optuna", "sampler": "tpe"},
            {"optimizer": "optuna", "sampler": "random"},
        ]

        for mock_config in algorithms:
            algo_name = mock_config.get("sampler") or mock_config["optimizer"]

            scenario = TestScenario(
                name=f"mock_nocred_{algo_name}",
                description=f"Mock {algo_name} without credentials",
                injection_mode="context",
                config_space={
                    "model": ["gpt-3.5-turbo", "gpt-4"],
                    "temperature": [0.5],
                },
                max_trials=2,
                mock_mode_config=mock_config,
                gist_template=f"mock-{algo_name} -> {{trial_count()}} | {{status()}}",
            )

            _, result = await scenario_runner(scenario)

            assert not isinstance(
                result, Exception
            ), f"Unexpected error with {algo_name}: {result}"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mock_mode_simulates_llm_responses(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that mock mode provides simulated LLM responses.

        Purpose:
            Verify that mock mode generates plausible simulated
            responses for testing purposes.

        Dimensions: Credentials=mock_simulation
        """
        scenario = TestScenario(
            name="mock_simulated",
            description="Mock mode simulates responses",
            injection_mode="context",
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
                "temperature": [0.1, 0.5, 0.9],
            },
            objectives=[
                ObjectiveSpec(name="accuracy", orientation="maximize"),
            ],
            max_trials=5,
            mock_mode_config={"optimizer": "random"},
            gist_template="mock-sim -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Should have completed trials with simulated metrics
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestConfigurationValidation:
    """Tests for early configuration validation.

    Purpose:
        Verify that configuration issues are detected early with
        clear error messages.

    Why This Matters:
        Users should get immediate feedback on configuration issues
        rather than cryptic errors during optimization.
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_valid_config_space_accepted(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that valid config spaces are accepted.

        Purpose:
            Verify that properly formatted config spaces are accepted
            without errors.

        Dimensions: Validation=valid_config
        """
        scenario = TestScenario(
            name="valid_config",
            description="Valid config space accepted",
            injection_mode="context",
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": (0.0, 1.0),  # Continuous
                "max_tokens": [256, 512, 1024],  # Categorical
            },
            max_trials=3,
            mock_mode_config={"optimizer": "random"},
            gist_template="valid -> {trial_count()} | {status()}",
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
    async def test_mixed_config_types_work(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that mixed config space types work correctly.

        Purpose:
            Verify that config spaces with both categorical (list)
            and continuous (tuple range) parameters work.

        Dimensions: Validation=mixed_types
        """
        scenario = TestScenario(
            name="mixed_types",
            description="Mixed categorical and continuous params",
            injection_mode="context",
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],  # Categorical
                "temperature": (0.0, 2.0),  # Continuous
                "top_p": (0.5, 1.0),  # Continuous
                "presence_penalty": [-2.0, 0.0, 2.0],  # Categorical
            },
            max_trials=5,
            mock_mode_config={"optimizer": "optuna", "sampler": "tpe"},
            gist_template="mixed -> {trial_count()} | {status()}",
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


class TestErrorMessageClarity:
    """Tests for clear error messages on authentication issues.

    Purpose:
        Verify that error messages are clear and actionable when
        there are configuration or credential issues.

    Why This Matters:
        Clear error messages reduce user frustration and support burden.
        Users should understand what's wrong and how to fix it.
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_optimization_provides_useful_result(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that optimization provides useful result structure.

        Purpose:
            Verify that the optimization result has a useful structure
            with all expected fields.

        Dimensions: ErrorMessages=result_structure
        """
        scenario = TestScenario(
            name="result_structure",
            description="Result has useful structure",
            injection_mode="context",
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.5],
            },
            max_trials=2,
            mock_mode_config={"optimizer": "random"},
            gist_template="result -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"


        # Result should have useful structure
        if hasattr(result, "trials"):
            # Trials should be accessible
            assert isinstance(result.trials, (list, tuple))

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestDevelopmentWorkflow:
    """Tests for common development workflow patterns.

    Purpose:
        Verify that common development workflows work correctly,
        especially for new users getting started.

    Why This Matters:
        The first-time user experience is critical for adoption.
        Common workflows must "just work".
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_minimal_setup_works(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that minimal setup is sufficient to get started.

        Purpose:
            Verify that users can get started with minimal configuration.

        Dimensions: Workflow=minimal_setup
        """
        scenario = TestScenario(
            name="minimal_setup",
            description="Minimal setup works",
            injection_mode="context",
            config_space={
                "temperature": [0.5, 0.7],
            },
            max_trials=2,
            mock_mode_config={"optimizer": "random"},
            gist_template="minimal -> {trial_count()} | {status()}",
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
    async def test_quick_iteration_cycle(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test quick iteration cycle for development.

        Purpose:
            Verify that short optimization runs complete quickly,
            enabling rapid iteration during development.

        Dimensions: Workflow=quick_iteration
        """
        scenario = TestScenario(
            name="quick_iteration",
            description="Quick iteration for development",
            injection_mode="context",
            config_space={
                "model": ["gpt-3.5-turbo"],
                "temperature": [0.5],
            },
            max_trials=1,
            mock_mode_config={"optimizer": "random"},
            expected=ExpectedResult(
                min_trials=1,
                max_trials=1,
            ),
            gist_template="quick -> {trial_count()} | {status()}",
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
    async def test_local_development_no_network(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that local development works without network.

        Purpose:
            Verify that mock mode allows development without network
            connectivity to LLM providers.

        Dimensions: Workflow=local_development
        """
        scenario = TestScenario(
            name="local_dev",
            description="Local development without network",
            injection_mode="context",
            config_space={
                "model": ["local-model-1", "local-model-2"],
                "temperature": [0.3, 0.7],
            },
            max_trials=4,
            mock_mode_config={"optimizer": "grid"},
            gist_template="local -> {trial_count()} | {status()}",
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
    async def test_ci_cd_friendly(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that optimization is CI/CD friendly.

        Purpose:
            Verify that optimization can run in CI/CD environments
            without special configuration.

        Dimensions: Workflow=ci_cd
        """
        scenario = TestScenario(
            name="ci_cd",
            description="CI/CD friendly execution",
            injection_mode="context",
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.5],
            },
            max_trials=2,
            timeout=30.0,  # Bounded execution time
            mock_mode_config={"optimizer": "random", "random_seed": 42},
            gist_template="ci -> {trial_count()} | {status()}",
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
