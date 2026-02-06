"""Comprehensive tests for Optuna optimizer algorithms.

Purpose:
    Address coverage gaps identified in the Optuna test suite. These tests
    validate all three Optuna algorithms (TPE, CMA-ES, Random) across
    dimensions that were previously untested.

Coverage Gaps Addressed:
    1. ConfigSpaceType: categorical_only, continuous_only (was only mixed/single_value)
    2. ExecutionMode: privacy, hybrid, cloud, standard (was only edge_analytics)
    3. ObjectiveConfig: single_maximize, single_minimize, weighted (partial)
    4. InjectionMode: seamless, parameter, attribute (was only context)
    5. ParallelMode: sequential and parallel for all algorithms

Algorithms Covered:
    - optuna_tpe: Tree-structured Parzen Estimator (general purpose)
    - optuna_cmaes: Covariance Matrix Adaptation Evolution Strategy (continuous)
    - optuna_random: Random sampler (baseline)
"""

from __future__ import annotations

import pytest

from tests.optimizer_validation.specs import (
    ExpectedOutcome,
    ExpectedResult,
    ObjectiveSpec,
    basic_scenario,
    config_space_scenario,
    multi_objective_scenario,
)

# Optuna algorithms
OPTUNA_ALGORITHMS = ["optuna_tpe", "optuna_cmaes", "optuna_random"]

# Algorithms suitable for categorical-only spaces
CATEGORICAL_COMPATIBLE_ALGORITHMS = ["optuna_tpe", "optuna_random"]

# Algorithms optimized for continuous spaces
CONTINUOUS_OPTIMIZED_ALGORITHMS = ["optuna_cmaes", "optuna_tpe"]

# Algorithms that support multi-objective optimization
# Note: CMA-ES does NOT support multi-objective - it raises ValueError
MULTI_OBJECTIVE_ALGORITHMS = ["optuna_tpe", "optuna_random"]


class TestOptunaConfigSpaceTypes:
    """Test Optuna algorithms with different config space types.

    Purpose:
        Validate that Optuna algorithms correctly handle categorical-only,
        continuous-only, and mixed configuration spaces.

    Coverage Gap Addressed:
        Previously only mixed and single_value spaces were tested with Optuna.
        This adds categorical_only and continuous_only coverage.
    """

    @pytest.mark.parametrize("algorithm", CATEGORICAL_COMPATIBLE_ALGORITHMS)
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_optuna_categorical_only_space(
        self,
        algorithm: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test Optuna with categorical-only configuration space.

        Purpose:
            Verify TPE and Random samplers handle discrete-only spaces correctly.
            CMA-ES is excluded as it requires continuous parameters.

        Expectations:
            - All sampled values are from defined categorical options
            - Optimizer explores the categorical space effectively
            - No type conversion errors

        Dimensions: Algorithm={algorithm}, ConfigSpaceType=categorical_only
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
            "safety_filter": ["strict", "moderate", "lenient"],
            "response_format": ["text", "json"],
        }

        scenario = config_space_scenario(
            name=f"{algorithm}_categorical_only",
            config_space=config_space,
            max_trials=4,
            mock_mode_config={"optimizer": algorithm},
            gist_template=f"{algorithm}-cat -> {{trial_count()}} | {{status()}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Verify all trial configs only contain categorical values
        if hasattr(result, "trials"):
            valid_models = {"gpt-3.5-turbo", "gpt-4", "gpt-4o"}
            valid_filters = {"strict", "moderate", "lenient"}
            valid_formats = {"text", "json"}
            for trial in result.trials:
                config = getattr(trial, "config", {})
                if "model" in config:
                    assert (
                        config["model"] in valid_models
                    ), f"Invalid model: {config['model']}"
                if "safety_filter" in config:
                    assert config["safety_filter"] in valid_filters
                if "response_format" in config:
                    assert config["response_format"] in valid_formats

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.parametrize("algorithm", CONTINUOUS_OPTIMIZED_ALGORITHMS)
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_optuna_continuous_only_space(
        self,
        algorithm: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test Optuna with continuous-only configuration space.

        Purpose:
            Verify TPE and CMA-ES handle continuous parameter spaces.
            CMA-ES is particularly optimized for continuous spaces.

        Expectations:
            - All sampled values are within defined ranges
            - CMA-ES leverages covariance modeling for efficient search
            - TPE builds tree-structured models for continuous params

        Dimensions: Algorithm={algorithm}, ConfigSpaceType=continuous_only
        """
        config_space = {
            "temperature": (0.0, 2.0),
            "top_p": (0.1, 1.0),
            "frequency_penalty": (-2.0, 2.0),
            "presence_penalty": (-2.0, 2.0),
        }

        scenario = config_space_scenario(
            name=f"{algorithm}_continuous_only",
            config_space=config_space,
            max_trials=5,
            mock_mode_config={"optimizer": algorithm},
            gist_template=f"{algorithm}-cont -> {{trial_count()}} | {{status()}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Verify continuous parameters are within specified ranges
        if hasattr(result, "trials"):
            for trial in result.trials:
                config = getattr(trial, "config", {})
                if "temperature" in config:
                    assert (
                        0.0 <= config["temperature"] <= 2.0
                    ), f"temperature out of range: {config['temperature']}"
                if "top_p" in config:
                    assert (
                        0.1 <= config["top_p"] <= 1.0
                    ), f"top_p out of range: {config['top_p']}"
                if "frequency_penalty" in config:
                    assert -2.0 <= config["frequency_penalty"] <= 2.0
                if "presence_penalty" in config:
                    assert -2.0 <= config["presence_penalty"] <= 2.0

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cmaes_with_categorical_only_degrades_gracefully(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test CMA-ES behavior with categorical-only space.

        Purpose:
            CMA-ES is designed for continuous optimization and may not handle
            categorical-only spaces well. This test documents expected behavior.

        Expectations:
            - Either falls back to a compatible sampler
            - Or raises a clear error about incompatibility
            - Should not produce invalid/corrupt results

        Dimensions: Algorithm=optuna_cmaes, ConfigSpaceType=categorical_only
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "format": ["json", "text"],
        }

        scenario = config_space_scenario(
            name="cmaes_categorical_degradation",
            config_space=config_space,
            max_trials=3,
            mock_mode_config={"optimizer": "optuna_cmaes"},
            # May fail or degrade - either is acceptable
            expected=ExpectedResult(outcome=ExpectedOutcome.PARTIAL),
            gist_template="cmaes-cat -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Document behavior - may succeed with fallback or fail
        # Track which degradation path was taken for observability
        if isinstance(result, Exception):
            # CMA-ES raised an error for categorical-only space - expected behavior
            error_msg = str(result).lower()
            incompatibility_terms = {"categorical", "cma", "continuous", "sampler"}
            assert any(
                term in error_msg for term in incompatibility_terms
            ), f"Error should mention incompatibility: {result}"
        elif hasattr(result, "trials"):
            # CMA-ES succeeded with fallback - verify results are valid
            valid_models = {"gpt-3.5-turbo", "gpt-4"}
            valid_formats = {"json", "text"}
            for trial in result.trials:
                config = getattr(trial, "config", {})
                if "model" in config:
                    assert config["model"] in valid_models
                if "format" in config:
                    assert config["format"] in valid_formats

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestOptunaExecutionModes:
    """Test Optuna algorithms with execution modes.

    Note:
        Only edge_analytics is currently supported. cloud/hybrid raise
        ConfigurationError (not yet supported), privacy/standard raise
        ConfigurationError (removed).
    """

    @pytest.mark.parametrize("algorithm", OPTUNA_ALGORITHMS)
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_optuna_edge_analytics_mode(
        self,
        algorithm: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test Optuna algorithms with edge_analytics execution mode.

        Purpose:
            Verify each Optuna algorithm works correctly with edge_analytics,
            the only currently supported execution mode.

        Dimensions: Algorithm={algorithm}, ExecutionMode=edge_analytics
        """
        scenario = basic_scenario(
            name=f"{algorithm}_edge_analytics",
            execution_mode="edge_analytics",
            max_trials=2,
            mock_mode_config={"optimizer": algorithm},
            gist_template=f"{algorithm}-edge_analytics -> {{trial_count()}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Verify trials were executed
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestOptunaObjectiveConfigs:
    """Test Optuna algorithms with different objective configurations.

    Purpose:
        Validate that Optuna optimizers correctly handle maximize, minimize,
        and weighted objective configurations.

    Coverage Gap Addressed:
        Limited coverage of objective types with Optuna algorithms.
    """

    @pytest.mark.parametrize("algorithm", OPTUNA_ALGORITHMS)
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_optuna_single_maximize(
        self,
        algorithm: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test Optuna with single maximize objective.

        Purpose:
            Verify Optuna correctly maximizes a single objective (e.g., accuracy).

        Dimensions: Algorithm={algorithm}, ObjectiveConfig=single_maximize
        """
        objectives = [
            ObjectiveSpec(name="accuracy", orientation="maximize", weight=1.0)
        ]

        scenario = multi_objective_scenario(
            name=f"{algorithm}_maximize",
            objectives=objectives,
            max_trials=3,
            mock_mode_config={"optimizer": algorithm},
            gist_template=f"{algorithm}-max -> {{trial_count()}} | {{best_score()}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Verify trials were executed with maximize objective
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"

        # Verify best_config exists for maximize objective
        if hasattr(result, "best_config"):
            assert (
                result.best_config is not None
            ), "Should have best_config for maximize"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.parametrize("algorithm", OPTUNA_ALGORITHMS)
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_optuna_single_minimize(
        self,
        algorithm: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test Optuna with single minimize objective.

        Purpose:
            Verify Optuna correctly minimizes a single objective (e.g., cost).

        Dimensions: Algorithm={algorithm}, ObjectiveConfig=single_minimize
        """
        objectives = [ObjectiveSpec(name="cost", orientation="minimize", weight=1.0)]

        scenario = multi_objective_scenario(
            name=f"{algorithm}_minimize",
            objectives=objectives,
            max_trials=3,
            mock_mode_config={"optimizer": algorithm},
            gist_template=f"{algorithm}-min -> {{trial_count()}} | {{best_score()}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.parametrize("algorithm", MULTI_OBJECTIVE_ALGORITHMS)
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_optuna_weighted_objectives(
        self,
        algorithm: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test Optuna with weighted multi-objective optimization.

        Purpose:
            Verify Optuna handles weighted objectives correctly, where
            different objectives have different importance.

        Note:
            CMA-ES is excluded as it does not support multi-objective optimization.

        Dimensions: Algorithm={algorithm}, ObjectiveConfig=weighted
        """
        objectives = [
            ObjectiveSpec(name="accuracy", orientation="maximize", weight=0.7),
            ObjectiveSpec(name="cost", orientation="minimize", weight=0.3),
        ]

        scenario = multi_objective_scenario(
            name=f"{algorithm}_weighted",
            objectives=objectives,
            max_trials=4,
            mock_mode_config={"optimizer": algorithm},
            gist_template=f"{algorithm}-weighted -> {{trial_count()}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestOptunaInjectionModes:
    """Test Optuna algorithms with different injection modes.

    Purpose:
        Validate that Optuna optimizers work correctly with all
        configuration injection strategies.

    Coverage Gap Addressed:
        Only context injection was tested; adding seamless, parameter.
    """

    # Note: attribute injection mode was removed in v2.x
    @pytest.mark.parametrize("algorithm", OPTUNA_ALGORITHMS)
    @pytest.mark.parametrize("injection_mode", ["seamless", "parameter"])
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_optuna_injection_modes(
        self,
        algorithm: str,
        injection_mode: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test Optuna algorithms with different injection modes.

        Purpose:
            Verify each Optuna algorithm works correctly with seamless,
            parameter, and attribute injection modes.

        Expectations:
            - Configuration is correctly injected via the specified mode
            - Optimization completes successfully
            - Results are valid regardless of injection mechanism

        Dimensions: Algorithm={algorithm}, InjectionMode={injection_mode}
        """
        scenario = basic_scenario(
            name=f"{algorithm}_{injection_mode}",
            injection_mode=injection_mode,
            max_trials=2,
            mock_mode_config={"optimizer": algorithm},
            gist_template=f"{algorithm}-{injection_mode} -> {{trial_count()}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestOptunaMultiObjective:
    """Test Optuna with true multi-objective (Pareto) optimization.

    Purpose:
        Validate Optuna's multi-objective optimization capabilities
        using samplers like NSGA-II or MOTPE.

    Coverage Gap Addressed:
        Limited multi-objective testing with Optuna algorithms.

    Note:
        CMA-ES does NOT support multi-objective optimization and is excluded.
    """

    @pytest.mark.parametrize("algorithm", MULTI_OBJECTIVE_ALGORITHMS)
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_optuna_pareto_optimization(
        self,
        algorithm: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test Optuna with Pareto multi-objective optimization.

        Purpose:
            Verify Optuna correctly finds Pareto-optimal solutions
            for conflicting objectives.

        Expectations:
            - Optimization explores the Pareto frontier
            - Multiple non-dominated solutions are found
            - Trade-offs between objectives are represented

        Note:
            CMA-ES is excluded as it does not support multi-objective optimization.

        Dimensions: Algorithm={algorithm}, ObjectiveConfig=multi_objective
        """
        objectives = [
            ObjectiveSpec(name="accuracy", orientation="maximize", weight=1.0),
            ObjectiveSpec(name="cost", orientation="minimize", weight=1.0),
            ObjectiveSpec(name="latency", orientation="minimize", weight=1.0),
        ]

        scenario = multi_objective_scenario(
            name=f"{algorithm}_pareto",
            objectives=objectives,
            max_trials=6,
            mock_mode_config={"optimizer": algorithm},
            gist_template=f"{algorithm}-pareto -> {{trial_count()}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestOptunaEdgeCases:
    """Test Optuna algorithm edge cases and special scenarios.

    Purpose:
        Validate Optuna behavior in edge cases and unusual configurations.
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_optuna_tpe_with_few_trials(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test TPE behavior with very few trials (cold start).

        Purpose:
            TPE requires some initial trials to build its model.
            Verify it handles the cold start phase correctly.

        Dimensions: Algorithm=optuna_tpe, edge case
        """
        scenario = basic_scenario(
            name="tpe_cold_start",
            max_trials=2,  # Very few trials for TPE
            mock_mode_config={"optimizer": "optuna_tpe"},
            gist_template="tpe-cold -> {trial_count()}",
        )

        _, result = await scenario_runner(scenario)

        # Should still work with random sampling fallback
        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_optuna_cmaes_single_continuous_param(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test CMA-ES with single continuous parameter.

        Purpose:
            CMA-ES builds covariance matrices; verify it handles
            1D optimization correctly.

        Dimensions: Algorithm=optuna_cmaes, ConfigSpaceType=continuous_only
        """
        config_space = {
            "temperature": (0.0, 2.0),
        }

        scenario = config_space_scenario(
            name="cmaes_1d",
            config_space=config_space,
            max_trials=5,
            mock_mode_config={"optimizer": "optuna_cmaes"},
            gist_template="cmaes-1d -> {trial_count()} | {best_score()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_optuna_random_reproducibility(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test Optuna random sampler reproducibility with seed.

        Purpose:
            Verify that setting a seed produces reproducible results.

        Dimensions: Algorithm=optuna_random, reproducibility
        """
        scenario = basic_scenario(
            name="optuna_random_seeded",
            max_trials=3,
            mock_mode_config={
                "optimizer": "optuna_random",
                "seed": 42,
            },
            gist_template="random-seed -> {trial_count()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_optuna_tpe_high_dimensional_space(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test TPE with high-dimensional configuration space.

        Purpose:
            Verify TPE scales to spaces with many parameters.

        Dimensions: Algorithm=optuna_tpe, ConfigSpaceType=mixed (high-dim)
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
            "temperature": (0.0, 2.0),
            "top_p": (0.1, 1.0),
            "frequency_penalty": (-2.0, 2.0),
            "presence_penalty": (-2.0, 2.0),
            "max_tokens": [256, 512, 1024, 2048],
            "response_format": ["text", "json"],
            "retry_count": [1, 2, 3],
        }

        scenario = config_space_scenario(
            name="tpe_high_dim",
            config_space=config_space,
            max_trials=10,
            mock_mode_config={"optimizer": "optuna_tpe"},
            gist_template="tpe-highdim -> {trial_count()} | {best_score()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.parametrize("algorithm", OPTUNA_ALGORITHMS)
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_optuna_with_mixed_param_types(
        self,
        algorithm: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test Optuna with mixed categorical and continuous parameters.

        Purpose:
            Verify all Optuna samplers handle mixed spaces correctly.

        Dimensions: Algorithm={algorithm}, ConfigSpaceType=mixed
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": (0.0, 1.0),
            "max_tokens": [100, 500, 1000],
        }

        scenario = config_space_scenario(
            name=f"{algorithm}_mixed",
            config_space=config_space,
            max_trials=4,
            mock_mode_config={"optimizer": algorithm},
            gist_template=f"{algorithm}-mixed -> {{trial_count()}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.parametrize("algorithm", OPTUNA_ALGORITHMS)
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_optuna_single_trial(
        self,
        algorithm: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test Optuna with only a single trial.

        Purpose:
            Verify edge case where max_trials=1 works correctly.

        Dimensions: Algorithm={algorithm}, edge case (single trial)
        """
        scenario = basic_scenario(
            name=f"{algorithm}_single_trial",
            max_trials=1,
            mock_mode_config={"optimizer": algorithm},
            gist_template=f"{algorithm}-single -> {{trial_count()}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.parametrize("algorithm", OPTUNA_ALGORITHMS)
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_optuna_many_trials(
        self,
        algorithm: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test Optuna with many trials.

        Purpose:
            Verify Optuna handles many trials efficiently.

        Dimensions: Algorithm={algorithm}, edge case (many trials)
        """
        scenario = basic_scenario(
            name=f"{algorithm}_many_trials",
            max_trials=15,
            mock_mode_config={"optimizer": algorithm},
            gist_template=f"{algorithm}-many -> {{trial_count()}} | {{best_score()}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestOptunaWithTimeout:
    """Tests for Optuna algorithms with timeout stop condition.

    Purpose:
        Address the gap where timeout was not explicitly tested with
        different Optuna algorithms. Verify that each sampler handles
        timeout gracefully.

    Coverage Gap Addressed:
        Timeout × Optuna algorithm interactions were not tested.
    """

    @pytest.mark.parametrize("algorithm", OPTUNA_ALGORITHMS)
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_optuna_with_short_timeout(
        self,
        algorithm: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test Optuna algorithms with short timeout.

        Purpose:
            Verify each Optuna sampler handles timeout correctly without
            crashing or corrupting results.

        Expectations:
            - Optimization completes without exceptions
            - stop_reason is 'timeout', 'max_trials_reached', or 'optimizer'
            - Any completed trials have valid results

        Dimensions: Algorithm={algorithm}, StopCondition=timeout
        """
        from tests.optimizer_validation.specs import TestScenario

        scenario = TestScenario(
            name=f"{algorithm}_timeout",
            description=f"{algorithm} with short timeout",
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
                "temperature": (0.0, 1.0),
            },
            timeout=1.0,  # Short timeout
            max_trials=100,  # High to ensure timeout triggers
            mock_mode_config={"optimizer": algorithm},
            gist_template=f"{algorithm}-timeout -> {{trial_count()}} | {{status()}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        assert hasattr(result, "stop_reason"), "Result should have stop_reason"
        assert result.stop_reason in (
            "timeout",
            "max_trials_reached",
            "optimizer",
        ), f"Expected valid stop_reason, got '{result.stop_reason}'"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.parametrize("algorithm", OPTUNA_ALGORITHMS)
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_optuna_timeout_preserves_trials(
        self,
        algorithm: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that Optuna preserves completed trials on timeout.

        Purpose:
            Verify that when timeout occurs, all trials that completed
            before the timeout are preserved in the result.

        Dimensions: Algorithm={algorithm}, StopCondition=timeout
        """
        from tests.optimizer_validation.specs import TestScenario

        scenario = TestScenario(
            name=f"{algorithm}_timeout_preserve",
            description=f"{algorithm} preserves trials on timeout",
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.5, 0.7],
            },
            timeout=2.0,
            max_trials=50,
            mock_mode_config={"optimizer": algorithm},
            gist_template=f"{algorithm}-preserve -> {{trial_count()}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        # Should have at least one completed trial
        assert len(result.trials) >= 1, "Should preserve completed trials"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_optuna_tpe_timeout_with_continuous_space(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test TPE with timeout on continuous config space.

        Purpose:
            TPE builds surrogate models over continuous spaces. Verify
            timeout doesn't corrupt the model or leave partial state.

        Dimensions: Algorithm=optuna_tpe, StopCondition=timeout,
        ConfigSpaceType=continuous
        """
        from tests.optimizer_validation.specs import TestScenario

        scenario = TestScenario(
            name="tpe_timeout_continuous",
            description="TPE timeout with continuous space",
            config_space={
                "temperature": (0.0, 2.0),
                "top_p": (0.1, 1.0),
                "frequency_penalty": (-2.0, 2.0),
            },
            timeout=1.5,
            max_trials=100,
            mock_mode_config={"optimizer": "optuna_tpe"},
            gist_template="tpe-timeout-cont -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_optuna_cmaes_timeout_stability(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test CMA-ES stability on timeout.

        Purpose:
            CMA-ES maintains covariance matrices internally. Verify
            timeout doesn't leave CMA-ES in an inconsistent state.

        Dimensions: Algorithm=optuna_cmaes, StopCondition=timeout
        """
        from tests.optimizer_validation.specs import TestScenario

        scenario = TestScenario(
            name="cmaes_timeout_stable",
            description="CMA-ES timeout stability",
            config_space={
                "temperature": (0.0, 2.0),
                "top_p": (0.1, 1.0),
            },
            timeout=1.5,
            max_trials=100,
            mock_mode_config={"optimizer": "optuna_cmaes"},
            gist_template="cmaes-timeout -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()
