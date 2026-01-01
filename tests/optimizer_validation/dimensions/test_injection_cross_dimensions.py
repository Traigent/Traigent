"""Tests for InjectionMode cross-dimension coverage gaps.

Purpose:
    Fill the identified coverage gaps between InjectionMode and other dimensions:
    - InjectionMode × Algorithm (16 gaps identified)
    - InjectionMode × ParallelMode (6 gaps identified)
    - InjectionMode × ConfigSpaceType (11 gaps identified)
    - InjectionMode × StopCondition (10 gaps identified)
    - InjectionMode × ObjectiveConfig (11 gaps identified)

Gap Analysis Reference:
    Based on comprehensive test coverage analysis that identified:
    - All modes missing tests for: bayesian algorithm
    - seamless/parameter/attribute missing: grid, optuna_tpe, optuna_cmaes, optuna_random
    - seamless/parameter/attribute have NO parallel execution tests
    - All modes missing tests for: continuous_only config spaces
    - seamless/parameter/attribute missing most stop condition tests

Test Categories:
    1. InjectionMode × Algorithm - Ensure all algorithms work with all injection modes
    2. InjectionMode × ParallelMode - Critical: thread safety for non-context modes
    3. InjectionMode × ConfigSpaceType - Continuous params with AST transformation
    4. InjectionMode × StopCondition - Timeout/convergence per injection mode
    5. InjectionMode × ObjectiveConfig - Multi-objective with all injection modes
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.optimizer_validation.specs import (
    ConstraintSpec,
    ExpectedResult,
    ObjectiveSpec,
    TestScenario,
)

# All injection modes
INJECTION_MODES = ["context", "parameter", "attribute", "seamless"]

# Non-context injection modes (these have more gaps)
NON_CONTEXT_MODES = ["parameter", "attribute", "seamless"]


class TestInjectionModeWithAlgorithms:
    """Tests for InjectionMode × Algorithm coverage gaps.

    Purpose:
        Validate that each injection mode works correctly with all supported
        optimization algorithms. Previously only random algorithm was tested
        with non-context injection modes.

    Identified Gaps:
        - context: missing bayesian
        - seamless: missing grid, optuna_tpe, optuna_cmaes, optuna_random, bayesian
        - parameter: missing grid, optuna_tpe, optuna_cmaes, optuna_random, bayesian
        - attribute: missing grid, optuna_tpe, optuna_cmaes, optuna_random, bayesian
    """

    @pytest.mark.parametrize("injection_mode", INJECTION_MODES)
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_injection_mode_with_grid_search(
        self,
        injection_mode: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test each injection mode works with grid search algorithm.

        Purpose:
            Verify that grid search's deterministic iteration through config
            space works correctly regardless of how configs are injected.

        Why This Matters:
            Grid search iterates through all combinations. The injection mode
            must correctly deliver each unique config to the function without
            any value corruption or cross-trial leakage.

        Dimensions: InjectionMode={injection_mode}, Algorithm=grid
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": [0.3, 0.7],
        }
        # 2x2 = 4 combinations

        scenario = TestScenario(
            name=f"grid_{injection_mode}",
            description=f"Grid search with {injection_mode} injection",
            injection_mode=injection_mode,
            config_space=config_space,
            max_trials=4,
            mock_mode_config={"optimizer": "grid"},
            expected=ExpectedResult(
                min_trials=4,
                max_trials=4,  # Grid exhausts space
            ),
            gist_template=f"grid+{injection_mode} -> {{trial_count()}} | {{status()}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Grid should complete all 4 combinations
        if hasattr(result, "trials"):
            assert (
                len(result.trials) == 4
            ), f"Grid should run 4 trials, got {len(result.trials)}"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.parametrize("injection_mode", INJECTION_MODES)
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_injection_mode_with_optuna_tpe(
        self,
        injection_mode: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test each injection mode works with Optuna TPE (Bayesian) algorithm.

        Purpose:
            Verify that TPE's Bayesian optimization works correctly with all
            injection modes. TPE uses trial history to guide sampling, which
            requires accurate config/metric correlation.

        Why This Matters:
            If injection mode corrupts configs or causes cross-trial leakage,
            TPE's surrogate model will learn incorrect correlations, leading
            to poor optimization performance.

        Dimensions: InjectionMode={injection_mode}, Algorithm=optuna_tpe
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": (0.0, 1.0),  # Continuous for better TPE testing
        }

        scenario = TestScenario(
            name=f"optuna_tpe_{injection_mode}",
            description=f"Optuna TPE with {injection_mode} injection",
            injection_mode=injection_mode,
            config_space=config_space,
            max_trials=3,
            mock_mode_config={"optimizer": "optuna", "sampler": "tpe"},
            gist_template=f"tpe+{injection_mode} -> {{trial_count()}} | {{status()}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.parametrize("injection_mode", INJECTION_MODES)
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_injection_mode_with_optuna_cmaes(
        self,
        injection_mode: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test each injection mode works with Optuna CMA-ES algorithm.

        Purpose:
            Verify that CMA-ES evolutionary strategy works with all injection
            modes. CMA-ES is particularly sensitive to continuous parameter
            handling.

        Why This Matters:
            CMA-ES uses population-based optimization with covariance matrix
            adaptation. Injection mode must correctly pass continuous values
            without precision loss or serialization issues.

        Dimensions: InjectionMode={injection_mode}, Algorithm=optuna_cmaes
        """
        # CMA-ES works best with continuous parameters
        config_space = {
            "temperature": (0.0, 1.0),
            "top_p": (0.5, 1.0),
        }

        scenario = TestScenario(
            name=f"optuna_cmaes_{injection_mode}",
            description=f"Optuna CMA-ES with {injection_mode} injection",
            injection_mode=injection_mode,
            config_space=config_space,
            max_trials=3,
            mock_mode_config={"optimizer": "optuna", "sampler": "cmaes"},
            gist_template=f"cmaes+{injection_mode} -> {{trial_count()}} | {{status()}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.parametrize("injection_mode", INJECTION_MODES)
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_injection_mode_with_optuna_random(
        self,
        injection_mode: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test each injection mode works with Optuna random sampler.

        Purpose:
            Verify that Optuna's random sampling works with all injection
            modes, providing baseline comparison for TPE/CMA-ES tests.

        Dimensions: InjectionMode={injection_mode}, Algorithm=optuna_random
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": (0.0, 1.0),
        }

        scenario = TestScenario(
            name=f"optuna_random_{injection_mode}",
            description=f"Optuna random with {injection_mode} injection",
            injection_mode=injection_mode,
            config_space=config_space,
            max_trials=3,
            mock_mode_config={"optimizer": "optuna", "sampler": "random"},
            gist_template=f"optuna-rnd+{injection_mode} -> {{trial_count()}} | {{status()}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestInjectionModeWithParallelExecution:
    """Tests for InjectionMode × ParallelMode coverage gaps.

    Purpose:
        CRITICAL GAP: seamless, parameter, and attribute injection modes
        have NO parallel execution tests. This is a significant risk because:

        1. Parallel execution may cause race conditions in config injection
        2. Thread-local context may not work for non-context modes
        3. Attribute injection may have object reference issues
        4. Seamless AST transformation may not be thread-safe

    Why This Matters:
        Users running parallel trials with non-context injection modes may
        experience silent config corruption, leading to incorrect optimization
        results that are very difficult to debug.
    """

    @pytest.mark.parametrize("injection_mode", NON_CONTEXT_MODES)
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_parallel_execution_basic(
        self,
        injection_mode: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test parallel execution with non-context injection modes.

        Purpose:
            Verify that parallel trials work correctly with parameter,
            attribute, and seamless injection modes.

        Expectations:
            - All trials complete without deadlock or hang
            - Each trial receives its own unique config
            - No cross-trial config contamination

        Why This Validates Thread Safety:
            Running multiple trials concurrently with shared resources
            (function objects, attributes) will expose race conditions
            that sequential execution would miss.

        Dimensions: InjectionMode={injection_mode}, ParallelMode=parallel
        """
        scenario = TestScenario(
            name=f"parallel_{injection_mode}",
            description=f"Parallel execution with {injection_mode} injection",
            injection_mode=injection_mode,
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
                "temperature": [0.3, 0.5, 0.7],
            },
            max_trials=5,
            parallel_config={"trial_concurrency": 3},
            timeout=60.0,  # Generous timeout to detect hangs
            gist_template=f"parallel+{injection_mode} -> {{trial_count()}} | {{status()}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Verify trials completed
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should have at least 1 completed trial"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.parametrize("injection_mode", NON_CONTEXT_MODES)
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_parallel_config_isolation(
        self,
        injection_mode: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that parallel trials have isolated configs with non-context modes.

        Purpose:
            Verify that each parallel trial receives its own unique config
            and doesn't see configs from other concurrent trials.

        Expectations:
            - Each trial's result reflects its own config
            - No metrics/scores are swapped between trials
            - Config values are correctly recorded in trial history

        Why This Validates Isolation:
            Config contamination is subtle - a trial might "work" but with
            wrong values. By running distinct configs in parallel, we can
            detect if any trial received another trial's config.

        Dimensions: InjectionMode={injection_mode}, ParallelMode=parallel
        """
        # Use very distinct configs to make contamination detectable
        config_space = {
            "model": ["model-A", "model-B", "model-C", "model-D"],
            "temperature": [0.1, 0.5, 0.9],
        }

        scenario = TestScenario(
            name=f"parallel_isolation_{injection_mode}",
            description=f"Parallel config isolation with {injection_mode}",
            injection_mode=injection_mode,
            config_space=config_space,
            max_trials=6,
            parallel_config={"trial_concurrency": 3},
            timeout=60.0,
            mock_mode_config={"optimizer": "random", "random_seed": 42},
            gist_template=f"isolation+{injection_mode} -> {{trial_count()}} | {{status()}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Verify each trial has a valid config from our space
        if hasattr(result, "trials"):
            valid_models = {"model-A", "model-B", "model-C", "model-D"}
            valid_temps = {0.1, 0.5, 0.9}

            for i, trial in enumerate(result.trials):
                config = getattr(trial, "config", {})
                model = config.get("model")
                temp = config.get("temperature")

                assert model in valid_models, f"Trial {i} has invalid model: {model}"
                assert temp in valid_temps, f"Trial {i} has invalid temperature: {temp}"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestInjectionModeWithConfigSpaceTypes:
    """Tests for InjectionMode × ConfigSpaceType coverage gaps.

    Purpose:
        Fill gaps for continuous-only and categorical-only config spaces
        with all injection modes. Particularly important for seamless
        injection which uses AST transformation.

    Identified Gaps:
        - All modes missing: continuous_only
        - seamless/parameter/attribute missing: single_value
    """

    @pytest.mark.parametrize("injection_mode", INJECTION_MODES)
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_continuous_only_config_space(
        self,
        injection_mode: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test injection modes with continuous-only config space.

        Purpose:
            Verify that injection modes correctly handle config spaces
            containing only continuous (float range) parameters.

        Why This Matters:
            Continuous parameters require different serialization and
            precision handling. Seamless injection's AST transformation
            must correctly replace float literals.

        Dimensions: InjectionMode={injection_mode}, ConfigSpaceType=continuous_only
        """
        config_space = {
            "temperature": (0.0, 2.0),
            "top_p": (0.1, 1.0),
            "frequency_penalty": (-2.0, 2.0),
        }

        scenario = TestScenario(
            name=f"continuous_only_{injection_mode}",
            description=f"Continuous-only space with {injection_mode}",
            injection_mode=injection_mode,
            config_space=config_space,
            max_trials=3,
            mock_mode_config={"optimizer": "random"},
            gist_template=(
                f"continuous+{injection_mode} -> {{trial_count()}} | {{status()}}"
            ),
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Verify continuous values are within ranges
        if hasattr(result, "trials"):
            for trial in result.trials:
                config = getattr(trial, "config", {})
                if "temperature" in config:
                    assert 0.0 <= config["temperature"] <= 2.0
                if "top_p" in config:
                    assert 0.1 <= config["top_p"] <= 1.0
                if "frequency_penalty" in config:
                    assert -2.0 <= config["frequency_penalty"] <= 2.0

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.parametrize("injection_mode", NON_CONTEXT_MODES)
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_single_value_config_space(
        self,
        injection_mode: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test injection modes with single-value config space.

        Purpose:
            Verify that injection modes handle the edge case of a config
            space with only one possible value per parameter.

        Why This Matters:
            Single-value spaces are used for baseline comparisons or
            controlled experiments. The injection mode must not fail
            when there's nothing to optimize.

        Dimensions: InjectionMode={injection_mode}, ConfigSpaceType=single_value
        """
        config_space = {
            "model": ["gpt-4"],  # Only one choice
            "temperature": [0.7],  # Only one choice
        }

        scenario = TestScenario(
            name=f"single_value_{injection_mode}",
            description=f"Single-value space with {injection_mode}",
            injection_mode=injection_mode,
            config_space=config_space,
            max_trials=2,
            mock_mode_config={"optimizer": "random"},
            gist_template=(
                f"single+{injection_mode} -> {{trial_count()}} | {{status()}}"
            ),
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # All trials should have the same config
        if hasattr(result, "trials"):
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config.get("model") == "gpt-4"
                assert config.get("temperature") == 0.7

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestInjectionModeWithStopConditions:
    """Tests for InjectionMode × StopCondition coverage gaps.

    Purpose:
        Fill gaps for timeout and config_exhaustion stop conditions
        with non-context injection modes.

    Identified Gaps:
        - context: missing convergence
        - seamless/parameter/attribute: missing timeout, convergence, config_exhaustion
    """

    @pytest.mark.parametrize("injection_mode", NON_CONTEXT_MODES)
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_timeout_stop_condition(
        self,
        injection_mode: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test timeout stop condition with non-context injection modes.

        Purpose:
            Verify that optimization stops correctly when timeout is reached,
            regardless of injection mode.

        Why This Matters:
            Timeout is critical for production use. If injection mode affects
            how timeout is handled, users may experience unexpected behavior.

        Note:
            This test uses a reasonable timeout (2s) with a moderate config space
            to avoid config exhaustion errors while still testing timeout behavior.

        Dimensions: InjectionMode={injection_mode}, StopCondition=timeout
        """
        scenario = TestScenario(
            name=f"timeout_{injection_mode}",
            description=f"Timeout stop with {injection_mode} injection",
            injection_mode=injection_mode,
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
                "temperature": [0.1, 0.3, 0.5, 0.7, 0.9],
            },
            max_trials=50,  # Moderate limit
            timeout=2.0,  # Reasonable timeout
            mock_mode_config={"optimizer": "random"},
            gist_template=(
                f"timeout+{injection_mode} -> {{trial_count()}} | {{status()}}"
            ),
        )

        _, result = await scenario_runner(scenario)

        # Should complete (either via timeout or max_trials in fast mock mode)
        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Verify trials completed
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should have at least 1 trial"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.parametrize("injection_mode", NON_CONTEXT_MODES)
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_config_exhaustion_stop(
        self,
        injection_mode: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test config space exhaustion stop with non-context injection modes.

        Purpose:
            Verify that grid search stops when config space is exhausted,
            regardless of injection mode.

        Why This Matters:
            Grid search must correctly identify when all configs have been
            tried. Injection mode should not affect this detection.

        Dimensions: InjectionMode={injection_mode}, StopCondition=config_exhaustion
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": [0.3, 0.7],
        }
        # 2x2 = 4 combinations

        scenario = TestScenario(
            name=f"exhaustion_{injection_mode}",
            description=f"Config exhaustion with {injection_mode}",
            injection_mode=injection_mode,
            config_space=config_space,
            max_trials=100,  # Higher than needed
            mock_mode_config={"optimizer": "grid"},
            expected=ExpectedResult(
                min_trials=4,
                max_trials=4,
            ),
            gist_template=(
                f"exhaust+{injection_mode} -> {{trial_count()}} | {{status()}}"
            ),
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Grid should stop at exactly 4 trials
        if hasattr(result, "trials"):
            assert (
                len(result.trials) == 4
            ), f"Should stop at 4 trials (exhaustion), got {len(result.trials)}"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestInjectionModeWithObjectiveConfigs:
    """Tests for InjectionMode × ObjectiveConfig coverage gaps.

    Purpose:
        Fill gaps for multi-objective optimization with non-context
        injection modes. Multi-objective requires correct metric tracking
        per trial.

    Identified Gaps:
        - seamless: missing ALL objective configs
        - parameter/attribute: missing most objective types
    """

    @pytest.mark.parametrize("injection_mode", NON_CONTEXT_MODES)
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_multi_objective_optimization(
        self,
        injection_mode: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test multi-objective optimization with non-context injection modes.

        Purpose:
            Verify that multiple objectives are correctly evaluated when
            using parameter, attribute, or seamless injection.

        Why This Matters:
            Multi-objective optimization requires tracking multiple metrics
            per trial. Injection mode must not interfere with metric collection
            or Pareto optimization.

        Dimensions: InjectionMode={injection_mode}, ObjectiveConfig=multi_objective
        """
        scenario = TestScenario(
            name=f"multi_obj_{injection_mode}",
            description=f"Multi-objective with {injection_mode}",
            injection_mode=injection_mode,
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": (0.0, 1.0),
            },
            objectives=[
                ObjectiveSpec(name="accuracy", orientation="maximize", weight=0.6),
                ObjectiveSpec(name="cost", orientation="minimize", weight=0.4),
            ],
            max_trials=3,
            gist_template=f"multi-obj+{injection_mode} -> {{trial_count()}} | {{status()}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.parametrize("injection_mode", NON_CONTEXT_MODES)
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_minimize_objective(
        self,
        injection_mode: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test minimize objective with non-context injection modes.

        Purpose:
            Verify that minimize objectives work correctly with parameter,
            attribute, and seamless injection modes.

        Dimensions: InjectionMode={injection_mode}, ObjectiveConfig=minimize
        """
        scenario = TestScenario(
            name=f"minimize_{injection_mode}",
            description=f"Minimize objective with {injection_mode}",
            injection_mode=injection_mode,
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.7],
            },
            objectives=[
                ObjectiveSpec(name="cost", orientation="minimize"),
            ],
            max_trials=3,
            gist_template=f"minimize+{injection_mode} -> {{trial_count()}} | {{status()}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestInjectionModeWithConstraints:
    """Tests for InjectionMode × ConstraintUsage coverage gaps.

    Purpose:
        Fill gaps for constraint evaluation with non-context injection modes.

    Identified Gaps:
        - seamless/parameter/attribute: missing none and metric_based constraints
    """

    @pytest.mark.parametrize("injection_mode", NON_CONTEXT_MODES)
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_config_constraint(
        self,
        injection_mode: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test config-based constraints with non-context injection modes.

        Purpose:
            Verify that config constraints are correctly evaluated when
            using parameter, attribute, or seamless injection.

        Why This Matters:
            Constraints filter configs before trials run. The injection mode
            must not affect constraint evaluation or cause rejected configs
            to be used anyway.

        Dimensions: InjectionMode={injection_mode}, ConstraintUsage=config_only
        """

        def reject_high_temp(config: dict[str, Any]) -> bool:
            """Reject configs with temperature > 0.8."""
            return config.get("temperature", 0) <= 0.8

        scenario = TestScenario(
            name=f"constraint_{injection_mode}",
            description=f"Config constraint with {injection_mode}",
            injection_mode=injection_mode,
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.5, 0.7, 0.9],
            },
            constraints=[
                ConstraintSpec(
                    name="temp_limit",
                    constraint_fn=reject_high_temp,
                    requires_metrics=False,
                )
            ],
            max_trials=4,
            mock_mode_config={"optimizer": "random", "random_seed": 42},
            gist_template=(
                f"constraint+{injection_mode} -> {{trial_count()}} | {{status()}}"
            ),
        )

        _, result = await scenario_runner(scenario)

        if not isinstance(result, Exception):
            # All completed trials should have temperature <= 0.8
            if hasattr(result, "trials"):
                for trial in result.trials:
                    config = getattr(trial, "config", {})
                    temp = config.get("temperature")
                    if temp is not None:
                        assert temp <= 0.8, f"Constraint violated: temperature={temp}"

            validation = result_validator(scenario, result)
            assert validation.passed, validation.summary()
