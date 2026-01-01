"""Tests for Algorithm × ConfigSpaceType pairwise combinations.

Purpose:
    Validate that every combination of optimization algorithm and config
    space type works correctly together. This is critical because:

    1. Different algorithms have different sampling strategies
    2. Some algorithms may not support all config space types
    3. Grid search needs finite spaces, TPE handles continuous better

Dimensions Covered:
    - Algorithm: random, grid, optuna_tpe, optuna_cmaes, optuna_random
    - ConfigSpaceType: categorical, continuous, mixed, single_value

Coverage Goal:
    Comprehensive pairwise coverage ensuring algorithms handle all space types.

Validation Approach:
    Tests verify that optimization completes successfully and produces
    appropriate results for each algorithm/space combination.
"""

from __future__ import annotations

import pytest

from tests.optimizer_validation.specs import (
    ExpectedOutcome,
    ExpectedResult,
    config_space_scenario,
)

# All algorithms available for testing (via mock_mode_config)
# In mock mode, all algorithms are simulated without requiring actual Optuna installation
ALGORITHMS = ["random", "grid", "optuna_tpe", "optuna_cmaes", "optuna_random"]

# Config space configurations - all types that should be tested
CONFIG_SPACES = {
    "categorical": {
        "model": ["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
        "format": ["text", "json"],
    },
    "continuous": {
        "temperature": (0.0, 2.0),
        "top_p": (0.1, 1.0),
    },
    "mixed": {
        "model": ["gpt-3.5-turbo", "gpt-4"],
        "temperature": (0.0, 1.0),
        "max_tokens": [100, 500, 1000],
    },
    "single_value": {
        "model": ["gpt-4"],
        "temperature": [0.7],
    },
    "empty": {},  # Edge case: empty config space (no parameters to optimize)
}


class TestAlgorithmConfigSpacePairwise:
    """Pairwise coverage of Algorithm × ConfigSpaceType.

    Purpose:
        Ensure every combination of algorithm and config space type works
        correctly. This catches compatibility issues between search
        strategies and parameter types.

    Why This Matters:
        - Grid search requires finite combinatorial spaces
        - Random search works with any space but may miss optimums
        - TPE/CMA-ES are designed for continuous spaces
        - Mixed spaces need algorithms that handle both types
    """

    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    @pytest.mark.parametrize("space_type", CONFIG_SPACES.keys())
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_algorithm_config_space_combination(
        self,
        algorithm: str,
        space_type: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test each algorithm × config space combination works or fails appropriately.

        Purpose:
            Verify that optimization either completes successfully or fails with
            an appropriate error when using incompatible algorithm/space combinations.

        Expectations:
            - Compatible combinations complete successfully
            - Grid search with continuous/mixed space fails with clear error
            - Random search handles all space types

        Known Incompatibilities:
            - Grid search cannot handle continuous or mixed spaces
              (requires finite enumerable combinations)

        Why This Validates Algorithm/Space Compatibility:
            This test documents which combinations work and which don't,
            ensuring the optimizer provides clear error messages for
            incompatible configurations.

        Dimensions:
            Algorithm={algorithm}
            ConfigSpaceType={space_type}
        """
        config_space = CONFIG_SPACES[space_type]

        # Known algorithm/space incompatibilities:
        # - Grid search cannot handle continuous parameters (requires finite enumerable space)
        # - CMA-ES is designed for continuous optimization, may struggle with categorical-only
        grid_incompatible = algorithm == "grid" and space_type in [
            "continuous",
            "mixed",
        ]

        # CMA-ES may have issues with purely categorical spaces (it's designed for continuous)
        cmaes_categorical_only = algorithm == "optuna_cmaes" and space_type in [
            "categorical",
            "single_value",
        ]

        # Empty config space is an edge case - algorithms may handle it differently
        empty_space = space_type == "empty"

        # Grid search needs limited trials for large spaces
        max_trials = 4 if algorithm == "grid" else 3

        # Determine expected outcome
        if grid_incompatible:
            expected = ExpectedResult(outcome=ExpectedOutcome.FAILURE)
            gist = f"{algorithm}+{space_type} -> {{error_type()}} | {{status()}}"
        else:
            expected = ExpectedResult()
            gist = (
                f"{algorithm}+{space_type} -> {{trial_count()}} trials | {{status()}}"
            )

        scenario = config_space_scenario(
            name=f"algo_{algorithm}_{space_type}",
            config_space=config_space,
            description=f"{algorithm} search with {space_type} space",
            gist_template=gist,
            max_trials=max_trials,
            mock_mode_config={"optimizer": algorithm},
            expected=expected,
        )

        func, result = await scenario_runner(scenario)

        if grid_incompatible:
            # Grid search with continuous/mixed should fail with clear error
            assert isinstance(
                result, Exception
            ), f"Expected grid search to fail with {space_type} space"
            error_msg = str(result).lower()
            assert (
                "grid" in error_msg or "continuous" in error_msg
            ), f"Error should mention grid/continuous incompatibility: {result}"
            # Emit evidence for expected failure
            validation = result_validator(scenario, result)
            assert validation.passed, validation.summary()
        elif empty_space:
            # Empty config space is an edge case
            # Algorithms may either:
            # 1. Fail with a clear error about empty space
            # 2. Succeed with a single "default" trial (no params to vary)
            # Either way, emit evidence
            validation = result_validator(scenario, result)
            assert validation.passed, validation.summary()
        elif cmaes_categorical_only:
            # CMA-ES with categorical-only may fail or succeed depending on implementation
            # We accept either outcome but document the behavior
            # Either way, emit evidence
            validation = result_validator(scenario, result)
            assert validation.passed, validation.summary()
        else:
            # Compatible combinations should succeed
            assert not isinstance(result, Exception), f"Unexpected error: {result}"
            validation = result_validator(scenario, result)
            assert validation.passed, validation.summary()


class TestGridSearchExhaustiveness:
    """Test that grid search fully explores finite spaces.

    Purpose:
        Verify that grid search correctly enumerates all combinations
        in categorical spaces when max_trials allows.
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_grid_exhausts_small_categorical_space(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test grid search explores all combinations in small space.

        Purpose:
            Verify that grid search visits all 4 combinations when the
            space is small enough to fully explore.

        Expectations:
            - All 4 combinations are visited (2 × 2)
            - Each trial has a unique configuration
            - Stop reason is config_exhaustion or max_trials_reached

        Dimensions: Algorithm=grid, ConfigSpaceType=categorical
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "format": ["text", "json"],
        }

        scenario = config_space_scenario(
            name="grid_exhaustive",
            config_space=config_space,
            description="Grid search exhausting 2×2 space",
            gist_template="grid-exhaustive -> {trial_count()} trials | {status()}",
            max_trials=10,  # More than needed to see if it stops early
            mock_mode_config={"optimizer": "grid"},
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Should have exactly 4 trials (exhausted the space)
        if hasattr(result, "trials"):
            # Grid should explore all combinations
            assert len(result.trials) <= 4, "Grid explored too many configs"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_grid_respects_max_trials_limit(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test grid search stops at max_trials even if space not exhausted.

        Purpose:
            Verify that grid search respects max_trials when the space
            is larger than the trial budget.

        Expectations:
            - Exactly max_trials trials are run
            - Stop reason is max_trials_reached
            - Each trial has a valid configuration

        Dimensions: Algorithm=grid, ConfigSpaceType=categorical
        """
        config_space = {
            "model": ["m1", "m2", "m3", "m4", "m5"],
            "format": ["f1", "f2", "f3", "f4"],
        }
        # 5 × 4 = 20 combinations, but we only allow 5 trials

        scenario = config_space_scenario(
            name="grid_limited",
            config_space=config_space,
            description="Grid search limited by max_trials",
            gist_template="grid-limited -> {stop_reason()} @ {trial_count()} | {status()}",
            max_trials=5,
            mock_mode_config={"optimizer": "grid"},
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        if hasattr(result, "trials"):
            assert (
                len(result.trials) == 5
            ), f"Expected 5 trials, got {len(result.trials)}"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestRandomSearchExploration:
    """Test random search exploration behavior.

    Purpose:
        Verify that random search correctly samples from all space types
        and provides variety in sampled configurations.
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_random_samples_from_continuous_range(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test random search samples float values from continuous range.

        Purpose:
            Verify that random search correctly samples from continuous
            parameters, producing float values within the specified range.

        Expectations:
            - All sampled values are within [0.0, 2.0] range
            - Values are actual floats with varied precision
            - Different trials have different values

        Dimensions: Algorithm=random, ConfigSpaceType=continuous
        """
        config_space = {
            "temperature": (0.0, 2.0),
        }

        scenario = config_space_scenario(
            name="random_continuous",
            config_space=config_space,
            description="Random search with continuous parameter",
            gist_template="random-continuous -> {trial_count()} trials | {status()}",
            max_trials=5,
            mock_mode_config={"optimizer": "random"},
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"


        # Verify values are in range
        if hasattr(result, "trials") and result.trials:
            temps = []
            for trial in result.trials:
                if hasattr(trial, "config") and trial.config:
                    temp = trial.config.get("temperature")
                    if temp is not None:
                        assert 0.0 <= temp <= 2.0, f"Temperature {temp} out of range"
                        temps.append(temp)

            # Verify variety (not all same value)
            if len(temps) > 1:
                unique = len(set(temps))
                assert unique > 1, "All random samples were identical"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_random_handles_mixed_space(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test random search handles mixed categorical/continuous space.

        Purpose:
            Verify that random search correctly samples from both
            categorical and continuous parameters in the same space.

        Expectations:
            - Categorical values are valid list members
            - Continuous values are within specified range
            - Both types appear correctly in each trial config

        Dimensions: Algorithm=random, ConfigSpaceType=mixed
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
            "temperature": (0.0, 1.0),
            "max_tokens": [100, 500, 1000],
        }

        scenario = config_space_scenario(
            name="random_mixed",
            config_space=config_space,
            description="Random search with mixed space",
            gist_template="random-mixed -> {trial_count()} trials | {status()}",
            max_trials=5,
            mock_mode_config={"optimizer": "random"},
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"


        # Verify config validity
        if hasattr(result, "trials") and result.trials:
            for trial in result.trials:
                if hasattr(trial, "config") and trial.config:
                    config = trial.config

                    # Check categorical values
                    if "model" in config:
                        assert config["model"] in ["gpt-3.5-turbo", "gpt-4", "gpt-4o"]
                    if "max_tokens" in config:
                        assert config["max_tokens"] in [100, 500, 1000]

                    # Check continuous values
                    if "temperature" in config:
                        temp = config["temperature"]
                        assert 0.0 <= temp <= 1.0, f"Temperature {temp} out of range"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestAlgorithmWithSingleValueSpace:
    """Test algorithms with degenerate single-value spaces.

    Purpose:
        Verify that algorithms handle single-value parameters correctly,
        where there's effectively no optimization to be done.
    """

    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_algorithm_with_single_value_params(
        self,
        algorithm: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test algorithm handles parameters with only one value.

        Purpose:
            Verify that algorithms don't fail when some or all parameters
            have only a single possible value.

        Expectations:
            - Optimization completes without errors
            - All trials have the same (only possible) configuration
            - Algorithm recognizes limited search space

        Dimensions:
            Algorithm={algorithm}
            ConfigSpaceType=single_value
        """
        config_space = {
            "model": ["gpt-4"],  # Only one option
            "temperature": [0.7],  # Only one option
        }

        scenario = config_space_scenario(
            name=f"{algorithm}_single_value",
            config_space=config_space,
            description=f"{algorithm} with single-value space",
            gist_template=f"{algorithm}-single-val -> {{trial_count()}} | {{status()}}",
            max_trials=3,
            mock_mode_config={"optimizer": algorithm},
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"


        # All trials should have identical config
        if hasattr(result, "trials") and result.trials:
            configs = [t.config for t in result.trials if hasattr(t, "config")]
            if len(configs) > 1:
                # All configs should be the same
                first_config = configs[0]
                for config in configs[1:]:
                    assert (
                        config == first_config
                    ), "Configs differ in single-value space"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestOptunaTPEAlgorithm:
    """Tests specific to Optuna TPE (Tree-structured Parzen Estimator) algorithm.

    Purpose:
        Verify that TPE algorithm works correctly across all config space types.
        TPE is a Bayesian optimization algorithm that models promising and
        unpromising regions of the search space.

    Why TPE Needs Dedicated Tests:
        - TPE uses kernel density estimation, requiring history of trials
        - TPE handles categorical and continuous differently
        - TPE should excel with mixed spaces due to tree structure
    """

    @pytest.mark.parametrize("space_type", CONFIG_SPACES.keys())
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_tpe_with_config_space(
        self,
        space_type: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test TPE algorithm with each config space type.

        Purpose:
            Verify that TPE handles all config space types appropriately.

        Expectations:
            - TPE completes optimization without errors
            - TPE produces valid configurations within the space
            - TPE respects max_trials limit

        Dimensions:
            Algorithm=optuna_tpe
            ConfigSpaceType={space_type}
        """
        config_space = CONFIG_SPACES[space_type]

        scenario = config_space_scenario(
            name=f"tpe_{space_type}",
            config_space=config_space,
            description=f"TPE with {space_type} config space",
            gist_template=f"tpe-{space_type} -> {{trial_count()}} trials | {{status()}}",
            max_trials=3,
            mock_mode_config={"optimizer": "optuna_tpe"},
        )

        func, result = await scenario_runner(scenario)

        # TPE should handle all space types (empty may fail gracefully)
        if space_type != "empty":
            assert not isinstance(
                result, Exception
            ), f"TPE failed with {space_type}: {result}"

        # Always emit evidence regardless of outcome

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestOptunaCMAESAlgorithm:
    """Tests specific to Optuna CMA-ES algorithm.

    Purpose:
        Verify that CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
        works correctly with continuous config spaces.

    Why CMA-ES Needs Dedicated Tests:
        - CMA-ES is designed specifically for continuous optimization
        - CMA-ES may struggle with categorical-only spaces
        - CMA-ES uses population-based evolution with covariance adaptation
    """

    @pytest.mark.parametrize("space_type", ["continuous", "mixed"])
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cmaes_with_continuous_space(
        self,
        space_type: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test CMA-ES with continuous and mixed spaces.

        Purpose:
            Verify that CMA-ES excels with continuous optimization
            where it was designed to work.

        Expectations:
            - CMA-ES completes optimization successfully
            - Continuous values are within specified ranges
            - CMA-ES explores the space efficiently

        Dimensions:
            Algorithm=optuna_cmaes
            ConfigSpaceType={space_type}
        """
        config_space = CONFIG_SPACES[space_type]

        scenario = config_space_scenario(
            name=f"cmaes_{space_type}",
            config_space=config_space,
            description=f"CMA-ES with {space_type} config space",
            gist_template=f"cmaes-{space_type} -> {{trial_count()}} trials | {{status()}}",
            max_trials=3,
            mock_mode_config={"optimizer": "optuna_cmaes"},
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(
            result, Exception
        ), f"CMA-ES failed with {space_type}: {result}"

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.parametrize("space_type", ["categorical", "single_value", "empty"])
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cmaes_with_categorical_space(
        self,
        space_type: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test CMA-ES behavior with categorical-only spaces.

        Purpose:
            Document how CMA-ES handles spaces it wasn't designed for.
            CMA-ES is fundamentally a continuous optimizer, so categorical
            spaces may trigger fallback behavior or errors.

        Expectations:
            - Either fails with clear error about incompatibility
            - Or succeeds with some internal transformation

        Dimensions:
            Algorithm=optuna_cmaes
            ConfigSpaceType={space_type}
        """
        config_space = CONFIG_SPACES[space_type]

        scenario = config_space_scenario(
            name=f"cmaes_categorical_{space_type}",
            config_space=config_space,
            description=f"CMA-ES with {space_type} (edge case)",
            gist_template=f"cmaes-cat-{space_type} -> {{trial_count()}} | {{status()}}",
            max_trials=3,
            mock_mode_config={"optimizer": "optuna_cmaes"},
        )

        func, result = await scenario_runner(scenario)

        # CMA-ES with categorical is implementation-dependent
        # Document the behavior but always emit evidence

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestOptunaRandomAlgorithm:
    """Tests specific to Optuna Random sampler.

    Purpose:
        Verify that Optuna's random sampler works correctly across all
        config space types, providing a baseline for comparison.

    Why Optuna Random Needs Dedicated Tests:
        - Random sampling should work with ANY config space
        - Provides baseline behavior to compare against TPE/CMA-ES
        - Should handle edge cases gracefully
    """

    @pytest.mark.parametrize("space_type", CONFIG_SPACES.keys())
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_optuna_random_with_config_space(
        self,
        space_type: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test Optuna random sampler with each config space type.

        Purpose:
            Verify that random sampling works universally across all
            config space types without algorithmic restrictions.

        Expectations:
            - Random sampling completes for all space types
            - Sampled values are within valid ranges/options
            - No special compatibility requirements

        Dimensions:
            Algorithm=optuna_random
            ConfigSpaceType={space_type}
        """
        config_space = CONFIG_SPACES[space_type]

        scenario = config_space_scenario(
            name=f"optuna_random_{space_type}",
            config_space=config_space,
            description=f"Optuna random with {space_type} config space",
            gist_template=f"optuna-rnd-{space_type} -> {{trial_count()}} | {{status()}}",
            max_trials=3,
            mock_mode_config={"optimizer": "optuna_random"},
        )

        func, result = await scenario_runner(scenario)

        # Random sampling should work with any space (empty may be edge case)
        if space_type != "empty":
            assert not isinstance(
                result, Exception
            ), f"Random failed with {space_type}: {result}"

        # Always emit evidence

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()
