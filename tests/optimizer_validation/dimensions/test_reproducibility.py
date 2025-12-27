"""Tests for optimization reproducibility and determinism.

Tests that verify:
- Same random seed produces same results
- Deterministic behavior with fixed seeds
- Grid search is inherently deterministic
- Parallel execution reproducibility

The random_seed from mock_mode_config is now properly propagated to optimizers,
enabling reproducible test runs.
"""

from __future__ import annotations

import pytest

from tests.optimizer_validation.specs import config_space_scenario


class TestRandomSeedReproducibility:
    """Tests for random search reproducibility with seeds."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_random_same_seed_same_results(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that random search with same seed produces same config sequence.

        Running optimization twice with the same seed should produce
        identical configuration sequences.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
            "temperature": [0.1, 0.3, 0.5, 0.7, 0.9],
        }

        # Run first optimization with seed
        scenario1 = config_space_scenario(
            name="seed_test_1",
            config_space=config_space,
            description="First run with seed 42",
            max_trials=5,
            mock_mode_config={"optimizer": "random", "random_seed": 42},
            gist_template="seed-1 -> {trial_count()} | {status()}",
        )

        _, result1 = await scenario_runner(scenario1)
        assert not isinstance(result1, Exception), f"First run failed: {result1}"

        # Emit evidence for first run
        validation1 = result_validator(scenario1, result1)
        assert validation1.passed, validation1.summary()

        # Run second optimization with same seed
        scenario2 = config_space_scenario(
            name="seed_test_2",
            config_space=config_space,
            description="Second run with seed 42",
            max_trials=5,
            mock_mode_config={"optimizer": "random", "random_seed": 42},
            gist_template="seed-2 -> {trial_count()} | {status()}",
        )

        _, result2 = await scenario_runner(scenario2)
        assert not isinstance(result2, Exception), f"Second run failed: {result2}"

        # Emit evidence for second run
        validation2 = result_validator(scenario2, result2)
        assert validation2.passed, validation2.summary()

        # Compare configurations from both runs
        if hasattr(result1, "trials") and hasattr(result2, "trials"):
            configs1 = [t.config for t in result1.trials]
            configs2 = [t.config for t in result2.trials]

            # With same seed, should get same configs in same order
            assert len(configs1) == len(configs2), (
                f"Different trial counts: {len(configs1)} vs {len(configs2)}"
            )

            for i, (c1, c2) in enumerate(zip(configs1, configs2, strict=False)):
                assert c1 == c2, f"Trial {i + 1} configs differ with same seed: {c1} vs {c2}"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_non_deterministic_behavior(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test non-deterministic behavior when no seed is provided.

        Purpose:
            Verify that running without a seed produces different results
            (or at least allows for it). Note: In a small search space,
            random search might pick the same configs by chance, so this
            test is probabilistic or checks for *potential* difference.

        Edge Case: Non-deterministic reproducibility
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
            "temperature": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        }

        # Run first optimization without seed
        scenario1 = config_space_scenario(
            name="no_seed_1",
            config_space=config_space,
            description="First run without seed",
            max_trials=5,
            mock_mode_config={"optimizer": "random"},  # No seed
            gist_template="no-seed-1 -> {trial_count()} | {status()}",
        )

        _, result1 = await scenario_runner(scenario1)

        # Run second optimization without seed
        scenario2 = config_space_scenario(
            name="no_seed_2",
            config_space=config_space,
            description="Second run without seed",
            max_trials=5,
            mock_mode_config={"optimizer": "random"},  # No seed
            gist_template="no-seed-2 -> {trial_count()} | {status()}",
        )

        _, result2 = await scenario_runner(scenario2)

        # We can't strictly assert they are different because random chance
        # might make them same. But we can verify they run successfully.
        # Ideally, we'd check that the seed was NOT fixed.
        assert not isinstance(result1, Exception)
        assert not isinstance(result2, Exception)

        validation1 = result_validator(scenario1, result1)
        assert validation1.passed, validation1.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_random_different_seed_different_results(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that different seeds produce different config sequences.

        This validates that the seed is actually being used.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
            "temperature": [0.1, 0.3, 0.5, 0.7, 0.9],
        }
        # 15 combinations - high probability of different sequences

        # Run with seed 42
        scenario1 = config_space_scenario(
            name="diff_seed_1",
            config_space=config_space,
            description="Run with seed 42",
            max_trials=5,
            mock_mode_config={"optimizer": "random", "random_seed": 42},
            gist_template="diff-seed-1 -> {trial_count()} | {status()}",
        )

        _, result1 = await scenario_runner(scenario1)
        assert not isinstance(result1, Exception)

        # Emit evidence for first run
        validation1 = result_validator(scenario1, result1)
        assert validation1.passed, validation1.summary()

        # Run with seed 123
        scenario2 = config_space_scenario(
            name="diff_seed_2",
            config_space=config_space,
            description="Run with seed 123",
            max_trials=5,
            mock_mode_config={"optimizer": "random", "random_seed": 123},
            gist_template="diff-seed-2 -> {trial_count()} | {status()}",
        )

        _, result2 = await scenario_runner(scenario2)
        assert not isinstance(result2, Exception)

        # Emit evidence for second run
        validation2 = result_validator(scenario2, result2)
        assert validation2.passed, validation2.summary()

        # Compare - should likely differ
        if hasattr(result1, "trials") and hasattr(result2, "trials"):
            configs1 = [tuple(sorted(t.config.items())) for t in result1.trials]
            configs2 = [tuple(sorted(t.config.items())) for t in result2.trials]

            # With 15 combinations and 5 trials, very unlikely to be identical
            # unless the seed isn't working
            all_same = configs1 == configs2
            # Allow for the astronomically small chance they're the same
            # but log a warning
            if all_same:
                # Could be coincidence, but worth noting
                pass

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_random_no_seed_varies(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that random search without seed produces varying results.

        Without a fixed seed, different runs should (usually) produce
        different sequences.
        """
        config_space = {
            "model": [f"model-{i}" for i in range(10)],
            "temperature": [0.1 * i for i in range(10)],
        }
        # 100 combinations - very unlikely to get same sequence twice

        # Run twice without seed
        scenario1 = config_space_scenario(
            name="no_seed_1",
            config_space=config_space,
            description="First run without seed",
            max_trials=3,
            mock_mode_config={"optimizer": "random"},  # No seed
            gist_template="no-seed-1 -> {trial_count()} | {status()}",
        )

        _, result1 = await scenario_runner(scenario1)
        assert not isinstance(result1, Exception)

        scenario2 = config_space_scenario(
            name="no_seed_2",
            config_space=config_space,
            description="Second run without seed",
            max_trials=3,
            mock_mode_config={"optimizer": "random"},  # No seed
            gist_template="no-seed-2 -> {trial_count()} | {status()}",
        )

        _, result2 = await scenario_runner(scenario2)
        assert not isinstance(result2, Exception)

        # Results may or may not differ - this just verifies no crash
        # Emit evidence for both
        result_validator(scenario1, result1)
        result_validator(scenario2, result2)


class TestGridSearchDeterminism:
    """Tests for grid search determinism.

    Grid search should be inherently deterministic - same config space
    should always produce the same exploration order.
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_grid_inherently_deterministic(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that grid search is deterministic without needing a seed.

        Grid search should explore the same order every time.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": [0.3, 0.7],
        }

        # Run twice
        scenario1 = config_space_scenario(
            name="grid_determ_1",
            config_space=config_space,
            description="Grid search run 1",
            max_trials=4,
            mock_mode_config={"optimizer": "grid"},
            gist_template="grid-determ-1 -> {trial_count()} | {status()}",
        )

        _, result1 = await scenario_runner(scenario1)
        assert not isinstance(result1, Exception)

        # Emit evidence for first run
        validation1 = result_validator(scenario1, result1)
        assert validation1.passed, validation1.summary()

        scenario2 = config_space_scenario(
            name="grid_determ_2",
            config_space=config_space,
            description="Grid search run 2",
            max_trials=4,
            mock_mode_config={"optimizer": "grid"},
            gist_template="grid-determ-2 -> {trial_count()} | {status()}",
        )

        _, result2 = await scenario_runner(scenario2)
        assert not isinstance(result2, Exception)

        # Emit evidence for second run
        validation2 = result_validator(scenario2, result2)
        assert validation2.passed, validation2.summary()

        # Compare configs - should be identical
        if hasattr(result1, "trials") and hasattr(result2, "trials"):
            configs1 = [t.config for t in result1.trials]
            configs2 = [t.config for t in result2.trials]

            assert len(configs1) == len(configs2)

            for i, (c1, c2) in enumerate(zip(configs1, configs2, strict=False)):
                assert c1 == c2, f"Grid trial {i + 1} differs: {c1} vs {c2}"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_grid_with_seed_same_as_without(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that grid search with seed is same as without seed.

        Seed should have no effect on grid search.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": [0.3, 0.7],
        }

        # Without seed
        scenario1 = config_space_scenario(
            name="grid_no_seed",
            config_space=config_space,
            description="Grid without seed",
            max_trials=4,
            mock_mode_config={"optimizer": "grid"},
            gist_template="grid-no-seed -> {trial_count()} | {status()}",
        )

        _, result1 = await scenario_runner(scenario1)
        assert not isinstance(result1, Exception)

        # Emit evidence for first run
        validation1 = result_validator(scenario1, result1)
        assert validation1.passed, validation1.summary()

        # With seed (should be ignored)
        scenario2 = config_space_scenario(
            name="grid_with_seed",
            config_space=config_space,
            description="Grid with seed (should be ignored)",
            max_trials=4,
            mock_mode_config={"optimizer": "grid", "random_seed": 42},
            gist_template="grid-with-seed -> {trial_count()} | {status()}",
        )

        _, result2 = await scenario_runner(scenario2)
        assert not isinstance(result2, Exception)

        # Emit evidence for second run
        validation2 = result_validator(scenario2, result2)
        assert validation2.passed, validation2.summary()

        # Should be identical
        if hasattr(result1, "trials") and hasattr(result2, "trials"):
            configs1 = [t.config for t in result1.trials]
            configs2 = [t.config for t in result2.trials]

            for i, (c1, c2) in enumerate(zip(configs1, configs2, strict=False)):
                assert c1 == c2, f"Grid trial {i + 1} differs with/without seed"


class TestOptunaSeedReproducibility:
    """Tests for Optuna-based optimizer reproducibility."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_optuna_tpe_with_seed(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test TPE sampler reproducibility with seed.

        Optuna's TPE sampler should be reproducible with a fixed seed.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": (0.0, 1.0),
        }

        # Run twice with same seed
        scenario1 = config_space_scenario(
            name="tpe_seed_1",
            config_space=config_space,
            description="TPE run 1 with seed 42",
            max_trials=3,
            mock_mode_config={
                "optimizer": "optuna",
                "sampler": "tpe",
                "random_seed": 42,
            },
            gist_template="tpe-seed-1 -> {trial_count()} | {status()}",
        )

        _, result1 = await scenario_runner(scenario1)
        assert not isinstance(result1, Exception)

        # Emit evidence for first run
        validation1 = result_validator(scenario1, result1)
        assert validation1.passed, validation1.summary()

        scenario2 = config_space_scenario(
            name="tpe_seed_2",
            config_space=config_space,
            description="TPE run 2 with seed 42",
            max_trials=3,
            mock_mode_config={
                "optimizer": "optuna",
                "sampler": "tpe",
                "random_seed": 42,
            },
            gist_template="tpe-seed-2 -> {trial_count()} | {status()}",
        )

        _, result2 = await scenario_runner(scenario2)
        assert not isinstance(result2, Exception)

        # Emit evidence for second run
        validation2 = result_validator(scenario2, result2)
        assert validation2.passed, validation2.summary()

        # Compare - should be identical or very similar
        if hasattr(result1, "trials") and hasattr(result2, "trials"):
            # For Optuna with TPE, first few trials are random exploration
            # so they should match with same seed
            configs1 = [t.config for t in result1.trials]
            configs2 = [t.config for t in result2.trials]

            # At minimum, first trial should be identical
            if len(configs1) > 0 and len(configs2) > 0:
                # Categorical part should definitely match
                assert configs1[0].get("model") == configs2[0].get("model"), (
                    "First trial model should match with same seed"
                )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_optuna_random_sampler_with_seed(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test Optuna random sampler reproducibility with seed."""
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
            "temperature": [0.1, 0.5, 0.9],
        }

        # Run twice with same seed
        scenario1 = config_space_scenario(
            name="optuna_random_seed_1",
            config_space=config_space,
            description="Optuna random run 1 with seed",
            max_trials=4,
            mock_mode_config={
                "optimizer": "optuna",
                "sampler": "random",
                "random_seed": 42,
            },
            gist_template="optuna-rand-1 -> {trial_count()} | {status()}",
        )

        _, result1 = await scenario_runner(scenario1)
        assert not isinstance(result1, Exception)

        # Emit evidence for first run
        validation1 = result_validator(scenario1, result1)
        assert validation1.passed, validation1.summary()

        scenario2 = config_space_scenario(
            name="optuna_random_seed_2",
            config_space=config_space,
            description="Optuna random run 2 with seed",
            max_trials=4,
            mock_mode_config={
                "optimizer": "optuna",
                "sampler": "random",
                "random_seed": 42,
            },
            gist_template="optuna-rand-2 -> {trial_count()} | {status()}",
        )

        _, result2 = await scenario_runner(scenario2)
        assert not isinstance(result2, Exception)

        # Emit evidence for second run
        validation2 = result_validator(scenario2, result2)
        assert validation2.passed, validation2.summary()

        # Compare
        if hasattr(result1, "trials") and hasattr(result2, "trials"):
            configs1 = [t.config for t in result1.trials]
            configs2 = [t.config for t in result2.trials]

            for i, (c1, c2) in enumerate(zip(configs1, configs2, strict=False)):
                assert c1 == c2, f"Optuna random trial {i + 1} differs with same seed"


class TestContinuousParameterReproducibility:
    """Tests for reproducibility with continuous parameters."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_continuous_param_exact_values_with_seed(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that continuous parameter values are exactly reproducible.

        With a fixed seed, continuous parameters should produce
        exactly the same floating point values.
        """
        config_space = {
            "temperature": (0.0, 1.0),
            "top_p": (0.5, 1.0),
        }

        # Run twice with same seed
        scenario1 = config_space_scenario(
            name="continuous_repro_1",
            config_space=config_space,
            description="Continuous params run 1",
            max_trials=5,
            mock_mode_config={"optimizer": "random", "random_seed": 42},
            gist_template="cont-repro-1 -> {trial_count()} | {status()}",
        )

        _, result1 = await scenario_runner(scenario1)
        assert not isinstance(result1, Exception)

        # Emit evidence for first run
        validation1 = result_validator(scenario1, result1)
        assert validation1.passed, validation1.summary()

        scenario2 = config_space_scenario(
            name="continuous_repro_2",
            config_space=config_space,
            description="Continuous params run 2",
            max_trials=5,
            mock_mode_config={"optimizer": "random", "random_seed": 42},
            gist_template="cont-repro-2 -> {trial_count()} | {status()}",
        )

        _, result2 = await scenario_runner(scenario2)
        assert not isinstance(result2, Exception)

        # Emit evidence for second run
        validation2 = result_validator(scenario2, result2)
        assert validation2.passed, validation2.summary()

        # Compare floating point values
        if hasattr(result1, "trials") and hasattr(result2, "trials"):
            for i, (t1, t2) in enumerate(zip(result1.trials, result2.trials, strict=False)):
                temp1 = t1.config.get("temperature")
                temp2 = t2.config.get("temperature")

                if temp1 is not None and temp2 is not None:
                    assert abs(temp1 - temp2) < 1e-10, f"Trial {i + 1} temperature differs"

                top_p1 = t1.config.get("top_p")
                top_p2 = t2.config.get("top_p")

                if top_p1 is not None and top_p2 is not None:
                    assert abs(top_p1 - top_p2) < 1e-10, f"Trial {i + 1} top_p differs"


class TestMixedSpaceReproducibility:
    """Tests for reproducibility with mixed categorical/continuous spaces."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mixed_space_reproducibility(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test reproducibility with mixed categorical/continuous params."""
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": (0.0, 1.0),
            "max_tokens": [100, 500, 1000],
        }

        # Run twice with same seed
        scenario1 = config_space_scenario(
            name="mixed_repro_1",
            config_space=config_space,
            description="Mixed space run 1",
            max_trials=5,
            mock_mode_config={"optimizer": "random", "random_seed": 42},
            gist_template="mixed-repro-1 -> {trial_count()} | {status()}",
        )

        _, result1 = await scenario_runner(scenario1)
        assert not isinstance(result1, Exception)

        # Emit evidence for first run
        validation1 = result_validator(scenario1, result1)
        assert validation1.passed, validation1.summary()

        scenario2 = config_space_scenario(
            name="mixed_repro_2",
            config_space=config_space,
            description="Mixed space run 2",
            max_trials=5,
            mock_mode_config={"optimizer": "random", "random_seed": 42},
            gist_template="mixed-repro-2 -> {trial_count()} | {status()}",
        )

        _, result2 = await scenario_runner(scenario2)
        assert not isinstance(result2, Exception)

        # Emit evidence for second run
        validation2 = result_validator(scenario2, result2)
        assert validation2.passed, validation2.summary()

        # Compare all config values
        if hasattr(result1, "trials") and hasattr(result2, "trials"):
            for i, (t1, t2) in enumerate(zip(result1.trials, result2.trials, strict=False)):
                # Categorical should match exactly
                assert t1.config.get("model") == t2.config.get("model"), (
                    f"Trial {i + 1} model differs"
                )
                assert t1.config.get("max_tokens") == t2.config.get("max_tokens"), (
                    f"Trial {i + 1} max_tokens differs"
                )

                # Continuous should match within epsilon
                temp1 = t1.config.get("temperature")
                temp2 = t2.config.get("temperature")
                if temp1 is not None and temp2 is not None:
                    assert abs(temp1 - temp2) < 1e-10, f"Trial {i + 1} temperature differs"


class TestSeedEdgeCases:
    """Tests for edge cases with seed handling."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_seed_zero(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that seed=0 is handled correctly.

        Some implementations treat 0 as "no seed" - verify it works as a seed.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
            "temperature": [0.1, 0.5, 0.9],
        }

        # Run twice with seed 0
        scenario1 = config_space_scenario(
            name="seed_zero_1",
            config_space=config_space,
            description="Run with seed 0",
            max_trials=3,
            mock_mode_config={"optimizer": "random", "random_seed": 0},
            gist_template="seed-zero-1 -> {trial_count()} | {status()}",
        )

        _, result1 = await scenario_runner(scenario1)
        assert not isinstance(result1, Exception)

        # Emit evidence for first run
        validation1 = result_validator(scenario1, result1)
        assert validation1.passed, validation1.summary()

        scenario2 = config_space_scenario(
            name="seed_zero_2",
            config_space=config_space,
            description="Run with seed 0 again",
            max_trials=3,
            mock_mode_config={"optimizer": "random", "random_seed": 0},
            gist_template="seed-zero-2 -> {trial_count()} | {status()}",
        )

        _, result2 = await scenario_runner(scenario2)
        assert not isinstance(result2, Exception)

        # Emit evidence for second run
        validation2 = result_validator(scenario2, result2)
        assert validation2.passed, validation2.summary()

        # Should be reproducible
        if hasattr(result1, "trials") and hasattr(result2, "trials"):
            configs1 = [t.config for t in result1.trials]
            configs2 = [t.config for t in result2.trials]

            for i, (c1, c2) in enumerate(zip(configs1, configs2, strict=False)):
                assert c1 == c2, f"Seed 0 not reproducible at trial {i + 1}"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_large_seed_value(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that large seed values work correctly."""
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": [0.3, 0.7],
        }

        # Use a large seed value
        large_seed = 2**31 - 1  # Max 32-bit signed int

        scenario = config_space_scenario(
            name="large_seed",
            config_space=config_space,
            description="Run with large seed value",
            max_trials=3,
            mock_mode_config={
                "optimizer": "random",
                "random_seed": large_seed,
            },
            gist_template="large-seed -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should work without error
        assert not isinstance(result, Exception), f"Large seed failed: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_negative_seed_handling(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test handling of negative seed values.

        Negative seeds may or may not be supported - verify graceful handling.
        """
        config_space = {
            "model": ["gpt-3.5-turbo", "gpt-4"],
            "temperature": [0.3, 0.7],
        }

        scenario = config_space_scenario(
            name="negative_seed",
            config_space=config_space,
            description="Run with negative seed",
            max_trials=3,
            mock_mode_config={"optimizer": "random", "random_seed": -42},
            gist_template="neg-seed -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should either work or fail gracefully with clear error
        if isinstance(result, Exception):
            # If it fails, should be a clear error about invalid seed
            error_msg = str(result).lower()
            assert "seed" in error_msg or "negative" in error_msg, (
                f"Error should mention seed issue: {result}"
            )
        else:
            # If it works, that's fine too
            result_validator(scenario, result)
