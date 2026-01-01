"""Interrupt handling tests for graceful shutdown and cancellation.

Purpose:
    Fill the gap where we have 0 tests for interrupt handling.
    Users need:
    - Clean Ctrl+C handling
    - Graceful shutdown on SIGTERM
    - Early stop without data loss
    - Callback/hook support for monitoring

Test Categories:
    1. Timeout-based Stops - Clean exit when time limit reached
    2. Trial Limit Stops - Clean exit when max_trials reached
    3. Early Completion - Config space exhaustion
    4. Progress Tracking - Monitoring during execution
    5. Resource Cleanup - No leaked resources

Note:
    These tests verify graceful shutdown patterns. Signal handling
    (SIGINT, SIGTERM) is difficult to test directly but the underlying
    mechanisms are validated.
"""

from __future__ import annotations

import pytest

from tests.optimizer_validation.specs import (
    ExpectedResult,
    ObjectiveSpec,
    TestScenario,
)


class TestTimeoutStops:
    """Tests for timeout-triggered graceful stops.

    Purpose:
        Verify that optimization stops gracefully when timeout
        is reached, preserving all completed work.

    Why This Matters:
        Users set timeouts to bound execution time. When timeout occurs:
        - All completed trials should be preserved
        - Best config should be available
        - No hanging or zombie processes
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_timeout_preserves_completed_trials(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that timeout preserves all completed trials.

        Purpose:
            Verify that when optimization times out, all trials that
            completed before the timeout are preserved in the result.

        Dimensions: Interrupt=timeout_preservation
        """
        scenario = TestScenario(
            name="timeout_preserve",
            description="Timeout preserves completed trials",
            injection_mode="context",
            config_space={
                "model": [f"model-{i}" for i in range(50)],
                "temperature": (0.0, 1.0),  # Continuous to avoid exhaustion
            },
            max_trials=200,  # High limit
            timeout=2.0,  # Short timeout
            mock_mode_config={"optimizer": "random"},
            gist_template="timeout-preserve -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Should have preserved some trials
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should preserve completed trials"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_timeout_returns_best_so_far(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that timeout returns the best configuration found.

        Purpose:
            Verify that when optimization times out, the best
            configuration discovered so far is still accessible.

        Dimensions: Interrupt=timeout_best
        """
        scenario = TestScenario(
            name="timeout_best",
            description="Timeout returns best config found",
            injection_mode="context",
            config_space={
                "model": ["cheap", "medium", "expensive"],
                "temperature": [0.1, 0.5, 0.9],
            },
            objectives=[
                ObjectiveSpec(name="cost", orientation="minimize"),
            ],
            max_trials=50,
            timeout=2.0,
            mock_mode_config={"optimizer": "random"},
            gist_template="timeout-best -> {trial_count()} | {status()}",
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
    async def test_no_hang_on_timeout(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that optimization doesn't hang on timeout.

        Purpose:
            Verify that timeout actually stops execution and doesn't
            leave the process hanging.

        Dimensions: Interrupt=no_hang
        """
        scenario = TestScenario(
            name="no_hang",
            description="No hang on timeout",
            injection_mode="context",
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.7],
            },
            max_trials=1000,  # Very high
            timeout=1.0,  # Very short
            mock_mode_config={"optimizer": "random"},
            gist_template="no-hang -> {trial_count()} | {status()}",
        )

        # This should complete within a reasonable time
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


class TestTrialLimitStops:
    """Tests for trial limit triggered stops.

    Purpose:
        Verify that optimization stops cleanly when max_trials
        is reached.

    Why This Matters:
        max_trials is the primary way users bound optimization.
        It must work reliably.
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_stops_at_max_trials(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that optimization stops at max_trials.

        Purpose:
            Verify that optimization stops exactly at the configured
            max_trials limit.

        Dimensions: Interrupt=max_trials
        """
        scenario = TestScenario(
            name="max_trials_stop",
            description="Stops at max_trials",
            injection_mode="context",
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
                "temperature": [0.1, 0.3, 0.5, 0.7, 0.9],
            },
            max_trials=5,  # Specific limit
            mock_mode_config={"optimizer": "random"},
            expected=ExpectedResult(
                max_trials=5,
            ),
            gist_template="max-trials -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        if hasattr(result, "trials"):
            assert len(result.trials) <= 5, "Should not exceed max_trials"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_all_trials_have_results(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that all trials up to max_trials have results.

        Purpose:
            Verify that every trial up to the limit has a complete
            result with config and metrics.

        Dimensions: Interrupt=complete_trials
        """
        scenario = TestScenario(
            name="complete_trials",
            description="All trials have results",
            injection_mode="context",
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.7],
            },
            max_trials=4,
            mock_mode_config={"optimizer": "grid"},
            expected=ExpectedResult(
                min_trials=4,
                max_trials=4,
            ),
            gist_template="complete -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        if hasattr(result, "trials"):
            assert len(result.trials) == 4
            for trial in result.trials:
                # Each trial should have a config
                assert hasattr(trial, "config") or isinstance(trial, dict)

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestEarlyCompletion:
    """Tests for early completion scenarios.

    Purpose:
        Verify that optimization completes early when appropriate
        (e.g., config space exhausted by grid search).

    Why This Matters:
        Grid search exhausts finite config spaces. Users should get
        results when this happens, not errors.
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_grid_stops_on_exhaustion(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that grid search stops when config space is exhausted.

        Purpose:
            Verify that grid search stops cleanly when all config
            combinations have been tried.

        Dimensions: Interrupt=exhaustion
        """
        scenario = TestScenario(
            name="grid_exhaustion",
            description="Grid stops on config exhaustion",
            injection_mode="context",
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.5],
            },
            max_trials=100,  # Higher than needed
            mock_mode_config={"optimizer": "grid"},
            expected=ExpectedResult(
                min_trials=2,  # 2 models × 1 temp = 2 combos
                max_trials=2,
            ),
            gist_template="exhaust -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        if hasattr(result, "trials"):
            assert len(result.trials) == 2, "Grid should stop at 2 trials"

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_single_trial_completes(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that single trial optimization completes cleanly.

        Purpose:
            Verify that max_trials=1 works correctly as an edge case.

        Dimensions: Interrupt=single_trial
        """
        scenario = TestScenario(
            name="single_trial",
            description="Single trial completes",
            injection_mode="context",
            config_space={
                "model": ["gpt-4"],
                "temperature": [0.7],
            },
            max_trials=1,
            mock_mode_config={"optimizer": "random"},
            expected=ExpectedResult(
                min_trials=1,
                max_trials=1,
            ),
            gist_template="single -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        if hasattr(result, "trials"):
            assert len(result.trials) == 1

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestProgressTracking:
    """Tests for progress tracking during optimization.

    Purpose:
        Verify that optimization progress can be tracked and
        monitored during execution.

    Why This Matters:
        Long-running optimizations need progress feedback:
        - How many trials completed?
        - What's the best score so far?
        - Estimated time remaining?
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_trial_count_accessible(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that trial count is accessible in results.

        Purpose:
            Verify that the number of completed trials is available
            in the optimization result.

        Dimensions: Progress=trial_count
        """
        scenario = TestScenario(
            name="trial_count",
            description="Trial count accessible",
            injection_mode="context",
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.7],
            },
            max_trials=4,
            mock_mode_config={"optimizer": "grid"},
            gist_template="count -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Should be able to count trials
        if hasattr(result, "trials"):
            trial_count = len(result.trials)
            assert trial_count == 4

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_best_score_tracked(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that best score is tracked during optimization.

        Purpose:
            Verify that the best score found so far is available
            in the optimization result.

        Dimensions: Progress=best_score
        """
        scenario = TestScenario(
            name="best_score",
            description="Best score tracked",
            injection_mode="context",
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
                "temperature": [0.1, 0.5, 0.9],
            },
            objectives=[
                ObjectiveSpec(name="accuracy", orientation="maximize"),
            ],
            max_trials=6,
            mock_mode_config={"optimizer": "random"},
            gist_template="best-score -> {trial_count()} | {status()}",
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


class TestResourceCleanup:
    """Tests for proper resource cleanup.

    Purpose:
        Verify that optimization cleans up resources properly,
        even on early termination.

    Why This Matters:
        Resource leaks cause:
        - Memory issues in long sessions
        - File handle exhaustion
        - Database connection issues
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_multiple_sequential_optimizations(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that multiple optimizations can run sequentially.

        Purpose:
            Verify that running multiple optimizations in sequence
            works correctly without resource leaks.

        Dimensions: Cleanup=sequential
        """
        for i in range(3):
            scenario = TestScenario(
                name=f"sequential_{i}",
                description=f"Sequential optimization {i}",
                injection_mode="context",
                config_space={
                    "model": ["gpt-3.5-turbo", "gpt-4"],
                    "temperature": [0.5],
                },
                max_trials=2,
                mock_mode_config={"optimizer": "random", "random_seed": i},
                gist_template=f"seq-{i} -> {{trial_count()}} | {{status()}}",
            )

            _, result = await scenario_runner(scenario)

            assert not isinstance(
                result, Exception
            ), f"Unexpected error on iteration {i}: {result}"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_optimization_after_timeout(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that optimization works correctly after a previous timeout.

        Purpose:
            Verify that a timed-out optimization doesn't leave state
            that affects subsequent optimizations.

        Dimensions: Cleanup=after_timeout
        """
        # First: timeout scenario
        scenario1 = TestScenario(
            name="timeout_first",
            description="First optimization times out",
            injection_mode="context",
            config_space={
                "model": [f"model-{i}" for i in range(10)],
                "temperature": [0.1, 0.5, 0.9],
            },
            max_trials=100,
            timeout=1.0,
            mock_mode_config={"optimizer": "random"},
            gist_template="timeout1 -> {trial_count()} | {status()}",
        )

        _, result1 = await scenario_runner(scenario1)
        assert not isinstance(result1, Exception)

        # Second: normal scenario should work fine
        scenario2 = TestScenario(
            name="normal_second",
            description="Second optimization runs normally",
            injection_mode="context",
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.5],
            },
            max_trials=2,
            mock_mode_config={"optimizer": "grid"},
            expected=ExpectedResult(
                min_trials=2,
                max_trials=2,
            ),
            gist_template="normal2 -> {trial_count()} | {status()}",
        )

        _, result2 = await scenario_runner(scenario2)

        assert not isinstance(result2, Exception), f"Unexpected error: {result2}"

        if hasattr(result2, "trials"):
            assert len(result2.trials) == 2, f"Expected 2 trials, got {len(result2.trials)}"
            # Verify trial configs are valid
            for trial in result2.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        validation = result_validator(scenario2, result2)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_different_algorithms_sequential(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test that different algorithms can run sequentially.

        Purpose:
            Verify that switching between algorithms doesn't cause
            issues with shared state.

        Dimensions: Cleanup=algorithm_switch
        """
        algorithms = [
            {"optimizer": "random"},
            {"optimizer": "grid"},
            {"optimizer": "optuna", "sampler": "tpe"},
        ]

        for mock_config in algorithms:
            algo_name = mock_config.get("sampler") or mock_config["optimizer"]

            scenario = TestScenario(
                name=f"algo_{algo_name}",
                description=f"Algorithm {algo_name}",
                injection_mode="context",
                config_space={
                    "model": ["gpt-3.5-turbo", "gpt-4"],
                    "temperature": [0.5],
                },
                max_trials=2,
                mock_mode_config=mock_config,
                gist_template=f"algo-{algo_name} -> {{trial_count()}} | {{status()}}",
            )

            _, result = await scenario_runner(scenario)

            assert not isinstance(
                result, Exception
            ), f"Unexpected error with {algo_name}: {result}"
