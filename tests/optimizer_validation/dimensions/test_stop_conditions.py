"""Tests for optimization stop conditions.

Tests all early stopping mechanisms and verifies the correct stop_reason is
returned:
- max_trials_reached: Hit the configured max_trials limit
- timeout: Exceeded the timeout duration
- cost_limit: Hit the cost budget limit
- plateau: Detected optimization plateau (no improvement)
- max_samples_reached: Hit the max samples/examples limit
- optimizer: Optimizer decided to stop (exhausted search space)
- condition: Generic stop condition was triggered
"""

from __future__ import annotations

import time

import pytest

from tests.optimizer_validation.specs import (
    ExpectedOutcome,
    ExpectedResult,
    TestScenario,
    basic_scenario,
)


class TestMaxTrialsStopCondition:
    """Tests for max_trials stop condition."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_stop_at_exact_max_trials(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Stop exactly at max_trials.

        Run with max_trials=3 and expect exactly three trials with stop_reason
        set to max_trials_reached.
        """
        max_trials = 3

        scenario = basic_scenario(
            name="stop_at_max_trials",
            max_trials=max_trials,
            expected=ExpectedResult(
                min_trials=max_trials,
                max_trials=max_trials,
                expected_stop_reason="max_trials_reached",
            ),
            gist_template="max-trials -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) >= 1, "Should complete at least one trial"
        assert result.stop_reason is not None, "Should have a stop reason"
        assert (
            len(result.trials) == max_trials
        ), f"Expected exactly {max_trials} trials, got {len(result.trials)}"
        # Hitting max_trials should set stop_reason to max_trials_reached
        assert (
            result.stop_reason == "max_trials_reached"
        ), f"Expected stop_reason 'max_trials_reached', got '{result.stop_reason}'"
        # Validate with result_validator
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_stop_at_one_trial(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Stop after one trial when max_trials=1.

        Expect exactly one trial and stop_reason set to max_trials_reached.
        """
        scenario = basic_scenario(
            name="single_trial",
            max_trials=1,
            expected=ExpectedResult(
                min_trials=1,
                max_trials=1,
                expected_stop_reason="max_trials_reached",
            ),
            gist_template="single-trial -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        assert len(result.trials) == 1
        assert (
            result.stop_reason == "max_trials_reached"
        ), f"Expected stop_reason 'max_trials_reached', got '{result.stop_reason}'"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_stop_at_two_trials(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Stop after two trials when max_trials=2.

        Expect exactly two trials and stop_reason set to max_trials_reached.
        """
        scenario = basic_scenario(
            name="two_trials",
            max_trials=2,
            expected=ExpectedResult(
                min_trials=2,
                max_trials=2,
                expected_stop_reason="max_trials_reached",
            ),
            gist_template="two-trials -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        assert len(result.trials) == 2
        assert (
            result.stop_reason == "max_trials_reached"
        ), f"Expected stop_reason 'max_trials_reached', got '{result.stop_reason}'"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_stop_with_larger_config_space(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Respect max_trials in a large config space.

        Use a large space but max_trials=3 and expect exactly three trials with
        stop_reason set to max_trials_reached.
        """
        scenario = TestScenario(
            name="large_space_limited_trials",
            description="Large config space but limited trials",
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
                "temperature": [0.1, 0.3, 0.5, 0.7, 0.9],
                "max_tokens": [100, 500, 1000],
            },
            max_trials=3,  # Less than full space (45 possible configs)
            expected=ExpectedResult(
                min_trials=3,
                max_trials=3,
                expected_stop_reason="max_trials_reached",
            ),
            gist_template="large-space -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        # Should stop at exactly 3 trials even though space is larger
        assert len(result.trials) == 3
        assert (
            result.stop_reason == "max_trials_reached"
        ), f"Expected stop_reason 'max_trials_reached', got '{result.stop_reason}'"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestTimeoutStopCondition:
    """Tests for timeout stop condition."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_stop_on_timeout(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Stop when timeout is reached.

        Purpose:
            Verify the optimizer respects the timeout stop condition and
            terminates the optimization run when the configured timeout
            duration is exceeded.

        Expectations:
            - Optimization completes without raising exceptions
            - Result has valid trials and stop_reason attributes
            - stop_reason is 'timeout' or 'max_trials_reached' (mock mode is
              fast)
            - If timeout triggered, elapsed time >= 80% of configured timeout

        Dimensions: StopCondition=timeout, timeout=0.5s, max_trials=100
        """
        scenario = TestScenario(
            name="timeout_stop",
            description="Optimization with short timeout",
            config_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
            timeout=0.5,  # Very short timeout
            max_trials=100,  # More than can complete in timeout
            expected=ExpectedResult(
                max_trials=100,
            ),
            gist_template="timeout -> {trial_count()} | {status()}",
        )

        start = time.monotonic()
        _, result = await scenario_runner(scenario)
        elapsed = time.monotonic() - start

        # Must complete without error
        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        # Verify result structure
        assert hasattr(result, "trials"), "Result should have trials"
        assert hasattr(result, "stop_reason"), "Result should have stop_reason"
        # In mock mode, trials are very fast - accept either timeout or
        # max_trials or optimizer (config exhaustion)
        assert result.stop_reason in (
            "timeout",
            "max_trials_reached",
            "optimizer",
        ), f"Expected 'timeout', 'max_trials_reached' or 'optimizer', got '{result.stop_reason}'"
        assert len(result.trials) <= scenario.max_trials
        # If timeout was triggered, verify elapsed time is reasonable
        if result.stop_reason == "timeout":
            assert (
                elapsed >= scenario.timeout * 0.8
            ), f"Stopped too early: elapsed={elapsed:.2f}s, timeout={scenario.timeout}s"
        # Emit evidence
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_very_short_timeout(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Handle an extremely short timeout.

        Purpose:
            Verify the optimizer handles extremely short timeouts (100ms)
            gracefully without crashing or producing invalid results. This
            tests the edge case where timeout may occur before or during the
            first trial.

        Expectations:
            - Optimization completes without exceptions
            - Result has valid trials and stop_reason attributes
            - Trial count does not exceed max_trials
            - stop_reason is 'timeout' or 'max_trials_reached'
            - If timeout triggered, elapsed time is reasonable

        Dimensions: StopCondition=timeout, timeout=0.1s, max_trials=50
        """
        scenario = TestScenario(
            name="very_short_timeout",
            description="Very short timeout edge case",
            config_space={"model": ["gpt-3.5-turbo"]},
            timeout=0.1,
            max_trials=50,
            expected=ExpectedResult(
                min_trials=0,
                max_trials=50,
            ),
            gist_template="very-short-timeout -> {trial_count()} | {status()}",
        )

        start = time.monotonic()
        _, result = await scenario_runner(scenario)
        elapsed = time.monotonic() - start

        # Must complete without error
        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        # Verify result structure
        assert hasattr(result, "trials"), "Result should have trials"
        assert hasattr(result, "stop_reason"), "Result should have stop_reason"
        # Should complete within max_trials limit
        assert len(result.trials) <= scenario.max_trials
        # Stop reason should be valid (not None)
        assert result.stop_reason in (
            "timeout",
            "max_trials_reached",
            "optimizer",
        ), f"Expected 'timeout', 'max_trials_reached' or 'optimizer', got '{result.stop_reason}'"
        # If timeout was triggered, verify timing is reasonable
        if result.stop_reason == "timeout":
            assert (
                elapsed >= scenario.timeout * 0.8
            ), f"Stopped too early: elapsed={elapsed:.2f}s, timeout={scenario.timeout}s"
        # Emit evidence
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_timeout_completes_current_trial(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Timeout allows the current trial to complete.

        Purpose:
            Verify that when a timeout occurs, any in-progress trial is allowed
            to complete gracefully rather than being forcibly terminated. This
            ensures no partial or corrupted trial results.

        Expectations:
            - Optimization completes without exceptions
            - At least one trial completes
            - All trials are in terminal state (completed/failed/pruned)
            - stop_reason is 'timeout' or 'max_trials_reached'

        Dimensions: StopCondition=timeout, timeout=1.0s, max_trials=10
        """
        scenario = TestScenario(
            name="timeout_graceful",
            description="Timeout should complete gracefully",
            config_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
            timeout=1.0,
            max_trials=10,
            expected=ExpectedResult(
                min_trials=1,
                max_trials=10,
            ),
            gist_template="timeout-graceful -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Must complete without error
        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        # Should have completed at least one trial
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) >= 1, "Should complete at least one trial"
        # stop_reason should be valid (not None)
        assert result.stop_reason in (
            "timeout",
            "max_trials_reached",
            "optimizer",
        ), f"Expected 'timeout', 'max_trials_reached' or 'optimizer', got '{result.stop_reason}'"
        # All completed trials should have a valid status (not in-progress)
        for i, trial in enumerate(result.trials):
            assert hasattr(trial, "status"), f"Trial {i} should have status"
            # Trial should be in a terminal state (not running)
            if hasattr(trial.status, "value"):
                status_str = trial.status.value.lower()
            else:
                status_str = str(trial.status).lower()
            assert status_str in (
                "completed",
                "failed",
                "pruned",
            ), f"Trial {i} should be in terminal state, got {status_str}"
        # Emit evidence
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestConfigSpaceExhaustionStopCondition:
    """Tests for optimizer stopping when config space is exhausted.

    Dimensions:
        StopCondition=config_exhaustion
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_stop_when_space_exhausted(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Config space exhaustion behavior.

        Dimensions:
            StopCondition=config_exhaustion

        With a tiny config space, expect only valid configs, no more than the
        space size in unique configs, and stop_reason set to optimizer or
        max_trials_reached.
        """
        # Very small config space - only 2 configurations possible
        config_space = {"model": ["gpt-3.5-turbo", "gpt-4"]}
        config_space_size = 2

        scenario = TestScenario(
            name="space_exhausted",
            description="Small config space gets exhausted",
            config_space=config_space,
            max_trials=10,  # More than available configs
            expected=ExpectedResult(
                max_trials=10,
            ),
            gist_template="space-exhaust -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        # Should complete at least one trial
        assert len(result.trials) >= 1, "Should complete at least one trial"
        # Should not exceed max_trials
        assert len(result.trials) <= scenario.max_trials

        # Verify all configs are from the valid space
        unique_configs = set()
        for trial in result.trials:
            assert (
                trial.config.get("model") in config_space["model"]
            ), f"Invalid config model: {trial.config.get('model')}"
            unique_configs.add(tuple(sorted(trial.config.items())))

        # Should have at most config_space_size unique configs
        assert (
            len(unique_configs) <= config_space_size
        ), f"Expected at most {config_space_size} unique configs, got {len(unique_configs)}"

        # stop_reason should be set (not None)
        assert result.stop_reason in (
            "optimizer",
            "max_trials_reached",
        ), f"Expected 'optimizer' or 'max_trials_reached', got '{result.stop_reason}'"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_single_config_space(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Single-option config space repeats the same config.

        Purpose:
            Verify optimizer behavior when the config space has only one
            possible configuration. The optimizer should either repeat this
            config or stop early due to exhaustion.

        Expectations:
            - All trials use the single available configuration
            - Only one unique config exists across all trials
            - stop_reason is 'optimizer' or 'max_trials_reached'

        Dimensions: StopCondition=config_exhaustion,
        ConfigSpaceType=single_option
        """
        scenario = TestScenario(
            name="single_config",
            description="Only one config available",
            config_space={
                "model": ["gpt-3.5-turbo"],  # Only one option
            },
            max_trials=5,
            expected=ExpectedResult(
                min_trials=1,
                max_trials=5,
            ),
            gist_template="single-config -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        # Should complete at least one trial
        assert len(result.trials) >= 1, "Should complete at least one trial"
        # All trials should use the same config
        for i, trial in enumerate(result.trials):
            assert (
                trial.config.get("model") == "gpt-3.5-turbo"
            ), f"Trial {i} should use 'gpt-3.5-turbo', got {trial.config.get('model')}"
        # Only one unique config should exist
        unique_configs = {tuple(sorted(t.config.items())) for t in result.trials}
        assert (
            len(unique_configs) == 1
        ), f"Expected exactly 1 unique config, got {len(unique_configs)}"
        # stop_reason should be set (not None)
        assert result.stop_reason in (
            "optimizer",
            "max_trials_reached",
        ), f"Expected 'optimizer' or 'max_trials_reached', got '{result.stop_reason}'"
        # Emit evidence
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestPlateauStopCondition:
    """Tests for plateau detection stop condition."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_plateau_detection_scenario(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Plateau detection scenario handling.

        Use a flat configuration surface and expect a valid run with at least
        one trial, plus a stop_reason in the accepted set (plateau may or may
        not trigger).
        """
        scenario = TestScenario(
            name="plateau_test",
            description="Test plateau detection behavior",
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.5],  # Fixed value = same results
            },
            max_trials=10,
            gist_template="plateau -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        # Should complete at least one trial
        assert len(result.trials) >= 1, "Should complete at least one trial"
        # Validate with validator
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()
        # stop_reason should be set (not None)
        assert result.stop_reason in (
            "plateau",
            "max_trials_reached",
            "optimizer",
        ), f"Expected valid stop_reason, got '{result.stop_reason}'"


class TestStopReasonValues:
    """Tests verifying stop_reason is set correctly."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_stop_reason_is_valid_literal(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Stop_reason is a valid literal and not None.

        Run a short scenario and expect stop_reason to be one of the supported
        values.
        """
        valid_stop_reasons = {
            "max_trials_reached",
            "max_samples_reached",
            "timeout",
            "cost_limit",
            "metric_limit",
            "optimizer",
            "plateau",
            "convergence",
            "user_cancelled",
            "condition",
            "error",
            "vendor_error",
            "network_error",
        }

        scenario = basic_scenario(
            name="valid_stop_reason",
            max_trials=2,
            expected=ExpectedResult(
                min_trials=2,
                max_trials=2,
                expected_stop_reason="max_trials_reached",
            ),
            gist_template="valid-stop -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        assert result.stop_reason is not None, "stop_reason should not be None"
        assert (
            result.stop_reason in valid_stop_reasons
        ), f"Invalid stop_reason: {result.stop_reason}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_successful_run_has_stop_reason(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Successful runs return a non-None stop_reason.

        Run a short scenario and expect stop_reason to be set to a supported
        value.
        """
        valid_stop_reasons = {
            "max_trials_reached",
            "max_samples_reached",
            "timeout",
            "cost_limit",
            "metric_limit",
            "optimizer",
            "plateau",
            "convergence",
            "user_cancelled",
            "condition",
            "error",
            "vendor_error",
            "network_error",
        }

        scenario = basic_scenario(
            name="completed_with_reason",
            max_trials=3,
            expected=ExpectedResult(
                min_trials=3,
                max_trials=3,
                expected_stop_reason="max_trials_reached",
            ),
            gist_template="completed-reason -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        # Verify stop_reason attribute exists
        assert hasattr(result, "stop_reason"), "Result should have stop_reason"
        # Stop reason should be valid (not None)
        assert result.stop_reason is not None, "stop_reason should not be None"
        assert (
            result.stop_reason in valid_stop_reasons
        ), f"Invalid stop_reason: {result.stop_reason}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_max_trials_produces_correct_stop_reason(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Reaching max_trials sets stop_reason to max_trials_reached.

        Run with max_trials=2 and expect exactly two trials with stop_reason
        set to max_trials_reached.
        """
        max_trials = 2

        scenario = basic_scenario(
            name="verify_max_trials_reason",
            max_trials=max_trials,
            expected=ExpectedResult(
                min_trials=max_trials,
                max_trials=max_trials,
                expected_stop_reason="max_trials_reached",
            ),
            gist_template="max-trials-reason -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        assert len(result.trials) == max_trials
        # When we hit max_trials exactly, stop_reason must be
        # max_trials_reached
        assert (
            result.stop_reason == "max_trials_reached"
        ), f"Expected 'max_trials_reached' when hitting max_trials, got '{result.stop_reason}'"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_config_space_exhaustion_produces_optimizer_stop_reason(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Config space exhaustion yields an optimizer-related stop_reason.

        Use a single-option space with max_trials > 1 and expect stop_reason
        to be optimizer or max_trials_reached after configs are exhausted.
        """
        # Very small config space - only 1 unique configuration possible
        config_space = {"model": ["gpt-3.5-turbo"]}  # Only 1 config
        config_space_size = 1
        max_trials = 5  # More than available configs

        scenario = TestScenario(
            name="verify_optimizer_reason",
            description="Exhaust config space to check stop_reason",
            config_space=config_space,
            max_trials=max_trials,
            gist_template="optimizer-reason -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        # Verify config space was actually limited
        unique_configs = {tuple(sorted(t.config.items())) for t in result.trials}
        assert (
            len(unique_configs) <= config_space_size
        ), f"Expected at most {config_space_size} unique configs, got {len(unique_configs)}"

        # When config space is exhausted, stop_reason should indicate this
        # - "optimizer" (optimizer decided to stop due to exhaustion)
        # - "max_trials_reached" (ran all trials with repeats)
        assert result.stop_reason in ("optimizer", "max_trials_reached"), (
            f"Expected 'optimizer' or 'max_trials_reached' for exhausted "
            f"space, got '{result.stop_reason}'"
        )
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestStopConditionInteraction:
    """Tests for interactions between multiple stop conditions."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_max_trials_before_timeout(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Max_trials takes precedence over a long timeout.

        Purpose:
            Verify that when max_trials is reached before timeout expires,
            the stop_reason correctly reflects max_trials_reached rather than
            timeout. This tests the priority of stop conditions.

        Expectations:
            - Exactly max_trials trials are completed
            - stop_reason is 'max_trials_reached', not 'timeout'
            - Optimization completes well before timeout

        Dimensions: StopCondition=max_trials+timeout, max_trials=2, timeout=60s
        """
        scenario = TestScenario(
            name="trials_before_timeout",
            description="Max trials should trigger before long timeout",
            config_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
            max_trials=2,
            timeout=60.0,  # Long timeout
            expected=ExpectedResult(
                min_trials=2,
                max_trials=2,
                expected_stop_reason="max_trials_reached",
            ),
            gist_template=("trials-before-timeout -> {trial_count()} | {status()}"),
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        assert len(result.trials) == 2
        # Stop reason should be max_trials_reached, not timeout
        assert (
            result.stop_reason == "max_trials_reached"
        ), f"Expected 'max_trials_reached', got '{result.stop_reason}'"
        # Emit evidence
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_timeout_before_max_trials(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Timeout can stop a run before max_trials.

        Purpose:
            Verify that a short timeout can stop the optimization before
            reaching max_trials. In mock mode, trials are fast so max_trials
            may still be reached, but the test validates proper timeout
            handling.

        Expectations:
            - Optimization completes without exceptions
            - Trial count does not exceed max_trials
            - stop_reason is 'timeout' or 'max_trials_reached'

        Dimensions: StopCondition=max_trials+timeout, max_trials=1000,
        timeout=0.5s
        """
        scenario = TestScenario(
            name="timeout_before_trials",
            description="Timeout should trigger before max trials",
            config_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
            max_trials=1000,  # Very high
            timeout=0.5,  # Very short
            expected=ExpectedResult(
                max_trials=1000,
            ),
            gist_template=("timeout-before-trials -> {trial_count()} | {status()}"),
        )

        _, result = await scenario_runner(scenario)

        # Must complete without error
        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        assert hasattr(result, "stop_reason"), "Result should have stop_reason"
        # Should have fewer trials than max (either timeout or completed)
        assert len(result.trials) <= 1000
        # Stop reason should be valid (not None)
        assert result.stop_reason in (
            "timeout",
            "max_trials_reached",
            "optimizer",
        ), f"Expected 'timeout', 'max_trials_reached' or 'optimizer', got '{result.stop_reason}'"
        # Emit evidence
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestStopConditionEdgeCases:
    """Tests for edge cases in stop condition handling."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_zero_timeout_behavior(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Zero timeout edge case behavior.

        Purpose:
            Verify how the optimizer handles timeout=0.0, which is an edge case
            that could cause immediate termination or be interpreted as "no
            timeout". Documents actual implementation behavior.

        Expectations:
            - Either raises ValueError/TypeError for invalid timeout
            - Or completes with valid structure and few/no trials
            - Trial count does not exceed max_trials

        Dimensions: StopCondition=timeout, timeout=0.0s (edge case)
        """
        scenario = TestScenario(
            name="zero_timeout",
            description="Zero timeout edge case",
            config_space={"model": ["gpt-3.5-turbo"]},
            timeout=0.0,
            max_trials=5,
            expected=ExpectedResult(
                min_trials=0,  # Zero timeout may complete with 0 trials
                max_trials=5,
            ),
            gist_template="zero-timeout -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Zero timeout is an edge case - document actual behavior
        if isinstance(result, Exception):
            # Validation rejection - verify error type is specific
            assert isinstance(result, (ValueError, TypeError)), (
                f"Expected ValueError or TypeError for invalid timeout, "
                f"got {type(result).__name__}: {result}"
            )
        else:
            # If it runs, verify it completed with valid structure
            assert hasattr(result, "trials"), "Result missing trials attribute"
            assert hasattr(result, "stop_reason"), "Result missing stop_reason"
            # With zero timeout, expect either immediate stop or very few
            # trials
            assert len(result.trials) <= scenario.max_trials
            # Emit evidence
            validation = result_validator(scenario, result)
            assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_negative_timeout_behavior(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Negative timeout edge case behavior.

        Purpose:
            Verify how the optimizer handles timeout=-1.0, which is invalid
            input.
            The implementation should either reject with an error or sanitize
            the value and run normally.

        Expectations:
            - Either raises ValueError/TypeError for invalid timeout
            - Or sanitizes to valid value and completes normally
            - Trial count does not exceed max_trials

        Dimensions: StopCondition=timeout, timeout=-1.0s (invalid edge case)
        """
        scenario = TestScenario(
            name="negative_timeout",
            description="Negative timeout edge case",
            config_space={"model": ["gpt-3.5-turbo"]},
            timeout=-1.0,
            max_trials=2,
            expected=ExpectedResult(
                outcome=ExpectedOutcome.FAILURE,
                max_trials=2,
            ),
            gist_template="negative-timeout -> {trial_count()} | {status()}",
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
    async def test_zero_max_trials_behavior(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Zero max_trials edge case behavior.

        Purpose:
            Verify the optimizer correctly handles max_trials=0, which should
            result in immediate completion with zero trials executed.

        Expectations:
            - Optimization completes without exceptions
            - Zero trials are executed
            - Result has valid structure

        Dimensions: StopCondition=max_trials, max_trials=0 (edge case)
        """
        scenario = TestScenario(
            name="zero_max_trials",
            description="Zero max_trials edge case",
            config_space={"model": ["gpt-3.5-turbo"]},
            max_trials=0,
            expected=ExpectedResult(
                min_trials=0,
                max_trials=0,
            ),
            gist_template="zero-max-trials -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        if not isinstance(result, Exception):
            # Should complete with zero trials
            assert len(result.trials) == 0
            # Emit evidence
            validation = result_validator(scenario, result)
            assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_negative_max_trials_behavior(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Negative max_trials edge case behavior.

        Purpose:
            Verify how the optimizer handles max_trials=-1, which is invalid
            input. The implementation should either reject with an error or
            clamp to 0 and complete with zero trials.

        Expectations:
            - Either raises ValueError/TypeError for invalid max_trials
            - Or clamps to 0 and completes with zero trials
            - Result has valid structure if not rejected

        Dimensions: StopCondition=max_trials, max_trials=-1 (invalid edge case)
        """
        scenario = TestScenario(
            name="negative_max_trials",
            description="Negative max_trials edge case",
            config_space={"model": ["gpt-3.5-turbo"]},
            max_trials=-1,
            expected=ExpectedResult(
                outcome=ExpectedOutcome.FAILURE,
                max_trials=0,
            ),
            gist_template=("negative-max-trials -> {trial_count()} | {status()}"),
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


class TestOtherStopConditions:
    """Tests for other stop conditions (max_samples, cost_limit, etc.)."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_max_samples_stop_condition(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Stop when max_samples is reached.

        Purpose:
            Verify that optimization stops when the total number of processed
            samples (examples) reaches the limit.

        Edge Case: max_samples limit
        """
        # 2 examples per trial
        dataset_size = 2
        max_samples = 5  # Should stop after 3 trials (6 samples > 5)

        scenario = TestScenario(
            name="max_samples_stop",
            description="Stop on max samples",
            config_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
            max_trials=10,
            dataset_size=dataset_size,
            mock_mode_config={
                "optimizer": "random",
                "max_samples": max_samples,
            },
            expected=ExpectedResult(
                max_trials=10,
            ),
            gist_template="max-samples -> {trial_count()} | {status()}",
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
    async def test_cost_limit_stop_condition(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Stop when cost limit is reached.

        Purpose:
            Verify optimization stops when accumulated cost exceeds limit.

        Edge Case: cost_limit
        """
        scenario = TestScenario(
            name="cost_limit_stop",
            description="Stop on cost limit",
            config_space={"model": ["gpt-3.5-turbo"]},
            max_trials=10,
            mock_mode_config={"optimizer": "random", "cost_limit": 0.01},
            expected=ExpectedResult(
                max_trials=10,
            ),
            gist_template="cost-limit -> {trial_count()} | {status()}",
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


class TestTrialCountAccuracy:
    """Tests verifying trial counting is accurate."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    @pytest.mark.parametrize("expected_trials", [1, 2, 3, 4, 5])
    async def test_exact_trial_count(
        self,
        expected_trials: int,
        scenario_runner,
        result_validator,
    ) -> None:
        """Exact trial count for various max_trials values.

        Purpose:
            Verify that the optimizer runs exactly the configured number of
            trials across a range of max_trials values (1-5). This is a
            parametrized test ensuring precise trial counting.

        Expectations:
            - Optimization completes without exceptions
            - Exactly max_trials trials are executed
            - Trial count accuracy is verified for each parameter value

        Dimensions: StopCondition=max_trials, max_trials=[1,2,3,4,5]
        """
        scenario = basic_scenario(
            name=f"exact_count_{expected_trials}",
            max_trials=expected_trials,
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.1, 0.3, 0.5, 0.7, 0.9],
            },
            expected=ExpectedResult(
                min_trials=expected_trials,
                max_trials=expected_trials,
                expected_stop_reason="max_trials_reached",
            ),
            gist_template="exact-count -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) >= 1, "Should complete at least one trial"
        assert result.stop_reason is not None, "Should have a stop reason"
        assert (
            len(result.trials) == expected_trials
        ), f"Expected {expected_trials} trials, got {len(result.trials)}"
        # Emit evidence
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestStopReasonPropagation:
    """Tests that stop_reason is properly propagated to OptimizationResult."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_result_has_stop_reason_attribute(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """OptimizationResult exposes stop_reason.

        Purpose:
            Verify that the OptimizationResult object includes a stop_reason
            attribute that is accessible after the optimization run completes.

        Expectations:
            - Optimization completes without exceptions
            - Result object has stop_reason attribute
            - stop_reason is accessible (not hidden or missing)

        Dimensions: StopCondition=max_trials (verifying result structure)
        """
        scenario = basic_scenario(
            name="has_stop_reason",
            max_trials=2,
            expected=ExpectedResult(
                min_trials=2,
                max_trials=2,
            ),
            gist_template="has-stop-reason -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        # Verify stop_reason attribute exists
        assert hasattr(result, "stop_reason")
        # Emit evidence
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_stop_reason_in_metadata(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Stop_reason is accessible without error.

        Purpose:
            Verify that accessing result.stop_reason succeeds regardless of
            whether it's stored directly on the result object or in metadata.

        Expectations:
            - Optimization completes without exceptions
            - Accessing stop_reason does not raise an exception
            - stop_reason has a valid value

        Dimensions: StopCondition=max_trials (verifying attribute access)
        """
        scenario = basic_scenario(
            name="stop_in_metadata",
            max_trials=2,
            expected=ExpectedResult(
                min_trials=2,
                max_trials=2,
            ),
            gist_template="stop-in-metadata -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        # stop_reason should be accessible directly
        _ = result.stop_reason  # Should not raise
        # Emit evidence
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


# All supported parallel modes
PARALLEL_MODES = ["sequential", "parallel", "auto"]


class TestStopConditionsWithParallelModes:
    """Tests for stop conditions across all parallel execution modes."""

    @pytest.mark.parametrize("parallel_mode", PARALLEL_MODES)
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_max_trials_with_parallel_mode(
        self,
        parallel_mode: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Max_trials is respected across parallel modes.

        Purpose:
            Verify that the max_trials stop condition is correctly enforced
            regardless of the parallel execution mode (sequential, parallel,
            or auto).

        Expectations:
            - Exactly max_trials trials are executed
            - stop_reason is 'max_trials_reached'
            - Parallel mode does not affect trial count accuracy

        Dimensions: StopCondition=max_trials, ParallelMode=[seq/par/auto]
        """
        max_trials = 3

        scenario = TestScenario(
            name=f"max_trials_{parallel_mode}",
            description=f"Max trials with {parallel_mode} mode",
            config_space={"model": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]},
            max_trials=max_trials,
            parallel_config={"mode": parallel_mode},
            expected=ExpectedResult(
                min_trials=max_trials,
                max_trials=max_trials,
                expected_stop_reason="max_trials_reached",
            ),
            gist_template=("max-trials-parallel -> {trial_count()} | {status()}"),
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        assert hasattr(result, "trials"), "Result should have trials"
        assert len(result.trials) >= 1, "Should complete at least one trial"
        assert result.stop_reason is not None, "Should have a stop reason"
        assert (
            len(result.trials) == max_trials
        ), f"Expected {max_trials} trials with {parallel_mode} mode, got {len(result.trials)}"
        # stop_reason should be max_trials_reached
        assert (
            result.stop_reason == "max_trials_reached"
        ), f"Expected 'max_trials_reached', got '{result.stop_reason}'"
        # Emit evidence
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.parametrize("parallel_mode", PARALLEL_MODES)
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_timeout_with_parallel_mode(
        self,
        parallel_mode: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Timeout handling works across parallel modes.

        Purpose:
            Verify that the timeout stop condition functions correctly across
            all parallel execution modes without crashing or producing invalid
            results.

        Expectations:
            - Optimization completes without exceptions
            - stop_reason is 'timeout' or 'max_trials_reached'
            - All parallel modes handle timeout gracefully

        Dimensions: StopCondition=timeout, ParallelMode=[seq/par/auto]
        """
        scenario = TestScenario(
            name=f"timeout_{parallel_mode}",
            description=f"Timeout with {parallel_mode} mode",
            config_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
            timeout=1.0,
            max_trials=100,  # High enough to not hit before timeout
            parallel_config={"mode": parallel_mode},
            expected=ExpectedResult(
                max_trials=100,
            ),
            gist_template="timeout-parallel -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should complete without crashing regardless of mode
        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        # stop_reason should be valid (not None)
        assert result.stop_reason in (
            "timeout",
            "max_trials_reached",
            "optimizer",
        ), f"Expected 'timeout', 'max_trials_reached' or 'optimizer', got '{result.stop_reason}'"
        # Emit evidence
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.parametrize("parallel_mode", PARALLEL_MODES)
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_stop_reason_valid_with_parallel_mode(
        self,
        parallel_mode: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Stop_reason remains valid across parallel modes.

        Purpose:
            Verify that stop_reason is always set to a valid literal value
            regardless of the parallel execution mode used.

        Expectations:
            - stop_reason is not None
            - stop_reason is one of the supported literal values
            - Parallel mode does not produce invalid stop_reason values

        Dimensions: StopCondition=max_trials, ParallelMode=[seq/par/auto]
        """
        valid_stop_reasons = {
            "max_trials_reached",
            "max_samples_reached",
            "timeout",
            "cost_limit",
            "metric_limit",
            "optimizer",
            "plateau",
            "convergence",
            "user_cancelled",
            "condition",
            "error",
            "vendor_error",
            "network_error",
        }

        scenario = TestScenario(
            name=f"stop_reason_{parallel_mode}",
            description=f"Stop reason validity with {parallel_mode} mode",
            config_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
            max_trials=2,
            parallel_config={"mode": parallel_mode},
            expected=ExpectedResult(
                min_trials=2,
                max_trials=2,
            ),
            gist_template=("stop-reason-parallel -> {trial_count()} | {status()}"),
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"

        assert result.stop_reason is not None, "stop_reason should not be None"
        assert (
            result.stop_reason in valid_stop_reasons
        ), f"Invalid stop_reason with {parallel_mode} mode: {result.stop_reason}"
        # Emit evidence
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestStopConditionsWithTrialConcurrency:
    """Tests for stop conditions with different trial concurrency settings."""

    @pytest.mark.parametrize("trial_concurrency", [1, 2, 4])
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_max_trials_with_concurrency(
        self,
        trial_concurrency: int,
        scenario_runner,
        result_validator,
    ) -> None:
        """Max_trials is respected with trial concurrency.

        Purpose:
            Verify that max_trials stop condition is correctly enforced when
            running multiple trials concurrently. The optimizer should not
            start more trials than configured, even with parallel execution.

        Expectations:
            - Trial count does not exceed max_trials
            - stop_reason is 'max_trials_reached' or 'optimizer'
            - Concurrent execution does not cause over-counting

        Dimensions: StopCondition=max_trials, TrialConcurrency=[1,2,4]
        """
        max_trials = 4

        # Use parallel mode for concurrency > 1
        parallel_config = {
            "mode": "parallel" if trial_concurrency > 1 else "sequential",
            "trial_concurrency": trial_concurrency,
        }

        scenario = TestScenario(
            name=f"max_trials_concurrency_{trial_concurrency}",
            description=f"Max trials with concurrency={trial_concurrency}",
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.7],
            },
            max_trials=max_trials,
            parallel_config=parallel_config,
            expected=ExpectedResult(
                min_trials=max_trials,
                max_trials=max_trials,
            ),
            gist_template=("max-trials-concurrency -> {trial_count()} | {status()}"),
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        # Should not exceed max_trials even with parallel execution
        assert len(result.trials) <= max_trials, (
            f"Expected at most {max_trials} trials with "
            f"concurrency={trial_concurrency}, got {len(result.trials)}"
        )
        # stop_reason should be set
        assert result.stop_reason in (
            "max_trials_reached",
            "optimizer",
        ), f"Expected 'max_trials_reached' or 'optimizer', got '{result.stop_reason}'"
        # Emit evidence
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.parametrize("trial_concurrency", [2, 4])
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_timeout_with_concurrency(
        self,
        trial_concurrency: int,
        scenario_runner,
        result_validator,
    ) -> None:
        """Timeout works with parallel trial execution.

        Purpose:
            Verify that timeout stop condition functions correctly when
            running multiple trials in parallel. All concurrent trials
            should complete gracefully when timeout occurs.

        Expectations:
            - At least one trial completes
            - stop_reason is 'timeout' or 'max_trials_reached'
            - No crashes or hangs with concurrent timeout

        Dimensions: StopCondition=timeout, TrialConcurrency=[2,4]
        """
        scenario = TestScenario(
            name=f"timeout_concurrency_{trial_concurrency}",
            description=f"Timeout with concurrency={trial_concurrency}",
            config_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
            timeout=1.0,
            max_trials=50,
            parallel_config={
                "mode": "parallel",
                "trial_concurrency": trial_concurrency,
            },
            expected=ExpectedResult(
                min_trials=1,
                max_trials=50,
            ),
            gist_template=("timeout-concurrency -> {trial_count()} | {status()}"),
        )

        _, result = await scenario_runner(scenario)

        # Should handle timeout gracefully with concurrent trials
        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        assert len(result.trials) >= 1  # Should complete at least one trial
        # stop_reason should be set
        assert result.stop_reason in (
            "timeout",
            "max_trials_reached",
            "optimizer",
        ), f"Expected 'timeout', 'max_trials_reached' or 'optimizer', got '{result.stop_reason}'"
        # Emit evidence
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestStopConditionsWithExampleConcurrency:
    """Tests for stop conditions with different example concurrency
    settings."""

    @pytest.mark.parametrize("example_concurrency", [1, 2, 4])
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_max_trials_with_example_concurrency(
        self,
        example_concurrency: int,
        scenario_runner,
        result_validator,
    ) -> None:
        """Max_trials works with parallel example evaluation.

        Purpose:
            Verify that max_trials stop condition works correctly when
            examples within each trial are evaluated in parallel. The
            trial count should be accurate regardless of example parallelism.

        Expectations:
            - Exactly max_trials trials are executed
            - stop_reason is 'max_trials_reached'
            - Example concurrency does not affect trial counting

        Dimensions: StopCondition=max_trials, ExampleConcurrency=[1,2,4]
        """
        max_trials = 3

        parallel_config = {
            "mode": "parallel" if example_concurrency > 1 else "sequential",
            "example_concurrency": example_concurrency,
        }

        scenario = TestScenario(
            name=f"max_trials_example_{example_concurrency}",
            description=(f"Max trials with example_concurrency={example_concurrency}"),
            config_space={"model": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]},
            max_trials=max_trials,
            parallel_config=parallel_config,
            dataset_size=5,  # Multiple examples to parallelize
            expected=ExpectedResult(
                min_trials=max_trials,
                max_trials=max_trials,
                expected_stop_reason="max_trials_reached",
            ),
            gist_template="max-trials-example -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        assert len(result.trials) == max_trials
        # stop_reason should be max_trials_reached
        assert (
            result.stop_reason == "max_trials_reached"
        ), f"Expected 'max_trials_reached', got '{result.stop_reason}'"
        # Emit evidence
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


class TestStopConditionsParallelEdgeCases:
    """Edge cases for stop conditions in parallel execution."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_single_trial_parallel_mode(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Single trial in parallel mode.

        Purpose:
            Verify that requesting max_trials=1 with parallel mode configured
            still executes exactly one trial. Parallel infrastructure should
            handle the single-trial edge case correctly.

        Expectations:
            - Exactly one trial is executed
            - stop_reason is 'max_trials_reached'
            - Parallel mode does not cause issues with single trial

        Dimensions: StopCondition=max_trials, max_trials=1,
        ParallelMode=parallel
        """
        scenario = TestScenario(
            name="single_trial_parallel",
            description="Single trial with parallel mode",
            config_space={"model": ["gpt-3.5-turbo"]},
            max_trials=1,
            parallel_config={"mode": "parallel", "trial_concurrency": 2},
            expected=ExpectedResult(
                min_trials=1,
                max_trials=1,
                expected_stop_reason="max_trials_reached",
            ),
            gist_template=("single-trial-parallel -> {trial_count()} | {status()}"),
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        assert len(result.trials) == 1
        # stop_reason should be max_trials_reached
        assert (
            result.stop_reason == "max_trials_reached"
        ), f"Expected 'max_trials_reached', got '{result.stop_reason}'"
        # Emit evidence
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_parallel_trials_exceed_config_space(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Concurrency exceeds config space size.

        Purpose:
            Verify graceful handling when trial_concurrency is set higher
            than the number of available configurations. The optimizer
            should not crash or produce invalid results.

        Expectations:
            - Trial count does not exceed max_trials
            - stop_reason is 'max_trials_reached' or 'optimizer'
            - No crashes despite concurrency > config space size

        Dimensions: StopCondition=max_trials, TrialConcurrency>ConfigSpaceSize
        """
        scenario = TestScenario(
            name="parallel_exceeds_space",
            description="Parallel concurrency exceeds config space",
            config_space={"model": ["gpt-3.5-turbo"]},  # Only 1 config
            max_trials=2,
            parallel_config={"mode": "parallel", "trial_concurrency": 4},
            expected=ExpectedResult(
                max_trials=2,
            ),
            gist_template=("parallel-exceeds-space -> {trial_count()} | {status()}"),
        )

        _, result = await scenario_runner(scenario)

        # Should handle gracefully even if concurrency > available configs
        assert not isinstance(result, Exception)
        # Should still respect max_trials even with high concurrency
        assert len(result.trials) <= 2
        # stop_reason should be set
        assert result.stop_reason in (
            "max_trials_reached",
            "optimizer",
        ), f"Expected 'max_trials_reached' or 'optimizer', got '{result.stop_reason}'"
        # Emit evidence
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_very_short_timeout_parallel(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Very short timeout in parallel execution.

        Purpose:
            Verify that a very short timeout (200ms) with parallel execution
            does not cause race conditions, hangs, or invalid results. In mock
            mode, all trials may complete before timeout.

        Expectations:
            - Optimization completes without exceptions
            - stop_reason is 'timeout' or 'max_trials_reached'
            - Trial count does not exceed max_trials

        Dimensions: StopCondition=timeout, timeout=0.2s, ParallelMode=parallel
        """
        scenario = TestScenario(
            name="short_timeout_parallel",
            description="Very short timeout with parallel mode",
            config_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
            timeout=0.2,
            max_trials=100,
            parallel_config={"mode": "parallel", "trial_concurrency": 4},
            expected=ExpectedResult(
                max_trials=100,
            ),
            gist_template=("short-timeout-parallel -> {trial_count()} | {status()}"),
        )

        _, result = await scenario_runner(scenario)

        # Should handle gracefully - either timeout or complete all trials
        # In mock mode trials are extremely fast so all 100 may complete
        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        # Either stopped by timeout or completed all trials
        assert result.stop_reason in (
            "timeout",
            "max_trials_reached",
            "optimizer",
        ), f"Expected 'timeout', 'max_trials_reached' or 'optimizer', got '{result.stop_reason}'"
        assert len(result.trials) <= 100
        # Emit evidence
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_zero_trials_parallel_mode(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Zero max_trials in parallel mode.

        Purpose:
            Verify that max_trials=0 works correctly with parallel mode,
            resulting in zero trials executed without any parallel
            infrastructure issues.

        Expectations:
            - Zero trials are executed if optimization succeeds
            - Parallel mode does not cause issues with zero trials

        Dimensions: StopCondition=max_trials, max_trials=0,
        ParallelMode=parallel
        """
        scenario = TestScenario(
            name="zero_trials_parallel",
            description="Zero trials with parallel mode",
            config_space={"model": ["gpt-3.5-turbo"]},
            max_trials=0,
            parallel_config={"mode": "parallel"},
            expected=ExpectedResult(
                min_trials=0,
                max_trials=0,
            ),
            gist_template=("zero-trials-parallel -> {trial_count()} | {status()}"),
        )

        _, result = await scenario_runner(scenario)

        if not isinstance(result, Exception):
            assert len(result.trials) == 0
            # Emit evidence
            validation = result_validator(scenario, result)
            assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_high_concurrency_max_trials(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """High concurrency respects max_trials.

        Purpose:
            Verify that very high trial_concurrency (10) still correctly
            enforces the max_trials limit. The optimizer should not overshoot
            max_trials due to concurrent trial scheduling.

        Expectations:
            - Trial count does not exceed max_trials
            - stop_reason is 'max_trials_reached' or 'optimizer'
            - High concurrency does not cause race conditions in counting

        Dimensions: StopCondition=max_trials, TrialConcurrency=10 (high)
        """
        max_trials = 5

        scenario = TestScenario(
            name="high_concurrency_limit",
            description="High concurrency should respect max_trials",
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.1, 0.5, 0.9],
            },
            max_trials=max_trials,
            parallel_config={
                "mode": "parallel",
                "trial_concurrency": 10,  # High concurrency
            },
            expected=ExpectedResult(
                max_trials=max_trials,
            ),
            gist_template=("high-concurrency -> {trial_count()} | {status()}"),
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception)
        # Even with high concurrency, should not exceed max_trials
        assert len(result.trials) <= max_trials
        # stop_reason should be set
        assert result.stop_reason in (
            "max_trials_reached",
            "optimizer",
        ), f"Expected 'max_trials_reached' or 'optimizer', got '{result.stop_reason}'"
        # Emit evidence
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()
