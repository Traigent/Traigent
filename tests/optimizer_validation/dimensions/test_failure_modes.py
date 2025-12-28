"""Tests for failure modes during optimization.

Tests how the optimizer handles various failure scenarios:
- timeout: Optimization exceeds time limit (and fails instead of stopping
  gracefully)
"""

from __future__ import annotations

import pytest

from tests.optimizer_validation.specs import ExpectedResult, TestScenario


class TestTimeoutFailureMode:
    """Tests for timeout failure modes."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_strict_timeout_failure(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Optimization fails when timeout is strictly enforced.

        Purpose:
            Verify that when a strict timeout is set and the optimization
            cannot complete a single trial or initialization within that time,
            it reports a failure or handles it gracefully.

        Edge Case: Strict timeout
        """
        # Very short timeout that should trigger before anything useful happens
        timeout = 0.0001

        scenario = TestScenario(
            name="strict_timeout_failure",
            description="Optimization fails due to strict timeout",
            config_space={"model": ["gpt-3.5-turbo"]},
            timeout=timeout,
            max_trials=10,
            expected=ExpectedResult(
                # Depending on implementation, this might be a failure or just
                # 0 trials. We expect it to handle it without crashing, but
                # maybe with 0 trials.
                min_trials=0,
                max_trials=0,
            ),
            gist_template="strict-timeout -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # It should not crash
        assert not isinstance(result, Exception)

        # Should have stop reason timeout or similar
        if hasattr(result, "stop_reason"):
            assert result.stop_reason in [
                "timeout",
                "max_trials_reached",
                "optimizer",
            ]

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()
