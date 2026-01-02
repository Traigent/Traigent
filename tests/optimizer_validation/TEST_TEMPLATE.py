"""Test template with mandatory behavior assertions.

This template demonstrates the required assertion patterns for optimizer validation tests.
Copy and customize this template when creating new tests.

MANDATORY SECTIONS:
    1. Exception check (always)
    2. Trial verification (always)
    3. Behavior-specific assertions (customize per test)
    4. Validator call with assertion (always)

DO NOT:
    - Call result_validator without asserting validation.passed
    - Use only "assert not isinstance(result, Exception)" as success check
    - Use vacuous assertions like "len(x) >= 0" or "assert True"
    - Skip explicit assertions and rely solely on the validator
"""

from __future__ import annotations

import pytest

from tests.optimizer_validation.specs import (
    ExpectedResult,
    ObjectiveSpec,
    TestScenario,
)


class TestTemplateExample:
    """Example test class following the required pattern."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_example_with_full_assertions(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Example test with all mandatory assertion sections.

        Test Purpose:
            [Describe what behavior this test verifies]

        Expected Behavior:
            [Describe expected outcome and how to verify it]

        Dimensions Tested:
            [List the dimensions/features being tested]
        """
        # =====================================================================
        # SECTION 1: Scenario Definition
        # =====================================================================
        scenario = TestScenario(
            name="example_test",
            description="Example demonstrating required assertion patterns",
            objectives=[ObjectiveSpec(name="accuracy", orientation="maximize")],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.7],
            },
            max_trials=4,
            mock_mode_config={"optimizer": "grid"},
            expected=ExpectedResult(
                min_trials=4,
                max_trials=4,
                expected_stop_reason="max_trials_reached",  # RECOMMENDED: Set this
                # best_score_range=(0.8, 1.0),  # RECOMMENDED: Set for optimization tests
            ),
            gist_template="example -> {trial_count()} | {status()}",
        )

        # =====================================================================
        # SECTION 2: Execute Scenario
        # =====================================================================
        _, result = await scenario_runner(scenario)

        # =====================================================================
        # SECTION 3: MANDATORY - Exception Check
        # =====================================================================
        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # =====================================================================
        # SECTION 4: MANDATORY - Trial Verification
        # =====================================================================
        # Every test MUST verify trials were executed
        assert hasattr(result, "trials"), "Result should have trials attribute"
        assert len(result.trials) >= 1, "Should complete at least one trial"

        # Verify trial structure
        for i, trial in enumerate(result.trials):
            config = getattr(trial, "config", {})
            assert config, f"Trial {i} should have config"

            # Verify config values are from expected space
            if "model" in config:
                assert config["model"] in [
                    "gpt-3.5-turbo",
                    "gpt-4",
                ], f"Trial {i} has invalid model: {config['model']}"
            if "temperature" in config:
                assert config["temperature"] in [
                    0.3,
                    0.7,
                ], f"Trial {i} has invalid temperature: {config['temperature']}"

        # =====================================================================
        # SECTION 5: RECOMMENDED - Behavior-Specific Assertions
        # =====================================================================
        # Add assertions specific to the feature being tested

        # Example: Verify stop reason (for stop condition tests)
        if hasattr(result, "stop_reason") and result.stop_reason:
            # Customize based on expected stop condition
            valid_reasons = ["max_trials_reached", "config_exhaustion", "optimizer"]
            assert (
                result.stop_reason in valid_reasons
            ), f"Unexpected stop reason: {result.stop_reason}"

        # Example: Verify best_config (for optimization tests)
        if result.best_config is not None:
            assert "model" in result.best_config, "best_config should have model"
            assert (
                "temperature" in result.best_config
            ), "best_config should have temperature"

        # Example: Verify metrics (for evaluator tests)
        for trial in result.trials:
            if trial.metrics:
                assert "accuracy" in trial.metrics, f"Trial should have accuracy metric"
                acc = trial.metrics["accuracy"]
                assert 0.0 <= acc <= 1.0, f"Accuracy should be in [0,1], got {acc}"

        # =====================================================================
        # SECTION 6: MANDATORY - Validator with Assertion
        # =====================================================================
        # ALWAYS capture the validation result and assert it passed
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


# =============================================================================
# QUICK REFERENCE: Common Assertion Patterns
# =============================================================================

"""
TRIAL VERIFICATION:
    assert len(result.trials) >= expected_min, "Message"
    assert len(result.trials) == exact_count, "Message"

CONFIG VERIFICATION:
    for trial in result.trials:
        assert trial.config, "Trial should have config"
        assert "key" in trial.config, "Config should have key"
        assert trial.config["key"] in valid_values, "Invalid value"

STOP REASON VERIFICATION:
    assert result.stop_reason == "expected_reason", f"Got: {result.stop_reason}"
    assert result.stop_reason in ["reason1", "reason2"], "Unexpected stop"

BEST SCORE VERIFICATION:
    assert result.best_score is not None, "Should have best_score"
    assert result.best_score >= min_expected, f"Score too low: {result.best_score}"
    assert min_val <= result.best_score <= max_val, "Score out of range"

BEST CONFIG VERIFICATION:
    assert result.best_config is not None, "Should have best_config"
    assert "param" in result.best_config, "Missing param in best_config"

METRICS VERIFICATION:
    for trial in result.trials:
        assert trial.metrics, f"Trial {trial.trial_id} has no metrics"
        assert "metric_name" in trial.metrics, "Missing metric"
        assert not math.isnan(trial.metrics["metric_name"]), "Metric is NaN"

FAILURE VERIFICATION (for expected failures):
    assert isinstance(result, expected_exception_type), f"Wrong exception: {type(result)}"
    assert "expected_text" in str(result), f"Wrong message: {result}"

VALIDATOR (always last):
    validation = result_validator(scenario, result)
    assert validation.passed, validation.summary()
"""

# =============================================================================
# ANTI-PATTERNS TO AVOID
# =============================================================================

"""
DON'T:
    result_validator(scenario, result)  # No assertion on result!

DO:
    validation = result_validator(scenario, result)
    assert validation.passed, validation.summary()

---

DON'T:
    assert len(result.trials) >= 0  # Vacuous - always true

DO:
    assert len(result.trials) >= 1  # Meaningful minimum

---

DON'T:
    assert True  # No-op assertion

DO:
    assert actual == expected  # Meaningful comparison

---

DON'T:
    # Only exception check, no behavior verification
    assert not isinstance(result, Exception)
    validation = result_validator(scenario, result)
    assert validation.passed

DO:
    assert not isinstance(result, Exception)
    assert len(result.trials) >= 1  # Explicit behavior check
    # ... more behavior assertions ...
    validation = result_validator(scenario, result)
    assert validation.passed
"""
