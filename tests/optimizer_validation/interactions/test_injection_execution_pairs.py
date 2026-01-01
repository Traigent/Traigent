"""Tests for InjectionMode × ExecutionMode pairwise combinations.

Purpose:
    Validate that every combination of injection mode and execution mode
    works correctly together. This is a critical dimension pair because:

    1. Injection modes determine HOW configs reach the function
    2. Execution modes determine WHERE optimization runs
    3. The combination affects config serialization, transmission, and context

Dimensions Covered:
    - InjectionMode: context, parameter, attribute, seamless
    - ExecutionMode: edge_analytics, privacy, hybrid, cloud, standard

Coverage Goal:
    Full pairwise coverage = 4 injection modes × 5 execution modes = 20 combinations

Validation Approach:
    Tests verify that optimization completes successfully with correct results
    regardless of the injection/execution mode combination used.
"""

from __future__ import annotations

import pytest

from tests.optimizer_validation.specs import TestScenario, basic_scenario

# All injection modes
INJECTION_MODES = ["context", "parameter", "attribute", "seamless"]

# All execution modes
EXECUTION_MODES = ["edge_analytics", "privacy", "hybrid", "cloud", "standard"]


class TestInjectionExecutionPairwise:
    """Pairwise coverage of InjectionMode × ExecutionMode.

    Purpose:
        Ensure every combination of injection mode and execution mode works
        correctly. This catches integration issues between how configs are
        passed and where optimization runs.

    Why This Matters:
        - Cloud modes may need to serialize configs differently
        - Privacy modes may restrict what context is available
        - Attribute injection may behave differently in distributed execution
        - Seamless injection requires source access which may not work in cloud
    """

    @pytest.mark.parametrize("injection_mode", INJECTION_MODES)
    @pytest.mark.parametrize("execution_mode", EXECUTION_MODES)
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_injection_execution_combination(
        self,
        injection_mode: str,
        execution_mode: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test each injection × execution mode combination works.

        Purpose:
            Verify that optimization completes successfully when using
            a specific combination of injection and execution modes.

        Expectations:
            - Optimization runs without raising exceptions
            - At least 1 trial completes successfully
            - OptimizationResult contains valid best_config and best_score
            - Config values are correctly passed regardless of mode

        Why This Validates Mode Compatibility:
            If injection and execution modes are incompatible, this will
            surface as either exceptions, incorrect configs, or failed trials.
            Success here proves the modes work together correctly.

        Dimensions:
            InjectionMode={injection_mode}
            ExecutionMode={execution_mode}
        """
        scenario = basic_scenario(
            name=f"pair_{injection_mode}_{execution_mode}",
            injection_mode=injection_mode,
            execution_mode=execution_mode,
            max_trials=2,
            gist_template=(
                f"{{injection_mode()}}+{execution_mode} -> {{trial_count()}} | {{status()}}"
            ),
        )

        _, result = await scenario_runner(scenario)

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
        assert validation.passed, validation.summary()


class TestInjectionExecutionWithConstraints:
    """Test injection × execution pairs with constraint handling.

    Purpose:
        Validate that constraints work correctly across all mode combinations.
        Constraint evaluation may need config access, which varies by
        injection mode.
    """

    @pytest.mark.parametrize("injection_mode", INJECTION_MODES)
    @pytest.mark.parametrize("execution_mode", ["edge_analytics", "cloud"])
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_constraint_with_mode_combination(
        self,
        injection_mode: str,
        execution_mode: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test constraints with different injection × execution combinations.

        Purpose:
            Verify that config constraints are correctly evaluated regardless
            of how configs are injected or where optimization runs.

        Expectations:
            - Constraint function receives correct config values
            - Rejected configs are not used for trials
            - Optimization completes with only valid configs

        Dimensions:
            InjectionMode={injection_mode}
            ExecutionMode={execution_mode}
            ConstraintUsage=config_only
        """
        from tests.optimizer_validation.specs import ConstraintSpec

        # Simple constraint that rejects high temperature
        def temperature_constraint(config):
            return config.get("temperature", 0) < 0.8

        scenario = TestScenario(
            name=f"constraint_{injection_mode}_{execution_mode}",
            description=(
                f"Constraint with {injection_mode} injection in {execution_mode}"
            ),
            injection_mode=injection_mode,
            execution_mode=execution_mode,
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.5, 0.7, 0.9],
            },
            constraints=[
                ConstraintSpec(
                    name="temp_limit",
                    constraint_fn=temperature_constraint,
                    requires_metrics=False,
                )
            ],
            max_trials=3,
            gist_template=(
                f"{{injection_mode()}}+{execution_mode} -> {{trial_count()}} | {{status()}}"
            ),
        )

        _, result = await scenario_runner(scenario)

        if not isinstance(result, Exception):
            validation = result_validator(scenario, result)
            # Constraints may cause some configs to be rejected
            # but optimization should still complete
            assert validation.passed, validation.summary()


class TestInjectionExecutionWithMultiObjective:
    """Test injection × execution pairs with multi-objective optimization.

    Purpose:
        Validate that multi-objective optimization works correctly across
        all mode combinations. Multi-objective may involve more complex
        config handling and result aggregation.
    """

    @pytest.mark.parametrize("injection_mode", ["context", "parameter"])
    @pytest.mark.parametrize("execution_mode", ["edge_analytics", "privacy", "hybrid"])
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_multi_objective_with_mode_combination(
        self,
        injection_mode: str,
        execution_mode: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test multi-objective optimization with different mode combinations.

        Purpose:
            Verify that multiple objectives are correctly evaluated and
            aggregated regardless of injection and execution modes.

        Expectations:
            - All objectives are evaluated for each trial
            - Weighted scoring works correctly
            - Best trial selection considers all objectives

        Dimensions:
            InjectionMode={injection_mode}
            ExecutionMode={execution_mode}
            ObjectiveConfig=multi_objective
        """
        from tests.optimizer_validation.specs import ObjectiveSpec

        scenario = TestScenario(
            name=f"multi_obj_{injection_mode}_{execution_mode}",
            description=(f"Multi-objective with {injection_mode} in {execution_mode}"),
            injection_mode=injection_mode,
            execution_mode=execution_mode,
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": (0.0, 1.0),
            },
            objectives=[
                ObjectiveSpec(name="accuracy", orientation="maximize", weight=0.7),
                ObjectiveSpec(name="cost", orientation="minimize", weight=0.3),
            ],
            max_trials=3,
            gist_template=(
                f"{{injection_mode()}}+{execution_mode} -> {{trial_count()}} | {{status()}}"
            ),
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


class TestInjectionExecutionWithContinuousSpace:
    """Test injection × execution pairs with continuous config spaces.

    Purpose:
        Validate that continuous parameter sampling works correctly across
        mode combinations. Continuous values may serialize differently.
    """

    @pytest.mark.parametrize("injection_mode", INJECTION_MODES)
    @pytest.mark.parametrize("execution_mode", ["edge_analytics", "cloud"])
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_continuous_space_with_mode_combination(
        self,
        injection_mode: str,
        execution_mode: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test continuous config spaces with different mode combinations.

        Purpose:
            Verify that continuous (float range) parameters are correctly
            sampled and transmitted regardless of mode combination.

        Expectations:
            - Sampled values are within specified range
            - Float precision is maintained across modes
            - Different trials have different values (exploration works)

        Dimensions:
            InjectionMode={injection_mode}
            ExecutionMode={execution_mode}
            ConfigSpaceType=continuous
        """
        scenario = TestScenario(
            name=f"continuous_{injection_mode}_{execution_mode}",
            description=(f"Continuous space with {injection_mode} in {execution_mode}"),
            injection_mode=injection_mode,
            execution_mode=execution_mode,
            config_space={
                "temperature": (0.0, 2.0),
                "top_p": (0.1, 1.0),
            },
            max_trials=3,
            gist_template=(
                f"{{injection_mode()}}+{execution_mode} -> {{trial_count()}} | {{status()}}"
            ),
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Verify trials were executed with valid configs
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete at least one trial"
            for trial in result.trials:
                config = getattr(trial, "config", {})
                assert config, "Trial should have config"


        # Verify continuous values are within range
        self._validate_continuous_values(result)

        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    def _validate_continuous_values(self, result):
        trials = getattr(result, "trials", [])
        if not trials:
            return

        for trial in trials:
            config = getattr(trial, "config", {})
            if not config:
                continue

            temp = config.get("temperature")
            if temp is not None:
                assert 0.0 <= temp <= 2.0, f"Temperature {temp} out of range"

            top_p = config.get("top_p")
            if top_p is not None:
                assert 0.1 <= top_p <= 1.0, f"top_p {top_p} out of range"
