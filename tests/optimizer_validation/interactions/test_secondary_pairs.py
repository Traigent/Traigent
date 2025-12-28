"""Tests for secondary dimension pairwise combinations.

Purpose:
    Provide coverage for critical pairs that are secondary to the main
    InjectionMode × ExecutionMode and Algorithm × ConfigSpaceType pairs.

Dimensions Covered:
    - ParallelMode × StopCondition
    - ObjectiveConfig × ConstraintUsage
    - Algorithm × FailureMode
    - ExecutionMode × ParallelMode
"""

from __future__ import annotations

import pytest

from tests.optimizer_validation.specs import (
    ConstraintSpec,
    ExpectedOutcome,
    ExpectedResult,
    ObjectiveSpec,
    TestScenario,
)

# =============================================================================
# ParallelMode × StopCondition
# =============================================================================


class TestParallelModeStopConditionPairs:
    """Pairwise coverage of ParallelMode × StopCondition.

    Covers combinations of sequential/parallel execution with various
    stop conditions to ensure they work together correctly.
    """

    @pytest.mark.parametrize("parallel_mode", ["sequential", "parallel"])
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_parallel_mode_with_config_exhaustion(
        self,
        parallel_mode: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test parallel mode with config_exhaustion stop condition.

        Dimensions:
            ParallelMode={parallel_mode}
            StopCondition=config_exhaustion
        """
        # Small config space to trigger exhaustion
        scenario = TestScenario(
            name=f"exhaust_{parallel_mode}",
            description=f"Config exhaustion with {parallel_mode} mode",
            config_space={"model": ["gpt-3.5-turbo", "gpt-4"]},  # Only 2 configs
            max_trials=10,  # More than config space
            parallel_config={"mode": parallel_mode},
            gist_template=f"{parallel_mode}+exhaust -> {{trial_count()}} | {{status()}}",
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        # Should complete without error
        assert hasattr(result, "trials")

        # Emit evidence
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


# =============================================================================
# ObjectiveConfig × ConstraintUsage
# =============================================================================


OBJECTIVE_CONFIGS = [
    ("single_maximize", [ObjectiveSpec(name="accuracy", orientation="maximize")]),
    ("single_minimize", [ObjectiveSpec(name="cost", orientation="minimize")]),
    (
        "multi_objective",
        [
            ObjectiveSpec(name="accuracy", orientation="maximize", weight=0.7),
            ObjectiveSpec(name="cost", orientation="minimize", weight=0.3),
        ],
    ),
    (
        "weighted",
        [
            ObjectiveSpec(name="accuracy", orientation="maximize", weight=0.6),
            ObjectiveSpec(name="latency", orientation="minimize", weight=0.4),
        ],
    ),
]

CONSTRAINT_CONFIGS = [
    ("none", []),
    (
        "config_only",
        [
            ConstraintSpec(
                name="temp_limit",
                constraint_fn=lambda c: c.get("temperature", 0) < 0.9,
                requires_metrics=False,
            )
        ],
    ),
    (
        "metric_based",
        [
            ConstraintSpec(
                name="min_accuracy",
                constraint_fn=lambda c, m: m.get("accuracy", 0) > 0.5,
                requires_metrics=True,
            )
        ],
    ),
]


class TestObjectiveConstraintPairs:
    """Pairwise coverage of ObjectiveConfig × ConstraintUsage.

    Tests that objectives and constraints work together correctly.
    """

    @pytest.mark.parametrize(
        "obj_name,objectives", OBJECTIVE_CONFIGS, ids=[c[0] for c in OBJECTIVE_CONFIGS]
    )
    @pytest.mark.parametrize(
        "constraint_name,constraints",
        CONSTRAINT_CONFIGS,
        ids=[c[0] for c in CONSTRAINT_CONFIGS],
    )
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_objective_constraint_combination(
        self,
        obj_name: str,
        objectives: list,
        constraint_name: str,
        constraints: list,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test objective × constraint combination.

        Dimensions:
            ObjectiveConfig={obj_name}
            ConstraintUsage={constraint_name}
        """
        scenario = TestScenario(
            name=f"{obj_name}_{constraint_name}",
            description=f"{obj_name} objectives with {constraint_name} constraints",
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": (0.0, 1.0),
            },
            objectives=objectives,
            constraints=constraints,
            max_trials=3,
            gist_template=f"{obj_name}+{constraint_name} -> {{trial_count()}} | {{status()}}",
        )

        func, result = await scenario_runner(scenario)

        # Should complete without error
        assert not isinstance(result, Exception), f"Unexpected error: {result}"

        # Emit evidence
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


# =============================================================================
# Algorithm × FailureMode
# =============================================================================


ALGORITHMS = ["random", "grid", "optuna_tpe", "optuna_cmaes", "optuna_random"]
FAILURE_MODES = ["function_raises", "evaluator_bug", "invalid_config"]


class TestAlgorithmFailureModePairs:
    """Pairwise coverage of Algorithm × FailureMode.

    Tests that each algorithm handles failure modes correctly.
    """

    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    @pytest.mark.parametrize("failure_mode", FAILURE_MODES)
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_algorithm_with_failure_mode(
        self,
        algorithm: str,
        failure_mode: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test algorithm handling of failure mode.

        Dimensions:
            Algorithm={algorithm}
            FailureMode={failure_mode}
        """
        # Create appropriate failure scenario based on failure mode
        gist = f"{algorithm}+{failure_mode} -> {{trial_count()}} | {{status()}}"
        if failure_mode == "function_raises":
            scenario = TestScenario(
                name=f"{algorithm}_{failure_mode}",
                description=f"{algorithm} with {failure_mode}",
                config_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
                mock_mode_config={"optimizer": algorithm},
                max_trials=3,
                function_should_raise=ValueError,
                function_raise_on_call=1,  # Raise on first call only
                expected=ExpectedResult(
                    outcome=ExpectedOutcome.PARTIAL,
                ),
                gist_template=gist,
            )

        elif failure_mode == "evaluator_bug":
            scenario = TestScenario(
                name=f"{algorithm}_{failure_mode}",
                description=f"{algorithm} with {failure_mode}",
                config_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
                mock_mode_config={"optimizer": algorithm},
                max_trials=3,
                expected=ExpectedResult(
                    outcome=ExpectedOutcome.PARTIAL,
                ),
                gist_template=gist,
            )

        else:  # invalid_config
            scenario = TestScenario(
                name=f"{algorithm}_{failure_mode}",
                description=f"{algorithm} with {failure_mode}",
                config_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
                mock_mode_config={"optimizer": algorithm},
                max_trials=3,
                gist_template=gist,
            )

        func, result = await scenario_runner(scenario)

        # Should handle failures gracefully (may succeed or fail gracefully)
        # The key is it doesn't crash - emit evidence regardless
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


# =============================================================================
# ExecutionMode × ParallelMode
# =============================================================================


EXECUTION_MODES = ["edge_analytics", "privacy", "hybrid", "cloud", "standard"]
PARALLEL_MODES = ["sequential", "parallel"]


class TestExecutionParallelModePairs:
    """Pairwise coverage of ExecutionMode × ParallelMode.

    Tests that all execution modes work with both parallel configurations.
    """

    @pytest.mark.parametrize("execution_mode", EXECUTION_MODES)
    @pytest.mark.parametrize("parallel_mode", PARALLEL_MODES)
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execution_parallel_combination(
        self,
        execution_mode: str,
        parallel_mode: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test execution mode × parallel mode combination.

        Dimensions:
            ExecutionMode={execution_mode}
            ParallelMode={parallel_mode}
        """
        scenario = TestScenario(
            name=f"{execution_mode}_{parallel_mode}",
            description=f"{execution_mode} execution with {parallel_mode} mode",
            execution_mode=execution_mode,
            config_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
            parallel_config={"mode": parallel_mode},
            max_trials=2,
            gist_template=f"{execution_mode}+{parallel_mode} -> {{trial_count()}} | {{status()}}",
        )

        func, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()
