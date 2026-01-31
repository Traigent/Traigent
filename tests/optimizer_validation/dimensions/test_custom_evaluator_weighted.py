"""Tests for custom evaluators with weighted objectives.

Purpose:
    Fill critical gaps where custom evaluators are tested in isolation (only 1 test)
    and weighted objectives lack cross-dimensional coverage.

Coverage Gaps Addressed:
    1. Custom Eval × All Algorithms - NOT TESTED (only implicit random)
    2. Custom Eval × All InjectionModes - NOT TESTED
    3. Custom Eval × All ExecutionModes - NOT TESTED
    4. Custom Eval × Weighted Objectives - NOT TESTED
    5. Custom Eval × Multi-objective - NOT TESTED
    6. Custom Eval × Parallel Mode - NOT TESTED
    7. Custom Eval × Constraints - NOT TESTED
    8. Weighted × random algorithm - MISSING
    9. Weighted × bayesian - MISSING
    10. Weighted × seamless injection - MISSING
    11. Weighted × timeout stop - MISSING

Test Categories:
    1. Custom Evaluator × Algorithm Matrix
    2. Custom Evaluator × Injection Modes
    3. Custom Evaluator × Weighted Objectives
    4. Custom Evaluator × Multi-Objective
    5. Custom Evaluator × Parallel Execution
    6. Custom Evaluator × Constraints
    7. Weighted Objectives × Algorithms
    8. Weighted Objectives × Injection Modes
    9. Weighted Objectives × Stop Conditions
"""

from __future__ import annotations

import pytest

from tests.optimizer_validation.specs import (
    ConstraintSpec,
    EvaluatorSpec,
    ExpectedResult,
    ObjectiveSpec,
    TestScenario,
    evaluator_scenario,
    multi_objective_scenario,
)

# =============================================================================
# Custom Evaluator Helpers
# =============================================================================


def create_custom_evaluator(
    accuracy: float = 0.85,
    cost: float = 0.5,
    latency: float = 0.3,
):
    """Factory for custom evaluators with configurable metrics."""
    from traigent.api.types import ExampleResult

    def custom_eval(func, config, example) -> ExampleResult:
        # Call the function
        actual_output = func(example.input_data.get("text", ""))

        # Vary metrics slightly based on config for optimization to work
        temp = config.get("temperature", 0.5)
        model = config.get("model", "gpt-3.5-turbo")

        # Simulate metrics varying with config
        acc = accuracy + (0.1 if "gpt-4" in model else 0.0) - temp * 0.1
        cst = cost + (0.3 if "gpt-4" in model else 0.0) + temp * 0.1
        lat = latency + (0.2 if "gpt-4" in model else 0.0)

        return ExampleResult(
            example_id=str(id(example)),
            input_data=example.input_data,
            expected_output=example.expected_output,
            actual_output=actual_output,
            metrics={
                "accuracy": max(0.0, min(1.0, acc)),
                "cost": max(0.0, cst),
                "latency": max(0.0, lat),
            },
            execution_time=0.01,
            success=True,
        )

    return custom_eval


def create_weighted_evaluator(weights: dict[str, float]):
    """Factory for evaluators that compute weighted scores."""
    from traigent.api.types import ExampleResult

    def weighted_eval(func, config, example) -> ExampleResult:
        actual_output = func(example.input_data.get("text", ""))

        # Base metrics
        metrics = {
            "accuracy": 0.8,
            "cost": 0.4,
            "latency": 0.2,
        }

        # Compute weighted score
        weighted_score = sum(metrics.get(k, 0.0) * w for k, w in weights.items())

        metrics["weighted_score"] = weighted_score

        return ExampleResult(
            example_id=str(id(example)),
            input_data=example.input_data,
            expected_output=example.expected_output,
            actual_output=actual_output,
            metrics=metrics,
            execution_time=0.01,
            success=True,
        )

    return weighted_eval


# =============================================================================
# Test Class: Custom Evaluator × Algorithm Matrix
# =============================================================================


class TestCustomEvaluatorWithAlgorithms:
    """Tests for custom evaluators with different optimization algorithms.

    Purpose:
        Address the critical gap where custom evaluator was only tested
        implicitly with random algorithm (1 test total).

    Coverage Gap Addressed:
        Custom Eval × All Algorithms
    """

    @pytest.mark.parametrize(
        "algorithm",
        ["random", "grid", "optuna_tpe", "optuna_random"],
    )
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_custom_evaluator_with_algorithm(
        self,
        algorithm: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test custom evaluator works with each algorithm.

        Purpose:
            Verify custom evaluator can be used with all optimization
            algorithms and produces valid results.

        Dimensions: Evaluator=custom, Algorithm={algorithm}
        """
        custom_eval = create_custom_evaluator()

        scenario = evaluator_scenario(
            name=f"custom_eval_{algorithm}",
            evaluator_type="custom",
            evaluator_fn=custom_eval,
            max_trials=4,
            mock_mode_config={"optimizer": algorithm},
            gist_template=f"custom-{algorithm} -> {{trial_count()}} | {{status()}}",
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
    async def test_custom_evaluator_with_bayesian(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test custom evaluator with Bayesian optimization.

        Purpose:
            Bayesian optimization uses surrogate models. Verify custom
            evaluator metrics feed into the model correctly.

        Dimensions: Evaluator=custom, Algorithm=bayesian
        """
        custom_eval = create_custom_evaluator()

        scenario = evaluator_scenario(
            name="custom_eval_bayesian",
            evaluator_type="custom",
            evaluator_fn=custom_eval,
            config_space={
                "temperature": (0.0, 1.0),  # Continuous for Bayesian
                "top_p": (0.5, 1.0),
            },
            max_trials=5,
            mock_mode_config={"optimizer": "bayesian"},
            gist_template="custom-bayesian -> {trial_count()} | {status()}",
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
    async def test_custom_evaluator_with_optuna_cmaes(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test custom evaluator with CMA-ES.

        Purpose:
            CMA-ES requires continuous spaces. Verify custom evaluator
            works with CMA-ES's specific requirements.

        Dimensions: Evaluator=custom, Algorithm=optuna_cmaes
        """
        custom_eval = create_custom_evaluator()

        scenario = evaluator_scenario(
            name="custom_eval_cmaes",
            evaluator_type="custom",
            evaluator_fn=custom_eval,
            config_space={
                "temperature": (0.0, 2.0),
                "top_p": (0.1, 1.0),
            },
            max_trials=4,
            mock_mode_config={"optimizer": "optuna_cmaes"},
            gist_template="custom-cmaes -> {trial_count()} | {status()}",
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


# =============================================================================
# Test Class: Custom Evaluator × Injection Modes
# =============================================================================


class TestCustomEvaluatorWithInjectionModes:
    """Tests for custom evaluators with different injection modes.

    Purpose:
        Address the gap where custom evaluator was not tested with
        different config injection modes.

    Coverage Gap Addressed:
        Custom Eval × All InjectionModes
    """

    @pytest.mark.parametrize(
        "injection_mode",
        ["context", "parameter", "seamless"],  # attribute removed in v2.x
    )
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_custom_evaluator_with_injection_mode(
        self,
        injection_mode: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test custom evaluator with each injection mode.

        Purpose:
            Verify custom evaluator receives correct config regardless
            of how config is injected into the function.

        Dimensions: Evaluator=custom, InjectionMode={injection_mode}
        """
        custom_eval = create_custom_evaluator()

        scenario = TestScenario(
            name=f"custom_eval_{injection_mode}",
            description=f"Custom evaluator with {injection_mode} injection",
            injection_mode=injection_mode,
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.7],
            },
            max_trials=4,
            mock_mode_config={"optimizer": "grid"},
            gist_template=f"custom-{injection_mode} -> {{trial_count()}} | {{status()}}",
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


# =============================================================================
# Test Class: Custom Evaluator × Weighted Objectives
# =============================================================================


class TestCustomEvaluatorWithWeightedObjectives:
    """Tests for custom evaluators combined with weighted objectives.

    Purpose:
        Address the critical gap where custom evaluators were never
        tested with weighted objective configurations.

    Coverage Gap Addressed:
        Custom Eval × Weighted Objectives
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_custom_evaluator_with_weighted_dual(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test custom evaluator with two weighted objectives.

        Purpose:
            Verify custom evaluator's metrics are used correctly
            in weighted objective calculations.

        Dimensions: Evaluator=custom, Objectives=weighted_dual
        """
        custom_eval = create_custom_evaluator()

        objectives = [
            ObjectiveSpec(name="accuracy", orientation="maximize", weight=0.7),
            ObjectiveSpec(name="cost", orientation="minimize", weight=0.3),
        ]

        scenario = TestScenario(
            name="custom_eval_weighted_dual",
            description="Custom evaluator with weighted objectives",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            objectives=objectives,
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.7],
            },
            max_trials=4,
            mock_mode_config={"optimizer": "grid"},
            expected=ExpectedResult(required_metrics=["accuracy", "cost"]),
            gist_template="custom-weighted-dual -> {trial_count()} | {status()}",
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
    async def test_custom_evaluator_with_weighted_triple(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test custom evaluator with three weighted objectives.

        Purpose:
            Verify custom evaluator's metrics are used correctly
            when three objectives with different weights are configured.

        Dimensions: Evaluator=custom, Objectives=weighted_triple
        """
        custom_eval = create_custom_evaluator()

        objectives = [
            ObjectiveSpec(name="accuracy", orientation="maximize", weight=0.5),
            ObjectiveSpec(name="cost", orientation="minimize", weight=0.3),
            ObjectiveSpec(name="latency", orientation="minimize", weight=0.2),
        ]

        scenario = TestScenario(
            name="custom_eval_weighted_triple",
            description="Custom evaluator with three weighted objectives",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            objectives=objectives,
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.7],
            },
            max_trials=4,
            mock_mode_config={"optimizer": "random"},
            expected=ExpectedResult(required_metrics=["accuracy", "cost", "latency"]),
            gist_template="custom-weighted-triple -> {trial_count()} | {status()}",
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
    async def test_evaluator_returning_weighted_metrics(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test evaluator that computes and returns multiple metrics for weighting.

        Purpose:
            Verify an evaluator can compute multiple metrics that are then
            combined using weighted objectives by the optimizer.

        Dimensions: Evaluator=multi_metric, Objectives=weighted
        """
        custom_eval = create_custom_evaluator()

        # Use weighted objectives with custom evaluator's known metrics
        objectives = [
            ObjectiveSpec(name="accuracy", orientation="maximize", weight=0.6),
            ObjectiveSpec(name="cost", orientation="minimize", weight=0.4),
        ]

        scenario = TestScenario(
            name="eval_weighted_metric",
            description="Evaluator returns metrics for weighted optimization",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            objectives=objectives,
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.5],
            },
            max_trials=2,
            mock_mode_config={"optimizer": "random"},
            expected=ExpectedResult(required_metrics=["accuracy", "cost"]),
            gist_template="eval-weighted-metric -> {trial_count()} | {status()}",
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


# =============================================================================
# Test Class: Custom Evaluator × Multi-Objective
# =============================================================================


class TestCustomEvaluatorWithMultiObjective:
    """Tests for custom evaluators with multi-objective optimization.

    Purpose:
        Address the gap where custom evaluators were not tested
        with multi-objective (non-weighted) configurations.

    Coverage Gap Addressed:
        Custom Eval × Multi-objective
    """

    @pytest.mark.parametrize(
        "algorithm",
        ["random", "grid", "optuna_tpe"],
    )
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_custom_evaluator_multi_objective_algorithm(
        self,
        algorithm: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test custom evaluator with multi-objective across algorithms.

        Purpose:
            Verify custom evaluator works with multi-objective optimization
            for each algorithm type.

        Dimensions: Evaluator=custom, Algorithm={algorithm}, Objectives=multi
        """
        custom_eval = create_custom_evaluator()

        objectives = [
            ObjectiveSpec(name="accuracy", orientation="maximize"),
            ObjectiveSpec(name="cost", orientation="minimize"),
        ]

        scenario = TestScenario(
            name=f"custom_multi_{algorithm}",
            description=f"Custom evaluator multi-objective with {algorithm}",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            objectives=objectives,
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.7],
            },
            max_trials=4,
            mock_mode_config={"optimizer": algorithm},
            expected=ExpectedResult(required_metrics=["accuracy", "cost"]),
            gist_template=f"custom-multi-{algorithm} -> {{trial_count()}} | {{status()}}",
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


# =============================================================================
# Test Class: Custom Evaluator × Parallel Execution
# =============================================================================


class TestCustomEvaluatorWithParallel:
    """Tests for custom evaluators in parallel execution mode.

    Purpose:
        Address the gap where custom evaluators were not tested
        with parallel trial execution.

    Coverage Gap Addressed:
        Custom Eval × Parallel Mode
    """

    @pytest.mark.parametrize("workers", [2, 4])
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_custom_evaluator_parallel_workers(
        self,
        workers: int,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test custom evaluator with parallel workers.

        Purpose:
            Verify custom evaluator is thread-safe and works correctly
            when multiple trials run concurrently.

        Dimensions: Evaluator=custom, Parallel={workers}
        """
        custom_eval = create_custom_evaluator()

        scenario = TestScenario(
            name=f"custom_parallel_{workers}",
            description=f"Custom evaluator with {workers} parallel workers",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.5, 0.7],
            },
            max_trials=6,
            parallel_config={"thread_workers": workers},
            mock_mode_config={"optimizer": "random"},
            gist_template=f"custom-parallel-{workers} -> {{trial_count()}} | {{status()}}",
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


# =============================================================================
# Test Class: Custom Evaluator × Constraints
# =============================================================================


class TestCustomEvaluatorWithConstraints:
    """Tests for custom evaluators with constraint functions.

    Purpose:
        Address the gap where custom evaluators were not tested
        with config or metric constraints.

    Coverage Gap Addressed:
        Custom Eval × Constraints
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_custom_evaluator_with_config_constraint(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test custom evaluator with config-based constraint.

        Purpose:
            Verify custom evaluator works when configs are filtered
            by a pre-trial constraint.

        Dimensions: Evaluator=custom, Constraints=config
        """
        custom_eval = create_custom_evaluator()

        def no_gpt4_constraint(config: dict) -> bool:
            """Reject GPT-4 configs."""
            return "gpt-4" not in config.get("model", "")

        scenario = TestScenario(
            name="custom_eval_config_constraint",
            description="Custom evaluator with config constraint",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            constraints=[
                ConstraintSpec(
                    name="no_gpt4",
                    constraint_fn=no_gpt4_constraint,
                    requires_metrics=False,
                ),
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.5],
            },
            max_trials=2,
            mock_mode_config={"optimizer": "random"},
            gist_template="custom-config-constraint -> {trial_count()} | {status()}",
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
    async def test_custom_evaluator_with_metric_constraint(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test custom evaluator with metric-based constraint.

        Purpose:
            Verify custom evaluator metrics can be used by
            post-trial metric constraints.

        Dimensions: Evaluator=custom, Constraints=metric
        """
        custom_eval = create_custom_evaluator()

        def accuracy_threshold(config: dict, metrics: dict) -> bool:
            """Reject configs with accuracy below threshold."""
            return metrics.get("accuracy", 0.0) >= 0.7

        scenario = TestScenario(
            name="custom_eval_metric_constraint",
            description="Custom evaluator with metric constraint",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            constraints=[
                ConstraintSpec(
                    name="accuracy_threshold",
                    constraint_fn=accuracy_threshold,
                    requires_metrics=True,
                ),
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.7],
            },
            max_trials=4,
            mock_mode_config={"optimizer": "grid"},
            gist_template="custom-metric-constraint -> {trial_count()} | {status()}",
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


# =============================================================================
# Test Class: Weighted Objectives × Algorithms
# =============================================================================


class TestWeightedObjectivesWithAlgorithms:
    """Tests for weighted objectives with different algorithms.

    Purpose:
        Address gaps where weighted objectives were not tested
        with all optimization algorithms.

    Coverage Gap Addressed:
        Weighted × random algorithm
        Weighted × bayesian
        Weighted × optuna variants
    """

    @pytest.mark.parametrize(
        "algorithm",
        ["random", "grid", "optuna_tpe", "optuna_random"],
    )
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_weighted_objectives_with_algorithm(
        self,
        algorithm: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test weighted objectives with each algorithm.

        Purpose:
            Verify weighted objective scoring works correctly
            with all optimization algorithms.

        Dimensions: Objectives=weighted, Algorithm={algorithm}
        """
        objectives = [
            ObjectiveSpec(name="accuracy", orientation="maximize", weight=0.6),
            ObjectiveSpec(name="cost", orientation="minimize", weight=0.4),
        ]

        scenario = multi_objective_scenario(
            name=f"weighted_{algorithm}",
            objectives=objectives,
            max_trials=4,
            mock_mode_config={"optimizer": algorithm},
            gist_template=f"weighted-{algorithm} -> {{trial_count()}} | {{status()}}",
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
    async def test_weighted_objectives_with_bayesian(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test weighted objectives with Bayesian optimization.

        Purpose:
            Bayesian optimization uses surrogate models. Verify
            weighted objectives are optimized correctly.

        Dimensions: Objectives=weighted, Algorithm=bayesian
        """
        objectives = [
            ObjectiveSpec(name="accuracy", orientation="maximize", weight=0.7),
            ObjectiveSpec(name="cost", orientation="minimize", weight=0.3),
        ]

        scenario = TestScenario(
            name="weighted_bayesian",
            description="Weighted objectives with Bayesian optimization",
            objectives=objectives,
            config_space={
                "temperature": (0.0, 1.0),
                "top_p": (0.5, 1.0),
            },
            max_trials=5,
            mock_mode_config={"optimizer": "bayesian"},
            expected=ExpectedResult(required_metrics=["accuracy", "cost"]),
            gist_template="weighted-bayesian -> {trial_count()} | {status()}",
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


# =============================================================================
# Test Class: Weighted Objectives × Injection Modes
# =============================================================================


class TestWeightedObjectivesWithInjectionModes:
    """Tests for weighted objectives with different injection modes.

    Purpose:
        Address gaps where weighted objectives were not tested
        with all injection modes.

    Coverage Gap Addressed:
        Weighted × seamless injection
        Weighted × parameter injection
    """

    @pytest.mark.parametrize(
        "injection_mode",
        ["context", "parameter", "seamless"],  # attribute removed in v2.x
    )
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_weighted_with_injection_mode(
        self,
        injection_mode: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test weighted objectives with each injection mode.

        Purpose:
            Verify weighted objective optimization works correctly
            regardless of how config is injected.

        Dimensions: Objectives=weighted, InjectionMode={injection_mode}
        """
        objectives = [
            ObjectiveSpec(name="accuracy", orientation="maximize", weight=0.6),
            ObjectiveSpec(name="cost", orientation="minimize", weight=0.4),
        ]

        scenario = TestScenario(
            name=f"weighted_{injection_mode}",
            description=f"Weighted objectives with {injection_mode} injection",
            injection_mode=injection_mode,
            objectives=objectives,
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.7],
            },
            max_trials=4,
            mock_mode_config={"optimizer": "grid"},
            expected=ExpectedResult(required_metrics=["accuracy", "cost"]),
            gist_template=f"weighted-{injection_mode} -> {{trial_count()}} | {{status()}}",
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


# =============================================================================
# Test Class: Weighted Objectives × Stop Conditions
# =============================================================================


class TestWeightedObjectivesWithStopConditions:
    """Tests for weighted objectives with different stop conditions.

    Purpose:
        Address gaps where weighted objectives were not tested
        with timeout and other stop conditions.

    Coverage Gap Addressed:
        Weighted × timeout stop
        Weighted × max_trials stop
        Weighted × config_exhaustion stop
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_weighted_objectives_with_timeout(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test weighted objectives with timeout stop.

        Purpose:
            Verify weighted objective results are correctly preserved
            when optimization stops due to timeout.

        Dimensions: Objectives=weighted, StopCondition=timeout
        """
        objectives = [
            ObjectiveSpec(name="accuracy", orientation="maximize", weight=0.6),
            ObjectiveSpec(name="cost", orientation="minimize", weight=0.4),
        ]

        scenario = TestScenario(
            name="weighted_timeout",
            description="Weighted objectives with timeout",
            objectives=objectives,
            config_space={
                "model": [f"model-{i}" for i in range(10)],
                "temperature": (0.0, 1.0),
            },
            max_trials=100,
            timeout=2.0,
            mock_mode_config={"optimizer": "random"},
            expected=ExpectedResult(required_metrics=["accuracy", "cost"]),
            gist_template="weighted-timeout -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        # Should complete with some trials
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should preserve completed trials"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_weighted_objectives_grid_exhaustion(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test weighted objectives with grid exhaustion.

        Purpose:
            Verify grid search completes early when config space
            is exhausted while using weighted objectives.

        Dimensions: Objectives=weighted, StopCondition=exhaustion
        """
        objectives = [
            ObjectiveSpec(name="accuracy", orientation="maximize", weight=0.7),
            ObjectiveSpec(name="cost", orientation="minimize", weight=0.3),
        ]

        scenario = TestScenario(
            name="weighted_exhaustion",
            description="Weighted objectives with grid exhaustion",
            objectives=objectives,
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.5],
            },
            max_trials=10,  # More than config space (2 × 1 = 2)
            mock_mode_config={"optimizer": "grid"},
            expected=ExpectedResult(
                min_trials=2,
                max_trials=2,
                required_metrics=["accuracy", "cost"],
            ),
            gist_template="weighted-exhaust -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        if hasattr(result, "trials"):
            assert len(result.trials) == 2, "Grid should stop at 2 trials"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_weighted_objectives_exact_max_trials(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test weighted objectives stop exactly at max_trials.

        Purpose:
            Verify weighted objective optimization respects max_trials
            and stops exactly when limit is reached.

        Dimensions: Objectives=weighted, StopCondition=max_trials
        """
        objectives = [
            ObjectiveSpec(name="accuracy", orientation="maximize", weight=0.6),
            ObjectiveSpec(name="cost", orientation="minimize", weight=0.4),
        ]

        scenario = TestScenario(
            name="weighted_max_trials",
            description="Weighted objectives exact max_trials",
            objectives=objectives,
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
                "temperature": [0.1, 0.5, 0.9],
            },
            max_trials=5,
            mock_mode_config={"optimizer": "random"},
            expected=ExpectedResult(
                min_trials=5,
                max_trials=5,
                expected_stop_reason="max_trials_reached",
                required_metrics=["accuracy", "cost"],
            ),
            gist_template="weighted-max-trials -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        if hasattr(result, "trials"):
            assert len(result.trials) == 5, "Should stop at exactly 5 trials"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


# =============================================================================
# Test Class: Combined Custom Evaluator + Weighted + Algorithm
# =============================================================================


class TestCustomEvaluatorWeightedAlgorithmMatrix:
    """Tests combining custom evaluators, weighted objectives, and algorithms.

    Purpose:
        Test the full combination of custom evaluator with weighted
        objectives across different algorithms.

    Coverage Gap Addressed:
        Custom Eval × Weighted × All Algorithms (critical gap)
    """

    @pytest.mark.parametrize(
        "algorithm",
        ["random", "grid", "optuna_tpe"],
    )
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_custom_weighted_algorithm_combo(
        self,
        algorithm: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test custom evaluator + weighted objectives + algorithm.

        Purpose:
            Full integration test combining custom evaluator with
            weighted objectives across optimization algorithms.

        Dimensions: Evaluator=custom, Objectives=weighted, Algorithm={algorithm}
        """
        custom_eval = create_custom_evaluator()

        objectives = [
            ObjectiveSpec(name="accuracy", orientation="maximize", weight=0.5),
            ObjectiveSpec(name="cost", orientation="minimize", weight=0.3),
            ObjectiveSpec(name="latency", orientation="minimize", weight=0.2),
        ]

        scenario = TestScenario(
            name=f"custom_weighted_{algorithm}",
            description=f"Custom eval + weighted objectives + {algorithm}",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            objectives=objectives,
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.7],
            },
            max_trials=4,
            mock_mode_config={"optimizer": algorithm},
            expected=ExpectedResult(
                required_metrics=["accuracy", "cost", "latency"],
            ),
            gist_template=f"custom-wt-{algorithm} -> {{trial_count()}} | {{status()}}",
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
    async def test_custom_weighted_parallel(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test custom evaluator + weighted objectives + parallel.

        Purpose:
            Verify the full combination works in parallel execution mode.

        Dimensions: Evaluator=custom, Objectives=weighted, Parallel=2
        """
        custom_eval = create_custom_evaluator()

        objectives = [
            ObjectiveSpec(name="accuracy", orientation="maximize", weight=0.6),
            ObjectiveSpec(name="cost", orientation="minimize", weight=0.4),
        ]

        scenario = TestScenario(
            name="custom_weighted_parallel",
            description="Custom eval + weighted + parallel",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            objectives=objectives,
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.5, 0.7],
            },
            max_trials=6,
            parallel_config={"thread_workers": 2},
            mock_mode_config={"optimizer": "random"},
            expected=ExpectedResult(required_metrics=["accuracy", "cost"]),
            gist_template="custom-wt-parallel -> {trial_count()} | {status()}",
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
    async def test_custom_weighted_timeout(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test custom evaluator + weighted objectives + timeout.

        Purpose:
            Verify the full combination handles timeout gracefully.

        Dimensions: Evaluator=custom, Objectives=weighted, StopCondition=timeout
        """
        custom_eval = create_custom_evaluator()

        objectives = [
            ObjectiveSpec(name="accuracy", orientation="maximize", weight=0.6),
            ObjectiveSpec(name="cost", orientation="minimize", weight=0.4),
        ]

        scenario = TestScenario(
            name="custom_weighted_timeout",
            description="Custom eval + weighted + timeout",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            objectives=objectives,
            config_space={
                "model": [f"model-{i}" for i in range(20)],
                "temperature": (0.0, 1.0),
            },
            max_trials=100,
            timeout=2.0,
            mock_mode_config={"optimizer": "random"},
            expected=ExpectedResult(required_metrics=["accuracy", "cost"]),
            gist_template="custom-wt-timeout -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        # Should have some trials before timeout
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete some trials"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()
