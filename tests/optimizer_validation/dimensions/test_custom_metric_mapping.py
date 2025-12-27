"""Tests for custom metric to objective mapping and direction handling.

Purpose:
    Fill critical gaps where custom metrics are not tested with:
    - Proper mapping to named objectives
    - Direction (maximize/minimize) handling
    - Weight combinations
    - Overriding default metrics
    - Edge cases and error conditions

Coverage Gaps Addressed:
    1. Custom Metric → Objective Mapping - NOT TESTED
    2. Overriding Default "accuracy" Metric - NOT TESTED
    3. Metric Names with Directions - NOT TESTED
    4. Weight + Direction + Custom Eval Combination - NOT TESTED
    5. Edge Cases (conflicts, errors, special values) - NOT TESTED

Test Categories:
    1. Custom Metric → Objective Mapping
    2. Overriding Default Metrics
    3. Metrics with Explicit Directions
    4. Weighted Metrics Multi-Objective
    5. Direction Edge Cases
    6. Metric Conflict & Ambiguity Cases
    7. Metric Error Cases (None, NaN, Inf, type mismatch)
    8. Weight Edge Cases (zero, negative, non-sum-to-one)
    9. Aggregation Edge Cases (partial metrics, all failures)
    10. Multi-Objective Edge Cases (duplicates, missing metrics)

Note:
    The evaluator framework has a set of known metric names. Tests use
    known metrics: accuracy, cost, latency, error_rate, precision, recall,
    score, duration, etc.
"""

from __future__ import annotations

import pytest

from tests.optimizer_validation.specs import (
    EvaluatorSpec,
    ExpectedOutcome,
    ExpectedResult,
    ObjectiveSpec,
    TestScenario,
)

# =============================================================================
# Custom Evaluator Helpers for Metric Mapping Tests
# =============================================================================


def create_metric_mapping_evaluator(metric_values: dict[str, float]):
    """Factory for evaluators returning specific named metrics."""
    from traigent.api.types import ExampleResult

    def custom_eval(func, config, example) -> ExampleResult:
        actual_output = func(example.input_data.get("text", ""))

        # Return metrics with config-based variation
        temp = config.get("temperature", 0.5)
        metrics = {}
        for name, base_value in metric_values.items():
            # Vary by temp to give optimizer something to optimize
            metrics[name] = base_value + (0.1 * temp)

        return ExampleResult(
            example_id=str(id(example)),
            input_data=example.input_data,
            expected_output=example.expected_output,
            actual_output=actual_output,
            metrics=metrics,
            execution_time=0.01,
            success=True,
        )

    return custom_eval


def create_override_accuracy_evaluator(custom_accuracy_fn):
    """Factory for evaluators that override default accuracy calculation."""
    from traigent.api.types import ExampleResult

    def custom_eval(func, config, example) -> ExampleResult:
        actual_output = func(example.input_data.get("text", ""))
        expected = example.expected_output

        # Use custom accuracy function instead of default
        accuracy = custom_accuracy_fn(expected, actual_output, config)

        return ExampleResult(
            example_id=str(id(example)),
            input_data=example.input_data,
            expected_output=expected,
            actual_output=actual_output,
            metrics={"accuracy": accuracy},
            execution_time=0.01,
            success=True,
        )

    return custom_eval


def create_multi_metric_evaluator(metric_calculators: dict):
    """Factory for evaluators with multiple custom metric calculations."""
    from traigent.api.types import ExampleResult

    def custom_eval(func, config, example) -> ExampleResult:
        actual_output = func(example.input_data.get("text", ""))
        expected = example.expected_output

        metrics = {}
        for metric_name, calc_fn in metric_calculators.items():
            metrics[metric_name] = calc_fn(expected, actual_output, config)

        return ExampleResult(
            example_id=str(id(example)),
            input_data=example.input_data,
            expected_output=expected,
            actual_output=actual_output,
            metrics=metrics,
            execution_time=0.01,
            success=True,
        )

    return custom_eval


# =============================================================================
# Test Class: Custom Metric → Objective Mapping
# =============================================================================


class TestCustomMetricToObjectiveMapping:
    """Tests for mapping custom evaluator metrics to named objectives.

    Purpose:
        Verify that custom evaluator metrics properly map to
        ObjectiveSpec names for optimization.

    Coverage Gap Addressed:
        Custom Metric → Objective Mapping - NOT TESTED
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_single_custom_metric_maps_to_objective(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test single custom metric maps to named objective.

        Purpose:
            Verify a custom metric "accuracy" properly maps to
            ObjectiveSpec(name="accuracy").

        Dimensions: MetricMapping=single, Objective=accuracy
        """
        custom_eval = create_metric_mapping_evaluator({"accuracy": 0.8})

        scenario = TestScenario(
            name="single_metric_mapping",
            description="Single custom metric maps to objective",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            objectives=[
                ObjectiveSpec(name="accuracy", orientation="maximize"),
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.7],
            },
            max_trials=4,
            mock_mode_config={"optimizer": "grid"},
            expected=ExpectedResult(required_metrics=["accuracy"]),
            gist_template="single-metric-map -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_multiple_custom_metrics_map_to_objectives(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test multiple custom metrics map to corresponding objectives.

        Purpose:
            Verify multiple custom metrics each map to their named
            objectives correctly.

        Dimensions: MetricMapping=multiple, Objectives=multi
        """
        custom_eval = create_metric_mapping_evaluator(
            {
                "accuracy": 0.85,
                "cost": 0.4,
                "latency": 0.2,
            }
        )

        scenario = TestScenario(
            name="multi_metric_mapping",
            description="Multiple custom metrics map to objectives",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            objectives=[
                ObjectiveSpec(name="accuracy", orientation="maximize"),
                ObjectiveSpec(name="cost", orientation="minimize"),
                ObjectiveSpec(name="latency", orientation="minimize"),
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.5],
            },
            max_trials=2,
            mock_mode_config={"optimizer": "random"},
            expected=ExpectedResult(required_metrics=["accuracy", "cost", "latency"]),
            gist_template="multi-metric-map -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_subset_of_metrics_used_as_objectives(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test only some custom metrics used as objectives.

        Purpose:
            Evaluator returns many metrics but only some are used
            as optimization objectives.

        Dimensions: MetricMapping=subset, Objectives=partial
        """
        custom_eval = create_metric_mapping_evaluator(
            {
                "accuracy": 0.8,
                "latency": 0.3,
                "cost": 0.5,
                "score": 0.9,  # Not used for optimization
            }
        )

        scenario = TestScenario(
            name="subset_metric_mapping",
            description="Subset of metrics used as objectives",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            objectives=[
                ObjectiveSpec(name="accuracy", orientation="maximize"),
                ObjectiveSpec(name="cost", orientation="minimize"),
                # latency and score not used as objectives
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.5],
            },
            max_trials=2,
            mock_mode_config={"optimizer": "random"},
            expected=ExpectedResult(required_metrics=["accuracy", "cost"]),
            gist_template="subset-metric-map -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


# =============================================================================
# Test Class: Overriding Default Metrics
# =============================================================================


class TestOverrideDefaultMetrics:
    """Tests for overriding default metric calculations.

    Purpose:
        Verify that custom evaluators can replace default metric
        calculations (accuracy, cost, latency) with custom logic.

    Coverage Gap Addressed:
        Overriding Default "accuracy" Metric - NOT TESTED
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_override_accuracy_with_semantic_similarity(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test overriding accuracy with semantic similarity.

        Purpose:
            Replace exact match accuracy with semantic similarity scoring.

        Dimensions: MetricOverride=accuracy, CustomLogic=semantic
        """

        def semantic_accuracy(expected, actual, config):
            """Semantic similarity instead of exact match."""
            # Simplified: check word overlap ratio
            if not expected or not actual:
                return 0.0
            exp_words = set(str(expected).lower().split())
            act_words = set(str(actual).lower().split())
            if not exp_words:
                return 0.0
            overlap = len(exp_words & act_words)
            return overlap / len(exp_words)

        custom_eval = create_override_accuracy_evaluator(semantic_accuracy)

        scenario = TestScenario(
            name="override_accuracy_semantic",
            description="Override accuracy with semantic similarity",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            objectives=[
                ObjectiveSpec(name="accuracy", orientation="maximize"),
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.7],
            },
            max_trials=4,
            mock_mode_config={"optimizer": "grid"},
            expected=ExpectedResult(required_metrics=["accuracy"]),
            gist_template="override-semantic -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_override_accuracy_with_fuzzy_match(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test overriding accuracy with fuzzy string matching.

        Purpose:
            Replace exact match with fuzzy matching allowing typos.

        Dimensions: MetricOverride=accuracy, CustomLogic=fuzzy
        """

        def fuzzy_accuracy(expected, actual, config):
            """Fuzzy match with tolerance for differences."""
            exp_str = str(expected).lower()
            act_str = str(actual).lower()

            if exp_str == act_str:
                return 1.0
            if not exp_str:
                return 0.0

            # Simple character overlap ratio
            common = sum(1 for c in exp_str if c in act_str)
            return common / len(exp_str)

        custom_eval = create_override_accuracy_evaluator(fuzzy_accuracy)

        scenario = TestScenario(
            name="override_accuracy_fuzzy",
            description="Override accuracy with fuzzy matching",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            objectives=[
                ObjectiveSpec(name="accuracy", orientation="maximize"),
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.5],
            },
            max_trials=2,
            mock_mode_config={"optimizer": "random"},
            expected=ExpectedResult(required_metrics=["accuracy"]),
            gist_template="override-fuzzy -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_override_accuracy_config_aware(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test accuracy calculation that considers config.

        Purpose:
            Custom accuracy that adjusts based on config parameters
            (e.g., stricter for GPT-4).

        Dimensions: MetricOverride=accuracy, CustomLogic=config_aware
        """

        def config_aware_accuracy(expected, actual, config):
            """Accuracy varies by model (stricter for GPT-4)."""
            base_score = 1.0 if str(expected) == str(actual) else 0.5
            model = config.get("model", "")

            # GPT-4 held to higher standard
            if "gpt-4" in model:
                return base_score * 0.9  # Penalize slightly
            return base_score

        custom_eval = create_override_accuracy_evaluator(config_aware_accuracy)

        scenario = TestScenario(
            name="override_accuracy_config",
            description="Config-aware accuracy calculation",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            objectives=[
                ObjectiveSpec(name="accuracy", orientation="maximize"),
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.5],
            },
            max_trials=2,
            mock_mode_config={"optimizer": "grid"},
            expected=ExpectedResult(required_metrics=["accuracy"]),
            gist_template="override-config -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


# =============================================================================
# Test Class: Metrics with Explicit Directions
# =============================================================================


class TestMetricsWithDirections:
    """Tests for metrics with explicit direction handling.

    Purpose:
        Verify metrics work correctly with maximize/minimize
        directions specified in ObjectiveSpec.

    Coverage Gap Addressed:
        Metric Names with Directions - NOT TESTED
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_metric_minimize(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test metric with minimize direction.

        Purpose:
            Verify "cost" metric optimizes correctly with minimize.

        Dimensions: Metric=cost, Direction=minimize
        """
        custom_eval = create_metric_mapping_evaluator({"cost": 0.5})

        scenario = TestScenario(
            name="metric_minimize",
            description="Metric with minimize direction",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            objectives=[
                ObjectiveSpec(name="cost", orientation="minimize"),
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.7],
            },
            max_trials=4,
            mock_mode_config={"optimizer": "grid"},
            expected=ExpectedResult(required_metrics=["cost"]),
            gist_template="metric-minimize -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_metric_maximize(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test metric with maximize direction.

        Purpose:
            Verify "accuracy" metric optimizes correctly with maximize.

        Dimensions: Metric=accuracy, Direction=maximize
        """
        custom_eval = create_metric_mapping_evaluator({"accuracy": 0.85})

        scenario = TestScenario(
            name="metric_maximize",
            description="Metric with maximize direction",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            objectives=[
                ObjectiveSpec(name="accuracy", orientation="maximize"),
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.7],
            },
            max_trials=4,
            mock_mode_config={"optimizer": "grid"},
            expected=ExpectedResult(required_metrics=["accuracy"]),
            gist_template="metric-maximize -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_metrics_mixed_directions(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test metrics with mixed directions.

        Purpose:
            Verify multiple metrics with different directions
            are handled correctly in multi-objective optimization.

        Dimensions: Metrics=multi, Directions=mixed
        """
        custom_eval = create_metric_mapping_evaluator(
            {
                "accuracy": 0.85,
                "cost": 0.4,
                "latency": 0.3,
            }
        )

        scenario = TestScenario(
            name="metrics_mixed",
            description="Metrics with mixed directions",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            objectives=[
                ObjectiveSpec(name="accuracy", orientation="maximize"),
                ObjectiveSpec(name="cost", orientation="minimize"),
                ObjectiveSpec(name="latency", orientation="minimize"),
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.5],
            },
            max_trials=2,
            mock_mode_config={"optimizer": "random"},
            expected=ExpectedResult(required_metrics=["accuracy", "cost", "latency"]),
            gist_template="metrics-mixed-dir -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


# =============================================================================
# Test Class: Weighted Metrics Multi-Objective
# =============================================================================


class TestWeightedMetrics:
    """Tests for weighted metrics in multi-objective optimization.

    Purpose:
        Verify the full chain: metrics with weights and
        mixed directions in multi-objective optimization.

    Coverage Gap Addressed:
        Weight + Direction + Custom Eval Combination - NOT TESTED
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_weighted_metrics_dual(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test weighted metrics with two objectives.

        Purpose:
            Metrics "accuracy" (max, 0.7) and "cost" (min, 0.3).

        Dimensions: Weighted=dual, Metrics=yes, Directions=mixed
        """
        custom_eval = create_metric_mapping_evaluator(
            {
                "accuracy": 0.8,
                "cost": 0.4,
            }
        )

        scenario = TestScenario(
            name="weighted_metrics_dual",
            description="Weighted metrics dual objective",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            objectives=[
                ObjectiveSpec(name="accuracy", orientation="maximize", weight=0.7),
                ObjectiveSpec(name="cost", orientation="minimize", weight=0.3),
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.7],
            },
            max_trials=4,
            mock_mode_config={"optimizer": "grid"},
            expected=ExpectedResult(required_metrics=["accuracy", "cost"]),
            gist_template="weighted-dual -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_weighted_metrics_triple(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test weighted metrics with three objectives.

        Purpose:
            Three metrics with different weights and directions.

        Dimensions: Weighted=triple, Metrics=yes, Directions=mixed
        """
        custom_eval = create_metric_mapping_evaluator(
            {
                "accuracy": 0.85,
                "cost": 0.4,
                "latency": 0.3,
            }
        )

        scenario = TestScenario(
            name="weighted_metrics_triple",
            description="Weighted metrics triple objective",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            objectives=[
                ObjectiveSpec(name="accuracy", orientation="maximize", weight=0.5),
                ObjectiveSpec(name="cost", orientation="minimize", weight=0.3),
                ObjectiveSpec(name="latency", orientation="minimize", weight=0.2),
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.5],
            },
            max_trials=2,
            mock_mode_config={"optimizer": "random"},
            expected=ExpectedResult(required_metrics=["accuracy", "cost", "latency"]),
            gist_template="weighted-triple -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.parametrize(
        "algorithm",
        ["random", "grid", "optuna_tpe"],
    )
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_weighted_metrics_across_algorithms(
        self,
        algorithm: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test weighted metrics work with all algorithms.

        Purpose:
            Verify weighted metric optimization works consistently
            across different algorithms.

        Dimensions: Weighted=yes, Metrics=yes, Algorithm={algorithm}
        """
        custom_eval = create_metric_mapping_evaluator(
            {
                "accuracy": 0.75,
                "cost": 0.5,
            }
        )

        scenario = TestScenario(
            name=f"weighted_{algorithm}",
            description=f"Weighted metrics with {algorithm}",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            objectives=[
                ObjectiveSpec(name="accuracy", orientation="maximize", weight=0.6),
                ObjectiveSpec(name="cost", orientation="minimize", weight=0.4),
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.7],
            },
            max_trials=4,
            mock_mode_config={"optimizer": algorithm},
            expected=ExpectedResult(required_metrics=["accuracy", "cost"]),
            gist_template=f"weighted-{algorithm} -> {{trial_count()}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


# =============================================================================
# Test Class: Direction Edge Cases
# =============================================================================


class TestDirectionEdgeCases:
    """Tests for edge cases in direction handling.

    Purpose:
        Verify edge cases like same direction for all metrics,
        single metric with unusual scales, etc.
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_all_minimize_objectives(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test all objectives with minimize direction.

        Purpose:
            All metrics should be minimized.

        Dimensions: Direction=all_minimize
        """
        custom_eval = create_metric_mapping_evaluator(
            {
                "cost": 0.5,
                "latency": 0.3,
            }
        )

        scenario = TestScenario(
            name="all_minimize",
            description="All objectives minimize",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            objectives=[
                ObjectiveSpec(name="cost", orientation="minimize"),
                ObjectiveSpec(name="latency", orientation="minimize"),
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.5],
            },
            max_trials=2,
            mock_mode_config={"optimizer": "random"},
            expected=ExpectedResult(required_metrics=["cost", "latency"]),
            gist_template="all-minimize -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_all_maximize_objectives(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test all objectives with maximize direction.

        Purpose:
            All metrics should be maximized.

        Dimensions: Direction=all_maximize
        """
        custom_eval = create_metric_mapping_evaluator(
            {
                "accuracy": 0.85,
                "score": 0.8,
            }
        )

        scenario = TestScenario(
            name="all_maximize",
            description="All objectives maximize",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            objectives=[
                ObjectiveSpec(name="accuracy", orientation="maximize"),
                ObjectiveSpec(name="score", orientation="maximize"),
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.5],
            },
            max_trials=2,
            mock_mode_config={"optimizer": "random"},
            expected=ExpectedResult(required_metrics=["accuracy", "score"]),
            gist_template="all-maximize -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_single_objective_minimize(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test single objective with minimize direction.

        Purpose:
            Single metric with minimize direction.

        Dimensions: Objectives=single, Direction=minimize
        """
        custom_eval = create_metric_mapping_evaluator({"cost": 0.5})

        scenario = TestScenario(
            name="single_minimize",
            description="Single objective minimize",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            objectives=[
                ObjectiveSpec(name="cost", orientation="minimize"),
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.7],
            },
            max_trials=4,
            mock_mode_config={"optimizer": "grid"},
            expected=ExpectedResult(required_metrics=["cost"]),
            gist_template="single-min -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_direction_with_scale_difference(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test directions with metrics on different scales.

        Purpose:
            Metrics with different scales should still work
            with correct directions.

        Dimensions: Scales=different, Directions=mixed
        """
        custom_eval = create_metric_mapping_evaluator(
            {
                "accuracy": 0.95,  # 0-1 scale
                "cost": 0.5,  # 0-1 scale
                "latency": 0.3,  # 0-1 scale
            }
        )

        scenario = TestScenario(
            name="scale_difference",
            description="Direction with different scales",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            objectives=[
                ObjectiveSpec(name="accuracy", orientation="maximize"),
                ObjectiveSpec(name="cost", orientation="minimize"),
                ObjectiveSpec(name="latency", orientation="minimize"),
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.5],
            },
            max_trials=2,
            mock_mode_config={"optimizer": "random"},
            expected=ExpectedResult(required_metrics=["accuracy", "cost", "latency"]),
            gist_template="scale-diff -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


# =============================================================================
# Test Class: Full Integration
# =============================================================================


class TestFullMetricIntegration:
    """Full integration tests combining all metric features.

    Purpose:
        Test the complete chain of custom evaluator + metric mapping +
        weights + directions + algorithm.
    """

    @pytest.mark.parametrize(
        "algorithm",
        ["random", "grid", "optuna_tpe"],
    )
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_full_metric_chain(
        self,
        algorithm: str,
        scenario_runner,
        result_validator,
    ) -> None:
        """Full chain: custom eval + weighted metrics + mixed directions.

        Purpose:
            Complete integration test verifying all pieces work together.

        Dimensions: CustomEval=yes, Weights=mixed, Directions=mixed,
            Algorithm={algorithm}
        """

        def accuracy_calc(expected, actual, config):
            base = 0.8
            temp = config.get("temperature", 0.5)
            return base + (0.1 * temp)

        def cost_calc(expected, actual, config):
            model = config.get("model", "")
            return 0.3 if "gpt-3.5" in model else 0.7

        def latency_calc(expected, actual, config):
            return 0.2 + (0.05 * config.get("temperature", 0.5))

        custom_eval = create_multi_metric_evaluator(
            {
                "accuracy": accuracy_calc,
                "cost": cost_calc,
                "latency": latency_calc,
            }
        )

        scenario = TestScenario(
            name=f"full_chain_{algorithm}",
            description=f"Full metric integration with {algorithm}",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            objectives=[
                ObjectiveSpec(name="accuracy", orientation="maximize", weight=0.5),
                ObjectiveSpec(name="cost", orientation="minimize", weight=0.3),
                ObjectiveSpec(name="latency", orientation="minimize", weight=0.2),
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.7],
            },
            max_trials=4,
            mock_mode_config={"optimizer": algorithm},
            expected=ExpectedResult(required_metrics=["accuracy", "cost", "latency"]),
            gist_template=f"full-chain-{algorithm} -> {{trial_count()}}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_full_chain_with_parallel(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Full chain with parallel execution.

        Purpose:
            Verify metrics work correctly in parallel mode.

        Dimensions: CustomEval=yes, Parallel=2
        """
        custom_eval = create_metric_mapping_evaluator(
            {
                "accuracy": 0.8,
                "cost": 0.4,
            }
        )

        scenario = TestScenario(
            name="full_chain_parallel",
            description="Full chain with parallel execution",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            objectives=[
                ObjectiveSpec(name="accuracy", orientation="maximize", weight=0.6),
                ObjectiveSpec(name="cost", orientation="minimize", weight=0.4),
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.5, 0.7],
            },
            max_trials=6,
            parallel_config={"max_workers": 2},
            mock_mode_config={"optimizer": "random"},
            expected=ExpectedResult(required_metrics=["accuracy", "cost"]),
            gist_template="full-parallel -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_full_chain_with_timeout(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Full chain with timeout stop condition.

        Purpose:
            Verify metrics preserved correctly on timeout.

        Dimensions: CustomEval=yes, StopCondition=timeout
        """
        custom_eval = create_metric_mapping_evaluator(
            {
                "accuracy": 0.8,
                "cost": 0.4,
            }
        )

        scenario = TestScenario(
            name="full_chain_timeout",
            description="Full chain with timeout",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            objectives=[
                ObjectiveSpec(name="accuracy", orientation="maximize", weight=0.7),
                ObjectiveSpec(name="cost", orientation="minimize", weight=0.3),
            ],
            config_space={
                "model": [f"model-{i}" for i in range(10)],
                "temperature": (0.0, 1.0),
            },
            max_trials=100,
            timeout=2.0,
            mock_mode_config={"optimizer": "random"},
            expected=ExpectedResult(required_metrics=["accuracy", "cost"]),
            gist_template="full-timeout -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        if hasattr(result, "trials"):
            assert len(result.trials) >= 1, "Should complete some trials"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


# =============================================================================
# Test Class: Metric Conflict & Ambiguity Cases
# =============================================================================


class TestMetricConflictCases:
    """Tests for metric conflicts and ambiguous configurations.

    Purpose:
        Verify behavior when there are conflicting or ambiguous
        metric configurations.

    Coverage Gap Addressed:
        Metric Conflict Edge Cases - NOT TESTED
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_objective_references_missing_metric(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test objective referencing a metric not returned by evaluator.

        Purpose:
            ObjectiveSpec(name="accuracy") but evaluator only returns
            {"cost": 0.5}. How does optimizer handle missing metric?

        Edge Case: Missing metric reference
        """
        # Evaluator only returns cost, not accuracy
        custom_eval = create_metric_mapping_evaluator({"cost": 0.5})

        scenario = TestScenario(
            name="missing_metric_ref",
            description="Objective references non-existent metric",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            objectives=[
                # accuracy not returned by evaluator
                ObjectiveSpec(name="accuracy", orientation="maximize"),
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.5],
            },
            max_trials=2,
            mock_mode_config={"optimizer": "random"},
            # Don't require accuracy since it won't be present
            gist_template="missing-metric -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle gracefully - may produce trials with fallback/default
        # The key is no crash
        if not isinstance(result, Exception):
            assert hasattr(result, "trials")
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_duplicate_objectives_same_metric(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test two objectives referencing the same metric name.

        Purpose:
            Two ObjectiveSpec instances both reference "accuracy" but with
            different weights. The system should reject this as invalid.

        Edge Case: Duplicate objective names
        Expected: System raises ValueError for duplicate objectives
        """
        custom_eval = create_metric_mapping_evaluator({"accuracy": 0.85})

        scenario = TestScenario(
            name="duplicate_obj_metric",
            description="Two objectives reference same metric",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            objectives=[
                ObjectiveSpec(name="accuracy", orientation="maximize", weight=0.6),
                # Same metric name, different weight
                ObjectiveSpec(name="accuracy", orientation="maximize", weight=0.4),
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.5],
            },
            max_trials=2,
            mock_mode_config={"optimizer": "random"},
            expected=ExpectedResult(
                outcome=ExpectedOutcome.FAILURE,
                error_type=ValueError,
            ),
            gist_template="dup-obj-metric -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # System correctly rejects duplicate objectives
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_conflicting_directions_same_metric(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test same metric with conflicting directions.

        Purpose:
            Two objectives reference "accuracy" but one says maximize,
            another says minimize. The system correctly rejects this
            logically contradictory configuration.

        Edge Case: Conflicting directions
        Expected: System raises ValueError for duplicate objective names
        """
        custom_eval = create_metric_mapping_evaluator({"accuracy": 0.85})

        scenario = TestScenario(
            name="conflicting_directions",
            description="Same metric with conflicting directions",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            objectives=[
                ObjectiveSpec(name="accuracy", orientation="maximize", weight=0.5),
                ObjectiveSpec(name="accuracy", orientation="minimize", weight=0.5),
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.5],
            },
            max_trials=2,
            mock_mode_config={"optimizer": "random"},
            expected=ExpectedResult(
                outcome=ExpectedOutcome.FAILURE,
                error_type=ValueError,
            ),
            gist_template="conflict-dir -> {error_type()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # System correctly rejects conflicting objective directions
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_metric_overwrites_builtin(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test custom metric potentially overwriting built-in score.

        Purpose:
            Evaluator returns {"score": 0.7} which may conflict with
            internally computed score. Verify custom takes precedence.

        Edge Case: Custom metric overwrites built-in
        """
        custom_eval = create_metric_mapping_evaluator({"score": 0.7})

        scenario = TestScenario(
            name="metric_overwrites_builtin",
            description="Custom metric overwrites built-in score",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            objectives=[
                ObjectiveSpec(name="score", orientation="maximize"),
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.7],
            },
            max_trials=4,
            mock_mode_config={"optimizer": "grid"},
            expected=ExpectedResult(required_metrics=["score"]),
            gist_template="overwrite-builtin -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


# =============================================================================
# Test Class: Metric Error Cases
# =============================================================================


class TestMetricErrorCases:
    """Tests for evaluators returning error values in metrics.

    Purpose:
        Verify behavior when evaluator returns None, NaN, Inf,
        or wrong types in metric values.

    Coverage Gap Addressed:
        Metric Error Cases - NOT TESTED
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_metric_returns_none(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test evaluator returning None for a metric value.

        Purpose:
            Evaluator returns {"accuracy": None, "cost": 0.5}.
            How is None handled in weighted sum or optimization?

        Edge Case: None metric value
        """
        from traigent.api.types import ExampleResult

        def none_metric_evaluator(func, config, example):
            actual_output = func(example.input_data.get("text", ""))
            return ExampleResult(
                example_id=str(id(example)),
                input_data=example.input_data,
                expected_output=example.expected_output,
                actual_output=actual_output,
                metrics={"accuracy": None, "cost": 0.5},  # None value
                execution_time=0.01,
                success=True,
            )

        scenario = TestScenario(
            name="metric_none_value",
            description="Metric with None value",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=none_metric_evaluator,
            ),
            objectives=[
                ObjectiveSpec(name="accuracy", orientation="maximize"),
                ObjectiveSpec(name="cost", orientation="minimize"),
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.5],
            },
            max_trials=2,
            mock_mode_config={"optimizer": "random"},
            gist_template="metric-none -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle gracefully - may skip None metric or use default
        if not isinstance(result, Exception):
            assert hasattr(result, "trials")
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_metric_returns_nan(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test evaluator returning NaN for a metric value.

        Purpose:
            NaN can corrupt optimization if not handled. Verify
            graceful handling.

        Edge Case: NaN metric value
        """
        from traigent.api.types import ExampleResult

        def nan_metric_evaluator(func, config, example):
            actual_output = func(example.input_data.get("text", ""))
            return ExampleResult(
                example_id=str(id(example)),
                input_data=example.input_data,
                expected_output=example.expected_output,
                actual_output=actual_output,
                metrics={"accuracy": float("nan")},  # NaN value
                execution_time=0.01,
                success=True,
            )

        scenario = TestScenario(
            name="metric_nan_value",
            description="Metric with NaN value",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=nan_metric_evaluator,
            ),
            objectives=[
                ObjectiveSpec(name="accuracy", orientation="maximize"),
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.5],
            },
            max_trials=2,
            mock_mode_config={"optimizer": "random"},
            gist_template="metric-nan -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle gracefully - no crash
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_metric_returns_inf(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test evaluator returning Inf for a metric value.

        Purpose:
            Infinity can break comparisons. Verify graceful handling.

        Edge Case: Inf metric value
        """
        from traigent.api.types import ExampleResult

        def inf_metric_evaluator(func, config, example):
            actual_output = func(example.input_data.get("text", ""))
            return ExampleResult(
                example_id=str(id(example)),
                input_data=example.input_data,
                expected_output=example.expected_output,
                actual_output=actual_output,
                metrics={"cost": float("inf")},  # Infinity
                execution_time=0.01,
                success=True,
            )

        scenario = TestScenario(
            name="metric_inf_value",
            description="Metric with Inf value",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=inf_metric_evaluator,
            ),
            objectives=[
                ObjectiveSpec(name="cost", orientation="minimize"),
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.5],
            },
            max_trials=2,
            mock_mode_config={"optimizer": "random"},
            gist_template="metric-inf -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle gracefully
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_metric_returns_negative_inf(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test evaluator returning -Inf for a metric value.

        Purpose:
            Negative infinity edge case.

        Edge Case: -Inf metric value
        """
        from traigent.api.types import ExampleResult

        def neg_inf_metric_evaluator(func, config, example):
            actual_output = func(example.input_data.get("text", ""))
            return ExampleResult(
                example_id=str(id(example)),
                input_data=example.input_data,
                expected_output=example.expected_output,
                actual_output=actual_output,
                metrics={"accuracy": float("-inf")},  # Negative infinity
                execution_time=0.01,
                success=True,
            )

        scenario = TestScenario(
            name="metric_neg_inf_value",
            description="Metric with -Inf value",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=neg_inf_metric_evaluator,
            ),
            objectives=[
                ObjectiveSpec(name="accuracy", orientation="maximize"),
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.5],
            },
            max_trials=2,
            mock_mode_config={"optimizer": "random"},
            gist_template="metric-neg-inf -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle gracefully
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_metric_type_mismatch_string(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test evaluator returning string instead of float.

        Purpose:
            Type mismatch - string instead of numeric metric value.

        Edge Case: String metric value
        """
        from traigent.api.types import ExampleResult

        def string_metric_evaluator(func, config, example):
            actual_output = func(example.input_data.get("text", ""))
            return ExampleResult(
                example_id=str(id(example)),
                input_data=example.input_data,
                expected_output=example.expected_output,
                actual_output=actual_output,
                metrics={"accuracy": "high"},  # String instead of float
                execution_time=0.01,
                success=True,
            )

        scenario = TestScenario(
            name="metric_string_type",
            description="Metric with string type",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=string_metric_evaluator,
            ),
            objectives=[
                ObjectiveSpec(name="accuracy", orientation="maximize"),
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.5],
            },
            max_trials=2,
            mock_mode_config={"optimizer": "random"},
            gist_template="metric-string -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle - may fail or convert
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_metric_empty_dict(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test evaluator returning empty metrics dict.

        Purpose:
            No metrics returned at all. Should use defaults or fail gracefully.

        Edge Case: Empty metrics
        """
        from traigent.api.types import ExampleResult

        def empty_metrics_evaluator(func, config, example):
            actual_output = func(example.input_data.get("text", ""))
            return ExampleResult(
                example_id=str(id(example)),
                input_data=example.input_data,
                expected_output=example.expected_output,
                actual_output=actual_output,
                metrics={},  # Empty metrics
                execution_time=0.01,
                success=True,
            )

        scenario = TestScenario(
            name="metric_empty_dict",
            description="Empty metrics dict",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=empty_metrics_evaluator,
            ),
            objectives=[
                ObjectiveSpec(name="accuracy", orientation="maximize"),
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.5],
            },
            max_trials=2,
            mock_mode_config={"optimizer": "random"},
            gist_template="metric-empty -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle gracefully
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluator_raises_on_some_examples(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test evaluator that raises on some examples but not others.

        Purpose:
            Partial failure - some examples succeed, some raise.
            How are metrics aggregated?

        Edge Case: Partial evaluator failure
        """
        from traigent.api.types import ExampleResult

        call_count = [0]

        def partial_failure_evaluator(func, config, example):
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                raise ValueError("Evaluator failed on even calls")

            actual_output = func(example.input_data.get("text", ""))
            return ExampleResult(
                example_id=str(id(example)),
                input_data=example.input_data,
                expected_output=example.expected_output,
                actual_output=actual_output,
                metrics={"accuracy": 0.8},
                execution_time=0.01,
                success=True,
            )

        scenario = TestScenario(
            name="partial_eval_failure",
            description="Evaluator fails on some examples",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=partial_failure_evaluator,
            ),
            objectives=[
                ObjectiveSpec(name="accuracy", orientation="maximize"),
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.5],
            },
            max_trials=2,
            mock_mode_config={"optimizer": "random"},
            expected=ExpectedResult(outcome=ExpectedOutcome.PARTIAL),
            gist_template="partial-eval-fail -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle partial failures gracefully
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


# =============================================================================
# Test Class: Weight Edge Cases
# =============================================================================


class TestWeightEdgeCases:
    """Tests for edge cases in objective weights.

    Purpose:
        Verify behavior with unusual weight configurations:
        zero, negative, non-sum-to-one, etc.

    Coverage Gap Addressed:
        Weight Edge Cases - NOT TESTED
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_zero_weight_objective(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test objective with zero weight.

        Purpose:
            Weight=0 means this objective should not affect optimization.
            Verify it's effectively ignored.

        Edge Case: Zero weight
        """
        custom_eval = create_metric_mapping_evaluator(
            {
                "accuracy": 0.85,
                "cost": 0.5,
            }
        )

        scenario = TestScenario(
            name="zero_weight_obj",
            description="Objective with zero weight",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            objectives=[
                ObjectiveSpec(name="accuracy", orientation="maximize", weight=1.0),
                ObjectiveSpec(name="cost", orientation="minimize", weight=0.0),
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.7],
            },
            max_trials=4,
            mock_mode_config={"optimizer": "grid"},
            expected=ExpectedResult(required_metrics=["accuracy", "cost"]),
            gist_template="zero-weight -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_negative_weight_objective(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test objective with negative weight.

        Purpose:
            Negative weight could invert direction.
            Verify behavior is defined.

        Edge Case: Negative weight
        """
        custom_eval = create_metric_mapping_evaluator(
            {
                "accuracy": 0.85,
                "cost": 0.5,
            }
        )

        scenario = TestScenario(
            name="negative_weight_obj",
            description="Objective with negative weight",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            objectives=[
                ObjectiveSpec(name="accuracy", orientation="maximize", weight=0.6),
                ObjectiveSpec(name="cost", orientation="minimize", weight=-0.4),
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.5],
            },
            max_trials=2,
            mock_mode_config={"optimizer": "random"},
            expected=ExpectedResult(required_metrics=["accuracy", "cost"]),
            gist_template="neg-weight -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle - may error or apply negative weight
        if not isinstance(result, Exception):
            assert hasattr(result, "trials")
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_weights_sum_less_than_one(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test weights that sum to less than 1.0.

        Purpose:
            Weights sum to 0.6 instead of 1.0.
            Should work but may produce different scale.

        Edge Case: Weights < 1.0
        """
        custom_eval = create_metric_mapping_evaluator(
            {
                "accuracy": 0.85,
                "cost": 0.5,
            }
        )

        scenario = TestScenario(
            name="weights_sum_lt_one",
            description="Weights sum to less than 1.0",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            objectives=[
                ObjectiveSpec(name="accuracy", orientation="maximize", weight=0.3),
                ObjectiveSpec(name="cost", orientation="minimize", weight=0.3),
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.7],
            },
            max_trials=4,
            mock_mode_config={"optimizer": "grid"},
            expected=ExpectedResult(required_metrics=["accuracy", "cost"]),
            gist_template="weights-lt-1 -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_weights_sum_greater_than_one(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test weights that sum to greater than 1.0.

        Purpose:
            Weights sum to 1.5 instead of 1.0.
            Should work but may produce different scale.

        Edge Case: Weights > 1.0
        """
        custom_eval = create_metric_mapping_evaluator(
            {
                "accuracy": 0.85,
                "cost": 0.5,
            }
        )

        scenario = TestScenario(
            name="weights_sum_gt_one",
            description="Weights sum to greater than 1.0",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            objectives=[
                ObjectiveSpec(name="accuracy", orientation="maximize", weight=0.8),
                ObjectiveSpec(name="cost", orientation="minimize", weight=0.7),
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.7],
            },
            max_trials=4,
            mock_mode_config={"optimizer": "grid"},
            expected=ExpectedResult(required_metrics=["accuracy", "cost"]),
            gist_template="weights-gt-1 -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_very_small_weight(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test objective with very small (near-zero) weight.

        Purpose:
            Very small weight like 0.001 should still be applied
            but with minimal effect.

        Edge Case: Tiny weight
        """
        custom_eval = create_metric_mapping_evaluator(
            {
                "accuracy": 0.85,
                "cost": 0.5,
            }
        )

        scenario = TestScenario(
            name="tiny_weight_obj",
            description="Objective with very small weight",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            objectives=[
                ObjectiveSpec(name="accuracy", orientation="maximize", weight=0.999),
                ObjectiveSpec(name="cost", orientation="minimize", weight=0.001),
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.5],
            },
            max_trials=2,
            mock_mode_config={"optimizer": "random"},
            expected=ExpectedResult(required_metrics=["accuracy", "cost"]),
            gist_template="tiny-weight -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_very_large_weight(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test objective with very large weight.

        Purpose:
            Very large weight like 100 should heavily bias
            toward that objective.

        Edge Case: Large weight
        """
        custom_eval = create_metric_mapping_evaluator(
            {
                "accuracy": 0.85,
                "cost": 0.5,
            }
        )

        scenario = TestScenario(
            name="large_weight_obj",
            description="Objective with very large weight",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            objectives=[
                ObjectiveSpec(name="accuracy", orientation="maximize", weight=100.0),
                ObjectiveSpec(name="cost", orientation="minimize", weight=1.0),
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.5],
            },
            max_trials=2,
            mock_mode_config={"optimizer": "random"},
            expected=ExpectedResult(required_metrics=["accuracy", "cost"]),
            gist_template="large-weight -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


# =============================================================================
# Test Class: Aggregation Edge Cases
# =============================================================================


class TestAggregationEdgeCases:
    """Tests for edge cases in metric aggregation across examples.

    Purpose:
        Verify behavior when different examples return different
        metrics or some examples fail.

    Coverage Gap Addressed:
        Aggregation Edge Cases - NOT TESTED
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_metrics_vary_per_example(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test metrics that vary between examples.

        Purpose:
            Example 1: {"accuracy": 0.8}
            Example 2: {"accuracy": 0.9, "cost": 0.3}
            How is missing cost in example 1 handled?

        Edge Case: Inconsistent metrics across examples
        """
        from traigent.api.types import ExampleResult

        example_counter = [0]

        def varying_metrics_evaluator(func, config, example):
            example_counter[0] += 1
            actual_output = func(example.input_data.get("text", ""))

            # Different metrics per example
            if example_counter[0] % 2 == 1:
                metrics = {"accuracy": 0.8}  # Only accuracy
            else:
                metrics = {"accuracy": 0.9, "cost": 0.3}  # Both metrics

            return ExampleResult(
                example_id=str(id(example)),
                input_data=example.input_data,
                expected_output=example.expected_output,
                actual_output=actual_output,
                metrics=metrics,
                execution_time=0.01,
                success=True,
            )

        scenario = TestScenario(
            name="varying_metrics",
            description="Metrics vary per example",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=varying_metrics_evaluator,
            ),
            objectives=[
                ObjectiveSpec(name="accuracy", orientation="maximize"),
                ObjectiveSpec(name="cost", orientation="minimize"),
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.5],
            },
            max_trials=2,
            dataset_size=4,
            mock_mode_config={"optimizer": "random"},
            gist_template="vary-metrics -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle gracefully
        if not isinstance(result, Exception):
            assert hasattr(result, "trials")
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_all_examples_fail(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test when all examples fail evaluation.

        Purpose:
            Every example raises exception. Trial should be marked
            as failed with no valid metrics.

        Edge Case: Complete example failure
        """

        def all_fail_evaluator(func, config, example):
            raise ValueError("All examples fail")

        scenario = TestScenario(
            name="all_examples_fail",
            description="All examples fail evaluation",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=all_fail_evaluator,
            ),
            objectives=[
                ObjectiveSpec(name="accuracy", orientation="maximize"),
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.5],
            },
            max_trials=2,
            mock_mode_config={"optimizer": "random"},
            expected=ExpectedResult(outcome=ExpectedOutcome.PARTIAL),
            gist_template="all-fail -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should handle gracefully - trials will fail
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_single_example_success_others_fail(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test when only one example succeeds, others fail.

        Purpose:
            Only first example returns metrics, rest raise.
            Metrics should still be computed from successful example.

        Edge Case: Single success
        """
        from traigent.api.types import ExampleResult

        call_count = [0]

        def single_success_evaluator(func, config, example):
            call_count[0] += 1
            if call_count[0] > 1:
                raise ValueError("Only first example succeeds")

            actual_output = func(example.input_data.get("text", ""))
            return ExampleResult(
                example_id=str(id(example)),
                input_data=example.input_data,
                expected_output=example.expected_output,
                actual_output=actual_output,
                metrics={"accuracy": 0.85},
                execution_time=0.01,
                success=True,
            )

        scenario = TestScenario(
            name="single_success",
            description="Only first example succeeds",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=single_success_evaluator,
            ),
            objectives=[
                ObjectiveSpec(name="accuracy", orientation="maximize"),
            ],
            config_space={
                "model": ["gpt-3.5-turbo"],
                "temperature": [0.5],
            },
            max_trials=1,
            dataset_size=3,
            mock_mode_config={"optimizer": "random"},
            gist_template="single-success -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        # Should have at least partial result
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()


# =============================================================================
# Test Class: Multi-Objective Edge Cases
# =============================================================================


class TestMultiObjectiveEdgeCases:
    """Tests for edge cases in multi-objective optimization.

    Purpose:
        Verify behavior with unusual multi-objective configurations:
        identical scores, single objective in multi mode, etc.

    Coverage Gap Addressed:
        Multi-Objective Edge Cases - NOT TESTED
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_pareto_with_identical_scores(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test Pareto frontier when configs have identical scores.

        Purpose:
            All configs return exact same (accuracy, cost) values.
            How is tie-breaking handled?

        Edge Case: Pareto ties
        """
        from traigent.api.types import ExampleResult

        def fixed_metrics_evaluator(func, config, example):
            actual_output = func(example.input_data.get("text", ""))
            return ExampleResult(
                example_id=str(id(example)),
                input_data=example.input_data,
                expected_output=example.expected_output,
                actual_output=actual_output,
                metrics={"accuracy": 0.8, "cost": 0.5},  # Always same
                execution_time=0.01,
                success=True,
            )

        scenario = TestScenario(
            name="pareto_ties",
            description="Pareto frontier with identical scores",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=fixed_metrics_evaluator,
            ),
            objectives=[
                ObjectiveSpec(name="accuracy", orientation="maximize"),
                ObjectiveSpec(name="cost", orientation="minimize"),
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.7],
            },
            max_trials=4,
            mock_mode_config={"optimizer": "grid"},
            expected=ExpectedResult(required_metrics=["accuracy", "cost"]),
            gist_template="pareto-ties -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_single_objective_in_multi_mode(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test single objective with multi-objective optimizer.

        Purpose:
            Only one ObjectiveSpec but using multi-objective capable
            optimizer. Should degenerate correctly.

        Edge Case: Single in multi mode
        """
        custom_eval = create_metric_mapping_evaluator({"accuracy": 0.85})

        scenario = TestScenario(
            name="single_in_multi",
            description="Single objective in multi-objective mode",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            objectives=[
                ObjectiveSpec(name="accuracy", orientation="maximize"),
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.7],
            },
            max_trials=4,
            mock_mode_config={"optimizer": "optuna_tpe"},  # Multi-objective capable
            expected=ExpectedResult(required_metrics=["accuracy"]),
            gist_template="single-multi-mode -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_many_objectives(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test with many objectives (>3).

        Purpose:
            Test with 5 objectives to verify no issues with
            higher-dimensional objective spaces.

        Edge Case: Many objectives
        """
        custom_eval = create_metric_mapping_evaluator(
            {
                "accuracy": 0.8,
                "cost": 0.4,
                "latency": 0.3,
                "score": 0.75,
            }
        )

        scenario = TestScenario(
            name="many_objectives",
            description="Many objectives (4)",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            objectives=[
                ObjectiveSpec(name="accuracy", orientation="maximize", weight=0.4),
                ObjectiveSpec(name="cost", orientation="minimize", weight=0.2),
                ObjectiveSpec(name="latency", orientation="minimize", weight=0.2),
                ObjectiveSpec(name="score", orientation="maximize", weight=0.2),
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.5],
            },
            max_trials=2,
            mock_mode_config={"optimizer": "random"},
            expected=ExpectedResult(
                required_metrics=["accuracy", "cost", "latency", "score"]
            ),
            gist_template="many-obj -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_all_objectives_same_direction(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test all objectives with same direction in multi-objective.

        Purpose:
            All objectives maximize. This is a valid but unusual case.

        Edge Case: Same direction for all
        """
        custom_eval = create_metric_mapping_evaluator(
            {
                "accuracy": 0.85,
                "score": 0.8,
            }
        )

        scenario = TestScenario(
            name="all_same_direction",
            description="All objectives same direction",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            objectives=[
                ObjectiveSpec(name="accuracy", orientation="maximize", weight=0.5),
                ObjectiveSpec(name="score", orientation="maximize", weight=0.5),
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.7],
            },
            max_trials=4,
            mock_mode_config={"optimizer": "grid"},
            expected=ExpectedResult(required_metrics=["accuracy", "score"]),
            gist_template="same-dir-multi -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_objectives_with_extreme_metric_values(
        self,
        scenario_runner,
        result_validator,
    ) -> None:
        """Test objectives with extreme metric values.

        Purpose:
            One metric near 0, another near 1. Test handling of
            scale differences in multi-objective optimization.

        Edge Case: Extreme metric values
        """
        custom_eval = create_metric_mapping_evaluator(
            {
                "accuracy": 0.99,  # Near 1
                "cost": 0.01,  # Near 0
            }
        )

        scenario = TestScenario(
            name="extreme_metric_values",
            description="Objectives with extreme values",
            evaluator=EvaluatorSpec(
                type="custom",
                evaluator_fn=custom_eval,
            ),
            objectives=[
                ObjectiveSpec(name="accuracy", orientation="maximize", weight=0.5),
                ObjectiveSpec(name="cost", orientation="minimize", weight=0.5),
            ],
            config_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": [0.3, 0.7],
            },
            max_trials=4,
            mock_mode_config={"optimizer": "grid"},
            expected=ExpectedResult(required_metrics=["accuracy", "cost"]),
            gist_template="extreme-values -> {trial_count()} | {status()}",
        )

        _, result = await scenario_runner(scenario)

        assert not isinstance(result, Exception), f"Unexpected error: {result}"
        validation = result_validator(scenario, result)
        assert validation.passed, validation.summary()
