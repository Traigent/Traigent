"""Unit tests for OptimizationOrchestrator helper functions.

This test suite covers the constructor helper methods that were extracted
from the main orchestrator __init__ method for improved testability and
maintainability.

Tests cover:
- validate_constructor_arguments: Input validation for optimizer, evaluator, max_trials, timeout
- normalize_parallel_trials: Validate and normalize parallel_trials parameter
- prepare_objectives: Objectives list preparation and ObjectiveSchema creation
- allocate_parallel_ceilings: Allocate sample budget across parallel trials

Tests include both:
1. Direct module function tests (preferred)
2. Backward-compatible static method tests (for API stability)
"""

import pytest

from traigent.core.objectives import ObjectiveSchema, create_default_objectives
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.core.orchestrator_helpers import (
    allocate_parallel_ceilings,
    constraint_requires_metrics,
    enforce_constraints,
    extract_cost_from_results,
    extract_optuna_trial_id,
    normalize_parallel_trials,
    prepare_evaluation_config,
    prepare_objectives,
    validate_constructor_arguments,
)
from traigent.evaluators.base import BaseEvaluator
from traigent.optimizers.base import BaseOptimizer
from traigent.utils.exceptions import TVLConstraintError


class MockOptimizer(BaseOptimizer):
    """Mock optimizer for testing."""

    def __init__(self):
        self.config_space = {"temperature": [0.3, 0.5, 0.7]}
        self.objectives = ["accuracy"]

    def suggest_next_trial(self, trials: list) -> dict:
        return {"temperature": 0.5}

    def should_stop(self, trials: list) -> bool:
        return False


class MockEvaluator(BaseEvaluator):
    """Mock evaluator for testing."""

    async def evaluate(self, func, config, dataset, **kwargs):
        return type(
            "EvalResult", (), {"aggregated_metrics": {}, "example_results": []}
        )()


class TestValidateConstructorArguments:
    """Test _validate_constructor_arguments static method."""

    def test_valid_arguments(self):
        """Test that valid arguments pass without exception."""
        optimizer = MockOptimizer()
        evaluator = MockEvaluator()
        max_trials = 10
        timeout = 60.0

        # Should not raise any exception
        validate_constructor_arguments(optimizer, evaluator, max_trials, timeout)

    def test_none_max_trials_allowed(self):
        """Test that None max_trials is allowed (unlimited trials)."""
        optimizer = MockOptimizer()
        evaluator = MockEvaluator()

        # Should not raise any exception
        validate_constructor_arguments(
            optimizer, evaluator, max_trials=None, timeout=None
        )

    def test_zero_max_trials_allowed(self):
        """Test that zero max_trials is allowed (no trials)."""
        optimizer = MockOptimizer()
        evaluator = MockEvaluator()

        # Should not raise any exception
        validate_constructor_arguments(optimizer, evaluator, max_trials=0, timeout=None)

    def test_invalid_optimizer_type(self):
        """Test that invalid optimizer type raises TypeError."""
        evaluator = MockEvaluator()

        with pytest.raises(
            TypeError, match="optimizer must be an instance of BaseOptimizer"
        ):
            validate_constructor_arguments(
                optimizer="not_an_optimizer",  # Invalid type
                evaluator=evaluator,
                max_trials=10,
                timeout=None,
            )

    def test_none_optimizer_raises_error(self):
        """Test that None optimizer raises TypeError."""
        evaluator = MockEvaluator()

        with pytest.raises(
            TypeError, match="optimizer must be an instance of BaseOptimizer"
        ):
            validate_constructor_arguments(
                optimizer=None,
                evaluator=evaluator,
                max_trials=10,
                timeout=None,
            )

    def test_invalid_evaluator_type(self):
        """Test that invalid evaluator type raises TypeError."""
        optimizer = MockOptimizer()

        with pytest.raises(
            TypeError, match="evaluator must be an instance of BaseEvaluator"
        ):
            validate_constructor_arguments(
                optimizer=optimizer,
                evaluator="not_an_evaluator",  # Invalid type
                max_trials=10,
                timeout=None,
            )

    def test_none_evaluator_raises_error(self):
        """Test that None evaluator raises TypeError."""
        optimizer = MockOptimizer()

        with pytest.raises(
            TypeError, match="evaluator must be an instance of BaseEvaluator"
        ):
            validate_constructor_arguments(
                optimizer=optimizer,
                evaluator=None,
                max_trials=10,
                timeout=None,
            )

    def test_negative_max_trials_raises_error(self):
        """Test that negative max_trials raises ValueError."""
        optimizer = MockOptimizer()
        evaluator = MockEvaluator()

        with pytest.raises(ValueError, match="max_trials must be non-negative"):
            validate_constructor_arguments(
                optimizer=optimizer,
                evaluator=evaluator,
                max_trials=-1,
                timeout=None,
            )

    def test_negative_timeout_raises_error(self):
        """Test that negative timeout raises ValueError."""
        optimizer = MockOptimizer()
        evaluator = MockEvaluator()

        with pytest.raises(ValueError, match="timeout must be non-negative"):
            validate_constructor_arguments(
                optimizer=optimizer,
                evaluator=evaluator,
                max_trials=10,
                timeout=-1.0,
            )

    def test_zero_timeout_allowed(self):
        """Test that zero timeout is allowed (immediate timeout)."""
        optimizer = MockOptimizer()
        evaluator = MockEvaluator()

        # Should not raise any exception
        validate_constructor_arguments(optimizer, evaluator, max_trials=10, timeout=0.0)


class TestParallelTrialsNormalization:
    """Tests for parallel_trials validation in Initialization."""

    def test_parallel_trials_default_is_one(self):
        orchestrator = OptimizationOrchestrator(MockOptimizer(), MockEvaluator())
        assert orchestrator.parallel_trials == 1

    @pytest.mark.parametrize("value", [0, -1, 2.5, float("inf"), True])
    def test_invalid_parallel_trials_raise(self, value):
        with pytest.raises(
            ValueError, match="parallel_trials must be a positive integer"
        ):
            OptimizationOrchestrator(
                MockOptimizer(),
                MockEvaluator(),
                parallel_trials=value,
            )

    def test_valid_parallel_trials(self):
        orchestrator = OptimizationOrchestrator(
            MockOptimizer(),
            MockEvaluator(),
            parallel_trials=4,
        )
        assert orchestrator.parallel_trials == 4


class TestPrepareObjectives:
    """Test _prepare_objectives static method."""

    def test_default_objectives(self):
        """Test that default objectives is ['accuracy'] when None provided."""
        objectives, schema = prepare_objectives(objectives=None, objective_schema=None)

        assert objectives == ["accuracy"]
        assert schema is not None
        assert isinstance(schema, ObjectiveSchema)

    def test_empty_list_defaults_to_accuracy(self):
        """Test that empty objectives list defaults to ['accuracy']."""
        objectives, schema = prepare_objectives(objectives=[], objective_schema=None)

        assert objectives == ["accuracy"]
        assert schema is not None

    def test_single_objective(self):
        """Test single objective is preserved."""
        objectives, schema = prepare_objectives(
            objectives=["f1_score"], objective_schema=None
        )

        assert objectives == ["f1_score"]
        assert schema is not None

    def test_multiple_objectives(self):
        """Test multiple objectives are preserved."""
        objectives, schema = prepare_objectives(
            objectives=["accuracy", "latency", "cost"], objective_schema=None
        )

        assert objectives == ["accuracy", "latency", "cost"]
        assert schema is not None

    def test_filters_none_objectives(self):
        """Test that None values are filtered from objectives list."""
        objectives, schema = prepare_objectives(
            objectives=["accuracy", None, "latency", None], objective_schema=None
        )

        assert objectives == ["accuracy", "latency"]
        assert None not in objectives

    def test_explicit_objective_schema_preserved(self):
        """Test that explicitly provided ObjectiveSchema is preserved."""
        # Create a proper ObjectiveSchema using the factory function
        custom_schema = create_default_objectives(
            ["accuracy", "cost"],
            orientations={"accuracy": "maximize", "cost": "minimize"},
            weights={"accuracy": 0.7, "cost": 0.3},
        )

        objectives, schema = prepare_objectives(
            objectives=["accuracy", "cost"], objective_schema=custom_schema
        )

        assert objectives == ["accuracy", "cost"]
        assert schema is custom_schema

    def test_creates_default_schema_when_none_provided(self):
        """Test that default schema is created when not provided."""
        objectives, schema = prepare_objectives(
            objectives=["accuracy", "precision"], objective_schema=None
        )

        assert objectives == ["accuracy", "precision"]
        assert schema is not None
        assert isinstance(schema, ObjectiveSchema)

    def test_schema_creation_failure_handled_gracefully(self):
        """Test that schema creation failure is handled gracefully."""
        # Use an objective name that might cause schema creation issues
        objectives, schema = prepare_objectives(
            objectives=["custom_metric_123"], objective_schema=None
        )

        # Should still return objectives even if schema creation fails
        assert objectives == ["custom_metric_123"]
        # Schema might be None if creation failed, but objectives are preserved
        # This tests the graceful fallback behavior

    def test_mixed_valid_and_none_objectives(self):
        """Test handling of mixed valid and None objectives."""
        objectives, schema = prepare_objectives(
            objectives=[None, "accuracy", None, "f1_score", None], objective_schema=None
        )

        assert objectives == ["accuracy", "f1_score"]
        assert len(objectives) == 2


class TestConstructorHelperIntegration:
    """Integration tests for constructor helpers working together."""

    def test_helpers_used_in_orchestrator_creation(self):
        """Test that helpers are properly used during orchestrator creation."""
        optimizer = MockOptimizer()
        evaluator = MockEvaluator()

        # Create orchestrator - this will use the helpers internally
        orchestrator = OptimizationOrchestrator(
            optimizer=optimizer,
            evaluator=evaluator,
            max_trials=5,
            timeout=60.0,
            objectives=["accuracy", "latency"],
        )

        # Verify objectives were prepared correctly
        assert orchestrator.objectives == ["accuracy", "latency"]
        assert orchestrator.objective_schema is not None

        # Verify validation passed (orchestrator created successfully)
        assert orchestrator.optimizer is optimizer
        assert orchestrator.evaluator is evaluator
        assert orchestrator.max_trials == 5
        assert orchestrator.timeout == 60.0

    def test_orchestrator_creation_with_defaults(self):
        """Test orchestrator creation with default objectives."""
        optimizer = MockOptimizer()
        evaluator = MockEvaluator()

        orchestrator = OptimizationOrchestrator(
            optimizer=optimizer,
            evaluator=evaluator,
        )

        # Should default to ['accuracy']
        assert orchestrator.objectives == ["accuracy"]
        assert orchestrator.objective_schema is not None

    def test_orchestrator_creation_with_custom_schema(self):
        """Test orchestrator creation with custom objective schema."""
        optimizer = MockOptimizer()
        evaluator = MockEvaluator()

        # Create a proper ObjectiveSchema using the factory function
        custom_schema = create_default_objectives(
            ["accuracy", "latency"],
            orientations={"accuracy": "maximize", "latency": "minimize"},
            weights={"accuracy": 0.6, "latency": 0.4},
        )

        orchestrator = OptimizationOrchestrator(
            optimizer=optimizer,
            evaluator=evaluator,
            objectives=["accuracy", "latency"],
            objective_schema=custom_schema,
        )

        # Custom schema should be preserved
        assert orchestrator.objective_schema is custom_schema
        assert orchestrator.objectives == ["accuracy", "latency"]

    def test_orchestrator_validation_failure_during_creation(self):
        """Test that validation failures prevent orchestrator creation."""
        optimizer = MockOptimizer()

        with pytest.raises(
            TypeError, match="evaluator must be an instance of BaseEvaluator"
        ):
            OptimizationOrchestrator(
                optimizer=optimizer,
                evaluator="invalid_evaluator",
            )

    def test_orchestrator_with_none_max_trials(self):
        """Test orchestrator creation with None max_trials (unlimited)."""
        optimizer = MockOptimizer()
        evaluator = MockEvaluator()

        orchestrator = OptimizationOrchestrator(
            optimizer=optimizer,
            evaluator=evaluator,
            max_trials=None,
        )

        assert orchestrator.max_trials is None


class TestHelperMethodDocumentation:
    """Test that helper methods have proper documentation."""

    def test_validate_constructor_arguments_has_docstring(self):
        """Test that _validate_constructor_arguments has documentation."""
        method = validate_constructor_arguments
        assert method.__doc__ is not None
        assert len(method.__doc__) > 0

    def test_prepare_objectives_has_docstring(self):
        """Test that _prepare_objectives has documentation."""
        method = prepare_objectives
        assert method.__doc__ is not None
        assert len(method.__doc__) > 0


# =============================================================================
# Direct Module Function Tests (preferred over static method access)
# =============================================================================


class TestModuleValidateConstructorArguments:
    """Test validate_constructor_arguments module function directly."""

    def test_valid_arguments_pass(self):
        """Test that valid arguments pass without exception."""
        optimizer = MockOptimizer()
        evaluator = MockEvaluator()
        # Should not raise
        validate_constructor_arguments(optimizer, evaluator, 10, 100, 60.0)

    def test_invalid_optimizer_raises_type_error(self):
        """Test that non-BaseOptimizer raises TypeError."""
        with pytest.raises(
            TypeError, match="optimizer must be an instance of BaseOptimizer"
        ):
            validate_constructor_arguments("invalid", MockEvaluator())

    def test_invalid_evaluator_raises_type_error(self):
        """Test that non-BaseEvaluator raises TypeError."""
        with pytest.raises(
            TypeError, match="evaluator must be an instance of BaseEvaluator"
        ):
            validate_constructor_arguments(MockOptimizer(), "invalid")

    def test_negative_max_trials_raises_value_error(self):
        """Test that negative max_trials raises ValueError."""
        with pytest.raises(ValueError, match="max_trials must be non-negative"):
            validate_constructor_arguments(
                MockOptimizer(), MockEvaluator(), max_trials=-1
            )

    def test_negative_max_total_examples_raises_value_error(self):
        """Test that negative max_total_examples raises ValueError."""
        with pytest.raises(ValueError, match="max_total_examples must be non-negative"):
            validate_constructor_arguments(
                MockOptimizer(), MockEvaluator(), max_total_examples=-1
            )

    def test_negative_timeout_raises_value_error(self):
        """Test that negative timeout raises ValueError."""
        with pytest.raises(ValueError, match="timeout must be non-negative"):
            validate_constructor_arguments(
                MockOptimizer(), MockEvaluator(), timeout=-1.0
            )


class TestModuleNormalizeParallelTrials:
    """Test normalize_parallel_trials module function directly."""

    def test_none_defaults_to_one(self):
        """Test that None returns 1."""
        assert normalize_parallel_trials(None) == 1

    def test_positive_integer_preserved(self):
        """Test that positive integers are preserved."""
        assert normalize_parallel_trials(4) == 4
        assert normalize_parallel_trials(1) == 1

    @pytest.mark.parametrize("value", [0, -1, -10])
    def test_non_positive_raises_value_error(self, value):
        """Test that non-positive integers raise ValueError."""
        with pytest.raises(
            ValueError, match="parallel_trials must be a positive integer"
        ):
            normalize_parallel_trials(value)

    @pytest.mark.parametrize("value", [2.5, "3", True, False, [1]])
    def test_non_integer_raises_value_error(self, value):
        """Test that non-integer types raise ValueError."""
        with pytest.raises(
            ValueError, match="parallel_trials must be a positive integer"
        ):
            normalize_parallel_trials(value)


class TestModulePrepareObjectives:
    """Test prepare_objectives module function directly."""

    def test_none_defaults_to_accuracy(self):
        """Test that None objectives defaults to ['accuracy']."""
        objectives, schema = prepare_objectives(None, None)
        assert objectives == ["accuracy"]
        assert schema is not None

    def test_empty_list_defaults_to_accuracy(self):
        """Test that empty list defaults to ['accuracy']."""
        objectives, _ = prepare_objectives([], None)
        assert objectives == ["accuracy"]

    def test_filters_none_values(self):
        """Test that None values are filtered from objectives."""
        objectives, _ = prepare_objectives(["acc", None, "cost", None], None)
        assert objectives == ["acc", "cost"]

    def test_preserves_custom_schema(self):
        """Test that provided schema is preserved."""
        custom_schema = create_default_objectives(["accuracy"])
        _, schema = prepare_objectives(["accuracy"], custom_schema)
        assert schema is custom_schema


class TestModuleAllocateParallelCeilings:
    """Test allocate_parallel_ceilings module function directly."""

    def test_zero_budget_returns_zeros(self):
        """Test that zero budget returns all zeros."""
        result = allocate_parallel_ceilings([10, 20, 30], 0)
        assert result == [0, 0, 0]

    def test_negative_budget_returns_zeros(self):
        """Test that negative budget returns all zeros."""
        result = allocate_parallel_ceilings([10, 20], -5)
        assert result == [0, 0]

    def test_empty_dataset_sizes_returns_empty(self):
        """Test that empty dataset_sizes returns empty list."""
        result = allocate_parallel_ceilings([], 100)
        assert result == []

    def test_even_distribution(self):
        """Test even distribution of budget across trials."""
        result = allocate_parallel_ceilings([100, 100, 100], 30)
        assert result == [10, 10, 10]

    def test_uneven_distribution_with_remainder(self):
        """Test uneven distribution with remainder."""
        result = allocate_parallel_ceilings([100, 100, 100], 10)
        # 10 / 3 = 3 with remainder 1, first trial gets extra
        assert sum(result) == 10
        assert result[0] == 4
        assert result[1] == 3
        assert result[2] == 3

    def test_respects_dataset_size_limits(self):
        """Test that allocation respects individual dataset sizes."""
        result = allocate_parallel_ceilings([5, 100, 100], 30)
        # First dataset can only take 5, rest redistributed
        assert result[0] == 5
        assert sum(result) == 30

    def test_large_budget_capped_by_dataset_sizes(self):
        """Test that large budget is capped by total dataset size."""
        result = allocate_parallel_ceilings([10, 20, 30], 1000)
        assert result == [10, 20, 30]  # Capped at dataset sizes

    def test_single_trial(self):
        """Test allocation with single trial."""
        result = allocate_parallel_ceilings([100], 50)
        assert result == [50]

    def test_leftover_redistribution(self):
        """Test that leftover budget is redistributed to trials with room."""
        # First trial limited to 2, so remaining budget goes to others
        result = allocate_parallel_ceilings([2, 100, 100], 30)
        assert result[0] == 2
        # Remaining 28 should be distributed to trials with room
        assert sum(result) == 30


class TestAllocateParallelCeilingsSafetyMechanism:
    """Test allocate_parallel_ceilings as a safety mechanism for parallel trials.

    These tests verify that when spinning up N parallel trials with X remaining
    budget, each trial gets at most X/N samples. This prevents any single trial
    from consuming more than its fair share when running in parallel.
    """

    def test_single_trial_gets_full_budget(self):
        """Test that a single trial can use the full remaining budget."""
        # Single trial with 150 examples, budget of 10000
        # Should get full dataset size (capped by dataset, not budget)
        result = allocate_parallel_ceilings([150], 10000)
        assert result == [150]

    def test_parallel_trials_split_budget_evenly(self):
        """Test that parallel trials split the budget evenly."""
        # 3 parallel trials, each with 150 examples, budget of 300
        # Each trial should get 100 samples (300 / 3)
        result = allocate_parallel_ceilings([150, 150, 150], 300)
        assert result == [100, 100, 100]
        assert sum(result) == 300

    def test_parallel_trials_dont_exceed_budget(self):
        """Test that parallel trials don't exceed total budget."""
        # 5 parallel trials with large datasets, limited budget
        result = allocate_parallel_ceilings([1000, 1000, 1000, 1000, 1000], 100)
        # Each gets 20 (100 / 5)
        assert result == [20, 20, 20, 20, 20]
        assert sum(result) == 100

    def test_dataset_size_caps_allocation(self):
        """Test that dataset size still caps allocation."""
        # 2 parallel trials, one with small dataset
        # Budget of 200, but first dataset only has 50
        result = allocate_parallel_ceilings([50, 200], 200)
        # First gets 50 (capped by dataset), second gets 100 + leftover
        assert result[0] == 50
        assert result[1] == 150  # Gets its share (100) plus leftover (50)
        assert sum(result) == 200

    def test_leftover_redistribution(self):
        """Test that leftover from small datasets is redistributed."""
        # 3 trials: first two have small datasets, third has large
        result = allocate_parallel_ceilings([10, 20, 1000], 300)
        # Base allocation: 100, 100, 100
        # First capped at 10 (leftover 90), second capped at 20 (leftover 80)
        # Third gets 100 + 90 + 80 = 270
        assert result[0] == 10
        assert result[1] == 20
        assert result[2] == 270
        assert sum(result) == 300

    def test_all_datasets_smaller_than_share(self):
        """Test when all datasets are smaller than their budget share."""
        # 3 trials with small datasets, large budget
        result = allocate_parallel_ceilings([10, 20, 30], 1000)
        # Each gets capped at their dataset size
        assert result == [10, 20, 30]
        assert sum(result) == 60  # Total is less than budget

    def test_budget_exactly_divisible(self):
        """Test when budget divides evenly among trials."""
        result = allocate_parallel_ceilings([100, 100, 100], 30)
        assert result == [10, 10, 10]

    def test_budget_with_remainder(self):
        """Test when budget doesn't divide evenly."""
        result = allocate_parallel_ceilings([100, 100, 100], 10)
        # 10 / 3 = 3 with remainder 1, first trial gets extra
        assert sum(result) == 10
        assert result[0] == 4
        assert result[1] == 3
        assert result[2] == 3

    def test_small_budget_still_allocates(self):
        """Test allocation with very small budget."""
        result = allocate_parallel_ceilings([100, 100, 100], 5)
        # 5 / 3 = 1 with remainder 2
        assert result == [2, 2, 1]
        assert sum(result) == 5


# =============================================================================
# New Helper Functions (Phase 4)
# =============================================================================


class TestExtractOptunaTrialId:
    """Test extract_optuna_trial_id module function."""

    def test_returns_provided_id_when_not_none(self):
        """Test that provided optuna_trial_id takes precedence."""
        config = {"_optuna_trial_id": 100}
        result = extract_optuna_trial_id(config, 42)
        assert result == 42

    def test_extracts_from_config_when_id_none(self):
        """Test extraction from config when optuna_trial_id is None."""
        config = {"_optuna_trial_id": 123, "temperature": 0.5}
        result = extract_optuna_trial_id(config, None)
        assert result == 123

    def test_returns_none_when_not_in_config(self):
        """Test returns None when _optuna_trial_id not in config."""
        config = {"temperature": 0.5}
        result = extract_optuna_trial_id(config, None)
        assert result is None

    def test_returns_none_for_non_dict_config(self):
        """Test returns None for non-dict config."""
        result = extract_optuna_trial_id("not_a_dict", None)
        assert result is None

    def test_returns_none_for_empty_config(self):
        """Test returns None for empty config."""
        result = extract_optuna_trial_id({}, None)
        assert result is None

    def test_with_zero_trial_id(self):
        """Test handling of zero trial ID."""
        config = {"_optuna_trial_id": 0}
        result = extract_optuna_trial_id(config, None)
        assert result == 0

    def test_provided_id_zero_takes_precedence(self):
        """Test that provided ID of 0 still takes precedence."""
        config = {"_optuna_trial_id": 100}
        result = extract_optuna_trial_id(config, 0)
        assert result == 0


class TestPrepareEvaluationConfig:
    """Test prepare_evaluation_config module function."""

    def test_filters_optuna_keys(self):
        """Test that _optuna prefixed keys are filtered."""
        config = {
            "temperature": 0.5,
            "model": "gpt-4",
            "_optuna_trial_id": 123,
            "_optuna_distributions": {},
        }
        result = prepare_evaluation_config(config)
        assert result == {"temperature": 0.5, "model": "gpt-4"}

    def test_preserves_non_optuna_keys(self):
        """Test that non-optuna keys are preserved."""
        config = {
            "temperature": 0.5,
            "max_tokens": 100,
            "stream": True,
        }
        result = prepare_evaluation_config(config)
        assert result == config

    def test_handles_empty_config(self):
        """Test handling of empty config."""
        result = prepare_evaluation_config({})
        assert result == {}

    def test_handles_non_dict_config(self):
        """Test handling of non-dict config."""
        result = prepare_evaluation_config("not_a_dict")
        assert result == "not_a_dict"

    def test_only_filters_optuna_prefix(self):
        """Test that only _optuna prefix is filtered, not similar names."""
        config = {
            "optuna_setting": "value",  # No underscore prefix
            "_optimizer_config": "value",  # Different prefix
            "_optuna_trial_id": 123,  # Should be filtered
        }
        result = prepare_evaluation_config(config)
        assert "_optuna_trial_id" not in result
        assert result == {"optuna_setting": "value", "_optimizer_config": "value"}

    def test_preserves_nested_structures(self):
        """Test that nested structures are preserved."""
        config = {
            "nested": {"a": 1, "b": 2},
            "list_value": [1, 2, 3],
            "_optuna_trial_id": 123,
        }
        result = prepare_evaluation_config(config)
        assert result == {
            "nested": {"a": 1, "b": 2},
            "list_value": [1, 2, 3],
        }


class TestConstraintRequiresMetrics:
    """Test the constraint_requires_metrics function."""

    def test_single_param_function_returns_false(self):
        """Test that single-param function returns False."""

        def single_param_constraint(config):
            return True

        assert constraint_requires_metrics(single_param_constraint) is False

    def test_two_param_function_returns_true(self):
        """Test that two-param function returns True."""

        def two_param_constraint(config, metrics):
            return True

        assert constraint_requires_metrics(two_param_constraint) is True

    def test_metadata_explicit_requires_metrics_true(self):
        """Test explicit metadata setting takes precedence."""

        def constraint(config):
            return True

        constraint.__tvl_constraint__ = {"requires_metrics": True}
        assert constraint_requires_metrics(constraint) is True

    def test_metadata_explicit_requires_metrics_false(self):
        """Test explicit metadata setting takes precedence."""

        def constraint(config, metrics):
            return True

        constraint.__tvl_constraint__ = {"requires_metrics": False}
        assert constraint_requires_metrics(constraint) is False

    def test_lambda_single_param(self):
        """Test lambda with single parameter."""
        single = lambda c: True
        assert constraint_requires_metrics(single) is False

    def test_lambda_two_params(self):
        """Test lambda with two parameters."""
        double = lambda c, m: True
        assert constraint_requires_metrics(double) is True

    def test_non_callable_returns_false(self):
        """Test that non-callable without signature returns False."""
        # Built-in that doesn't support inspect.signature
        assert constraint_requires_metrics(len) is False


class TestEnforceConstraints:
    """Test the enforce_constraints function."""

    def test_empty_constraints_does_nothing(self):
        """Test that empty constraints list passes."""
        enforce_constraints({}, {}, [], "test")  # Should not raise

    def test_passing_constraint(self):
        """Test that passing constraint doesn't raise."""

        def always_pass(config):
            return True

        enforce_constraints({"a": 1}, None, [always_pass], "pre")

    def test_failing_constraint_raises_error(self):
        """Test that failing constraint raises TVLConstraintError."""

        def always_fail(config):
            return False

        with pytest.raises(TVLConstraintError) as exc_info:
            enforce_constraints({"a": 1}, None, [always_fail], "pre")
        assert "always_fail" in str(exc_info.value)

    def test_constraint_with_metrics(self):
        """Test constraint that uses metrics."""

        def metrics_constraint(config, metrics):
            return metrics.get("accuracy", 0) > 0.5

        enforce_constraints({"a": 1}, {"accuracy": 0.7}, [metrics_constraint], "post")

    def test_failing_metrics_constraint(self):
        """Test failing metrics constraint."""

        def metrics_constraint(config, metrics):
            return metrics.get("accuracy", 0) > 0.9

        with pytest.raises(TVLConstraintError):
            enforce_constraints(
                {"a": 1}, {"accuracy": 0.7}, [metrics_constraint], "post"
            )

    def test_constraint_exception_wrapped(self):
        """Test that constraint exception is wrapped in TVLConstraintError."""

        def bad_constraint(config):
            raise ValueError("Something went wrong")

        with pytest.raises(TVLConstraintError) as exc_info:
            enforce_constraints({"a": 1}, None, [bad_constraint], "pre")
        assert "Something went wrong" in str(exc_info.value)

    def test_custom_constraint_message(self):
        """Test that constraint with custom message uses it."""

        def custom_constraint(config):
            return False

        custom_constraint.__tvl_constraint__ = {
            "id": "my_constraint",
            "message": "Custom failure message",
        }

        with pytest.raises(TVLConstraintError) as exc_info:
            enforce_constraints({"a": 1}, None, [custom_constraint], "pre")
        assert "Custom failure message" in str(exc_info.value)

    def test_multiple_constraints_all_pass(self):
        """Test multiple constraints all passing."""

        def c1(config):
            return True

        def c2(config):
            return True

        enforce_constraints({"a": 1}, None, [c1, c2], "pre")

    def test_multiple_constraints_one_fails(self):
        """Test that first failing constraint raises."""

        def c1(config):
            return True

        def c2(config):
            return False

        with pytest.raises(TVLConstraintError):
            enforce_constraints({"a": 1}, None, [c1, c2], "pre")


class TestExtractCostFromResults:
    """Test the extract_cost_from_results function."""

    def test_empty_progress_state(self):
        """Test with no progress state."""

        class MockResult:
            pass

        result = MockResult()
        examples, cost = extract_cost_from_results(result, None, "trial_1")
        assert examples is None
        assert cost is None

    def test_progress_state_values(self):
        """Test extraction from progress state."""

        class MockResult:
            pass

        result = MockResult()
        progress = {"evaluated": 10, "total_cost": 0.05}
        examples, cost = extract_cost_from_results(result, progress, "trial_1")
        assert examples == 10
        assert abs(cost - 0.05) < 0.001

    def test_example_results_extraction(self):
        """Test extraction from example_results."""

        class MockExample:
            def __init__(self, cost):
                self.metrics = {"cost": cost}

        class MockResult:
            def __init__(self):
                self.example_results = [
                    MockExample(0.01),
                    MockExample(0.02),
                    MockExample(0.03),
                ]

        result = MockResult()
        examples, cost = extract_cost_from_results(result, None, "trial_1")
        assert examples == 3
        assert abs(cost - 0.06) < 0.001

    def test_aggregated_metrics_fallback(self):
        """Test fallback to aggregated_metrics."""

        class MockResult:
            def __init__(self):
                self.example_results = None
                self.aggregated_metrics = {"total_cost": 0.10}

        result = MockResult()
        examples, cost = extract_cost_from_results(result, None, "trial_1")
        assert examples is None
        assert abs(cost - 0.10) < 0.001

    def test_total_examples_override(self):
        """Test that total_examples overrides example_results count."""

        class MockExample:
            def __init__(self):
                self.metrics = {}

        class MockResult:
            def __init__(self):
                self.example_results = [MockExample() for _ in range(100)]
                self.total_examples = 50  # Override

        result = MockResult()
        examples, _cost = extract_cost_from_results(result, None, "trial_1")
        assert examples == 50  # Uses total_examples, not len(example_results)
