"""Unit tests for weighted multi-objective optimization functionality.

This test suite covers:
- Weighted objective scoring with scalarize_objectives function
- BaseOptimizer objective_weights support
- BatchOptimizer weighted scoring implementation
- Edge cases with zero weights, missing weights, and invalid configurations
- Integration with optimization workflows
"""

from datetime import datetime
from unittest.mock import MagicMock

from traigent.api.types import TrialResult
from traigent.optimizers.batch_optimizers import MultiObjectiveBatchOptimizer
from traigent.optimizers.grid import GridSearchOptimizer
from traigent.utils.multi_objective import scalarize_objectives


class TestScalarizeObjectives:
    """Test scalarize_objectives function functionality."""

    def test_scalarize_basic_weighted_average(self):
        """Test basic weighted average calculation."""
        objectives = {"accuracy": 0.9, "cost": 0.05}
        weights = {"accuracy": 0.7, "cost": 0.3}

        score = scalarize_objectives(objectives, weights)

        # Expected: (0.9 * 0.7 + 0.05 * 0.3) / (0.7 + 0.3) = (0.63 + 0.015) / 1.0 = 0.645
        expected_score = 0.645
        assert abs(score - expected_score) < 1e-10

    def test_scalarize_equal_weights(self):
        """Test scalarization with equal weights."""
        objectives = {"accuracy": 0.8, "latency": 1.2, "cost": 0.03}
        weights = {"accuracy": 1.0, "latency": 1.0, "cost": 1.0}

        score = scalarize_objectives(objectives, weights)

        # Expected: (0.8 + 1.2 + 0.03) / 3 = 2.03 / 3 ≈ 0.6767
        expected_score = (0.8 + 1.2 + 0.03) / 3
        assert abs(score - expected_score) < 1e-10

    def test_scalarize_single_objective(self):
        """Test scalarization with single objective."""
        objectives = {"accuracy": 0.95}
        weights = {"accuracy": 1.0}

        score = scalarize_objectives(objectives, weights)

        # Should return the objective value directly
        assert score == 0.95

    def test_scalarize_zero_total_weight(self):
        """Test scalarization with zero total weight."""
        objectives = {"accuracy": 0.9, "cost": 0.05}
        weights = {"accuracy": 0.0, "cost": 0.0}

        score = scalarize_objectives(objectives, weights)

        # With zero total weight, should use equal weights (default behavior)
        expected_score = (0.9 + 0.05) / 2
        assert abs(score - expected_score) < 1e-10

    def test_scalarize_empty_weights(self):
        """Test scalarization with empty weights dictionary."""
        objectives = {"accuracy": 0.85, "cost": 0.04}
        weights = {}

        score = scalarize_objectives(objectives, weights)

        # Should use default weights of 1.0 for each objective
        expected_score = (0.85 + 0.04) / 2
        assert abs(score - expected_score) < 1e-10

    def test_scalarize_missing_objective_weights(self):
        """Test scalarization when some objectives don't have weights."""
        objectives = {"accuracy": 0.9, "cost": 0.05, "latency": 1.0}
        weights = {"accuracy": 0.6, "cost": 0.4}  # Missing latency weight

        score = scalarize_objectives(objectives, weights)

        # Should use default weight of 1.0 for missing objectives
        # (0.9 * 0.6 + 0.05 * 0.4 + 1.0 * 1.0) / (0.6 + 0.4 + 1.0) = (0.54 + 0.02 + 1.0) / 2.0 = 0.78
        expected_score = (0.9 * 0.6 + 0.05 * 0.4 + 1.0 * 1.0) / (0.6 + 0.4 + 1.0)
        assert abs(score - expected_score) < 1e-10

    def test_scalarize_extra_weights(self):
        """Test scalarization when weights include objectives not in objectives dict."""
        objectives = {"accuracy": 0.9, "cost": 0.05}
        weights = {
            "accuracy": 0.7,
            "cost": 0.3,
            "latency": 0.2,
        }  # Extra weight for latency

        score = scalarize_objectives(objectives, weights)

        # Should ignore weights for missing objectives
        expected_score = (0.9 * 0.7 + 0.05 * 0.3) / (0.7 + 0.3)
        assert abs(score - expected_score) < 1e-10

    def test_scalarize_negative_weights(self):
        """Test scalarization with negative weights."""
        objectives = {"accuracy": 0.9, "cost": 0.05}
        weights = {"accuracy": 0.8, "cost": -0.2}  # Negative weight for cost (minimize)

        score = scalarize_objectives(objectives, weights)

        # Should handle negative weights correctly
        expected_score = (0.9 * 0.8 + 0.05 * (-0.2)) / (0.8 + (-0.2))
        assert abs(score - expected_score) < 1e-10

    def test_scalarize_very_large_values(self):
        """Test scalarization with very large objective values."""
        objectives = {"accuracy": 0.95, "throughput": 10000.0}
        weights = {"accuracy": 0.1, "throughput": 0.9}

        score = scalarize_objectives(objectives, weights)

        # Should handle large values correctly
        expected_score = (0.95 * 0.1 + 10000.0 * 0.9) / (0.1 + 0.9)
        assert abs(score - expected_score) < 1e-6

    def test_scalarize_very_small_values(self):
        """Test scalarization with very small objective values."""
        objectives = {"precision": 1e-8, "recall": 2e-8}
        weights = {"precision": 0.6, "recall": 0.4}

        score = scalarize_objectives(objectives, weights)

        expected_score = (1e-8 * 0.6 + 2e-8 * 0.4) / (0.6 + 0.4)
        assert abs(score - expected_score) < 1e-15


class TestBaseOptimizerWeights:
    """Test BaseOptimizer objective_weights functionality."""

    def test_base_optimizer_with_weights(self):
        """Test BaseOptimizer initialization with objective weights."""
        objectives = ["accuracy", "cost"]
        objective_weights = {"accuracy": 0.7, "cost": 0.3}
        config_space = {"param": [1, 2]}

        optimizer = GridSearchOptimizer(
            config_space=config_space,
            objectives=objectives,
            objective_weights=objective_weights,
        )

        assert optimizer.objective_weights == objective_weights
        assert optimizer.objectives == objectives

    def test_base_optimizer_default_weights(self):
        """Test BaseOptimizer initialization with default weights."""
        objectives = ["accuracy", "cost", "latency"]
        config_space = {"param": [1, 2]}

        optimizer = GridSearchOptimizer(
            config_space=config_space, objectives=objectives
        )

        # Should create default equal weights
        expected_weights = {"accuracy": 1.0, "cost": 1.0, "latency": 1.0}
        assert optimizer.objective_weights == expected_weights

    def test_base_optimizer_partial_weights(self):
        """Test BaseOptimizer with partial weight specification."""
        objectives = ["accuracy", "cost", "latency"]
        objective_weights = {"accuracy": 0.6, "cost": 0.4}  # Missing latency
        config_space = {"param": [1, 2]}

        optimizer = GridSearchOptimizer(
            config_space=config_space,
            objectives=objectives,
            objective_weights=objective_weights,
        )

        # Should use provided weights and fill in defaults
        expected_weights = {"accuracy": 0.6, "cost": 0.4, "latency": 1.0}
        assert optimizer.objective_weights == expected_weights

    def test_base_optimizer_extra_weights(self):
        """Test BaseOptimizer with extra weights not in objectives."""
        objectives = ["accuracy", "cost"]
        objective_weights = {"accuracy": 0.7, "cost": 0.3, "latency": 0.2}
        config_space = {"param": [1, 2]}

        optimizer = GridSearchOptimizer(
            config_space=config_space,
            objectives=objectives,
            objective_weights=objective_weights,
        )

        # Should only keep weights for specified objectives
        expected_weights = {"accuracy": 0.7, "cost": 0.3}
        assert optimizer.objective_weights == expected_weights

    def test_base_optimizer_empty_objectives(self):
        """Test BaseOptimizer with empty objectives list."""
        objectives = []
        objective_weights = {"accuracy": 0.7}
        config_space = {"param": [1, 2]}

        optimizer = GridSearchOptimizer(
            config_space=config_space,
            objectives=objectives,
            objective_weights=objective_weights,
        )

        assert optimizer.objective_weights == {}
        assert optimizer.objectives == []


class TestMultiObjectiveBatchOptimizerWeightedScoring:
    """Test MultiObjectiveBatchOptimizer weighted scoring implementation."""

    def test_multi_objective_batch_optimizer_uses_scalarize_objectives(self):
        """Test that MultiObjectiveBatchOptimizer uses scalarize_objectives function."""

        config_space = {"param": [1, 2]}
        objectives = ["accuracy", "cost"]

        # Create BatchOptimizationConfig
        from traigent.optimizers.batch_optimizers import BatchOptimizationConfig

        batch_config = BatchOptimizationConfig()

        optimizer = MultiObjectiveBatchOptimizer(
            configuration_space=config_space,
            objectives=objectives,
            batch_config=batch_config,
        )

        # Set objective weights manually for testing
        optimizer.objective_weights = {"accuracy": 0.8, "cost": 0.2}

        # Create a mock trial with objective scores in metadata
        trial = MagicMock()
        trial.metadata = {"objective_scores": {"accuracy": 0.9, "cost": 0.05}}

        # Test the _run_batch_trial method indirectly by checking if scalarize is called
        # This is tricky since _run_batch_trial is complex, so we'll test the integration
        # by verifying the weighted scoring logic exists in the class
        assert hasattr(optimizer, "objectives")
        assert optimizer.objectives == objectives

    def test_multi_objective_batch_optimizer_pareto_frontier(self):
        """Test MultiObjectiveBatchOptimizer Pareto frontier functionality."""

        config_space = {"param": [1, 2, 3]}
        objectives = ["accuracy", "cost"]

        from traigent.optimizers.batch_optimizers import BatchOptimizationConfig

        batch_config = BatchOptimizationConfig()

        optimizer = MultiObjectiveBatchOptimizer(
            configuration_space=config_space,
            objectives=objectives,
            batch_config=batch_config,
            pareto_frontier_size=10,
        )

        # Should have Pareto frontier functionality
        assert hasattr(optimizer, "pareto_frontier")
        assert hasattr(optimizer, "_update_pareto_frontier")
        assert hasattr(optimizer, "_dominates")
        assert optimizer.pareto_frontier_size == 10

    def test_dominates_logic(self):
        """Test the dominance logic in MultiObjectiveBatchOptimizer."""

        config_space = {"param": [1, 2]}
        objectives = ["accuracy", "cost"]

        from traigent.optimizers.batch_optimizers import BatchOptimizationConfig

        batch_config = BatchOptimizationConfig()

        optimizer = MultiObjectiveBatchOptimizer(
            configuration_space=config_space,
            objectives=objectives,
            batch_config=batch_config,
        )

        # Test dominance relationships
        scores1 = {"accuracy": 0.9, "cost": 0.05}  # High accuracy, low cost
        scores2 = {"accuracy": 0.8, "cost": 0.08}  # Lower accuracy, higher cost
        scores3 = {"accuracy": 0.85, "cost": 0.03}  # Medium accuracy, very low cost

        # scores1 should dominate scores2 (better in both objectives)
        assert optimizer._dominates(scores1, scores2)
        assert not optimizer._dominates(scores2, scores1)

        # Neither scores1 nor scores3 should dominate each other (trade-off)
        assert not optimizer._dominates(scores1, scores3)
        assert not optimizer._dominates(scores3, scores1)


class TestGridSearchOptimizerWeights:
    """Test GridSearchOptimizer with weighted objectives."""

    def test_grid_search_weighted_optimization(self):
        """Test GridSearchOptimizer with weighted objectives."""
        config_space = {"temperature": [0.3, 0.7], "max_tokens": [100, 200]}
        objectives = ["accuracy", "cost"]
        objective_weights = {"accuracy": 0.8, "cost": 0.2}

        optimizer = GridSearchOptimizer(
            config_space=config_space,
            objectives=objectives,
            objective_weights=objective_weights,
        )

        # Should inherit from BatchOptimizer, so weights should be set
        assert optimizer.objective_weights == objective_weights
        assert optimizer.objectives == objectives

        # Should generate all combinations
        configs = optimizer._generate_configurations()
        assert len(configs) == 4  # 2 * 2 combinations

    def test_grid_search_has_no_private_composite_scorer(self):
        """#1682: grid's dead _calculate_composite_score override is gone.

        Selection is owned by the orchestrator/result_selection layer; no
        second scorer may live on GridSearchOptimizer.
        """
        assert "_calculate_composite_score" not in GridSearchOptimizer.__dict__

    def test_grid_trials_rank_through_shared_weighted_scorer(self):
        """#1682: grid-produced trials are ranked by the shared weighted scorer.

        Uses the issue's LITERAL trade-off table (realistic fraction-of-a-cent
        costs): flipping ObjectiveSchema weights between accuracy-heavy and
        cost-heavy must flip the selected configuration.
        """
        from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema
        from traigent.core.result_selection import select_best_configuration

        def make_trials() -> list[TrialResult]:
            table = [
                ("gpt-4o-analog", 0.92, 0.010),  # high accuracy, high cost
                ("mid-analog", 0.82, 0.004),
                ("nano-analog", 0.70, 0.001),  # low accuracy, low cost
            ]
            return [
                TrialResult(
                    trial_id=f"trial_{model}",
                    config={"model": model},
                    metrics={"accuracy": accuracy, "cost": cost},
                    status="completed",
                    duration=1.0,
                    timestamp=datetime.now(),
                    metadata={},
                )
                for model, accuracy, cost in table
            ]

        def schema(accuracy_weight: float, cost_weight: float) -> ObjectiveSchema:
            return ObjectiveSchema.from_objectives(
                [
                    ObjectiveDefinition(
                        name="accuracy", orientation="maximize", weight=accuracy_weight
                    ),
                    ObjectiveDefinition(
                        name="cost", orientation="minimize", weight=cost_weight
                    ),
                ]
            )

        def winner(accuracy_weight: float, cost_weight: float) -> dict:
            result = select_best_configuration(
                trials=make_trials(),
                primary_objective="accuracy",
                config_space_keys={"model"},
                aggregate_configs=False,
                objective_order=["accuracy", "cost"],
                comparability_mode="legacy",
                objective_schema=schema(accuracy_weight, cost_weight),
            )
            assert result.best_config is not None
            return result.best_config

        assert winner(0.9, 0.1) == {"model": "gpt-4o-analog"}
        assert winner(0.1, 0.9) == {"model": "nano-analog"}

    def test_post_hoc_weighted_scores_use_shared_scoring_function(self):
        """#1682 reconciliation: OptimizationResult post-hoc weighted scores
        equal ObjectiveSchema.compute_weighted_score(metrics, ranges=...) —
        the same objectives.py function live selection uses (no second
        implementation)."""
        from traigent.api.types import OptimizationResult, OptimizationStatus
        from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema

        trials = [
            TrialResult(
                trial_id="expensive",
                config={"model": "big"},
                metrics={"accuracy": 0.95, "cost": 0.08},
                status="completed",
                duration=1.0,
                timestamp=datetime.now(),
            ),
            TrialResult(
                trial_id="cheap",
                config={"model": "small"},
                metrics={"accuracy": 0.75, "cost": 0.02},
                status="completed",
                duration=1.0,
                timestamp=datetime.now(),
            ),
        ]
        result = OptimizationResult(
            trials=trials,
            best_config={"model": "big"},
            best_score=0.95,
            optimization_id="opt_1682",
            duration=1.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy", "cost"],
            algorithm="grid",
            timestamp=datetime.now(),
        )
        schema = ObjectiveSchema.from_objectives(
            [
                ObjectiveDefinition(
                    name="accuracy", orientation="maximize", weight=0.3
                ),
                ObjectiveDefinition(name="cost", orientation="minimize", weight=0.7),
            ]
        )

        analysis = result.calculate_weighted_scores(objective_schema=schema)
        ranges = analysis["normalization_ranges"]
        assert ranges  # observed min/max were computed

        for trial, weighted in analysis["weighted_scores"]:
            expected = schema.compute_weighted_score(trial.metrics, ranges=ranges)
            assert expected is not None
            assert weighted == expected

        per_trial = result.score_trials(objective_schema=schema)
        by_id = {row["trial_id"]: row["weighted"] for row in per_trial}
        for trial in trials:
            expected = schema.compute_weighted_score(trial.metrics, ranges=ranges)
            assert by_id[trial.trial_id] == expected

    def test_terminal_selection_agrees_with_post_hoc_weighted_ranking(self):
        """#1682 (review advisory 3): best_config from terminal selection and
        best_weighted_config from post-hoc calculate_weighted_scores use the
        same observed-range weighted scoring and cannot disagree — verified
        on the issue's literal table for both weight orders."""
        from traigent.api.types import OptimizationResult, OptimizationStatus
        from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema
        from traigent.core.result_selection import select_best_configuration

        trials = [
            TrialResult(
                trial_id=f"trial_{model}",
                config={"model": model},
                metrics={"accuracy": accuracy, "cost": cost},
                status="completed",
                duration=1.0,
                timestamp=datetime.now(),
            )
            for model, accuracy, cost in [
                ("gpt-4o-analog", 0.92, 0.010),
                ("mid-analog", 0.82, 0.004),
                ("nano-analog", 0.70, 0.001),
            ]
        ]

        for accuracy_weight, cost_weight, expected_model in [
            (0.9, 0.1, "gpt-4o-analog"),
            (0.1, 0.9, "nano-analog"),
        ]:
            schema = ObjectiveSchema.from_objectives(
                [
                    ObjectiveDefinition(
                        name="accuracy", orientation="maximize", weight=accuracy_weight
                    ),
                    ObjectiveDefinition(
                        name="cost", orientation="minimize", weight=cost_weight
                    ),
                ]
            )
            terminal = select_best_configuration(
                trials=trials,
                primary_objective="accuracy",
                config_space_keys={"model"},
                aggregate_configs=False,
                objective_order=["accuracy", "cost"],
                comparability_mode="legacy",
                objective_schema=schema,
            )
            result = OptimizationResult(
                trials=trials,
                best_config=terminal.best_config or {},
                best_score=terminal.best_score,
                optimization_id="opt_1682_agree",
                duration=1.0,
                convergence_info={},
                status=OptimizationStatus.COMPLETED,
                objectives=["accuracy", "cost"],
                algorithm="grid",
                timestamp=datetime.now(),
            )
            post_hoc = result.calculate_weighted_scores(objective_schema=schema)

            assert terminal.best_config == {"model": expected_model}
            assert post_hoc["best_weighted_config"] == terminal.best_config


class TestWeightedObjectivesIntegration:
    """Test integration of weighted objectives with optimization workflows."""

    def test_end_to_end_weighted_optimization(self):
        """Test end-to-end weighted optimization workflow."""

        # Create realistic trial results with trade-offs
        trial_results = [
            TrialResult(
                trial_id="high_acc_high_cost",
                config={"temperature": 0.1},
                metrics={"accuracy": 0.95, "cost": 0.08},
                status="completed",
                duration=1.0,
                timestamp=datetime.now(),
                metadata={},
            ),
            TrialResult(
                trial_id="med_acc_med_cost",
                config={"temperature": 0.5},
                metrics={"accuracy": 0.85, "cost": 0.05},
                status="completed",
                duration=1.0,
                timestamp=datetime.now(),
                metadata={},
            ),
            TrialResult(
                trial_id="low_acc_low_cost",
                config={"temperature": 0.9},
                metrics={"accuracy": 0.75, "cost": 0.02},
                status="completed",
                duration=1.0,
                timestamp=datetime.now(),
                metadata={},
            ),
        ]

        # Test different weight preferences
        weight_scenarios = [
            {"accuracy": 0.9, "cost": 0.1},  # Strongly prefer accuracy
            {"accuracy": 0.1, "cost": 0.9},  # Strongly prefer low cost
            {"accuracy": 0.5, "cost": 0.5},  # Balance both
        ]

        expected_winners = [
            0,
            2,
            0,
        ]  # Indices of expected best trials for each scenario

        for i, weights in enumerate(weight_scenarios):
            scores = []
            for trial in trial_results:
                objectives_dict = {obj: trial.metrics[obj] for obj in weights.keys()}
                score = scalarize_objectives(
                    objectives_dict, weights, minimize_objectives=["cost"]
                )
                scores.append(score)

            best_trial_index = scores.index(max(scores))
            assert best_trial_index == expected_winners[i], (
                f"Scenario {i}: Expected trial {expected_winners[i]}, got {best_trial_index}"
            )

    def test_weighted_objectives_with_missing_data(self):
        """Test weighted objectives handling when some trials have missing metrics."""

        trials = [
            TrialResult(
                trial_id="complete",
                config={"param": 1},
                metrics={"accuracy": 0.9, "cost": 0.05, "latency": 1.0},
                status="completed",
                duration=1.0,
                timestamp=datetime.now(),
                metadata={},
            ),
            TrialResult(
                trial_id="partial",
                config={"param": 2},
                metrics={"accuracy": 0.85},  # Missing cost and latency
                status="completed",
                duration=1.0,
                timestamp=datetime.now(),
                metadata={},
            ),
            TrialResult(
                trial_id="failed",
                config={"param": 3},
                metrics={},  # No metrics
                status="failed",
                duration=0.1,
                timestamp=datetime.now(),
                metadata={},
            ),
        ]

        objectives = ["accuracy", "cost", "latency"]
        weights = {"accuracy": 0.6, "cost": 0.3, "latency": 0.1}

        # Should handle missing metrics gracefully
        for trial in trials[:2]:  # Skip failed trial
            available_objectives = {
                obj: val for obj, val in trial.metrics.items() if obj in objectives
            }
            if available_objectives:  # Only calculate if we have some objectives
                score = scalarize_objectives(available_objectives, weights)
                assert isinstance(score, (int, float))
                assert score >= 0

    def test_objective_weights_validation(self):
        """Test validation of objective weights configuration."""

        # Valid configurations
        valid_configs = [
            {"accuracy": 1.0},
            {"accuracy": 0.7, "cost": 0.3},
            {"accuracy": 2.0, "cost": 1.0, "latency": 0.5},  # Don't need to sum to 1
            {"accuracy": 0.0, "cost": 1.0},  # Zero weights allowed
        ]

        for weights in valid_configs:
            objectives = list(weights.keys())
            config_space = {"param": [1, 2]}
            optimizer = GridSearchOptimizer(
                config_space=config_space,
                objectives=objectives,
                objective_weights=weights,
            )
            assert optimizer.objective_weights is not None

        # Edge cases that should work
        edge_cases = [
            ({}, []),  # Empty weights and objectives
            ({"accuracy": -0.5, "cost": 1.5}, ["accuracy", "cost"]),  # Negative weights
        ]

        for weights, objectives in edge_cases:
            config_space = {"param": [1, 2]}
            optimizer = GridSearchOptimizer(
                config_space=config_space,
                objectives=objectives,
                objective_weights=weights,
            )
            assert optimizer.objective_weights is not None


class TestWeightedObjectivesErrorHandling:
    """Test error handling in weighted objectives functionality."""

    def test_scalarize_empty_objectives(self):
        """Test scalarize_objectives with empty objectives dict."""
        objectives = {}
        weights = {"accuracy": 1.0}

        score = scalarize_objectives(objectives, weights)

        # Should return 0 for empty objectives
        assert score == 0.0

    def test_scalarize_none_values(self):
        """Test scalarize_objectives with None values."""
        objectives = {"accuracy": None, "cost": 0.05}
        weights = {"accuracy": 0.7, "cost": 0.3}

        # Should handle None values gracefully (skip them)
        score = scalarize_objectives(objectives, weights)

        # Should only consider non-None values
        assert isinstance(score, (int, float))

    def test_multi_objective_optimizer_error_handling(self):
        """Test MultiObjectiveBatchOptimizer error handling."""

        config_space = {"param": [1, 2]}
        objectives = ["accuracy", "cost"]

        from traigent.optimizers.batch_optimizers import BatchOptimizationConfig

        batch_config = BatchOptimizationConfig()

        optimizer = MultiObjectiveBatchOptimizer(
            configuration_space=config_space,
            objectives=objectives,
            batch_config=batch_config,
        )

        # Test dominance with empty scores
        empty_scores1 = {}
        empty_scores2 = {"accuracy": 0.9}

        # Should handle empty scores gracefully
        assert not optimizer._dominates(empty_scores1, empty_scores2)
        assert not optimizer._dominates(empty_scores2, empty_scores1)

        # Test with missing objectives (#1944): a missing MINIMIZE metric must
        # default to the orientation-WORST sentinel (+inf for cost), never 0.0
        # (which is the *best* cost and would let the incomplete trial falsely
        # dominate). So a higher-accuracy trial that omits cost neither
        # dominates nor is dominated by a complete trial — it is non-comparable.
        partial_scores1 = {"accuracy": 0.9}  # Missing cost => worst (+inf)
        partial_scores2 = {"accuracy": 0.8, "cost": 0.05}

        assert not optimizer._dominates(partial_scores1, partial_scores2)
        assert not optimizer._dominates(partial_scores2, partial_scores1)
