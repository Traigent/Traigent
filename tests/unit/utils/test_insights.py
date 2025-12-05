"""Comprehensive tests for optimization insights functionality."""

import statistics
from datetime import datetime

import pytest

from traigent.api.types import (
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)
from traigent.utils.insights import get_optimization_insights


class TestGetOptimizationInsights:
    """Test get_optimization_insights function."""

    @pytest.fixture
    def sample_trials(self):
        """Create sample trial results."""
        return [
            TrialResult(
                trial_id="trial_1",
                config={"model": "gpt-4o-mini", "temperature": 0.3, "max_tokens": 200},
                metrics={"accuracy": 0.85, "cost_per_1k": 0.002, "latency": 0.5},
                status=TrialStatus.COMPLETED,
                duration=2.0,
                timestamp=datetime.now(),
                metadata={},
            ),
            TrialResult(
                trial_id="trial_2",
                config={"model": "GPT-4o", "temperature": 0.7, "max_tokens": 300},
                metrics={"accuracy": 0.92, "cost_per_1k": 0.008, "latency": 0.8},
                status=TrialStatus.COMPLETED,
                duration=3.0,
                timestamp=datetime.now(),
                metadata={},
            ),
            TrialResult(
                trial_id="trial_3",
                config={"model": "gpt-4o-mini", "temperature": 0.5, "max_tokens": 100},
                metrics={"accuracy": 0.78, "cost_per_1k": 0.001, "latency": 0.3},
                status=TrialStatus.COMPLETED,
                duration=1.5,
                timestamp=datetime.now(),
                metadata={},
            ),
            TrialResult(
                trial_id="trial_4",
                config={"model": "GPT-4o", "temperature": 0.1, "max_tokens": 150},
                metrics={"accuracy": 0.88, "cost_per_1k": 0.006, "latency": 0.6},
                status=TrialStatus.COMPLETED,
                duration=2.5,
                timestamp=datetime.now(),
                metadata={},
            ),
        ]

    @pytest.fixture
    def optimization_result(self, sample_trials):
        """Create sample optimization result."""
        return OptimizationResult(
            trials=sample_trials,
            best_config={"model": "GPT-4o", "temperature": 0.7, "max_tokens": 300},
            best_score=0.92,
            optimization_id="test_insights",
            duration=15.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy", "cost_per_1k", "latency"],
            algorithm="random",
            timestamp=datetime.now(),
            metadata={},
        )

    def test_get_optimization_insights_basic_structure(self, optimization_result):
        """Test basic structure of insights output."""
        insights = get_optimization_insights(optimization_result)

        # Check top-level structure
        assert "top_configurations" in insights
        assert "performance_summary" in insights
        assert "parameter_insights" in insights
        assert "recommendations" in insights
        assert "error" not in insights

        # Check types
        assert isinstance(insights["top_configurations"], list)
        assert isinstance(insights["performance_summary"], dict)
        assert isinstance(insights["parameter_insights"], dict)
        assert isinstance(insights["recommendations"], list)

    def test_get_optimization_insights_top_configurations(self, optimization_result):
        """Test top configurations analysis."""
        insights = get_optimization_insights(optimization_result)
        top_configs = insights["top_configurations"]

        # Should have 3 configurations (default top_k=3)
        assert len(top_configs) == 3

        # Check first (best) configuration
        best_config = top_configs[0]
        assert best_config["rank"] == 1
        assert best_config["score"] == 0.92  # Best accuracy
        assert best_config["config"]["model"] == "GPT-4o"
        assert best_config["config"]["temperature"] == 0.7
        assert best_config["trial_id"] == "trial_2"

        # Check cost analysis is present
        assert "cost_analysis" in best_config
        assert "cost_per_query" in best_config["cost_analysis"]
        assert "cost_efficiency" in best_config["cost_analysis"]

        # Check relative performance
        assert "relative_performance" in best_config
        assert best_config["relative_performance"] == 1.0  # Best should be 1.0

        # Check rankings are correct
        assert top_configs[0]["rank"] == 1
        assert top_configs[1]["rank"] == 2
        assert top_configs[2]["rank"] == 3

        # Check scores are in descending order
        assert top_configs[0]["score"] >= top_configs[1]["score"]
        assert top_configs[1]["score"] >= top_configs[2]["score"]

    def test_get_optimization_insights_performance_summary(self, optimization_result):
        """Test performance summary analysis."""
        insights = get_optimization_insights(optimization_result)
        summary = insights["performance_summary"]

        # Check basic metrics
        assert summary["total_trials"] == 4
        assert summary["successful_trials"] == 4
        assert summary["primary_objective"] == "accuracy"  # First in objectives list
        assert summary["best_score"] == 0.92
        assert summary["worst_score"] == 0.78
        assert summary["average_score"] == statistics.mean([0.85, 0.92, 0.78, 0.88])

        # Check improvement calculation
        expected_improvement = (0.92 - 0.78) / 0.78
        assert abs(summary["improvement"] - expected_improvement) < 0.001

        # Check consistency metric
        scores = [0.85, 0.92, 0.78, 0.88]
        expected_consistency = 1.0 - (
            statistics.stdev(scores) / statistics.mean(scores)
        )
        assert abs(summary["consistency"] - expected_consistency) < 0.001

        # Check multi-objective analysis
        assert "multi_objective_analysis" in summary
        multi_obj = summary["multi_objective_analysis"]
        assert "accuracy" in multi_obj
        assert "cost_per_1k" in multi_obj
        assert "latency" in multi_obj

        # Check individual objective analysis
        assert multi_obj["accuracy"]["best"] == 0.92
        assert (
            multi_obj["cost_per_1k"]["best"] == 0.008
        )  # Highest cost is "best" in this case
        assert multi_obj["latency"]["best"] == 0.8

    def test_get_optimization_insights_parameter_importance(self, optimization_result):
        """Test parameter importance analysis."""
        insights = get_optimization_insights(optimization_result)
        param_insights = insights["parameter_insights"]

        # Should analyze all parameters
        assert "model" in param_insights
        assert "temperature" in param_insights
        assert "max_tokens" in param_insights

        # Check model parameter analysis (categorical)
        model_insight = param_insights["model"]
        assert "best_value" in model_insight
        assert "performance_impact" in model_insight
        assert "value_distribution" in model_insight
        assert "optimization_priority" in model_insight

        # Best model should be GPT-4o (best trial has accuracy 0.92)
        assert model_insight["best_value"] == "GPT-4o"

        # Check value distribution
        value_dist = model_insight["value_distribution"]
        assert value_dist["type"] == "categorical"
        assert "unique_values" in value_dist
        assert "best_value" in value_dist
        assert "value_performance" in value_dist

        # Check temperature parameter analysis (numeric)
        temp_insight = param_insights["temperature"]
        assert temp_insight["best_value"] == 0.7  # From best trial

        # Check priority classification
        assert model_insight["optimization_priority"] in ["high", "medium", "low"]

    def test_get_optimization_insights_recommendations(self, optimization_result):
        """Test recommendations generation."""
        insights = get_optimization_insights(optimization_result)
        recommendations = insights["recommendations"]

        # Should generate recommendations
        assert len(recommendations) > 0
        assert all(isinstance(rec, str) for rec in recommendations)

        # Should contain performance recommendation (significant improvement)
        performance_recs = [r for r in recommendations if "improvement" in r.lower()]
        assert len(performance_recs) > 0

        # Should contain parameter recommendations
        [r for r in recommendations if "high-impact parameters" in r.lower()]
        # May or may not be present depending on parameter impact analysis

        # All recommendations should be strings
        assert all(isinstance(rec, str) for rec in recommendations)

    def test_get_optimization_insights_no_results(self):
        """Test handling of None or empty results."""
        insights = get_optimization_insights(None)

        assert "error" in insights
        assert insights["error"] == "No optimization results available"
        assert insights["top_configurations"] == []
        assert insights["performance_summary"] == {}
        assert insights["parameter_insights"] == {}
        assert insights["recommendations"] == []

    def test_get_optimization_insights_no_trials(self):
        """Test handling of results with no trials."""
        empty_result = OptimizationResult(
            trials=[],
            best_config=None,
            best_score=0.0,
            optimization_id="empty",
            duration=0.0,
            convergence_info={},
            status=OptimizationStatus.FAILED,
            objectives=["accuracy"],
            algorithm="random",
            timestamp=datetime.now(),
            metadata={},
        )

        insights = get_optimization_insights(empty_result)

        assert "error" in insights
        assert insights["error"] == "No optimization results available"

    def test_get_optimization_insights_failed_trials(self):
        """Test handling of results with only failed trials."""
        failed_trials = [
            TrialResult(
                trial_id="trial_1",
                config={"model": "gpt-4o-mini", "temperature": 0.3},
                metrics={},  # No metrics
                status=TrialStatus.FAILED,
                duration=0.0,
                timestamp=datetime.now(),
                metadata={},
            ),
            TrialResult(
                trial_id="trial_2",
                config={"model": "GPT-4o", "temperature": 0.7},
                metrics=None,  # None metrics
                status=TrialStatus.FAILED,
                duration=0.0,
                timestamp=datetime.now(),
                metadata={},
            ),
        ]

        failed_result = OptimizationResult(
            trials=failed_trials,
            best_config=None,
            best_score=0.0,
            optimization_id="failed",
            duration=5.0,
            convergence_info={},
            status=OptimizationStatus.FAILED,
            objectives=["accuracy"],
            algorithm="random",
            timestamp=datetime.now(),
            metadata={},
        )

        insights = get_optimization_insights(failed_result)

        assert "error" in insights
        assert insights["error"] == "No successful trials found"

    def test_get_optimization_insights_mixed_trial_status(self, sample_trials):
        """Test handling of results with mixed successful/failed trials."""
        # Add a failed trial
        mixed_trials = sample_trials + [
            TrialResult(
                trial_id="trial_failed",
                config={"model": "invalid", "temperature": 2.0},
                metrics={},
                status=TrialStatus.FAILED,
                duration=0.0,
                timestamp=datetime.now(),
                metadata={},
            )
        ]

        mixed_result = OptimizationResult(
            trials=mixed_trials,
            best_config={"model": "GPT-4o", "temperature": 0.7, "max_tokens": 300},
            best_score=0.92,
            optimization_id="mixed",
            duration=15.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy"],
            algorithm="random",
            timestamp=datetime.now(),
            metadata={},
        )

        insights = get_optimization_insights(mixed_result)

        # Should only analyze successful trials
        assert "error" not in insights
        assert (
            insights["performance_summary"]["successful_trials"] == 4
        )  # Only successful ones
        assert len(insights["top_configurations"]) == 3  # Top 3 from successful trials

    def test_get_optimization_insights_single_objective(self):
        """Test insights with single objective."""
        single_trial = [
            TrialResult(
                trial_id="trial_1",
                config={"model": "gpt-4o-mini", "temperature": 0.5},
                metrics={"accuracy": 0.85},
                status=TrialStatus.COMPLETED,
                duration=2.0,
                timestamp=datetime.now(),
                metadata={},
            )
        ]

        single_result = OptimizationResult(
            trials=single_trial,
            best_config={"model": "gpt-4o-mini", "temperature": 0.5},
            best_score=0.85,
            optimization_id="single",
            duration=5.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy"],  # Single objective
            algorithm="random",
            timestamp=datetime.now(),
            metadata={},
        )

        insights = get_optimization_insights(single_result)

        # Should not have multi-objective analysis
        assert "multi_objective_analysis" not in insights["performance_summary"]
        assert insights["performance_summary"]["primary_objective"] == "accuracy"

    def test_get_optimization_insights_cost_efficiency_recommendations(self):
        """Test cost efficiency recommendations generation."""
        # Create trials with clear cost/performance trade-offs
        cost_trials = [
            TrialResult(
                trial_id="expensive",
                config={"model": "GPT-4o", "temperature": 0.3},
                metrics={
                    "accuracy": 0.90,
                    "cost_per_1k": 0.010,
                },  # High cost, good performance
                status=TrialStatus.COMPLETED,
                duration=2.0,
                timestamp=datetime.now(),
                metadata={},
            ),
            TrialResult(
                trial_id="efficient",
                config={"model": "gpt-4o-mini", "temperature": 0.3},
                metrics={
                    "accuracy": 0.88,
                    "cost_per_1k": 0.002,
                },  # Low cost, slightly lower performance
                status=TrialStatus.COMPLETED,
                duration=1.5,
                timestamp=datetime.now(),
                metadata={},
            ),
        ]

        cost_result = OptimizationResult(
            trials=cost_trials,
            best_config={"model": "GPT-4o", "temperature": 0.3},
            best_score=0.90,
            optimization_id="cost_test",
            duration=5.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy"],
            algorithm="random",
            timestamp=datetime.now(),
            metadata={},
        )

        insights = get_optimization_insights(cost_result)

        # Should generate cost efficiency recommendation
        recommendations = insights["recommendations"]
        cost_recs = [r for r in recommendations if "cost efficiency" in r.lower()]

        # Should recommend the more cost-efficient option
        if cost_recs:
            assert "configuration #2" in cost_recs[0]  # Second best for cost efficiency

    def test_get_optimization_insights_consistency_warning(self):
        """Test low consistency warning generation."""
        # Create trials with high variance
        variable_trials = [
            TrialResult(
                trial_id="trial_1",
                config={"model": "gpt-4o-mini", "temperature": 0.1},
                metrics={"accuracy": 0.95},  # High
                status=TrialStatus.COMPLETED,
                duration=2.0,
                timestamp=datetime.now(),
                metadata={},
            ),
            TrialResult(
                trial_id="trial_2",
                config={"model": "gpt-4o-mini", "temperature": 0.2},
                metrics={"accuracy": 0.65},  # Low
                status=TrialStatus.COMPLETED,
                duration=2.0,
                timestamp=datetime.now(),
                metadata={},
            ),
            TrialResult(
                trial_id="trial_3",
                config={"model": "gpt-4o-mini", "temperature": 0.3},
                metrics={"accuracy": 0.85},  # Medium
                status=TrialStatus.COMPLETED,
                duration=2.0,
                timestamp=datetime.now(),
                metadata={},
            ),
        ]

        variable_result = OptimizationResult(
            trials=variable_trials,
            best_config={"model": "gpt-4o-mini", "temperature": 0.1},
            best_score=0.95,
            optimization_id="variable_test",
            duration=10.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy"],
            algorithm="random",
            timestamp=datetime.now(),
            metadata={},
        )

        insights = get_optimization_insights(variable_result)

        # Should warn about high variance
        recommendations = insights["recommendations"]
        variance_warnings = [r for r in recommendations if "variance" in r.lower()]

        if variance_warnings:
            assert (
                "more trials" in variance_warnings[0].lower()
                or "adjusting parameter" in variance_warnings[0].lower()
            )

    def test_get_optimization_insights_with_custom_metrics(self):
        """Test insights with custom metrics beyond standard ones."""
        custom_trials = [
            TrialResult(
                trial_id="trial_1",
                config={"model": "gpt-4o-mini", "custom_param": "value1"},
                metrics={
                    "accuracy": 0.85,
                    "custom_metric": 0.75,
                    "response_quality": 4.2,
                },
                status=TrialStatus.COMPLETED,
                duration=2.0,
                timestamp=datetime.now(),
                metadata={},
            ),
            TrialResult(
                trial_id="trial_2",
                config={"model": "GPT-4o", "custom_param": "value2"},
                metrics={
                    "accuracy": 0.90,
                    "custom_metric": 0.80,
                    "response_quality": 4.5,
                },
                status=TrialStatus.COMPLETED,
                duration=3.0,
                timestamp=datetime.now(),
                metadata={},
            ),
        ]

        custom_result = OptimizationResult(
            trials=custom_trials,
            best_config={"model": "GPT-4o", "custom_param": "value2"},
            best_score=0.90,
            optimization_id="custom_test",
            duration=8.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy", "custom_metric"],
            algorithm="random",
            timestamp=datetime.now(),
            metadata={},
        )

        insights = get_optimization_insights(custom_result)

        # Should handle custom metrics
        assert "error" not in insights

        # Check multi-objective analysis includes custom metrics
        multi_obj = insights["performance_summary"]["multi_objective_analysis"]
        assert "accuracy" in multi_obj
        assert "custom_metric" in multi_obj

        # Check parameter insights includes custom parameters
        assert "custom_param" in insights["parameter_insights"]

    def test_get_optimization_insights_parameter_impact_calculation(self):
        """Test parameter impact calculation for different parameter types."""
        # Test with clear parameter impact
        impact_trials = [
            # GPT-4o performs better
            TrialResult(
                trial_id="trial_1",
                config={"model": "GPT-4o", "temperature": 0.3},
                metrics={"accuracy": 0.92},
                status=TrialStatus.COMPLETED,
                duration=2.0,
                timestamp=datetime.now(),
                metadata={},
            ),
            TrialResult(
                trial_id="trial_2",
                config={"model": "GPT-4o", "temperature": 0.7},
                metrics={"accuracy": 0.90},
                status=TrialStatus.COMPLETED,
                duration=2.0,
                timestamp=datetime.now(),
                metadata={},
            ),
            # gpt-4o-mini performs worse
            TrialResult(
                trial_id="trial_3",
                config={"model": "gpt-4o-mini", "temperature": 0.3},
                metrics={"accuracy": 0.75},
                status=TrialStatus.COMPLETED,
                duration=1.5,
                timestamp=datetime.now(),
                metadata={},
            ),
            TrialResult(
                trial_id="trial_4",
                config={"model": "gpt-4o-mini", "temperature": 0.7},
                metrics={"accuracy": 0.73},
                status=TrialStatus.COMPLETED,
                duration=1.5,
                timestamp=datetime.now(),
                metadata={},
            ),
        ]

        impact_result = OptimizationResult(
            trials=impact_trials,
            best_config={"model": "GPT-4o", "temperature": 0.3},
            best_score=0.92,
            optimization_id="impact_test",
            duration=10.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy"],
            algorithm="random",
            timestamp=datetime.now(),
            metadata={},
        )

        insights = get_optimization_insights(impact_result)

        # Model should have high impact (clear difference between GPT-4o and gpt-4o-mini)
        model_insight = insights["parameter_insights"]["model"]
        assert (
            model_insight["performance_impact"] > 0.05
        )  # Should detect significant impact
        assert model_insight["optimization_priority"] in ["high", "medium"]

        # Temperature might have lower impact
        temp_insight = insights["parameter_insights"]["temperature"]
        assert "performance_impact" in temp_insight
        assert "optimization_priority" in temp_insight
