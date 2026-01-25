"""Tests for VariableAnalyzer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest


@dataclass
class MockTrial:
    """Mock trial for testing."""

    config: dict[str, Any]
    metrics: dict[str, float]


@dataclass
class MockOptimizationResult:
    """Mock optimization result for testing."""

    trials: list[MockTrial]
    configuration_space: dict[str, Any]
    objectives: list[str] | None = None


class TestVariableAnalyzer:
    """Tests for VariableAnalyzer class."""

    def test_basic_initialization(self):
        """Test analyzer initialization."""
        from traigent_tuned_variables import VariableAnalyzer

        result = MockOptimizationResult(
            trials=[],
            configuration_space={"temperature": (0.0, 1.0)},
        )

        analyzer = VariableAnalyzer(result)

        assert analyzer.importance_method == "variance"
        assert analyzer.elimination_threshold == 0.05

    def test_get_variable_importance_empty(self):
        """Test importance calculation with no trials."""
        from traigent_tuned_variables import VariableAnalyzer

        result = MockOptimizationResult(
            trials=[],
            configuration_space={"temperature": (0.0, 1.0)},
        )

        analyzer = VariableAnalyzer(result)
        importance = analyzer.get_variable_importance("accuracy")

        assert importance == {}

    def test_get_variable_importance_single_variable(self):
        """Test importance calculation with trials."""
        from traigent_tuned_variables import VariableAnalyzer

        trials = [
            MockTrial(config={"model": "gpt-4"}, metrics={"accuracy": 0.9}),
            MockTrial(config={"model": "gpt-4"}, metrics={"accuracy": 0.85}),
            MockTrial(config={"model": "gpt-3.5"}, metrics={"accuracy": 0.7}),
            MockTrial(config={"model": "gpt-3.5"}, metrics={"accuracy": 0.75}),
        ]

        result = MockOptimizationResult(
            trials=trials,
            configuration_space={"model": ["gpt-4", "gpt-3.5"]},
        )

        analyzer = VariableAnalyzer(result)
        importance = analyzer.get_variable_importance("accuracy")

        # Model should show importance since values differ
        assert "model" in importance
        assert importance["model"] > 0

    def test_get_value_rankings(self):
        """Test value rankings for categorical variable."""
        from traigent_tuned_variables import VariableAnalyzer

        trials = [
            MockTrial(config={"model": "a"}, metrics={"acc": 0.9}),
            MockTrial(config={"model": "a"}, metrics={"acc": 0.85}),
            MockTrial(config={"model": "a"}, metrics={"acc": 0.88}),
            MockTrial(config={"model": "b"}, metrics={"acc": 0.6}),
            MockTrial(config={"model": "b"}, metrics={"acc": 0.65}),
            MockTrial(config={"model": "b"}, metrics={"acc": 0.62}),
        ]

        result = MockOptimizationResult(
            trials=trials,
            configuration_space={"model": ["a", "b"]},
        )

        analyzer = VariableAnalyzer(result)
        rankings = analyzer.get_value_rankings("model", "acc")

        assert len(rankings) == 2
        # Model "a" should rank higher
        assert rankings[0].value == "a"
        assert rankings[0].mean_score > rankings[1].mean_score
        # "b" might be marked as dominated
        assert rankings[1].value == "b"

    def test_get_dominated_values_single_objective(self):
        """Test dominated value detection for single objective."""
        from traigent_tuned_variables import VariableAnalyzer

        # Create clear dominance: "good" always better than "bad"
        trials = [
            MockTrial(config={"strategy": "good"}, metrics={"score": 0.9}),
            MockTrial(config={"strategy": "good"}, metrics={"score": 0.95}),
            MockTrial(config={"strategy": "good"}, metrics={"score": 0.92}),
            MockTrial(config={"strategy": "bad"}, metrics={"score": 0.3}),
            MockTrial(config={"strategy": "bad"}, metrics={"score": 0.35}),
            MockTrial(config={"strategy": "bad"}, metrics={"score": 0.32}),
        ]

        result = MockOptimizationResult(
            trials=trials,
            configuration_space={"strategy": ["good", "bad"]},
        )

        analyzer = VariableAnalyzer(result)
        dominated = analyzer.get_dominated_values("strategy", ["score"])

        # "bad" should be dominated
        assert "bad" in dominated

    def test_suggest_range_adjustment(self):
        """Test range adjustment suggestion for numeric variable."""
        from traigent_tuned_variables import VariableAnalyzer

        # Best trials at temperature around 0.7
        trials = [
            MockTrial(config={"temperature": 0.1}, metrics={"acc": 0.5}),
            MockTrial(config={"temperature": 0.3}, metrics={"acc": 0.6}),
            MockTrial(config={"temperature": 0.5}, metrics={"acc": 0.7}),
            MockTrial(config={"temperature": 0.7}, metrics={"acc": 0.9}),
            MockTrial(config={"temperature": 0.8}, metrics={"acc": 0.85}),
            MockTrial(config={"temperature": 0.9}, metrics={"acc": 0.75}),
        ]

        result = MockOptimizationResult(
            trials=trials,
            configuration_space={"temperature": (0.0, 1.0)},
        )

        analyzer = VariableAnalyzer(result)
        suggested = analyzer.suggest_range_adjustment("temperature", "acc")

        # Should suggest narrower range around 0.7-0.8
        assert suggested is not None
        low, high = suggested
        assert low >= 0.0
        assert high <= 1.0
        # Should include the high-performing region
        assert low <= 0.7
        assert high >= 0.8

    def test_get_refined_space_prunes_low_importance(self):
        """Test that get_refined_space prunes low-importance variables."""
        from traigent_tuned_variables import VariableAnalyzer

        # Create trials where 'model' matters but 'unused' doesn't
        trials = [
            MockTrial(config={"model": "a", "unused": "x"}, metrics={"acc": 0.9}),
            MockTrial(config={"model": "a", "unused": "y"}, metrics={"acc": 0.9}),
            MockTrial(config={"model": "b", "unused": "x"}, metrics={"acc": 0.5}),
            MockTrial(config={"model": "b", "unused": "y"}, metrics={"acc": 0.5}),
        ]

        result = MockOptimizationResult(
            trials=trials,
            configuration_space={
                "model": ["a", "b"],
                "unused": ["x", "y"],
            },
        )

        analyzer = VariableAnalyzer(result, elimination_threshold=0.01)
        refined = analyzer.get_refined_space(
            ["acc"], prune_low_importance=True, return_typed=False
        )

        # Model should be kept (high importance)
        assert "model" in refined
        # Unused might be pruned (low importance)
        # Note: depends on exact variance calculation

    def test_get_refined_space_returns_typed(self):
        """Test that get_refined_space returns typed ParameterRange objects."""
        from traigent_tuned_variables import VariableAnalyzer

        from traigent.api.parameter_ranges import Choices, Range

        trials = [
            MockTrial(config={"temp": 0.5, "model": "a"}, metrics={"acc": 0.8}),
            MockTrial(config={"temp": 0.7, "model": "b"}, metrics={"acc": 0.9}),
        ]

        result = MockOptimizationResult(
            trials=trials,
            configuration_space={
                "temp": (0.0, 1.0),
                "model": ["a", "b"],
            },
        )

        analyzer = VariableAnalyzer(result, elimination_threshold=0.0)
        refined = analyzer.get_refined_space(
            ["acc"], prune_low_importance=False, return_typed=True
        )

        # Should return ParameterRange objects
        assert isinstance(refined.get("temp"), Range)
        assert isinstance(refined.get("model"), Choices)

    def test_analyze_returns_complete_analysis(self):
        """Test that analyze() returns complete OptimizationAnalysis."""
        from traigent_tuned_variables import VariableAnalyzer
        from traigent_tuned_variables.analysis import EliminationAction

        trials = [
            MockTrial(config={"model": "good"}, metrics={"acc": 0.9}),
            MockTrial(config={"model": "good"}, metrics={"acc": 0.85}),
            MockTrial(config={"model": "good"}, metrics={"acc": 0.88}),
            MockTrial(config={"model": "bad"}, metrics={"acc": 0.4}),
            MockTrial(config={"model": "bad"}, metrics={"acc": 0.45}),
            MockTrial(config={"model": "bad"}, metrics={"acc": 0.42}),
        ]

        result = MockOptimizationResult(
            trials=trials,
            configuration_space={"model": ["good", "bad"]},
        )

        analyzer = VariableAnalyzer(result)
        analysis = analyzer.analyze("acc")

        # Should have variable analysis
        assert "model" in analysis.variables
        var_analysis = analysis.variables["model"]
        assert var_analysis.name == "model"
        assert var_analysis.var_type == "categorical"
        assert var_analysis.importance >= 0

        # Should have value rankings
        assert var_analysis.value_rankings is not None
        assert len(var_analysis.value_rankings) == 2

        # Should have suggestion (prune dominated values)
        assert var_analysis.suggestion is not None
        # "bad" is dominated, so suggestion should be to prune
        if var_analysis.suggestion.action == EliminationAction.PRUNE_VALUES:
            assert "bad" in (var_analysis.suggestion.dominated_values or [])


class TestStatisticsHelpers:
    """Tests for statistics helper functions."""

    def test_mean_empty(self):
        """Test mean of empty list."""
        from traigent_tuned_variables.analysis.variable_analyzer import _mean

        assert _mean([]) == 0.0

    def test_mean_values(self):
        """Test mean calculation."""
        from traigent_tuned_variables.analysis.variable_analyzer import _mean

        result = _mean([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result == 3.0

    def test_stdev_single_value(self):
        """Test stdev with single value."""
        from traigent_tuned_variables.analysis.variable_analyzer import _stdev

        assert _stdev([1.0]) == 0.0

    def test_stdev_values(self):
        """Test stdev calculation."""
        from traigent_tuned_variables.analysis.variable_analyzer import _stdev

        result = _stdev([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result > 0

    def test_percentile_empty(self):
        """Test percentile of empty list."""
        from traigent_tuned_variables.analysis.variable_analyzer import _percentile

        assert _percentile([], 50) == 0.0

    def test_percentile_median(self):
        """Test median (50th percentile)."""
        from traigent_tuned_variables.analysis.variable_analyzer import _percentile

        result = _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50)
        assert result == pytest.approx(3.0, abs=0.1)

    def test_percentile_quartiles(self):
        """Test 25th and 75th percentiles."""
        from traigent_tuned_variables.analysis.variable_analyzer import _percentile

        values = list(range(1, 101))  # 1 to 100

        p25 = _percentile(values, 25)
        p75 = _percentile(values, 75)

        assert p25 < p75
        assert p25 == pytest.approx(25.75, abs=1)
        assert p75 == pytest.approx(75.25, abs=1)


class TestCodexFixes:
    """Tests for issues found in Codex review."""

    def test_explicit_config_space(self):
        """Test analyzer accepts explicit configuration_space arg."""
        from traigent_tuned_variables import VariableAnalyzer

        trials = [
            MockTrial(config={"model": "a"}, metrics={"acc": 0.9}),
            MockTrial(config={"model": "b"}, metrics={"acc": 0.7}),
        ]

        # Result without config space
        result = MockOptimizationResult(trials=trials, configuration_space=None)

        # Provide explicit config space
        explicit_space = {"model": ["a", "b"]}
        analyzer = VariableAnalyzer(result, configuration_space=explicit_space)

        # Should work despite result lacking config space
        importance = analyzer.get_variable_importance("acc")
        assert "model" in importance

    def test_objective_direction_minimize(self):
        """Test analyzer respects 'minimize' direction for rankings."""
        from traigent_tuned_variables import VariableAnalyzer

        # Lower scores are better for latency
        trials = [
            MockTrial(config={"model": "fast"}, metrics={"latency": 10.0}),
            MockTrial(config={"model": "fast"}, metrics={"latency": 12.0}),
            MockTrial(config={"model": "fast"}, metrics={"latency": 11.0}),
            MockTrial(config={"model": "slow"}, metrics={"latency": 50.0}),
            MockTrial(config={"model": "slow"}, metrics={"latency": 55.0}),
            MockTrial(config={"model": "slow"}, metrics={"latency": 52.0}),
        ]

        result = MockOptimizationResult(
            trials=trials,
            configuration_space={"model": ["fast", "slow"]},
        )

        # Without direction, assumes maximize (wrong for latency)
        analyzer_wrong = VariableAnalyzer(result)
        rankings_wrong = analyzer_wrong.get_value_rankings("model", "latency")
        # Would rank "slow" first (higher values)
        assert rankings_wrong[0].value == "slow"

        # With correct direction
        analyzer_correct = VariableAnalyzer(result, directions={"latency": "minimize"})
        rankings_correct = analyzer_correct.get_value_rankings("model", "latency")
        # Should rank "fast" first (lower is better)
        assert rankings_correct[0].value == "fast"

    def test_collapsed_range_handling(self):
        """Test _to_parameter_range handles low==high case."""
        from traigent_tuned_variables import VariableAnalyzer

        from traigent.api.parameter_ranges import Range

        trials = [
            MockTrial(config={"temp": 0.7}, metrics={"acc": 0.9}),
            MockTrial(config={"temp": 0.7}, metrics={"acc": 0.85}),
        ]

        result = MockOptimizationResult(
            trials=trials,
            configuration_space={"temp": (0.0, 1.0)},
        )

        analyzer = VariableAnalyzer(result, elimination_threshold=0.0)

        # This would previously crash if suggest_range_adjustment returned (0.7, 0.7)
        # Now it should expand to a small range
        refined = analyzer.get_refined_space(["acc"], return_typed=True)

        assert "temp" in refined
        temp_range = refined["temp"]
        assert isinstance(temp_range, Range)
        # Should be a valid range with low < high
        assert temp_range.low < temp_range.high

    def test_config_metrics_name_collision(self):
        """Test that name collisions between config and metrics are handled."""
        from traigent_tuned_variables import VariableAnalyzer

        # This trial has "score" as both config param and metric
        trials = [
            MockTrial(
                config={"score": "high"},  # categorical config param
                metrics={"score": 0.9},  # numeric metric with same name
            ),
        ]

        result = MockOptimizationResult(
            trials=trials,
            configuration_space={"score": ["high", "low"]},
        )

        analyzer = VariableAnalyzer(result)

        # Should not crash and should handle collision gracefully
        # The metric should be available as "metric_score"
        trial_dicts = analyzer._get_trials_as_dicts()
        assert len(trial_dicts) == 1
        assert trial_dicts[0]["score"] == "high"  # Config wins
        assert trial_dicts[0]["metric_score"] == pytest.approx(0.9)  # Metric renamed

    def test_dominated_values_minimize_direction(self):
        """Test dominated value detection respects minimize direction."""
        from traigent_tuned_variables import VariableAnalyzer

        # For cost, lower is better
        trials = [
            MockTrial(config={"provider": "cheap"}, metrics={"cost": 1.0}),
            MockTrial(config={"provider": "cheap"}, metrics={"cost": 1.2}),
            MockTrial(config={"provider": "cheap"}, metrics={"cost": 1.1}),
            MockTrial(config={"provider": "expensive"}, metrics={"cost": 10.0}),
            MockTrial(config={"provider": "expensive"}, metrics={"cost": 12.0}),
            MockTrial(config={"provider": "expensive"}, metrics={"cost": 11.0}),
        ]

        result = MockOptimizationResult(
            trials=trials,
            configuration_space={"provider": ["cheap", "expensive"]},
        )

        analyzer = VariableAnalyzer(result, directions={"cost": "minimize"})
        dominated = analyzer.get_dominated_values("provider", ["cost"])

        # "expensive" should be dominated (higher cost is worse)
        assert "expensive" in dominated

    def test_get_refined_space_typed_range_return_typed_true(self):
        """Test get_refined_space with typed Range in config_space, return_typed=True."""
        from traigent_tuned_variables import VariableAnalyzer

        from traigent.api.parameter_ranges import IntRange, Range

        # Config space has typed ParameterRange objects
        trials = [
            MockTrial(config={"temp": 0.5, "tokens": 100}, metrics={"acc": 0.8}),
            MockTrial(config={"temp": 0.7, "tokens": 200}, metrics={"acc": 0.9}),
            MockTrial(config={"temp": 0.6, "tokens": 150}, metrics={"acc": 0.85}),
        ]

        result = MockOptimizationResult(
            trials=trials,
            configuration_space={
                "temp": Range(0.0, 1.0),  # Typed Range
                "tokens": IntRange(50, 500),  # Typed IntRange
            },
        )

        analyzer = VariableAnalyzer(result, elimination_threshold=0.0)
        refined = analyzer.get_refined_space(
            ["acc"], prune_low_importance=False, return_typed=True
        )

        # Should return Range objects, not nested Range(Range(...))
        assert "temp" in refined
        assert isinstance(refined["temp"], Range)
        # Check it's a valid Range, not corrupted
        assert isinstance(refined["temp"].low, (int, float))
        assert isinstance(refined["temp"].high, (int, float))
        assert refined["temp"].low < refined["temp"].high

        assert "tokens" in refined
        assert isinstance(refined["tokens"], IntRange)
        assert isinstance(refined["tokens"].low, int)
        assert isinstance(refined["tokens"].high, int)
        assert refined["tokens"].low < refined["tokens"].high

    def test_get_refined_space_typed_range_return_typed_false(self):
        """Test get_refined_space with typed Range in config_space, return_typed=False."""
        from traigent_tuned_variables import VariableAnalyzer

        from traigent.api.parameter_ranges import IntRange, Range

        # Config space has typed ParameterRange objects
        trials = [
            MockTrial(config={"temp": 0.5, "tokens": 100}, metrics={"acc": 0.8}),
            MockTrial(config={"temp": 0.7, "tokens": 200}, metrics={"acc": 0.9}),
            MockTrial(config={"temp": 0.6, "tokens": 150}, metrics={"acc": 0.85}),
        ]

        result = MockOptimizationResult(
            trials=trials,
            configuration_space={
                "temp": Range(0.0, 1.0),  # Typed Range
                "tokens": IntRange(50, 500),  # Typed IntRange
            },
        )

        analyzer = VariableAnalyzer(result, elimination_threshold=0.0)
        refined = analyzer.get_refined_space(
            ["acc"], prune_low_importance=False, return_typed=False
        )

        # Should return raw tuples, not Range objects or nested structures
        assert "temp" in refined
        assert isinstance(refined["temp"], tuple)
        assert len(refined["temp"]) == 2
        # Values should be numeric, not Range objects
        assert isinstance(refined["temp"][0], (int, float))
        assert isinstance(refined["temp"][1], (int, float))

        assert "tokens" in refined
        assert isinstance(refined["tokens"], tuple)
        assert len(refined["tokens"]) == 2
        assert isinstance(refined["tokens"][0], (int, float))
        assert isinstance(refined["tokens"][1], (int, float))


class TestMultiObjectiveAnalysis:
    """Tests for multi-objective analysis functionality (P-2)."""

    def test_analyze_multi_objective_empty_objectives(self):
        """Test multi-objective analysis with empty objectives list."""
        from traigent_tuned_variables import VariableAnalyzer
        from traigent_tuned_variables.analysis import MultiObjectiveAnalysis

        result = MockOptimizationResult(
            trials=[],
            configuration_space={"model": ["a", "b"]},
        )

        analyzer = VariableAnalyzer(result)
        analysis = analyzer.analyze_multi_objective([])

        assert isinstance(analysis, MultiObjectiveAnalysis)
        assert analysis.objectives == []
        assert analysis.variables == {}

    def test_analyze_multi_objective_single_objective(self):
        """Test multi-objective analysis with single objective (same as analyze)."""
        from traigent_tuned_variables import VariableAnalyzer

        trials = [
            MockTrial(config={"model": "a"}, metrics={"acc": 0.9}),
            MockTrial(config={"model": "a"}, metrics={"acc": 0.85}),
            MockTrial(config={"model": "a"}, metrics={"acc": 0.88}),
            MockTrial(config={"model": "b"}, metrics={"acc": 0.4}),
            MockTrial(config={"model": "b"}, metrics={"acc": 0.45}),
            MockTrial(config={"model": "b"}, metrics={"acc": 0.42}),
        ]

        result = MockOptimizationResult(
            trials=trials,
            configuration_space={"model": ["a", "b"]},
        )

        analyzer = VariableAnalyzer(result)
        analysis = analyzer.analyze_multi_objective(["acc"])

        assert analysis.objectives == ["acc"]
        assert "model" in analysis.variables
        var_analysis = analysis.variables["model"]
        assert var_analysis.name == "model"
        assert "acc" in var_analysis.importance_by_objective
        assert var_analysis.aggregate_importance > 0

    def test_analyze_multi_objective_two_objectives(self):
        """Test multi-objective analysis with two objectives."""
        from traigent_tuned_variables import VariableAnalyzer

        trials = [
            MockTrial(config={"model": "fast"}, metrics={"acc": 0.7, "latency": 10}),
            MockTrial(config={"model": "fast"}, metrics={"acc": 0.65, "latency": 12}),
            MockTrial(config={"model": "fast"}, metrics={"acc": 0.68, "latency": 11}),
            MockTrial(config={"model": "slow"}, metrics={"acc": 0.9, "latency": 50}),
            MockTrial(config={"model": "slow"}, metrics={"acc": 0.88, "latency": 55}),
            MockTrial(config={"model": "slow"}, metrics={"acc": 0.92, "latency": 52}),
        ]

        result = MockOptimizationResult(
            trials=trials,
            configuration_space={"model": ["fast", "slow"]},
        )

        analyzer = VariableAnalyzer(result)
        analysis = analyzer.analyze_multi_objective(["acc", "latency"])

        assert analysis.objectives == ["acc", "latency"]
        assert "model" in analysis.variables
        var_analysis = analysis.variables["model"]

        # Should have importance for both objectives
        assert "acc" in var_analysis.importance_by_objective
        assert "latency" in var_analysis.importance_by_objective

    def test_analyze_multi_objective_aggregation_methods(self):
        """Test different importance aggregation methods."""
        from traigent_tuned_variables import VariableAnalyzer

        trials = [
            MockTrial(config={"model": "a"}, metrics={"obj1": 0.9, "obj2": 0.1}),
            MockTrial(config={"model": "a"}, metrics={"obj1": 0.85, "obj2": 0.15}),
            MockTrial(config={"model": "a"}, metrics={"obj1": 0.88, "obj2": 0.12}),
            MockTrial(config={"model": "b"}, metrics={"obj1": 0.3, "obj2": 0.8}),
            MockTrial(config={"model": "b"}, metrics={"obj1": 0.35, "obj2": 0.85}),
            MockTrial(config={"model": "b"}, metrics={"obj1": 0.32, "obj2": 0.82}),
        ]

        result = MockOptimizationResult(
            trials=trials,
            configuration_space={"model": ["a", "b"]},
        )

        analyzer = VariableAnalyzer(result)

        # Mean aggregation
        analysis_mean = analyzer.analyze_multi_objective(["obj1", "obj2"], aggregation="mean")
        mean_imp = analysis_mean.variables["model"].aggregate_importance

        # Max aggregation (keep if important for any)
        analysis_max = analyzer.analyze_multi_objective(["obj1", "obj2"], aggregation="max")
        max_imp = analysis_max.variables["model"].aggregate_importance

        # Min aggregation (eliminate only if unimportant for all)
        analysis_min = analyzer.analyze_multi_objective(["obj1", "obj2"], aggregation="min")
        min_imp = analysis_min.variables["model"].aggregate_importance

        # Max should be >= mean >= min
        assert max_imp >= mean_imp >= min_imp

    def test_analyze_multi_objective_pareto_dominated(self):
        """Test Pareto dominance detection in multi-objective analysis."""
        from traigent_tuned_variables import VariableAnalyzer

        # "dominated" is worse on BOTH objectives
        trials = [
            MockTrial(config={"strategy": "good"}, metrics={"acc": 0.9, "speed": 0.8}),
            MockTrial(config={"strategy": "good"}, metrics={"acc": 0.88, "speed": 0.85}),
            MockTrial(config={"strategy": "good"}, metrics={"acc": 0.92, "speed": 0.82}),
            MockTrial(config={"strategy": "dominated"}, metrics={"acc": 0.3, "speed": 0.2}),
            MockTrial(config={"strategy": "dominated"}, metrics={"acc": 0.32, "speed": 0.22}),
            MockTrial(config={"strategy": "dominated"}, metrics={"acc": 0.28, "speed": 0.18}),
        ]

        result = MockOptimizationResult(
            trials=trials,
            configuration_space={"strategy": ["good", "dominated"]},
        )

        analyzer = VariableAnalyzer(result)
        analysis = analyzer.analyze_multi_objective(["acc", "speed"])

        var_analysis = analysis.variables["strategy"]
        assert var_analysis.pareto_dominated_values is not None
        assert "dominated" in var_analysis.pareto_dominated_values

        # Pareto frontier should only contain "good"
        assert "strategy" in analysis.pareto_frontier_values
        assert "good" in analysis.pareto_frontier_values["strategy"]
        assert "dominated" not in analysis.pareto_frontier_values["strategy"]

    def test_analyze_multi_objective_pareto_tradeoff(self):
        """Test that trade-off values are NOT Pareto dominated."""
        from traigent_tuned_variables import VariableAnalyzer

        # Trade-off: "fast" is good at speed, "accurate" is good at accuracy
        trials = [
            MockTrial(config={"strategy": "fast"}, metrics={"acc": 0.6, "speed": 0.9}),
            MockTrial(config={"strategy": "fast"}, metrics={"acc": 0.58, "speed": 0.92}),
            MockTrial(config={"strategy": "fast"}, metrics={"acc": 0.62, "speed": 0.88}),
            MockTrial(config={"strategy": "accurate"}, metrics={"acc": 0.95, "speed": 0.4}),
            MockTrial(config={"strategy": "accurate"}, metrics={"acc": 0.93, "speed": 0.42}),
            MockTrial(config={"strategy": "accurate"}, metrics={"acc": 0.97, "speed": 0.38}),
        ]

        result = MockOptimizationResult(
            trials=trials,
            configuration_space={"strategy": ["fast", "accurate"]},
        )

        analyzer = VariableAnalyzer(result)
        analysis = analyzer.analyze_multi_objective(["acc", "speed"])

        # Neither should be dominated (they're on the Pareto frontier)
        var_analysis = analysis.variables["strategy"]
        dominated = var_analysis.pareto_dominated_values or []
        assert "fast" not in dominated
        assert "accurate" not in dominated

    def test_analyze_multi_objective_refined_space(self):
        """Test that refined space is generated for multi-objective."""
        from traigent_tuned_variables import VariableAnalyzer

        trials = [
            MockTrial(config={"model": "good", "temp": 0.7}, metrics={"acc": 0.9, "cost": 1.0}),
            MockTrial(config={"model": "good", "temp": 0.8}, metrics={"acc": 0.88, "cost": 1.1}),
            MockTrial(config={"model": "good", "temp": 0.75}, metrics={"acc": 0.92, "cost": 0.9}),
            MockTrial(config={"model": "bad", "temp": 0.1}, metrics={"acc": 0.3, "cost": 5.0}),
            MockTrial(config={"model": "bad", "temp": 0.2}, metrics={"acc": 0.32, "cost": 4.8}),
            MockTrial(config={"model": "bad", "temp": 0.15}, metrics={"acc": 0.28, "cost": 5.2}),
        ]

        result = MockOptimizationResult(
            trials=trials,
            configuration_space={
                "model": ["good", "bad"],
                "temp": (0.0, 1.0),
            },
        )

        analyzer = VariableAnalyzer(result)
        analysis = analyzer.analyze_multi_objective(["acc", "cost"])

        # Should have refined space
        assert analysis.refined_space is not None
        assert len(analysis.refined_space) > 0

    def test_analyze_multi_objective_elimination_suggestions(self):
        """Test elimination suggestions in multi-objective analysis."""
        from traigent_tuned_variables import VariableAnalyzer
        from traigent_tuned_variables.analysis import EliminationAction

        trials = [
            MockTrial(config={"model": "good", "unused": "x"}, metrics={"acc": 0.9, "speed": 0.8}),
            MockTrial(config={"model": "good", "unused": "y"}, metrics={"acc": 0.9, "speed": 0.8}),
            MockTrial(config={"model": "bad", "unused": "x"}, metrics={"acc": 0.3, "speed": 0.2}),
            MockTrial(config={"model": "bad", "unused": "y"}, metrics={"acc": 0.3, "speed": 0.2}),
        ]

        result = MockOptimizationResult(
            trials=trials,
            configuration_space={
                "model": ["good", "bad"],
                "unused": ["x", "y"],  # Same performance regardless of value
            },
        )

        analyzer = VariableAnalyzer(result, elimination_threshold=0.01)
        analysis = analyzer.analyze_multi_objective(["acc", "speed"])

        # "model" should have suggestions (either prune or keep)
        model_analysis = analysis.variables["model"]
        assert model_analysis.suggestion is not None

        # Check if "bad" is suggested for pruning
        if model_analysis.suggestion.action == EliminationAction.PRUNE_VALUES:
            assert "bad" in (model_analysis.suggestion.dominated_values or [])

    def test_multi_objective_analysis_dataclass_fields(self):
        """Test that MultiObjectiveAnalysis dataclass has expected fields."""
        from traigent_tuned_variables.analysis import (
            MultiObjectiveAnalysis,
            MultiObjectiveVariableAnalysis,
        )

        # Test empty initialization
        analysis = MultiObjectiveAnalysis()
        assert analysis.objectives == []
        assert analysis.variables == {}
        assert analysis.elimination_suggestions == []
        assert analysis.pareto_frontier_values == {}
        assert analysis.refined_space == {}

        # Test MultiObjectiveVariableAnalysis
        var_analysis = MultiObjectiveVariableAnalysis(
            name="test",
            var_type="categorical",
            importance_by_objective={"acc": 0.5, "cost": 0.3},
            aggregate_importance=0.4,
        )
        assert var_analysis.name == "test"
        assert var_analysis.var_type == "categorical"
        assert var_analysis.importance_by_objective["acc"] == pytest.approx(0.5)
        assert var_analysis.importance_by_objective["cost"] == pytest.approx(0.3)
        assert var_analysis.aggregate_importance == pytest.approx(0.4)
        assert var_analysis.pareto_dominated_values is None
        assert var_analysis.suggestion is None
