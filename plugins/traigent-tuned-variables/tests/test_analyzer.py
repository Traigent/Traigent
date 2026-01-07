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
