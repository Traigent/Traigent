"""Unit tests for traigent.utils.importance.

Tests for parameter importance analysis including variance-based,
correlation-based, and permutation-based methods.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Performance
# CONC-Quality-Observability FUNC-ANALYTICS REQ-ANLY-011
# SYNC-Observability

from __future__ import annotations

import statistics
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from traigent.api.types import OptimizationResult, TrialResult, TrialStatus
from traigent.utils.importance import ImportanceResult, ParameterImportanceAnalyzer


class TestImportanceResult:
    """Tests for ImportanceResult dataclass."""

    @pytest.fixture
    def result(self) -> ImportanceResult:
        """Create test ImportanceResult instance."""
        return ImportanceResult(
            parameter_name="learning_rate",
            importance_score=0.75,
            confidence_interval=(0.65, 0.85),
            method="variance_based",
            sample_size=100,
        )

    def test_initialization(self, result: ImportanceResult) -> None:
        """Test ImportanceResult initializes with correct values."""
        assert result.parameter_name == "learning_rate"
        assert result.importance_score == 0.75
        assert result.confidence_interval == (0.65, 0.85)
        assert result.method == "variance_based"
        assert result.sample_size == 100

    def test_str_representation(self, result: ImportanceResult) -> None:
        """Test string representation of ImportanceResult."""
        string_repr = str(result)
        assert "learning_rate" in string_repr
        assert "0.750" in string_repr
        assert "variance_based" in string_repr
        assert "n=100" in string_repr

    def test_all_fields_accessible(self, result: ImportanceResult) -> None:
        """Test all fields are accessible."""
        assert hasattr(result, "parameter_name")
        assert hasattr(result, "importance_score")
        assert hasattr(result, "confidence_interval")
        assert hasattr(result, "method")
        assert hasattr(result, "sample_size")


class TestParameterImportanceAnalyzer:
    """Tests for ParameterImportanceAnalyzer class."""

    @pytest.fixture
    def analyzer(self) -> ParameterImportanceAnalyzer:
        """Create test analyzer instance."""
        return ParameterImportanceAnalyzer(objective="accuracy")

    @pytest.fixture
    def simple_trials(self) -> list[TrialResult]:
        """Create simple trial results for testing."""
        trials = []
        for i in range(20):
            # Vary learning_rate significantly
            lr = 0.001 if i < 10 else 0.01
            # temperature less important
            temp = 0.7 if i % 2 == 0 else 0.8
            # Accuracy correlates with learning_rate
            accuracy = 0.6 + (0.3 if lr == 0.01 else 0.0) + (i % 5) * 0.01

            trials.append(
                TrialResult(
                    trial_id=f"trial_{i}",
                    config={"learning_rate": lr, "temperature": temp},
                    metrics={"accuracy": accuracy},
                    status=TrialStatus.COMPLETED,
                    duration=1.0,
                    timestamp=datetime.now(),
                )
            )
        return trials

    @pytest.fixture
    def categorical_trials(self) -> list[TrialResult]:
        """Create trials with categorical parameters."""
        trials = []
        models = ["gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo", "gpt-4"] * 5
        for i, model in enumerate(models):
            accuracy = 0.7 if model == "gpt-4" else 0.6
            trials.append(
                TrialResult(
                    trial_id=f"trial_{i}",
                    config={"model": model, "temperature": 0.7},
                    metrics={"accuracy": accuracy},
                    status=TrialStatus.COMPLETED,
                    duration=1.0,
                    timestamp=datetime.now(),
                )
            )
        return trials

    @pytest.fixture
    def mixed_status_trials(self) -> list[TrialResult]:
        """Create trials with mixed statuses."""
        trials = []
        for i in range(15):
            status = TrialStatus.COMPLETED if i < 10 else TrialStatus.FAILED
            metrics = (
                {"accuracy": 0.7 + i * 0.01} if status == TrialStatus.COMPLETED else {}
            )
            trials.append(
                TrialResult(
                    trial_id=f"trial_{i}",
                    config={"param1": i * 0.1},
                    metrics=metrics,
                    status=status,
                    duration=1.0,
                    timestamp=datetime.now(),
                )
            )
        return trials

    def test_initialization(self, analyzer: ParameterImportanceAnalyzer) -> None:
        """Test analyzer initializes with correct objective."""
        assert analyzer.objective == "accuracy"

    def test_initialization_default_objective(self) -> None:
        """Test analyzer uses default objective."""
        analyzer = ParameterImportanceAnalyzer()
        assert analyzer.objective == "accuracy"

    def test_analyze_variance_based_empty_trials(
        self, analyzer: ParameterImportanceAnalyzer
    ) -> None:
        """Test variance-based analysis with empty trials list."""
        result = analyzer.analyze_variance_based([])
        assert result == {}

    def test_analyze_variance_based_insufficient_trials(
        self, analyzer: ParameterImportanceAnalyzer
    ) -> None:
        """Test variance-based analysis with insufficient trials."""
        trials = [
            TrialResult(
                trial_id="trial_1",
                config={"param1": 0.5},
                metrics={"accuracy": 0.7},
                status=TrialStatus.COMPLETED,
                duration=1.0,
                timestamp=datetime.now(),
            )
        ]
        result = analyzer.analyze_variance_based(trials)
        assert result == {}

    def test_analyze_variance_based_filters_failed_trials(
        self,
        analyzer: ParameterImportanceAnalyzer,
        mixed_status_trials: list[TrialResult],
    ) -> None:
        """Test variance-based analysis filters out failed trials."""
        result = analyzer.analyze_variance_based(mixed_status_trials)
        assert isinstance(result, dict)

    def test_analyze_variance_based_zero_variance(
        self, analyzer: ParameterImportanceAnalyzer
    ) -> None:
        """Test variance-based analysis with zero variance in objective."""
        trials = []
        for i in range(15):
            trials.append(
                TrialResult(
                    trial_id=f"trial_{i}",
                    config={"param1": i * 0.1},
                    metrics={"accuracy": 0.7},  # Same value
                    status=TrialStatus.COMPLETED,
                    duration=1.0,
                    timestamp=datetime.now(),
                )
            )
        result = analyzer.analyze_variance_based(trials)
        assert result == {}

    def test_analyze_variance_based_single_param_value(
        self, analyzer: ParameterImportanceAnalyzer
    ) -> None:
        """Test variance-based analysis with single parameter value."""
        trials = []
        for i in range(15):
            trials.append(
                TrialResult(
                    trial_id=f"trial_{i}",
                    config={"param1": 0.5},  # Same value
                    metrics={"accuracy": 0.7 + i * 0.01},
                    status=TrialStatus.COMPLETED,
                    duration=1.0,
                    timestamp=datetime.now(),
                )
            )
        result = analyzer.analyze_variance_based(trials)
        # Should return empty as we need at least 2 different values
        assert result == {}

    def test_analyze_variance_based_success(
        self, analyzer: ParameterImportanceAnalyzer, simple_trials: list[TrialResult]
    ) -> None:
        """Test variance-based analysis with valid trials."""
        result = analyzer.analyze_variance_based(simple_trials)

        assert isinstance(result, dict)
        assert "learning_rate" in result or "temperature" in result

        # Check structure of results
        for param_result in result.values():
            assert isinstance(param_result, ImportanceResult)
            assert param_result.method == "variance_based"
            assert 0.0 <= param_result.importance_score <= 1.0
            assert len(param_result.confidence_interval) == 2
            assert param_result.sample_size > 0

    def test_calculate_parameter_variance_importance_insufficient_groups(
        self, analyzer: ParameterImportanceAnalyzer
    ) -> None:
        """Test _calculate_parameter_variance_importance with insufficient groups."""
        trials = []
        for i in range(10):
            trials.append(
                TrialResult(
                    trial_id=f"trial_{i}",
                    config={"param1": 0.5},
                    metrics={"accuracy": 0.7 + i * 0.01},
                    status=TrialStatus.COMPLETED,
                    duration=1.0,
                    timestamp=datetime.now(),
                )
            )
        objective_values = [0.7 + i * 0.01 for i in range(10)]
        result = analyzer._calculate_parameter_variance_importance(
            trials, "param1", objective_values, 0.01
        )
        assert result is None

    def test_calculate_parameter_variance_importance_zero_total_ss(
        self, analyzer: ParameterImportanceAnalyzer
    ) -> None:
        """Test _calculate_parameter_variance_importance with zero total sum of squares."""
        trials = []
        # Create trials with two groups, each with same value
        for i in range(10):
            param_value = 0.1 if i < 5 else 0.2
            trials.append(
                TrialResult(
                    trial_id=f"trial_{i}",
                    config={"param1": param_value},
                    metrics={"accuracy": 0.7},  # Same accuracy
                    status=TrialStatus.COMPLETED,
                    duration=1.0,
                    timestamp=datetime.now(),
                )
            )
        objective_values = [0.7] * 10
        result = analyzer._calculate_parameter_variance_importance(
            trials, "param1", objective_values, 0.0
        )
        # Zero total_ss means no variance to explain; result should indicate no importance
        assert result is not None
        assert result.importance_score == 0.0

    def test_calculate_parameter_variance_importance_confidence_interval(
        self, analyzer: ParameterImportanceAnalyzer, simple_trials: list[TrialResult]
    ) -> None:
        """Test confidence interval calculation is within bounds."""
        objective_values = [t.metrics["accuracy"] for t in simple_trials]
        total_variance = statistics.variance(objective_values)

        result = analyzer._calculate_parameter_variance_importance(
            simple_trials, "learning_rate", objective_values, total_variance
        )

        if result is not None:
            ci_low, ci_high = result.confidence_interval
            assert ci_low >= 0.0
            assert ci_high <= 1.0
            assert ci_low <= result.importance_score <= ci_high

    def test_analyze_correlation_based_empty_trials(
        self, analyzer: ParameterImportanceAnalyzer
    ) -> None:
        """Test correlation-based analysis with empty trials."""
        result = analyzer.analyze_correlation_based([])
        assert result == {}

    def test_analyze_correlation_based_insufficient_trials(
        self, analyzer: ParameterImportanceAnalyzer
    ) -> None:
        """Test correlation-based analysis with insufficient trials."""
        trials = [
            TrialResult(
                trial_id="trial_1",
                config={"param1": 0.5},
                metrics={"accuracy": 0.7},
                status=TrialStatus.COMPLETED,
                duration=1.0,
                timestamp=datetime.now(),
            )
        ]
        result = analyzer.analyze_correlation_based(trials)
        assert result == {}

    def test_analyze_correlation_based_success(
        self, analyzer: ParameterImportanceAnalyzer, simple_trials: list[TrialResult]
    ) -> None:
        """Test correlation-based analysis with valid trials."""
        result = analyzer.analyze_correlation_based(simple_trials)

        assert isinstance(result, dict)
        # Check structure of results
        for param_result in result.values():
            assert isinstance(param_result, ImportanceResult)
            assert param_result.method == "correlation_based"
            assert 0.0 <= param_result.importance_score <= 1.0
            assert len(param_result.confidence_interval) == 2

    def test_analyze_correlation_based_categorical_params(
        self,
        analyzer: ParameterImportanceAnalyzer,
        categorical_trials: list[TrialResult],
    ) -> None:
        """Test correlation-based analysis with categorical parameters."""
        result = analyzer.analyze_correlation_based(categorical_trials)

        assert isinstance(result, dict)
        if "model" in result:
            assert isinstance(result["model"], ImportanceResult)
            assert result["model"].method == "correlation_based"

    def test_analyze_correlation_based_boolean_params(
        self, analyzer: ParameterImportanceAnalyzer
    ) -> None:
        """Test correlation-based analysis with boolean parameters."""
        trials = []
        for i in range(15):
            use_cache = i % 2 == 0
            accuracy = 0.8 if use_cache else 0.7
            trials.append(
                TrialResult(
                    trial_id=f"trial_{i}",
                    config={"use_cache": use_cache},
                    metrics={"accuracy": accuracy},
                    status=TrialStatus.COMPLETED,
                    duration=1.0,
                    timestamp=datetime.now(),
                )
            )
        result = analyzer.analyze_correlation_based(trials)
        if "use_cache" in result:
            assert result["use_cache"].importance_score >= 0.0

    def test_calculate_parameter_correlation_importance_insufficient_values(
        self, analyzer: ParameterImportanceAnalyzer
    ) -> None:
        """Test correlation importance with insufficient parameter values."""
        trials = []
        for i in range(5):
            trials.append(
                TrialResult(
                    trial_id=f"trial_{i}",
                    config={"param1": 0.5},
                    metrics={"accuracy": 0.7},
                    status=TrialStatus.COMPLETED,
                    duration=1.0,
                    timestamp=datetime.now(),
                )
            )
        objective_values = [0.7] * 5
        result = analyzer._calculate_parameter_correlation_importance(
            trials, "param1", objective_values
        )
        assert result is None

    def test_calculate_parameter_correlation_importance_single_unique_value(
        self, analyzer: ParameterImportanceAnalyzer
    ) -> None:
        """Test correlation importance with single unique parameter value."""
        trials = []
        for i in range(15):
            trials.append(
                TrialResult(
                    trial_id=f"trial_{i}",
                    config={"param1": 0.5},
                    metrics={"accuracy": 0.7 + i * 0.01},
                    status=TrialStatus.COMPLETED,
                    duration=1.0,
                    timestamp=datetime.now(),
                )
            )
        objective_values = [0.7 + i * 0.01 for i in range(15)]
        result = analyzer._calculate_parameter_correlation_importance(
            trials, "param1", objective_values
        )
        assert result is None

    def test_calculate_parameter_correlation_importance_unsupported_type(
        self, analyzer: ParameterImportanceAnalyzer
    ) -> None:
        """Test correlation importance with unsupported parameter type."""
        trials = []
        for i in range(15):
            trials.append(
                TrialResult(
                    trial_id=f"trial_{i}",
                    config={"param1": {"nested": "value"}},  # Dict not supported
                    metrics={"accuracy": 0.7 + i * 0.01},
                    status=TrialStatus.COMPLETED,
                    duration=1.0,
                    timestamp=datetime.now(),
                )
            )
        objective_values = [0.7 + i * 0.01 for i in range(15)]
        result = analyzer._calculate_parameter_correlation_importance(
            trials, "param1", objective_values
        )
        # Should return None for unsupported types
        assert result is None

    def test_calculate_correlation_empty_lists(
        self, analyzer: ParameterImportanceAnalyzer
    ) -> None:
        """Test correlation calculation with empty lists."""
        result = analyzer._calculate_correlation([], [])
        assert result == 0.0

    def test_calculate_correlation_single_value(
        self, analyzer: ParameterImportanceAnalyzer
    ) -> None:
        """Test correlation calculation with single value."""
        result = analyzer._calculate_correlation([1.0], [2.0])
        assert result == 0.0

    def test_calculate_correlation_mismatched_lengths(
        self, analyzer: ParameterImportanceAnalyzer
    ) -> None:
        """Test correlation calculation with mismatched lengths."""
        result = analyzer._calculate_correlation([1.0, 2.0], [3.0])
        assert result == 0.0

    def test_calculate_correlation_zero_denominator(
        self, analyzer: ParameterImportanceAnalyzer
    ) -> None:
        """Test correlation calculation with zero denominator."""
        # Same values in x will cause zero variance
        result = analyzer._calculate_correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])
        assert result == 0.0

    def test_calculate_correlation_perfect_positive(
        self, analyzer: ParameterImportanceAnalyzer
    ) -> None:
        """Test correlation calculation with perfect positive correlation."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        result = analyzer._calculate_correlation(x, y)
        assert abs(result - 1.0) < 0.001

    def test_calculate_correlation_perfect_negative(
        self, analyzer: ParameterImportanceAnalyzer
    ) -> None:
        """Test correlation calculation with perfect negative correlation."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 8.0, 6.0, 4.0, 2.0]
        result = analyzer._calculate_correlation(x, y)
        assert abs(result + 1.0) < 0.001

    def test_calculate_correlation_no_correlation(
        self, analyzer: ParameterImportanceAnalyzer
    ) -> None:
        """Test correlation calculation with no correlation."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [3.0, 3.0, 3.0, 3.0, 3.0]
        result = analyzer._calculate_correlation(x, y)
        assert result == 0.0

    def test_analyze_permutation_based_empty_trials(
        self, analyzer: ParameterImportanceAnalyzer
    ) -> None:
        """Test permutation-based analysis with empty trials."""
        result = analyzer.analyze_permutation_based([])
        assert result == {}

    def test_analyze_permutation_based_insufficient_trials(
        self, analyzer: ParameterImportanceAnalyzer
    ) -> None:
        """Test permutation-based analysis with insufficient trials."""
        trials = []
        for i in range(10):
            trials.append(
                TrialResult(
                    trial_id=f"trial_{i}",
                    config={"param1": i * 0.1},
                    metrics={"accuracy": 0.7},
                    status=TrialStatus.COMPLETED,
                    duration=1.0,
                    timestamp=datetime.now(),
                )
            )
        result = analyzer.analyze_permutation_based(trials)
        assert result == {}

    @patch("random.shuffle")
    def test_analyze_permutation_based_success(
        self,
        mock_shuffle: Mock,
        analyzer: ParameterImportanceAnalyzer,
        simple_trials: list[TrialResult],
    ) -> None:
        """Test permutation-based analysis with valid trials."""
        # Use fewer permutations for speed
        result = analyzer.analyze_permutation_based(simple_trials, n_permutations=5)

        assert isinstance(result, dict)
        # Check structure of results
        for param_result in result.values():
            assert isinstance(param_result, ImportanceResult)
            assert param_result.method == "permutation_based"
            assert param_result.importance_score >= 0.0
            assert len(param_result.confidence_interval) == 2

    def test_calculate_baseline_score_empty_trials(
        self, analyzer: ParameterImportanceAnalyzer
    ) -> None:
        """Test baseline score calculation with empty trials."""
        result = analyzer._calculate_baseline_score([])
        assert result == 0.0

    def test_calculate_baseline_score_valid_trials(
        self, analyzer: ParameterImportanceAnalyzer, simple_trials: list[TrialResult]
    ) -> None:
        """Test baseline score calculation with valid trials."""
        result = analyzer._calculate_baseline_score(simple_trials)
        assert result > 0.0
        # Should be the mean of accuracy values
        expected = statistics.mean([t.metrics["accuracy"] for t in simple_trials])
        assert abs(result - expected) < 0.001

    def test_permute_parameter_empty_trials(
        self, analyzer: ParameterImportanceAnalyzer
    ) -> None:
        """Test parameter permutation with empty trials."""
        result = analyzer._permute_parameter([], "param1")
        assert result == []

    def test_permute_parameter_missing_param(
        self, analyzer: ParameterImportanceAnalyzer, simple_trials: list[TrialResult]
    ) -> None:
        """Test parameter permutation with missing parameter."""
        result = analyzer._permute_parameter(simple_trials, "nonexistent_param")
        # Should return original trials if parameter not found
        assert len(result) == len(simple_trials)

    @patch("random.shuffle")
    def test_permute_parameter_shuffles_values(
        self, mock_shuffle: Mock, analyzer: ParameterImportanceAnalyzer
    ) -> None:
        """Test parameter permutation shuffles values correctly."""
        trials = []
        for i in range(5):
            trials.append(
                TrialResult(
                    trial_id=f"trial_{i}",
                    config={"param1": i},
                    metrics={"accuracy": 0.7},
                    status=TrialStatus.COMPLETED,
                    duration=1.0,
                    timestamp=datetime.now(),
                )
            )

        # Mock shuffle to reverse the list
        def reverse_shuffle(lst: list) -> None:
            lst.reverse()

        mock_shuffle.side_effect = reverse_shuffle

        result = analyzer._permute_parameter(trials, "param1")

        assert len(result) == len(trials)
        # Values should be shuffled
        mock_shuffle.assert_called_once()

    def test_get_top_parameters_empty_results(
        self, analyzer: ParameterImportanceAnalyzer
    ) -> None:
        """Test getting top parameters with empty results."""
        result = analyzer.get_top_parameters({}, top_k=5)
        assert result == []

    def test_get_top_parameters_less_than_k(
        self, analyzer: ParameterImportanceAnalyzer
    ) -> None:
        """Test getting top parameters with fewer than k results."""
        results = {
            "param1": ImportanceResult(
                parameter_name="param1",
                importance_score=0.8,
                confidence_interval=(0.7, 0.9),
                method="variance_based",
                sample_size=100,
            ),
            "param2": ImportanceResult(
                parameter_name="param2",
                importance_score=0.6,
                confidence_interval=(0.5, 0.7),
                method="variance_based",
                sample_size=100,
            ),
        }
        result = analyzer.get_top_parameters(results, top_k=5)
        assert len(result) == 2

    def test_get_top_parameters_sorted_by_importance(
        self, analyzer: ParameterImportanceAnalyzer
    ) -> None:
        """Test top parameters are sorted by importance score."""
        results = {
            "param1": ImportanceResult(
                parameter_name="param1",
                importance_score=0.5,
                confidence_interval=(0.4, 0.6),
                method="variance_based",
                sample_size=100,
            ),
            "param2": ImportanceResult(
                parameter_name="param2",
                importance_score=0.9,
                confidence_interval=(0.8, 1.0),
                method="variance_based",
                sample_size=100,
            ),
            "param3": ImportanceResult(
                parameter_name="param3",
                importance_score=0.7,
                confidence_interval=(0.6, 0.8),
                method="variance_based",
                sample_size=100,
            ),
        }
        result = analyzer.get_top_parameters(results, top_k=2)

        assert len(result) == 2
        assert result[0].parameter_name == "param2"
        assert result[1].parameter_name == "param3"
        # Should be sorted descending
        assert result[0].importance_score >= result[1].importance_score

    def test_generate_importance_report_basic_structure(
        self, analyzer: ParameterImportanceAnalyzer, simple_trials: list[TrialResult]
    ) -> None:
        """Test importance report generation has correct structure."""
        from traigent.api.types import OptimizationStatus

        opt_result = OptimizationResult(
            trials=simple_trials,
            best_config={"learning_rate": 0.01},
            best_score=0.9,
            optimization_id="opt_123",
            duration=10.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy"],
            algorithm="grid_search",
            timestamp=datetime.now(),
        )

        report = analyzer.generate_importance_report(opt_result)

        assert isinstance(report, str)
        assert "Parameter Importance Analysis Report" in report
        assert "opt_123" in report
        assert "accuracy" in report
        assert str(len(simple_trials)) in report

    def test_generate_importance_report_includes_methods(
        self, analyzer: ParameterImportanceAnalyzer, simple_trials: list[TrialResult]
    ) -> None:
        """Test importance report includes different analysis methods."""
        from traigent.api.types import OptimizationStatus

        opt_result = OptimizationResult(
            trials=simple_trials,
            best_config={"learning_rate": 0.01},
            best_score=0.9,
            optimization_id="opt_123",
            duration=10.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy"],
            algorithm="grid_search",
            timestamp=datetime.now(),
        )

        report = analyzer.generate_importance_report(opt_result)

        assert "Variance-based Analysis" in report
        assert "Correlation-based Analysis" in report

    def test_generate_importance_report_includes_permutation_when_sufficient_data(
        self, analyzer: ParameterImportanceAnalyzer
    ) -> None:
        """Test permutation analysis included when sufficient trials."""
        from traigent.api.types import OptimizationStatus

        # Create 25 trials (more than threshold of 20)
        trials = []
        for i in range(25):
            lr = 0.001 if i < 12 else 0.01
            trials.append(
                TrialResult(
                    trial_id=f"trial_{i}",
                    config={"learning_rate": lr},
                    metrics={"accuracy": 0.6 + (0.3 if lr == 0.01 else 0.0)},
                    status=TrialStatus.COMPLETED,
                    duration=1.0,
                    timestamp=datetime.now(),
                )
            )

        opt_result = OptimizationResult(
            trials=trials,
            best_config={"learning_rate": 0.01},
            best_score=0.9,
            optimization_id="opt_123",
            duration=10.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy"],
            algorithm="grid_search",
            timestamp=datetime.now(),
        )

        report = analyzer.generate_importance_report(opt_result)
        assert "Permutation-based Analysis" in report

    def test_generate_importance_report_handles_errors(
        self, analyzer: ParameterImportanceAnalyzer, simple_trials: list[TrialResult]
    ) -> None:
        """Test importance report handles analysis errors gracefully."""
        from traigent.api.types import OptimizationStatus

        opt_result = OptimizationResult(
            trials=simple_trials,
            best_config={"learning_rate": 0.01},
            best_score=0.9,
            optimization_id="opt_123",
            duration=10.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy"],
            algorithm="grid_search",
            timestamp=datetime.now(),
        )

        # Patch one of the analysis methods to raise an exception
        with patch.object(
            analyzer, "analyze_variance_based", side_effect=Exception("Test error")
        ):
            report = analyzer.generate_importance_report(opt_result)
            assert "Error in analysis" in report

    def test_generate_importance_report_insufficient_data(
        self, analyzer: ParameterImportanceAnalyzer
    ) -> None:
        """Test importance report with insufficient data."""
        from traigent.api.types import OptimizationStatus

        # Only 5 trials
        trials = []
        for i in range(5):
            trials.append(
                TrialResult(
                    trial_id=f"trial_{i}",
                    config={"param1": i * 0.1},
                    metrics={"accuracy": 0.7},
                    status=TrialStatus.COMPLETED,
                    duration=1.0,
                    timestamp=datetime.now(),
                )
            )

        opt_result = OptimizationResult(
            trials=trials,
            best_config={"param1": 0.1},
            best_score=0.7,
            optimization_id="opt_123",
            duration=10.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy"],
            algorithm="grid_search",
            timestamp=datetime.now(),
        )

        report = analyzer.generate_importance_report(opt_result)
        assert "Insufficient data" in report or "No significant importance" in report

    def test_generate_importance_report_includes_recommendations(
        self, analyzer: ParameterImportanceAnalyzer, simple_trials: list[TrialResult]
    ) -> None:
        """Test importance report includes recommendations."""
        from traigent.api.types import OptimizationStatus

        opt_result = OptimizationResult(
            trials=simple_trials,
            best_config={"learning_rate": 0.01},
            best_score=0.9,
            optimization_id="opt_123",
            duration=10.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy"],
            algorithm="grid_search",
            timestamp=datetime.now(),
        )

        report = analyzer.generate_importance_report(opt_result)
        assert "Summary & Recommendations" in report

    def test_multiple_objectives_handled(
        self, analyzer: ParameterImportanceAnalyzer
    ) -> None:
        """Test analyzer handles trials with multiple objectives."""
        trials = []
        for i in range(15):
            lr = 0.001 if i < 7 else 0.01
            trials.append(
                TrialResult(
                    trial_id=f"trial_{i}",
                    config={"learning_rate": lr},
                    metrics={"accuracy": 0.7 + i * 0.01, "cost": 0.5 - i * 0.01},
                    status=TrialStatus.COMPLETED,
                    duration=1.0,
                    timestamp=datetime.now(),
                )
            )

        result = analyzer.analyze_variance_based(trials)
        assert isinstance(result, dict)

    def test_missing_objective_in_trials(self) -> None:
        """Test analyzer handles trials missing the target objective."""
        analyzer = ParameterImportanceAnalyzer(objective="loss")
        trials = []
        for i in range(15):
            trials.append(
                TrialResult(
                    trial_id=f"trial_{i}",
                    config={"param1": i * 0.1},
                    metrics={"accuracy": 0.7},  # Missing 'loss'
                    status=TrialStatus.COMPLETED,
                    duration=1.0,
                    timestamp=datetime.now(),
                )
            )

        result = analyzer.analyze_variance_based(trials)
        assert result == {}

    def test_confidence_interval_bounds(
        self, analyzer: ParameterImportanceAnalyzer, simple_trials: list[TrialResult]
    ) -> None:
        """Test all confidence intervals are within valid bounds."""
        result = analyzer.analyze_variance_based(simple_trials)

        for param_result in result.values():
            ci_low, ci_high = param_result.confidence_interval
            assert ci_low >= 0.0
            assert ci_high <= 1.0
            assert ci_low <= ci_high

    def test_edge_case_single_group_within_group_variance(
        self, analyzer: ParameterImportanceAnalyzer
    ) -> None:
        """Test handling of groups with single value for within-group variance."""
        trials = []
        # Create groups with single values
        for i in range(10):
            param_val = 0.1 if i < 5 else 0.2
            trials.append(
                TrialResult(
                    trial_id=f"trial_{i}",
                    config={
                        "param1": param_val,
                        "unique_param": i,
                    },  # Each has unique value
                    metrics={"accuracy": 0.7 + i * 0.02},
                    status=TrialStatus.COMPLETED,
                    duration=1.0,
                    timestamp=datetime.now(),
                )
            )

        result = analyzer.analyze_variance_based(trials)
        # Should handle groups with single values
        if "unique_param" in result:
            assert result["unique_param"].importance_score >= 0.0
