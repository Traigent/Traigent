"""Advanced tests for traigent.api.types module - targeting uncovered methods."""

from datetime import datetime

import pytest

from traigent.api.types import (
    ExperimentStats,
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)


class TestTrialStatusPruned:
    """Test PRUNED status enum value."""

    def test_pruned_status_value(self):
        """Test PRUNED status enum value."""
        assert TrialStatus.PRUNED == "pruned"

    def test_pruned_status_in_comparisons(self):
        """Test PRUNED status in comparisons."""
        assert TrialStatus.PRUNED != TrialStatus.CANCELLED
        assert TrialStatus.PRUNED != TrialStatus.COMPLETED
        assert TrialStatus.PRUNED == TrialStatus.PRUNED


class TestExperimentStatsToDict:
    """Test ExperimentStats.to_dict() method."""

    def test_to_dict_complete(self):
        """Test to_dict with all fields populated."""
        stats = ExperimentStats(
            total_duration=10.5,
            total_cost=2.5,
            unique_configurations=5,
            trial_counts={"completed": 10, "failed": 2},
            average_trial_duration=1.05,
            cost_per_configuration=0.5,
            success_rate=0.833,
            error_message=None,
        )

        result = stats.to_dict()

        assert isinstance(result, dict)
        assert result["total_duration"] == 10.5
        assert result["total_cost"] == 2.5
        assert result["unique_configurations"] == 5
        assert result["trial_counts"] == {"completed": 10, "failed": 2}
        assert result["average_trial_duration"] == 1.05
        assert result["cost_per_configuration"] == 0.5
        assert result["success_rate"] == 0.833
        assert result["error_message"] is None

    def test_to_dict_with_error(self):
        """Test to_dict with error message."""
        stats = ExperimentStats(
            total_duration=0.0,
            total_cost=0.0,
            unique_configurations=0,
            trial_counts={},
            error_message="Test error",
        )

        result = stats.to_dict()
        assert result["error_message"] == "Test error"

    def test_to_dict_none_values(self):
        """Test to_dict with None optional values."""
        stats = ExperimentStats(
            total_duration=5.0,
            total_cost=1.0,
            unique_configurations=3,
            trial_counts={"total": 3},
            average_trial_duration=None,
            cost_per_configuration=None,
            success_rate=None,
        )

        result = stats.to_dict()
        assert result["average_trial_duration"] is None
        assert result["cost_per_configuration"] is None
        assert result["success_rate"] is None


class TestOptimizationResultObjectiveRanges:
    """Test _calculate_objective_ranges method."""

    def setup_method(self):
        """Set up test data."""
        self.trials = [
            TrialResult(
                trial_id="t1",
                config={"model": "a"},
                metrics={"accuracy": 0.8, "cost": 0.1},
                status=TrialStatus.COMPLETED,
                duration=1.0,
                timestamp=datetime.now(),
            ),
            TrialResult(
                trial_id="t2",
                config={"model": "b"},
                metrics={"accuracy": 0.9, "cost": 0.05},
                status=TrialStatus.COMPLETED,
                duration=1.5,
                timestamp=datetime.now(),
            ),
            TrialResult(
                trial_id="t3",
                config={"model": "c"},
                metrics={"accuracy": 0.85, "cost": 0.08},
                status=TrialStatus.COMPLETED,
                duration=1.2,
                timestamp=datetime.now(),
            ),
        ]

    def test_calculate_objective_ranges_normal(self):
        """Test _calculate_objective_ranges with normal data."""
        result = OptimizationResult(
            trials=self.trials,
            best_config={"model": "b"},
            best_score=0.9,
            optimization_id="opt_001",
            duration=10.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy", "cost"],
            algorithm="random",
            timestamp=datetime.now(),
        )

        ranges = result._calculate_objective_ranges()

        assert "accuracy" in ranges
        assert "cost" in ranges
        assert ranges["accuracy"] == (0.8, 0.9)
        assert ranges["cost"] == (0.05, 0.1)

    def test_calculate_objective_ranges_with_none_values(self):
        """Test _calculate_objective_ranges with None metric values."""
        trials_with_none = self.trials + [
            TrialResult(
                trial_id="t4",
                config={"model": "d"},
                metrics={"accuracy": None, "cost": 0.06},
                status=TrialStatus.COMPLETED,
                duration=1.0,
                timestamp=datetime.now(),
            )
        ]

        result = OptimizationResult(
            trials=trials_with_none,
            best_config={"model": "b"},
            best_score=0.9,
            optimization_id="opt_001",
            duration=10.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy", "cost"],
            algorithm="random",
            timestamp=datetime.now(),
        )

        ranges = result._calculate_objective_ranges()

        # None values should be ignored
        assert ranges["accuracy"] == (0.8, 0.9)
        assert ranges["cost"] == (0.05, 0.1)

    def test_calculate_objective_ranges_no_values(self):
        """Test _calculate_objective_ranges with no valid values."""
        result = OptimizationResult(
            trials=[],
            best_config={},
            best_score=0.0,
            optimization_id="opt_001",
            duration=0.0,
            convergence_info={},
            status=OptimizationStatus.FAILED,
            objectives=["accuracy", "cost"],
            algorithm="random",
            timestamp=datetime.now(),
        )

        ranges = result._calculate_objective_ranges()

        # Should return default ranges
        assert ranges["accuracy"] == (0.0, 1.0)
        assert ranges["cost"] == (0.0, 1.0)

    def test_calculate_objective_ranges_single_value(self):
        """Test _calculate_objective_ranges with single value per objective."""
        single_trial = [self.trials[0]]

        result = OptimizationResult(
            trials=single_trial,
            best_config={"model": "a"},
            best_score=0.8,
            optimization_id="opt_001",
            duration=1.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy", "cost"],
            algorithm="random",
            timestamp=datetime.now(),
        )

        ranges = result._calculate_objective_ranges()

        # Min and max should be the same value
        assert ranges["accuracy"] == (0.8, 0.8)
        assert ranges["cost"] == (0.1, 0.1)


class TestOptimizationResultConfigHash:
    """Test _generate_config_hash method."""

    def test_generate_config_hash_simple(self):
        """Test _generate_config_hash with simple config."""
        config = {"model": "gpt-3.5", "temperature": 0.7}
        hash1 = OptimizationResult._generate_config_hash(config)

        assert isinstance(hash1, str)
        assert len(hash1) == 16  # SHA256 truncated to 16 chars

    def test_generate_config_hash_deterministic(self):
        """Test _generate_config_hash is deterministic."""
        config = {"model": "gpt-4", "temperature": 0.5, "max_tokens": 100}

        hash1 = OptimizationResult._generate_config_hash(config)
        hash2 = OptimizationResult._generate_config_hash(config)

        assert hash1 == hash2

    def test_generate_config_hash_order_independent(self):
        """Test _generate_config_hash is order-independent."""
        config1 = {"model": "gpt-4", "temperature": 0.7}
        config2 = {"temperature": 0.7, "model": "gpt-4"}

        hash1 = OptimizationResult._generate_config_hash(config1)
        hash2 = OptimizationResult._generate_config_hash(config2)

        assert hash1 == hash2

    def test_generate_config_hash_different_configs(self):
        """Test _generate_config_hash produces different hashes for different configs."""
        config1 = {"model": "gpt-3.5", "temperature": 0.7}
        config2 = {"model": "gpt-4", "temperature": 0.7}

        hash1 = OptimizationResult._generate_config_hash(config1)
        hash2 = OptimizationResult._generate_config_hash(config2)

        assert hash1 != hash2

    def test_generate_config_hash_empty_config(self):
        """Test _generate_config_hash with empty config."""
        hash1 = OptimizationResult._generate_config_hash({})
        hash2 = OptimizationResult._generate_config_hash(None)

        assert isinstance(hash1, str)
        assert len(hash1) == 16
        assert isinstance(hash2, str)
        assert len(hash2) == 16
        # Empty dict and None produce different hashes (implementation detail)


class TestOptimizationResultResponseTime:
    """Test _compute_average_response_time and _extract_response_time methods."""

    def test_compute_average_response_time_from_example_results(self):
        """Test _compute_average_response_time with example_results."""
        metadata = {
            "example_results": [
                {"response_time": 1.5},
                {"response_time": 2.0},
                {"response_time": 1.8},
            ]
        }

        avg = OptimizationResult._compute_average_response_time(metadata)
        assert avg == pytest.approx((1.5 + 2.0 + 1.8) / 3)

    def test_compute_average_response_time_from_measures(self):
        """Test _compute_average_response_time with measures."""
        metadata = {
            "measures": [
                {"response_time": 0.5},
                {"response_time": 0.8},
            ]
        }

        avg = OptimizationResult._compute_average_response_time(metadata)
        assert avg == pytest.approx((0.5 + 0.8) / 2)

    def test_compute_average_response_time_no_metadata(self):
        """Test _compute_average_response_time with no metadata."""
        assert OptimizationResult._compute_average_response_time(None) is None
        assert OptimizationResult._compute_average_response_time({}) is None

    def test_compute_average_response_time_no_times(self):
        """Test _compute_average_response_time with no response times."""
        metadata = {"example_results": [{"other_field": 1}, {"other_field": 2}]}

        assert OptimizationResult._compute_average_response_time(metadata) is None

    def test_extract_response_time_nested_metrics(self):
        """Test _extract_response_time with nested metrics."""
        entry = {"metrics": {"response_time": 1.5}}
        assert OptimizationResult._extract_response_time(entry) == pytest.approx(1.5)

    def test_extract_response_time_from_object_attribute(self):
        """Test _extract_response_time from object with execution_time attribute."""

        class DummyResult:
            def __init__(self):
                self.execution_time = 2.5
                self.metrics = {}

        result = DummyResult()
        assert OptimizationResult._extract_response_time(result) == pytest.approx(2.5)

    def test_extract_response_time_none_entry(self):
        """Test _extract_response_time with None entry."""
        assert OptimizationResult._extract_response_time(None) is None

    def test_extract_response_time_invalid_value(self):
        """Test _extract_response_time with invalid value."""
        entry = {"response_time": "invalid"}
        assert OptimizationResult._extract_response_time(entry) is None


class TestOptimizationResultAggregatedDataframe:
    """Test to_aggregated_dataframe method."""

    def setup_method(self):
        """Set up test data with repeated configs."""
        self.trials = [
            # Config A - 3 samples
            TrialResult(
                trial_id="t1",
                config={"model": "a", "temp": 0.5},
                metrics={"accuracy": 0.8, "cost": 0.1},
                status=TrialStatus.COMPLETED,
                duration=1.0,
                timestamp=datetime.now(),
                metadata={"example_results": [{"response_time": 0.5}]},
            ),
            TrialResult(
                trial_id="t2",
                config={"model": "a", "temp": 0.5},
                metrics={"accuracy": 0.85, "cost": 0.12},
                status=TrialStatus.COMPLETED,
                duration=1.2,
                timestamp=datetime.now(),
                metadata={"example_results": [{"response_time": 0.6}]},
            ),
            TrialResult(
                trial_id="t3",
                config={"model": "a", "temp": 0.5},
                metrics={"accuracy": 0.82, "cost": 0.11},
                status=TrialStatus.COMPLETED,
                duration=1.1,
                timestamp=datetime.now(),
                metadata={"example_results": [{"response_time": 0.55}]},
            ),
            # Config B - 2 samples
            TrialResult(
                trial_id="t4",
                config={"model": "b", "temp": 0.7},
                metrics={"accuracy": 0.9, "cost": 0.2},
                status=TrialStatus.COMPLETED,
                duration=1.5,
                timestamp=datetime.now(),
            ),
            TrialResult(
                trial_id="t5",
                config={"model": "b", "temp": 0.7},
                metrics={"accuracy": 0.92, "cost": 0.22},
                status=TrialStatus.COMPLETED,
                duration=1.6,
                timestamp=datetime.now(),
            ),
        ]

    def test_to_aggregated_dataframe_basic(self):
        """Test to_aggregated_dataframe groups and averages correctly."""
        result = OptimizationResult(
            trials=self.trials,
            best_config={"model": "b", "temp": 0.7},
            best_score=0.92,
            optimization_id="opt_001",
            duration=10.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy"],
            algorithm="random",
            timestamp=datetime.now(),
        )

        df = result.to_aggregated_dataframe()

        assert len(df) == 2  # 2 unique configs
        assert "samples_count" in df.columns
        assert "config_hash" in df.columns
        assert "accuracy" in df.columns
        assert "cost" in df.columns
        assert "duration" in df.columns

        # Check config A aggregation (3 samples)
        config_a_rows = df[df["model"] == "a"]
        assert len(config_a_rows) == 1
        assert config_a_rows.iloc[0]["samples_count"] == 3
        assert config_a_rows.iloc[0]["accuracy"] == pytest.approx(
            (0.8 + 0.85 + 0.82) / 3
        )
        assert config_a_rows.iloc[0]["cost"] == pytest.approx((0.1 + 0.12 + 0.11) / 3)
        assert config_a_rows.iloc[0]["duration"] == pytest.approx((1.0 + 1.2 + 1.1) / 3)

        # Check config B aggregation (2 samples)
        config_b_rows = df[df["model"] == "b"]
        assert len(config_b_rows) == 1
        assert config_b_rows.iloc[0]["samples_count"] == 2
        assert config_b_rows.iloc[0]["accuracy"] == pytest.approx((0.9 + 0.92) / 2)

    def test_to_aggregated_dataframe_with_primary_objective(self):
        """Test to_aggregated_dataframe with primary_objective sorting."""
        result = OptimizationResult(
            trials=self.trials,
            best_config={"model": "b", "temp": 0.7},
            best_score=0.92,
            optimization_id="opt_001",
            duration=10.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy"],
            algorithm="random",
            timestamp=datetime.now(),
        )

        df = result.to_aggregated_dataframe(primary_objective="accuracy")

        # Should be sorted by accuracy (descending for maximize)
        assert df.iloc[0]["model"] == "b"  # Higher accuracy
        assert df.iloc[1]["model"] == "a"  # Lower accuracy

    def test_to_aggregated_dataframe_empty_trials(self):
        """Test to_aggregated_dataframe with no trials."""
        result = OptimizationResult(
            trials=[],
            best_config={},
            best_score=0.0,
            optimization_id="opt_001",
            duration=0.0,
            convergence_info={},
            status=OptimizationStatus.FAILED,
            objectives=[],
            algorithm="random",
            timestamp=datetime.now(),
        )

        df = result.to_aggregated_dataframe()
        assert len(df) == 0

    def test_to_aggregated_dataframe_with_response_time(self):
        """Test to_aggregated_dataframe includes avg_response_time."""
        result = OptimizationResult(
            trials=self.trials[:3],  # Only config A with response times
            best_config={"model": "a", "temp": 0.5},
            best_score=0.85,
            optimization_id="opt_001",
            duration=5.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy"],
            algorithm="random",
            timestamp=datetime.now(),
        )

        df = result.to_aggregated_dataframe()

        assert "avg_response_time" in df.columns
        assert df.iloc[0]["avg_response_time"] == pytest.approx((0.5 + 0.6 + 0.55) / 3)


class TestOptimizationResultGetSummary:
    """Test get_summary method."""

    def test_get_summary_basic(self):
        """Test get_summary with basic trial data."""
        trials = [
            TrialResult(
                trial_id="t1",
                config={"model": "a"},
                metrics={"accuracy": 0.8, "total_cost": 0.5},
                status=TrialStatus.COMPLETED,
                duration=1.0,
                timestamp=datetime.now(),
                metadata={"examples_attempted": 10},
            ),
            TrialResult(
                trial_id="t2",
                config={"model": "b"},
                metrics={"accuracy": 0.9, "total_cost": 1.0},
                status=TrialStatus.COMPLETED,
                duration=1.5,
                timestamp=datetime.now(),
                metadata={"examples_attempted": 10},
            ),
            TrialResult(
                trial_id="t3",
                config={"model": "c"},
                metrics={},
                status=TrialStatus.FAILED,
                duration=0.5,
                timestamp=datetime.now(),
                error_message="Error",
            ),
        ]

        result = OptimizationResult(
            trials=trials,
            best_config={"model": "b"},
            best_score=0.9,
            optimization_id="opt_001",
            duration=10.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy"],
            algorithm="random",
            timestamp=datetime.now(),
        )

        summary = result.get_summary()

        assert summary["total_trials"] == 3
        assert summary["completed_trials"] == 2
        assert summary["failed_trials"] == 1
        assert summary["cancelled_trials"] == 0
        assert summary["pruned_trials"] == 0
        assert summary["total_duration"] == pytest.approx(3.0)
        assert summary["total_cost"] == pytest.approx(1.5)
        assert summary["total_examples_attempted"] == 20
        assert summary["non_failed_trials"] == 2
        assert summary["best_config"] == {"model": "b"}
        assert summary["best_score"] == 0.9

    def test_get_summary_trials_per_model(self):
        """Test get_summary tracks trials per model."""
        trials = [
            TrialResult(
                trial_id="t1",
                config={"model": "gpt-3.5"},
                metrics={},
                status=TrialStatus.COMPLETED,
                duration=1.0,
                timestamp=datetime.now(),
            ),
            TrialResult(
                trial_id="t2",
                config={"model": "gpt-3.5"},
                metrics={},
                status=TrialStatus.COMPLETED,
                duration=1.0,
                timestamp=datetime.now(),
            ),
            TrialResult(
                trial_id="t3",
                config={"model": "gpt-4"},
                metrics={},
                status=TrialStatus.COMPLETED,
                duration=1.0,
                timestamp=datetime.now(),
            ),
        ]

        result = OptimizationResult(
            trials=trials,
            best_config={"model": "gpt-4"},
            best_score=0.95,
            optimization_id="opt_001",
            duration=5.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy"],
            algorithm="random",
            timestamp=datetime.now(),
        )

        summary = result.get_summary()

        assert summary["trials_per_model"] == {"gpt-3.5": 2, "gpt-4": 1}

    def test_get_summary_cost_from_metadata(self):
        """Test get_summary extracts cost from metadata."""
        trials = [
            TrialResult(
                trial_id="t1",
                config={"model": "a"},
                metrics={},
                status=TrialStatus.COMPLETED,
                duration=1.0,
                timestamp=datetime.now(),
                metadata={"total_example_cost": 1.5},
            ),
            TrialResult(
                trial_id="t2",
                config={"model": "b"},
                metrics={},
                status=TrialStatus.COMPLETED,
                duration=1.0,
                timestamp=datetime.now(),
                metadata={"total_example_cost": 2.0},
            ),
        ]

        result = OptimizationResult(
            trials=trials,
            best_config={"model": "b"},
            best_score=0.9,
            optimization_id="opt_001",
            duration=5.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy"],
            algorithm="random",
            timestamp=datetime.now(),
        )

        summary = result.get_summary()
        assert summary["total_cost"] == pytest.approx(3.5)

    def test_get_summary_invalid_values(self):
        """Test get_summary handles invalid values gracefully."""
        trials = [
            TrialResult(
                trial_id="t1",
                config={"model": "a"},
                metrics={"total_cost": "invalid"},
                status=TrialStatus.COMPLETED,
                duration=1.0,
                timestamp=datetime.now(),
                metadata={"examples_attempted": "invalid"},
            ),
        ]

        result = OptimizationResult(
            trials=trials,
            best_config={"model": "a"},
            best_score=0.8,
            optimization_id="opt_001",
            duration=5.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy"],
            algorithm="random",
            timestamp=datetime.now(),
        )

        summary = result.get_summary()

        # Should not raise, invalid values should be skipped
        assert summary["total_cost"] == 0.0
        assert summary["total_examples_attempted"] == 0


class TestOptimizationResultNormalizationMethods:
    """Test normalization and weighted scoring helper methods."""

    def setup_method(self):
        """Set up test data."""
        self.trials = [
            TrialResult(
                trial_id="t1",
                config={"model": "a"},
                metrics={"accuracy": 0.8, "cost": 0.1},
                status=TrialStatus.COMPLETED,
                duration=1.0,
                timestamp=datetime.now(),
            ),
            TrialResult(
                trial_id="t2",
                config={"model": "b"},
                metrics={"accuracy": 0.9, "cost": 0.2},
                status=TrialStatus.COMPLETED,
                duration=1.5,
                timestamp=datetime.now(),
            ),
        ]

    def test_auto_detect_minimize_objectives(self):
        """Test _auto_detect_minimize_objectives detects common patterns."""
        result = OptimizationResult(
            trials=self.trials,
            best_config={},
            best_score=0.9,
            optimization_id="opt_001",
            duration=5.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy", "cost", "latency", "error_rate", "processing_time"],
            algorithm="random",
            timestamp=datetime.now(),
        )

        minimize = result._auto_detect_minimize_objectives()

        assert "cost" in minimize
        assert "latency" in minimize
        assert "error_rate" in minimize
        assert "processing_time" in minimize
        assert "accuracy" not in minimize

    def test_normalize_weight_map(self):
        """Test _normalize_weight_map normalizes weights to sum to 1."""
        weights = {"accuracy": 3.0, "cost": 1.0}

        normalized = OptimizationResult._normalize_weight_map(weights)

        assert normalized["accuracy"] == pytest.approx(0.75)
        assert normalized["cost"] == pytest.approx(0.25)
        assert sum(normalized.values()) == pytest.approx(1.0)

    def test_normalize_weight_map_zero_total(self):
        """Test _normalize_weight_map with zero total weight."""
        weights = {"accuracy": 0.0, "cost": 0.0}

        normalized = OptimizationResult._normalize_weight_map(weights)

        # Should return unchanged when total is 0
        assert normalized == weights

    def test_legacy_normalize_value_maximize(self):
        """Test _legacy_normalize_value for maximization."""
        # Value in middle of range
        normalized = OptimizationResult._legacy_normalize_value(
            value=0.5, min_val=0.0, max_val=1.0, minimize=False
        )
        assert normalized == pytest.approx(0.5)

        # Value at max
        normalized = OptimizationResult._legacy_normalize_value(
            value=1.0, min_val=0.0, max_val=1.0, minimize=False
        )
        assert normalized == pytest.approx(1.0)

        # Value at min
        normalized = OptimizationResult._legacy_normalize_value(
            value=0.0, min_val=0.0, max_val=1.0, minimize=False
        )
        assert normalized == pytest.approx(0.0)

    def test_legacy_normalize_value_minimize(self):
        """Test _legacy_normalize_value for minimization."""
        # Lower value should normalize higher for minimize
        normalized = OptimizationResult._legacy_normalize_value(
            value=0.2, min_val=0.0, max_val=1.0, minimize=True
        )
        assert normalized == pytest.approx(0.8)

        # Higher value should normalize lower for minimize
        normalized = OptimizationResult._legacy_normalize_value(
            value=0.8, min_val=0.0, max_val=1.0, minimize=True
        )
        assert normalized == pytest.approx(0.2)

    def test_legacy_normalize_value_same_min_max(self):
        """Test _legacy_normalize_value when min equals max."""
        normalized = OptimizationResult._legacy_normalize_value(
            value=0.5, min_val=0.5, max_val=0.5, minimize=False
        )
        assert normalized == pytest.approx(0.5)

    def test_legacy_normalize_value_preserves_out_of_range_values(self):
        """Test _legacy_normalize_value keeps out-of-range signal (no clipping)."""
        # Value below min
        normalized = OptimizationResult._legacy_normalize_value(
            value=-0.5, min_val=0.0, max_val=1.0, minimize=False
        )
        assert normalized == pytest.approx(-0.5)

        # Value above max
        normalized = OptimizationResult._legacy_normalize_value(
            value=1.5, min_val=0.0, max_val=1.0, minimize=False
        )
        assert normalized == pytest.approx(1.5)

    def test_legacy_normalize_value_invalid(self):
        """Test _legacy_normalize_value with invalid input."""
        assert OptimizationResult._legacy_normalize_value(None, 0.0, 1.0, False) is None
        assert (
            OptimizationResult._legacy_normalize_value("invalid", 0.0, 1.0, False)
            is None
        )

    def test_score_trial(self):
        """Test _score_trial calculates weighted score."""
        result = OptimizationResult(
            trials=self.trials,
            best_config={},
            best_score=0.9,
            optimization_id="opt_001",
            duration=5.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy", "cost"],
            algorithm="random",
            timestamp=datetime.now(),
        )

        normalized_metrics = {"accuracy": 0.8, "cost": 0.6}
        objective_weights = {"accuracy": 0.7, "cost": 0.3}

        score = result._score_trial(normalized_metrics, objective_weights)

        # 0.8 * 0.7 + 0.6 * 0.3 = 0.56 + 0.18 = 0.74
        assert score == pytest.approx(0.74)


class TestOptimizationResultHelperMethods:
    """Test various helper methods in OptimizationResult."""

    def test_coerce_float_valid(self):
        """Test _coerce_float with valid inputs."""
        assert OptimizationResult._coerce_float(1.5) == pytest.approx(1.5)
        assert OptimizationResult._coerce_float(10) == pytest.approx(10.0)
        assert OptimizationResult._coerce_float("3.14") == pytest.approx(3.14)

    def test_coerce_float_invalid(self):
        """Test _coerce_float with invalid inputs."""
        assert OptimizationResult._coerce_float(None) is None
        assert OptimizationResult._coerce_float("invalid") is None
        assert OptimizationResult._coerce_float([1, 2, 3]) is None

    def test_resolve_initial_duration(self):
        """Test _resolve_initial_duration method."""
        result = OptimizationResult(
            trials=[],
            best_config={},
            best_score=0.0,
            optimization_id="opt_001",
            duration=10.5,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=[],
            algorithm="random",
            timestamp=datetime.now(),
        )

        assert result._resolve_initial_duration() == pytest.approx(10.5)

    def test_resolve_initial_duration_none(self):
        """Test _resolve_initial_duration with None duration."""
        result = OptimizationResult(
            trials=[],
            best_config={},
            best_score=0.0,
            optimization_id="opt_001",
            duration=None,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=[],
            algorithm="random",
            timestamp=datetime.now(),
        )

        assert result._resolve_initial_duration() == pytest.approx(0.0)

    def test_resolve_initial_duration_invalid(self):
        """Test _resolve_initial_duration with invalid duration."""
        result = OptimizationResult(
            trials=[],
            best_config={},
            best_score=0.0,
            optimization_id="opt_001",
            duration="invalid",
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=[],
            algorithm="random",
            timestamp=datetime.now(),
        )

        assert result._resolve_initial_duration() == pytest.approx(0.0)

    def test_increment_status_count(self):
        """Test _increment_status_count increments correctly."""
        result = OptimizationResult(
            trials=[],
            best_config={},
            best_score=0.0,
            optimization_id="opt_001",
            duration=0.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=[],
            algorithm="random",
            timestamp=datetime.now(),
        )

        counts = {
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
            "running": 0,
            "pending": 0,
            "not_started": 0,
        }

        trial_completed = TrialResult(
            trial_id="t1",
            config={},
            metrics={},
            status=TrialStatus.COMPLETED,
            duration=1.0,
            timestamp=datetime.now(),
        )

        result._increment_status_count(trial_completed, counts)
        assert counts["completed"] == 1

        trial_failed = TrialResult(
            trial_id="t2",
            config={},
            metrics={},
            status=TrialStatus.FAILED,
            duration=1.0,
            timestamp=datetime.now(),
        )

        result._increment_status_count(trial_failed, counts)
        assert counts["failed"] == 1

    def test_extract_trial_cost_from_metrics(self):
        """Test _extract_trial_cost from metrics."""
        result = OptimizationResult(
            trials=[],
            best_config={},
            best_score=0.0,
            optimization_id="opt_001",
            duration=0.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=[],
            algorithm="random",
            timestamp=datetime.now(),
        )

        trial = TrialResult(
            trial_id="t1",
            config={},
            metrics={"total_cost": 1.5},
            status=TrialStatus.COMPLETED,
            duration=1.0,
            timestamp=datetime.now(),
        )

        cost = result._extract_trial_cost(trial)
        assert cost == pytest.approx(1.5)

    def test_extract_trial_cost_from_metadata_dict(self):
        """Test _extract_trial_cost from metadata cost dict."""
        result = OptimizationResult(
            trials=[],
            best_config={},
            best_score=0.0,
            optimization_id="opt_001",
            duration=0.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=[],
            algorithm="random",
            timestamp=datetime.now(),
        )

        trial = TrialResult(
            trial_id="t1",
            config={},
            metrics={},
            status=TrialStatus.COMPLETED,
            duration=1.0,
            timestamp=datetime.now(),
            metadata={"cost": {"total_cost": 2.0}},
        )

        cost = result._extract_trial_cost(trial)
        assert cost == pytest.approx(2.0)

    def test_extract_trial_cost_from_metadata_scalar(self):
        """Test _extract_trial_cost from metadata cost scalar."""
        result = OptimizationResult(
            trials=[],
            best_config={},
            best_score=0.0,
            optimization_id="opt_001",
            duration=0.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=[],
            algorithm="random",
            timestamp=datetime.now(),
        )

        trial = TrialResult(
            trial_id="t1",
            config={},
            metrics={},
            status=TrialStatus.COMPLETED,
            duration=1.0,
            timestamp=datetime.now(),
            metadata={"cost": 3.5},
        )

        cost = result._extract_trial_cost(trial)
        assert cost == pytest.approx(3.5)

    def test_extract_trial_cost_none(self):
        """Test _extract_trial_cost returns None when no cost."""
        result = OptimizationResult(
            trials=[],
            best_config={},
            best_score=0.0,
            optimization_id="opt_001",
            duration=0.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=[],
            algorithm="random",
            timestamp=datetime.now(),
        )

        trial = TrialResult(
            trial_id="t1",
            config={},
            metrics={},
            status=TrialStatus.COMPLETED,
            duration=1.0,
            timestamp=datetime.now(),
        )

        assert result._extract_trial_cost(trial) is None

    def test_resolve_total_cost_from_attribute(self):
        """Test _resolve_total_cost uses total_cost attribute."""
        result = OptimizationResult(
            trials=[],
            best_config={},
            best_score=0.0,
            optimization_id="opt_001",
            duration=0.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=[],
            algorithm="random",
            timestamp=datetime.now(),
            total_cost=10.5,
        )

        assert result._resolve_total_cost(5.0) == pytest.approx(10.5)

    def test_resolve_total_cost_from_computed(self):
        """Test _resolve_total_cost falls back to computed cost."""
        result = OptimizationResult(
            trials=[],
            best_config={},
            best_score=0.0,
            optimization_id="opt_001",
            duration=0.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=[],
            algorithm="random",
            timestamp=datetime.now(),
            total_cost=None,
        )

        assert result._resolve_total_cost(7.5) == pytest.approx(7.5)
