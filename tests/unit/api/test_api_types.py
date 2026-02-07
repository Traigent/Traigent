"""Comprehensive tests for traigent.api.types module."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from traigent.api.types import (
    ConfigSpace,
    ConfigurationComparison,
    ExampleResult,
    Metrics,
    Objectives,
    OptimizationJob,
    OptimizationResult,
    OptimizationStatus,
    ParetoFront,
    SensitivityAnalysis,
    StrategyConfig,
    Trial,
    TrialResult,
    TrialStatus,
)
from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema


class TestEnums:
    """Test enum types."""

    def test_optimization_status_values(self):
        """Test OptimizationStatus enum values."""
        assert OptimizationStatus.NOT_STARTED == "not_started"
        assert OptimizationStatus.PENDING == "pending"
        assert OptimizationStatus.RUNNING == "running"
        assert OptimizationStatus.COMPLETED == "completed"
        assert OptimizationStatus.FAILED == "failed"
        assert OptimizationStatus.CANCELLED == "cancelled"

    def test_trial_status_values(self):
        """Test TrialStatus enum values."""
        assert TrialStatus.NOT_STARTED == "not_started"
        assert TrialStatus.PENDING == "pending"
        assert TrialStatus.RUNNING == "running"
        assert TrialStatus.COMPLETED == "completed"
        assert TrialStatus.FAILED == "failed"
        assert TrialStatus.CANCELLED == "cancelled"

    def test_enum_comparison(self):
        """Test enum comparison."""
        assert TrialStatus.COMPLETED == TrialStatus.COMPLETED
        assert TrialStatus.COMPLETED != TrialStatus.FAILED
        assert OptimizationStatus.RUNNING != OptimizationStatus.COMPLETED

    def test_enum_string_representation(self):
        """Test enum string representation (StrEnum returns value directly)."""
        assert str(OptimizationStatus.RUNNING) == "running"
        assert str(TrialStatus.CANCELLED) == "cancelled"
        # Test the actual values
        assert OptimizationStatus.RUNNING.value == "running"
        assert TrialStatus.CANCELLED.value == "cancelled"


class TestTrial:
    """Test Trial dataclass."""

    def test_trial_creation(self):
        """Test creating a Trial instance."""
        now = datetime.now()
        trial = Trial(
            trial_id="trial_001",
            config={"model": "gpt-3.5", "temperature": 0.7},
            timestamp=now,
        )

        assert trial.trial_id == "trial_001"
        assert trial.config == {"model": "gpt-3.5", "temperature": 0.7}
        assert trial.timestamp == now
        assert trial.status == TrialStatus.PENDING
        assert trial.metadata == {}

    def test_trial_with_metadata(self):
        """Test Trial with custom metadata."""
        trial = Trial(
            trial_id="trial_002",
            config={"param": "value"},
            timestamp=datetime.now(),
            status=TrialStatus.RUNNING,
            metadata={"source": "test", "priority": "high"},
        )

        assert trial.status == TrialStatus.RUNNING
        assert trial.metadata["source"] == "test"
        assert trial.metadata["priority"] == "high"

    def test_trial_dataclass_fields(self):
        """Test Trial dataclass fields."""
        from dataclasses import fields

        field_names = {f.name for f in fields(Trial)}
        expected_fields = {"trial_id", "config", "timestamp", "status", "metadata"}
        assert expected_fields.issubset(field_names)


class TestTrialResult:
    """Test TrialResult dataclass."""

    def test_trial_result_creation(self):
        """Test creating a TrialResult instance."""
        now = datetime.now()
        result = TrialResult(
            trial_id="trial_001",
            config={"model": "gpt-4", "temperature": 0.5},
            metrics={"accuracy": 0.95, "cost": 0.05},
            status=TrialStatus.COMPLETED,
            duration=1.23,
            timestamp=now,
        )

        assert result.trial_id == "trial_001"
        assert result.config["model"] == "gpt-4"
        assert result.metrics["accuracy"] == 0.95
        assert result.status == TrialStatus.COMPLETED
        assert result.duration == 1.23
        assert result.error_message is None

    def test_trial_result_with_error(self):
        """Test TrialResult with error."""
        result = TrialResult(
            trial_id="trial_002",
            config={},
            metrics={},
            status=TrialStatus.FAILED,
            duration=0.5,
            timestamp=datetime.now(),
            error_message="Connection timeout",
        )

        assert result.status == TrialStatus.FAILED
        assert result.error_message == "Connection timeout"
        assert not result.is_successful

    def test_is_successful_property(self):
        """Test is_successful property."""
        # Successful trial
        success_result = TrialResult(
            trial_id="s1",
            config={},
            metrics={"score": 0.9},
            status=TrialStatus.COMPLETED,
            duration=1.0,
            timestamp=datetime.now(),
        )
        assert success_result.is_successful

        # Failed trial
        failed_result = TrialResult(
            trial_id="f1",
            config={},
            metrics={},
            status=TrialStatus.FAILED,
            duration=0.1,
            timestamp=datetime.now(),
        )
        assert not failed_result.is_successful

        # Pruned trial
        pruned_result = TrialResult(
            trial_id="p1",
            config={},
            metrics={},
            status=TrialStatus.CANCELLED,
            duration=0.5,
            timestamp=datetime.now(),
        )
        assert not pruned_result.is_successful

    def test_get_metric(self):
        """Test get_metric method."""
        result = TrialResult(
            trial_id="trial_001",
            config={},
            metrics={"accuracy": 0.95, "loss": 0.05},
            status=TrialStatus.COMPLETED,
            duration=1.0,
            timestamp=datetime.now(),
        )

        assert result.get_metric("accuracy") == 0.95
        assert result.get_metric("loss") == 0.05
        assert result.get_metric("nonexistent") is None
        assert result.get_metric("nonexistent", default=0.0) == 0.0


class TestExampleResult:
    """Test ExampleResult dataclass."""

    def test_example_result_creation(self):
        """Test creating an ExampleResult instance."""
        result = ExampleResult(
            example_id="ex_001",
            input_data={"text": "Hello world"},
            expected_output="positive",
            actual_output="positive",
            metrics={"accuracy": 1.0, "confidence": 0.98},
            execution_time=0.05,
            success=True,
        )

        assert result.example_id == "ex_001"
        assert result.input_data["text"] == "Hello world"
        assert result.expected_output == "positive"
        assert result.actual_output == "positive"
        assert result.metrics["accuracy"] == 1.0
        assert result.execution_time == 0.05
        assert result.success

    def test_example_result_with_error(self):
        """Test ExampleResult with error."""
        result = ExampleResult(
            example_id="ex_002",
            input_data={"text": "Test"},
            expected_output="output",
            actual_output=None,
            metrics={},
            execution_time=0.01,
            success=False,
            error_message="Model error",
        )

        assert not result.success
        assert result.error_message == "Model error"
        assert not result.is_successful

    def test_is_successful_property(self):
        """Test is_successful property."""
        # Successful example
        success = ExampleResult(
            example_id="s1",
            input_data={},
            expected_output="A",
            actual_output="A",
            metrics={"score": 1.0},
            execution_time=0.1,
            success=True,
        )
        assert success.is_successful

        # Failed example
        failed = ExampleResult(
            example_id="f1",
            input_data={},
            expected_output="A",
            actual_output="B",
            metrics={},
            execution_time=0.1,
            success=False,
        )
        assert not failed.is_successful

        # Example with error message
        error_example = ExampleResult(
            example_id="e1",
            input_data={},
            expected_output="A",
            actual_output="A",
            metrics={},
            execution_time=0.1,
            success=True,
            error_message="Some error",
        )
        assert not error_example.is_successful

    def test_get_metric(self):
        """Test get_metric method."""
        result = ExampleResult(
            example_id="ex_001",
            input_data={},
            expected_output="",
            actual_output="",
            metrics={"precision": 0.9, "recall": 0.85},
            execution_time=0.1,
            success=True,
        )

        assert result.get_metric("precision") == 0.9
        assert result.get_metric("recall") == 0.85
        assert result.get_metric("f1") is None
        assert result.get_metric("f1", default=0.5) == 0.5


class TestOptimizationResult:
    """Test OptimizationResult dataclass."""

    def setup_method(self):
        """Set up test data."""
        self.trials = [
            TrialResult(
                trial_id="t1",
                config={"model": "gpt-3.5"},
                metrics={"accuracy": 0.85, "cost": 0.02},
                status=TrialStatus.COMPLETED,
                duration=1.0,
                timestamp=datetime.now(),
            ),
            TrialResult(
                trial_id="t2",
                config={"model": "gpt-4"},
                metrics={"accuracy": 0.95, "cost": 0.10},
                status=TrialStatus.COMPLETED,
                duration=1.5,
                timestamp=datetime.now(),
            ),
            TrialResult(
                trial_id="t3",
                config={"model": "gpt-3.5-turbo"},
                metrics={},
                status=TrialStatus.FAILED,
                duration=0.5,
                timestamp=datetime.now(),
                error_message="API error",
            ),
        ]

    def test_optimization_result_creation(self):
        """Test creating an OptimizationResult instance."""
        result = OptimizationResult(
            trials=self.trials,
            best_config={"model": "gpt-4"},
            best_score=0.95,
            optimization_id="opt_001",
            duration=10.5,
            convergence_info={"iterations": 100, "converged": True},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy"],
            algorithm="bayesian",
            timestamp=datetime.now(),
        )

        assert len(result.trials) == 3
        assert result.best_config["model"] == "gpt-4"
        assert result.best_score == 0.95
        assert result.optimization_id == "opt_001"
        assert result.duration == 10.5
        assert result.status == OptimizationStatus.COMPLETED

    def test_successful_trials_property(self):
        """Test successful_trials property."""
        result = OptimizationResult(
            trials=self.trials,
            best_config={},
            best_score=0.95,
            optimization_id="opt_001",
            duration=10.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy"],
            algorithm="random",
            timestamp=datetime.now(),
        )

        successful = result.successful_trials
        assert len(successful) == 2
        assert all(t.is_successful for t in successful)

    def test_failed_trials_property(self):
        """Test failed_trials property."""
        result = OptimizationResult(
            trials=self.trials,
            best_config={},
            best_score=0.95,
            optimization_id="opt_001",
            duration=10.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy"],
            algorithm="random",
            timestamp=datetime.now(),
        )

        failed = result.failed_trials
        assert len(failed) == 1
        assert failed[0].trial_id == "t3"

    def test_success_rate_property(self):
        """Test success_rate property."""
        result = OptimizationResult(
            trials=self.trials,
            best_config={},
            best_score=0.95,
            optimization_id="opt_001",
            duration=10.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy"],
            algorithm="random",
            timestamp=datetime.now(),
        )

        assert result.success_rate == 2 / 3  # 2 successful out of 3

    def test_success_rate_empty_trials(self):
        """Test success_rate with no trials."""
        result = OptimizationResult(
            trials=[],
            best_config={},
            best_score=0.0,
            optimization_id="opt_001",
            duration=0.0,
            convergence_info={},
            status=OptimizationStatus.FAILED,
            objectives=["accuracy"],
            algorithm="random",
            timestamp=datetime.now(),
        )

        assert result.success_rate == 0.0

    def test_best_metrics_property(self):
        """Test best_metrics property."""
        result = OptimizationResult(
            trials=self.trials,
            best_config={},
            best_score=0.95,
            optimization_id="opt_001",
            duration=10.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy"],
            algorithm="random",
            timestamp=datetime.now(),
        )

        best_metrics = result.best_metrics
        assert best_metrics["accuracy"] == 0.95
        assert best_metrics["cost"] == 0.10

    def test_best_metrics_empty_trials(self):
        """Test best_metrics with no trials."""
        result = OptimizationResult(
            trials=[],
            best_config={},
            best_score=0.0,
            optimization_id="opt_001",
            duration=0.0,
            convergence_info={},
            status=OptimizationStatus.FAILED,
            objectives=["accuracy"],
            algorithm="random",
            timestamp=datetime.now(),
        )

        assert result.best_metrics == {}

    def test_best_metrics_no_objectives(self):
        """best_metrics should handle missing objectives gracefully."""
        result = OptimizationResult(
            trials=self.trials,
            best_config=self.trials[1].config,
            best_score=0.95,
            optimization_id="opt_001",
            duration=10.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=[],
            algorithm="random",
            timestamp=datetime.now(),
        )

        best_metrics = result.best_metrics
        assert best_metrics == self.trials[1].metrics

    def test_to_dataframe(self):
        """Test to_dataframe method."""
        result = OptimizationResult(
            trials=self.trials[:2],  # Use only successful trials
            best_config={},
            best_score=0.95,
            optimization_id="opt_001",
            duration=10.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy"],
            algorithm="random",
            timestamp=datetime.now(),
        )

        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "trial_id" in df.columns
        assert "status" in df.columns
        assert "model" in df.columns  # From config
        assert "accuracy" in df.columns  # From metrics
        assert "duration" in df.columns
        assert "timestamp" in df.columns

    def test_extract_response_time_from_mapping(self):
        """_extract_response_time should prioritise direct mapping fields."""
        entry = {"response_time": "1.25"}

        assert OptimizationResult._extract_response_time(entry) == pytest.approx(1.25)

    def test_extract_response_time_from_ms_and_metrics(self):
        """_extract_response_time should convert milliseconds and inspect nested metrics."""
        entry = {
            "metrics": {"response_time_ms": 450},
        }

        assert OptimizationResult._extract_response_time(entry) == pytest.approx(0.45)

    def test_extract_response_time_falls_back_to_execution_time(self):
        """_extract_response_time should fall back to execution_time on objects."""

        class DummyEntry:
            def __init__(self):
                self.metrics = {}
                self.execution_time = "2.0"

        fallback_entry = DummyEntry()

        assert OptimizationResult._extract_response_time(
            fallback_entry
        ) == pytest.approx(2.0)

    def test_calculate_weighted_scores_normalizes_weights(self):
        """Weighted score calculation should normalize provided weights and infer minimize goals."""
        result = OptimizationResult(
            trials=self.trials,
            best_config={},
            best_score=0.95,
            optimization_id="opt_001",
            duration=10.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy", "cost"],
            algorithm="random",
            timestamp=datetime.now(),
        )

        weighted = result.calculate_weighted_scores(objective_weights={"accuracy": 3.0})

        used_weights = weighted["objective_weights_used"]
        assert pytest.approx(sum(used_weights.values()), rel=1e-6) == 1.0
        assert pytest.approx(used_weights["accuracy"], rel=1e-6) == 0.75
        assert pytest.approx(used_weights["cost"], rel=1e-6) == 0.25
        assert "cost" in weighted["minimize_objectives"]

    def test_calculate_weighted_scores_uses_schema_preferences(self):
        """Objective schema should drive weights, orientation, and normalization."""
        objectives = [
            ObjectiveDefinition(name="accuracy", orientation="maximize", weight=0.2),
            ObjectiveDefinition(name="cost", orientation="minimize", weight=0.8),
        ]
        schema = ObjectiveSchema.from_objectives(objectives)

        result = OptimizationResult(
            trials=self.trials,
            best_config={},
            best_score=0.95,
            optimization_id="opt_001",
            duration=10.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy", "cost"],
            algorithm="random",
            timestamp=datetime.now(),
        )

        weighted = result.calculate_weighted_scores(objective_schema=schema)

        used_weights = weighted["objective_weights_used"]
        assert pytest.approx(sum(used_weights.values()), rel=1e-6) == 1.0
        assert pytest.approx(used_weights["accuracy"], rel=1e-6) == 0.2
        assert pytest.approx(used_weights["cost"], rel=1e-6) == 0.8
        assert weighted["minimize_objectives"] == ["cost"]


class TestSensitivityAnalysis:
    """Test SensitivityAnalysis dataclass."""

    def test_sensitivity_analysis_creation(self):
        """Test creating a SensitivityAnalysis instance."""
        analysis = SensitivityAnalysis(
            parameter_importance={"temperature": 0.8, "model": 0.6, "max_tokens": 0.3},
            parameter_interactions={("temperature", "model"): 0.4},
            most_important_parameter="temperature",
            statistical_significance={"temperature": 0.001, "model": 0.05},
            method="sobol",
            confidence_level=0.95,
        )

        assert analysis.parameter_importance["temperature"] == 0.8
        assert analysis.most_important_parameter == "temperature"
        assert analysis.method == "sobol"
        assert analysis.confidence_level == 0.95

    def test_get_top_parameters(self):
        """Test get_top_parameters method."""
        analysis = SensitivityAnalysis(
            parameter_importance={
                "param1": 0.9,
                "param2": 0.3,
                "param3": 0.6,
                "param4": 0.1,
                "param5": 0.7,
            },
            parameter_interactions={},
            most_important_parameter="param1",
            statistical_significance={},
            method="variance",
            confidence_level=0.95,
        )

        top_3 = analysis.get_top_parameters(n=3)
        assert len(top_3) == 3
        assert top_3[0] == ("param1", 0.9)
        assert top_3[1] == ("param5", 0.7)
        assert top_3[2] == ("param3", 0.6)

    def test_get_top_parameters_negative_values(self):
        """Test get_top_parameters with negative importance values."""
        analysis = SensitivityAnalysis(
            parameter_importance={
                "param1": -0.8,  # Negative correlation
                "param2": 0.5,
                "param3": -0.9,  # Strongest negative
            },
            parameter_interactions={},
            most_important_parameter="param3",
            statistical_significance={},
            method="correlation",
            confidence_level=0.95,
        )

        top_2 = analysis.get_top_parameters(n=2)
        assert top_2[0] == ("param3", -0.9)  # Absolute value ordering
        assert top_2[1] == ("param1", -0.8)

    def test_get_top_parameters_more_than_available(self):
        """Test get_top_parameters when n > available parameters."""
        analysis = SensitivityAnalysis(
            parameter_importance={"param1": 0.5, "param2": 0.3},
            parameter_interactions={},
            most_important_parameter="param1",
            statistical_significance={},
            method="variance",
            confidence_level=0.95,
        )

        top_5 = analysis.get_top_parameters(n=5)
        assert len(top_5) == 2  # Only 2 parameters available


class TestConfigurationComparison:
    """Test ConfigurationComparison dataclass."""

    def test_configuration_comparison_creation(self):
        """Test creating a ConfigurationComparison instance."""
        configs = [
            {"model": "gpt-3.5", "temperature": 0.5},
            {"model": "gpt-4", "temperature": 0.7},
            {"model": "gpt-3.5-turbo", "temperature": 0.6},
        ]

        comparison = ConfigurationComparison(
            configurations=configs,
            comparison_metrics={
                "accuracy": [0.85, 0.95, 0.90],
                "cost": [0.02, 0.10, 0.05],
            },
            statistical_tests={"accuracy_anova": {"p_value": 0.03, "f_statistic": 4.5}},
            significant_differences=[(0, 1, "accuracy"), (1, 2, "cost")],
            confidence_level=0.95,
        )

        assert len(comparison.configurations) == 3
        assert comparison.comparison_metrics["accuracy"][1] == 0.95
        assert len(comparison.significant_differences) == 2

    def test_get_best_configuration(self):
        """Test get_best_configuration method."""
        configs = [{"config": "A"}, {"config": "B"}, {"config": "C"}]

        comparison = ConfigurationComparison(
            configurations=configs,
            comparison_metrics={"score": [0.7, 0.9, 0.8], "speed": [10, 5, 8]},
            statistical_tests={},
            significant_differences=[],
            confidence_level=0.95,
        )

        # Best score
        best_idx, best_config = comparison.get_best_configuration("score")
        assert best_idx == 1
        assert best_config["config"] == "B"

        # Best speed
        best_idx, best_config = comparison.get_best_configuration("speed")
        assert best_idx == 0
        assert best_config["config"] == "A"

    def test_get_best_configuration_invalid_metric(self):
        """Test get_best_configuration with invalid metric."""
        comparison = ConfigurationComparison(
            configurations=[{}],
            comparison_metrics={"valid": [1.0]},
            statistical_tests={},
            significant_differences=[],
            confidence_level=0.95,
        )

        with pytest.raises(ValueError, match="Metric 'invalid' not found"):
            comparison.get_best_configuration("invalid")


class TestParetoFront:
    """Test ParetoFront dataclass."""

    def test_pareto_front_creation(self):
        """Test creating a ParetoFront instance."""
        configs = [{"model": "A"}, {"model": "B"}, {"model": "C"}]

        pareto = ParetoFront(
            configurations=configs,
            objective_values=np.array([[0.8, 0.2], [0.9, 0.5], [0.7, 0.1]]),
            objectives=["accuracy", "cost"],
            is_maximized=[True, False],  # Maximize accuracy, minimize cost
        )

        assert len(pareto.configurations) == 3
        assert pareto.objective_values.shape == (3, 2)
        assert pareto.objectives == ["accuracy", "cost"]

    def test_get_best_balanced_config(self):
        """Test get_best_balanced_config method."""
        configs = [{"id": "A"}, {"id": "B"}, {"id": "C"}]

        # B has best balance: high accuracy (0.85) and low cost (0.3)
        pareto = ParetoFront(
            configurations=configs,
            objective_values=np.array(
                [
                    [0.9, 0.8],  # A: highest accuracy but highest cost
                    [0.85, 0.3],  # B: good accuracy, low cost
                    [0.7, 0.1],  # C: lowest accuracy, lowest cost
                ]
            ),
            objectives=["accuracy", "cost"],
            is_maximized=[True, False],
        )

        best = pareto.get_best_balanced_config()
        assert best["id"] == "B"

    def test_get_best_balanced_config_empty(self):
        """Test get_best_balanced_config with no configurations."""
        pareto = ParetoFront(
            configurations=[],
            objective_values=np.array([]),
            objectives=["accuracy"],
            is_maximized=[True],
        )

        with pytest.raises(ValueError, match="No configurations in Pareto front"):
            pareto.get_best_balanced_config()

    def test_get_best_balanced_config_single_objective(self):
        """Test get_best_balanced_config with single objective."""
        configs = [{"id": "A"}, {"id": "B"}, {"id": "C"}]

        pareto = ParetoFront(
            configurations=configs,
            objective_values=np.array([[0.7], [0.9], [0.8]]),
            objectives=["accuracy"],
            is_maximized=[True],
        )

        best = pareto.get_best_balanced_config()
        assert best["id"] == "B"  # Highest accuracy

    def test_plot_trade_offs(self):
        """Test plot_trade_offs method (placeholder test)."""
        pareto = ParetoFront(
            configurations=[{"model": "A"}],
            objective_values=np.array([[0.8, 0.2]]),
            objectives=["accuracy", "cost"],
            is_maximized=[True, False],
        )

        # Method is not implemented, just check it exists and doesn't raise
        result = pareto.plot_trade_offs("accuracy", "cost")
        assert result is None  # Method returns None


class TestStrategyConfig:
    """Test StrategyConfig dataclass."""

    def test_strategy_config_creation(self):
        """Test creating a StrategyConfig instance."""
        config = StrategyConfig(
            algorithm="bayesian",
            algorithm_config={"acquisition_function": "ei"},
            parallel_workers=4,
            resource_limits={"max_memory": "4GB"},
        )

        assert config.algorithm == "bayesian"
        assert config.algorithm_config["acquisition_function"] == "ei"
        assert config.parallel_workers == 4
        assert config.resource_limits["max_memory"] == "4GB"

    def test_strategy_config_defaults(self):
        """Test StrategyConfig with defaults."""
        config = StrategyConfig(algorithm="random")

        assert config.algorithm == "random"
        assert config.algorithm_config == {}
        assert config.parallel_workers == 1
        assert config.resource_limits == {}

    def test_validate_valid_config(self):
        """Test validate with valid configuration."""
        config = StrategyConfig(algorithm="grid", parallel_workers=2)

        # Should not raise
        result = config.validate()
        assert result is None  # Method returns None on success

    def test_validate_invalid_workers(self):
        """Test validate with invalid parallel_workers."""
        config = StrategyConfig(algorithm="grid", parallel_workers=0)

        with pytest.raises(ValueError, match="parallel_workers must be >= 1"):
            config.validate()

    def test_validate_negative_workers(self):
        """Test validate with negative parallel_workers."""
        config = StrategyConfig(algorithm="grid", parallel_workers=-5)

        with pytest.raises(ValueError, match="parallel_workers must be >= 1"):
            config.validate()


class TestOptimizationJob:
    """Test OptimizationJob dataclass."""

    def test_optimization_job_creation(self):
        """Test creating an OptimizationJob instance."""
        job = OptimizationJob(
            job_id="job_001",
            status=OptimizationStatus.RUNNING,
            progress=0.45,
            estimated_completion=datetime.now(),
        )

        assert job.job_id == "job_001"
        assert job.status == OptimizationStatus.RUNNING
        assert job.progress == 0.45
        assert job.estimated_completion is not None

    def test_optimization_job_no_estimated_completion(self):
        """Test OptimizationJob without estimated completion."""
        job = OptimizationJob(
            job_id="job_002",
            status=OptimizationStatus.PENDING,
            progress=0.0,
            estimated_completion=None,
        )

        assert job.estimated_completion is None

    def test_is_complete(self):
        """Test is_complete method."""
        # Running job
        running_job = OptimizationJob(
            job_id="j1",
            status=OptimizationStatus.RUNNING,
            progress=0.5,
            estimated_completion=None,
        )
        assert not running_job.is_complete()

        # Pending job
        pending_job = OptimizationJob(
            job_id="j0",
            status=OptimizationStatus.PENDING,
            progress=0.0,
            estimated_completion=None,
        )
        assert not pending_job.is_complete()

        # Completed job
        completed_job = OptimizationJob(
            job_id="j2",
            status=OptimizationStatus.COMPLETED,
            progress=1.0,
            estimated_completion=None,
        )
        assert completed_job.is_complete()

        # Failed job
        failed_job = OptimizationJob(
            job_id="j3",
            status=OptimizationStatus.FAILED,
            progress=0.3,
            estimated_completion=None,
        )
        assert failed_job.is_complete()

        # Stopped job
        stopped_job = OptimizationJob(
            job_id="j4",
            status=OptimizationStatus.CANCELLED,
            progress=0.7,
            estimated_completion=None,
        )
        assert stopped_job.is_complete()

    def test_wait_not_implemented(self):
        """Test wait method raises NotImplementedError."""
        job = OptimizationJob(
            job_id="job_001",
            status=OptimizationStatus.RUNNING,
            progress=0.0,
            estimated_completion=None,
        )

        with pytest.raises(
            NotImplementedError, match="Background jobs not yet implemented"
        ):
            job.wait()

        with pytest.raises(NotImplementedError):
            job.wait(timeout=10.0)


class TestTypeAliases:
    """Test type aliases are defined correctly."""

    def test_type_aliases_exist(self):
        """Test that type aliases are defined."""
        # These are type aliases, so we just check they can be used
        config_space: ConfigSpace = {"param": [1, 2, 3]}
        metrics: Metrics = {"accuracy": 0.95}
        objectives: Objectives = ["accuracy", "cost"]

        assert isinstance(config_space, dict)
        assert isinstance(metrics, dict)
        assert isinstance(objectives, list)

    def test_config_space_types(self):
        """Test ConfigSpace type alias supports various value types."""
        # List of values
        config1: ConfigSpace = {"model": ["gpt-3.5", "gpt-4"]}

        # Tuple range
        config2: ConfigSpace = {"temperature": (0.0, 1.0)}

        # Single value
        config3: ConfigSpace = {"max_tokens": 100}

        # Mixed types
        config4: ConfigSpace = {
            "model": ["gpt-3.5", "gpt-4"],
            "temperature": (0.0, 1.0),
            "seed": 42,
        }

        assert all(isinstance(c, dict) for c in [config1, config2, config3, config4])
