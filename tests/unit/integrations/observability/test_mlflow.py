"""Unit tests for MLflow integration.

Tests for Traigent MLflow experiment tracking and observability.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Observability
# CONC-Quality-Compatibility FUNC-INTEGRATIONS FUNC-ANALYTICS
# REQ-INT-008 REQ-ANLY-011 SYNC-Observability

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from traigent.api.types import (
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)
from traigent.integrations.observability import mlflow as mlflow_module


class MockMLflowRun:
    """Mock MLflow run object."""

    def __init__(self, run_id: str = "mock_run_123") -> None:
        """Initialize mock run."""
        self.info = Mock()
        self.info.run_id = run_id
        self.info.run_name = f"run_{run_id}"
        self.data = Mock()
        self.data.metrics = {}
        self.data.params = {}
        self.data.tags = {}


class MockMLflowClient:
    """Mock MLflow tracking client."""

    def __init__(self) -> None:
        """Initialize mock client."""
        self.runs: dict[str, MockMLflowRun] = {}

    def get_run(self, run_id: str) -> MockMLflowRun:
        """Get run by ID."""
        if run_id not in self.runs:
            run = MockMLflowRun(run_id)
            self.runs[run_id] = run
        return self.runs[run_id]


class MockMLflow:
    """Mock MLflow module."""

    def __init__(self) -> None:
        """Initialize mock mlflow."""
        self.current_run: MockMLflowRun | None = None
        self.experiments: dict[str, Mock] = {}
        self.logged_params: dict[str, any] = {}
        self.logged_metrics: dict[str, list[tuple[float, int | None]]] = {}
        self.logged_tags: dict[str, str] = {}
        self.logged_dicts: list[tuple[dict, str]] = []
        self.logged_artifacts: list[tuple[str, str | None]] = []
        self.tracking_uri: str | None = None
        self.active_experiment: str | None = None
        self.tracking = Mock()
        self.tracking.MlflowClient = MockMLflowClient

    def set_tracking_uri(self, uri: str) -> None:
        """Set tracking URI."""
        self.tracking_uri = uri

    def get_experiment_by_name(self, name: str) -> Mock | None:
        """Get experiment by name."""
        return self.experiments.get(name)

    def create_experiment(self, name: str) -> str:
        """Create experiment."""
        experiment = Mock()
        experiment.experiment_id = f"exp_{len(self.experiments)}"
        self.experiments[name] = experiment
        return experiment.experiment_id

    def set_experiment(self, name: str) -> None:
        """Set active experiment."""
        self.active_experiment = name

    def start_run(self, run_name: str | None = None) -> MockMLflowRun:
        """Start MLflow run."""
        self.current_run = MockMLflowRun()
        return self.current_run

    def finish(self) -> None:
        """Finish current run."""
        self.current_run = None

    def end_run(self) -> None:
        """End current run."""
        self.finish()

    def set_tag(self, key: str, value: str) -> None:
        """Log tag."""
        self.logged_tags[key] = value

    def log_param(self, key: str, value: Any) -> None:
        """Log parameter."""
        self.logged_params[key] = value

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log metric."""
        if key not in self.logged_metrics:
            self.logged_metrics[key] = []
        self.logged_metrics[key].append((value, step))

    def log_dict(self, data: dict, filename: str) -> None:
        """Log dictionary as artifact."""
        self.logged_dicts.append((data, filename))

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        """Log artifact."""
        self.logged_artifacts.append((path, artifact_path))

    def search_runs(
        self,
        experiment_ids: list[str],
        filter_string: str | None = None,
        order_by: list[str] | None = None,
        max_results: int = 1,
    ) -> Mock:
        """Search runs."""
        import pandas as pd

        # Return empty DataFrame for testing
        return pd.DataFrame()


@pytest.fixture
def mock_mlflow() -> MockMLflow:
    """Create mock mlflow module."""
    return MockMLflow()


@pytest.fixture
def patched_mlflow(
    monkeypatch: pytest.MonkeyPatch, mock_mlflow: MockMLflow
) -> MockMLflow:
    """Patch mlflow module in mlflow integration."""
    monkeypatch.setattr(mlflow_module, "mlflow", mock_mlflow)
    monkeypatch.setattr(mlflow_module, "MLFLOW_AVAILABLE", True)
    return mock_mlflow


@pytest.fixture
def sample_trial() -> TrialResult:
    """Create sample trial result."""
    return TrialResult(
        trial_id="trial_001",
        config={"model": "gpt-4o", "temperature": 0.7},
        metrics={"accuracy": 0.85, "latency": 1.2},
        status=TrialStatus.COMPLETED,
        duration=5.5,
        timestamp=datetime.now(UTC),
    )


@pytest.fixture
def sample_optimization_result() -> OptimizationResult:
    """Create sample optimization result."""
    trials = [
        TrialResult(
            trial_id=f"trial_{i}",
            config={"model": "gpt-4o", "temperature": 0.5 + i * 0.1},
            metrics={"accuracy": 0.8 + i * 0.05, "cost": 0.1 - i * 0.01},
            status=TrialStatus.COMPLETED,
            duration=5.0 + i,
            timestamp=datetime.now(UTC),
        )
        for i in range(3)
    ]

    return OptimizationResult(
        trials=trials,
        best_config={"model": "gpt-4o", "temperature": 0.7},
        best_score=0.9,
        optimization_id="opt_123",
        duration=20.0,
        convergence_info={"converged": True},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="bayesian",
        timestamp=datetime.now(UTC),
    )


class TestTraigentMLflowTracker:
    """Tests for TraigentMLflowTracker."""

    def test_init_raises_when_mlflow_not_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test initialization raises ImportError when MLflow is not installed."""
        monkeypatch.setattr(mlflow_module, "MLFLOW_AVAILABLE", False)

        with pytest.raises(ImportError, match="MLflow is not installed"):
            mlflow_module.TraigentMLflowTracker()

    def test_init_with_defaults(self, patched_mlflow: MockMLflow) -> None:
        """Test initialization with default parameters."""
        tracker = mlflow_module.TraigentMLflowTracker()

        assert tracker.auto_log is True
        assert tracker.current_run is None
        assert patched_mlflow.active_experiment == "traigent_optimization"

    def test_init_with_custom_tracking_uri(self, patched_mlflow: MockMLflow) -> None:
        """Test initialization with custom tracking URI."""
        tracker = mlflow_module.TraigentMLflowTracker(
            tracking_uri="http://localhost:5000"
        )

        assert patched_mlflow.tracking_uri == "http://localhost:5000"
        assert tracker is not None

    def test_init_creates_new_experiment(self, patched_mlflow: MockMLflow) -> None:
        """Test initialization creates new experiment if it doesn't exist."""
        tracker = mlflow_module.TraigentMLflowTracker(experiment_name="new_experiment")

        assert "new_experiment" in patched_mlflow.experiments
        assert tracker.experiment_id is not None

    def test_init_uses_existing_experiment(self, patched_mlflow: MockMLflow) -> None:
        """Test initialization uses existing experiment."""
        # Create experiment first
        existing_exp = Mock()
        existing_exp.experiment_id = "exp_existing"
        patched_mlflow.experiments["my_experiment"] = existing_exp

        tracker = mlflow_module.TraigentMLflowTracker(experiment_name="my_experiment")

        assert tracker.experiment_id == "exp_existing"

    def test_init_handles_experiment_creation_failure(
        self, patched_mlflow: MockMLflow
    ) -> None:
        """Test initialization handles experiment creation failure."""

        def raise_error(name: str) -> str:
            raise RuntimeError("MLflow server unavailable")

        # Patch the method
        original_create = patched_mlflow.create_experiment
        patched_mlflow.create_experiment = raise_error  # type: ignore[assignment]

        tracker = mlflow_module.TraigentMLflowTracker()

        assert tracker.experiment_id is None

        # Restore original method
        patched_mlflow.create_experiment = original_create

    def test_start_optimization_run(self, patched_mlflow: MockMLflow) -> None:
        """Test starting optimization run."""
        tracker = mlflow_module.TraigentMLflowTracker()

        run_id = tracker.start_optimization_run(
            function_name="test_function",
            objectives=["accuracy", "latency"],
            configuration_space={"model": ["gpt-4o", "gpt-3.5"], "temp": [0.5, 1.0]},
        )

        assert run_id == "mock_run_123"
        assert tracker.current_run is not None
        assert patched_mlflow.logged_tags["traigent.function_name"] == "test_function"
        assert patched_mlflow.logged_tags["traigent.objectives"] == "accuracy,latency"
        assert patched_mlflow.logged_params["num_objectives"] == 2
        assert patched_mlflow.logged_params["config_space_size"] == 2

    def test_start_optimization_run_with_custom_name(
        self, patched_mlflow: MockMLflow
    ) -> None:
        """Test starting optimization run with custom run name."""
        tracker = mlflow_module.TraigentMLflowTracker()

        run_id = tracker.start_optimization_run(
            function_name="test_function",
            objectives=["accuracy"],
            configuration_space={},
            run_name="custom_run",
        )

        assert run_id == "mock_run_123"

    def test_start_optimization_run_with_tags(self, patched_mlflow: MockMLflow) -> None:
        """Test starting optimization run with custom tags."""
        tracker = mlflow_module.TraigentMLflowTracker()

        tracker.start_optimization_run(
            function_name="test_function",
            objectives=["accuracy"],
            configuration_space={},
            tags={"env": "test", "version": "1.0"},
        )

        assert patched_mlflow.logged_tags["env"] == "test"
        assert patched_mlflow.logged_tags["version"] == "1.0"

    def test_log_trial(
        self, patched_mlflow: MockMLflow, sample_trial: TrialResult
    ) -> None:
        """Test logging individual trial."""
        tracker = mlflow_module.TraigentMLflowTracker()
        tracker.start_optimization_run(
            function_name="test",
            objectives=["accuracy"],
            configuration_space={},
        )

        # Clear previous logs
        patched_mlflow.logged_params.clear()
        patched_mlflow.logged_metrics.clear()
        patched_mlflow.logged_artifacts.clear()

        with patch("builtins.open", create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file

            tracker.log_trial(sample_trial, trial_number=1)

        # Check parameters
        assert patched_mlflow.logged_params["trial_1_model"] == "gpt-4o"
        assert patched_mlflow.logged_params["trial_1_temperature"] == 0.7
        assert (
            patched_mlflow.logged_params["trial_1_status"]
            == TrialStatus.COMPLETED.value
        )

        # Check metrics
        assert ("trial_accuracy", 0.85, 1) in [
            (k, v, s)
            for k, vals in patched_mlflow.logged_metrics.items()
            for v, s in vals
        ]
        assert any(k == "trial_duration" for k in patched_mlflow.logged_metrics.keys())

    def test_log_trial_without_active_run(
        self, patched_mlflow: MockMLflow, sample_trial: TrialResult
    ) -> None:
        """Test logging trial without active run logs warning."""
        tracker = mlflow_module.TraigentMLflowTracker()

        tracker.log_trial(sample_trial, trial_number=1)

        # Should return early without logging
        assert len(patched_mlflow.logged_params) == 0

    def test_log_trial_with_custom_step(
        self, patched_mlflow: MockMLflow, sample_trial: TrialResult
    ) -> None:
        """Test logging trial with custom step number."""
        tracker = mlflow_module.TraigentMLflowTracker()
        tracker.start_optimization_run(
            function_name="test",
            objectives=["accuracy"],
            configuration_space={},
        )

        patched_mlflow.logged_metrics.clear()

        with patch("builtins.open", create=True), patch("pathlib.Path.unlink"):
            tracker.log_trial(sample_trial, trial_number=5, step=10)

        # Check that metrics were logged with correct step
        for metric_values in patched_mlflow.logged_metrics.values():
            for _, step in metric_values:
                assert step == 10

    def test_log_optimization_result(
        self, patched_mlflow: MockMLflow, sample_optimization_result: OptimizationResult
    ) -> None:
        """Test logging complete optimization result."""
        tracker = mlflow_module.TraigentMLflowTracker()
        tracker.start_optimization_run(
            function_name="test",
            objectives=["accuracy"],
            configuration_space={},
        )

        patched_mlflow.logged_params.clear()
        patched_mlflow.logged_metrics.clear()

        tracker.log_optimization_result(sample_optimization_result)

        # Check best config logged
        assert patched_mlflow.logged_params["best_model"] == "gpt-4o"
        assert patched_mlflow.logged_params["best_temperature"] == 0.7

        # Check best metrics logged
        assert any(k == "best_accuracy" for k in patched_mlflow.logged_metrics.keys())

        # Check summary metrics
        assert any(k == "total_trials" for k in patched_mlflow.logged_metrics.keys())
        assert any(
            k == "successful_trials" for k in patched_mlflow.logged_metrics.keys()
        )

    def test_log_optimization_result_without_active_run(
        self, patched_mlflow: MockMLflow, sample_optimization_result: OptimizationResult
    ) -> None:
        """Test logging optimization result without active run logs warning."""
        tracker = mlflow_module.TraigentMLflowTracker()

        tracker.log_optimization_result(sample_optimization_result)

        # Should return early without logging
        assert len(patched_mlflow.logged_params) == 0

    def test_log_optimization_result_with_dataset(
        self,
        patched_mlflow: MockMLflow,
        sample_optimization_result: OptimizationResult,
        tmp_path: Path,
    ) -> None:
        """Test logging optimization result with dataset path."""
        tracker = mlflow_module.TraigentMLflowTracker()
        tracker.start_optimization_run(
            function_name="test",
            objectives=["accuracy"],
            configuration_space={},
        )

        # Create dummy dataset file
        dataset_path = tmp_path / "dataset.json"
        dataset_path.write_text('{"data": "test"}')

        patched_mlflow.logged_artifacts.clear()

        tracker.log_optimization_result(
            sample_optimization_result, dataset_path=str(dataset_path)
        )

        # Check dataset was logged
        assert any(
            str(dataset_path) in artifact[0]
            for artifact in patched_mlflow.logged_artifacts
        )

    def test_log_optimization_result_with_metadata(
        self, patched_mlflow: MockMLflow
    ) -> None:
        """Test logging optimization result with metadata."""
        tracker = mlflow_module.TraigentMLflowTracker()
        tracker.start_optimization_run(
            function_name="test",
            objectives=["accuracy"],
            configuration_space={},
        )

        result = OptimizationResult(
            trials=[],
            best_config={},
            best_score=0.9,
            optimization_id="opt_123",
            duration=10.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy"],
            algorithm="bayesian",
            timestamp=datetime.now(UTC),
            metadata={"env": "test", "num_samples": 100},
        )

        patched_mlflow.logged_params.clear()
        patched_mlflow.logged_metrics.clear()

        tracker.log_optimization_result(result)

        # Check metadata logged
        assert patched_mlflow.logged_params["metadata_env"] == "test"
        assert any(
            k == "metadata_num_samples" for k in patched_mlflow.logged_metrics.keys()
        )

    def test_end_optimization_run(self, patched_mlflow: MockMLflow) -> None:
        """Test ending optimization run."""
        tracker = mlflow_module.TraigentMLflowTracker()
        tracker.start_optimization_run(
            function_name="test",
            objectives=["accuracy"],
            configuration_space={},
        )

        assert tracker.current_run is not None

        tracker.end_optimization_run()

        assert tracker.current_run is None
        assert patched_mlflow.current_run is None

    def test_end_optimization_run_without_active_run(
        self, patched_mlflow: MockMLflow
    ) -> None:
        """Test ending optimization run when no run is active."""
        tracker = mlflow_module.TraigentMLflowTracker()

        # Should not raise error
        tracker.end_optimization_run()

        assert tracker.current_run is None

    def test_compare_optimizations(self, patched_mlflow: MockMLflow) -> None:
        """Test comparing multiple optimization runs."""
        tracker = mlflow_module.TraigentMLflowTracker()

        # Create mock client with runs
        mock_client = MockMLflowClient()
        run1 = MockMLflowRun("run_1")
        run1.data.metrics = {"best_accuracy": 0.85, "total_trials": 10}
        run1.data.params = {"best_model": "gpt-4o", "optimization_status": "completed"}
        run1.data.tags = {"traigent.function_name": "test_func"}
        run1.info.run_name = "run_1"
        mock_client.runs["run_1"] = run1

        run2 = MockMLflowRun("run_2")
        run2.data.metrics = {"best_accuracy": 0.90, "total_trials": 15}
        run2.data.params = {"best_model": "gpt-3.5", "optimization_status": "completed"}
        run2.data.tags = {"traigent.function_name": "test_func"}
        run2.info.run_name = "run_2"
        mock_client.runs["run_2"] = run2

        patched_mlflow.tracking.MlflowClient = lambda: mock_client

        comparison = tracker.compare_optimizations(["run_1", "run_2"])

        assert "run_1" in comparison
        assert "run_2" in comparison
        assert comparison["run_1"]["best_metrics"]["accuracy"] == 0.85
        assert comparison["run_2"]["best_metrics"]["accuracy"] == 0.90

    def test_compare_optimizations_with_error(self, patched_mlflow: MockMLflow) -> None:
        """Test comparing optimizations handles errors gracefully."""
        tracker = mlflow_module.TraigentMLflowTracker()

        mock_client = MockMLflowClient()

        def raise_error(run_id: str) -> MockMLflowRun:
            raise RuntimeError("Run not found")

        mock_client.get_run = raise_error  # type: ignore[assignment]
        patched_mlflow.tracking.MlflowClient = lambda: mock_client

        comparison = tracker.compare_optimizations(["invalid_run"])

        assert "invalid_run" in comparison
        assert "error" in comparison["invalid_run"]

    def test_compare_optimizations_when_mlflow_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test compare_optimizations raises when MLflow not available."""
        monkeypatch.setattr(mlflow_module, "MLFLOW_AVAILABLE", False)

        # Create tracker with mocked availability during init
        with patch.object(mlflow_module, "MLFLOW_AVAILABLE", True):
            tracker = mlflow_module.TraigentMLflowTracker()

        # Now set it to False for the comparison
        monkeypatch.setattr(mlflow_module, "MLFLOW_AVAILABLE", False)

        with pytest.raises(ImportError, match="MLflow not available"):
            tracker.compare_optimizations(["run_1"])

    def test_get_best_run(self, patched_mlflow: MockMLflow) -> None:
        """Test getting best run from experiment."""
        import pandas as pd

        tracker = mlflow_module.TraigentMLflowTracker()

        # Create mock experiment
        experiment = Mock()
        experiment.experiment_id = "exp_123"
        patched_mlflow.experiments["test_experiment"] = experiment

        # Mock search_runs to return DataFrame with best run
        best_run_data = {
            "run_id": ["run_123"],
            "tags.mlflow.runName": ["best_run"],
            "tags.traigent.function_name": ["test_func"],
            "metrics.best_accuracy": [0.95],
            "metrics.total_trials": [20.0],
            "metrics.optimization_duration": [100.0],
        }
        mock_df = pd.DataFrame(best_run_data)

        patched_mlflow.search_runs = lambda **kwargs: mock_df

        result = tracker.get_best_run("test_experiment", "accuracy", maximize=True)

        assert result is not None
        assert result["run_id"] == "run_123"
        assert result["best_metric_value"] == 0.95

    def test_get_best_run_returns_none_for_missing_experiment(
        self, patched_mlflow: MockMLflow
    ) -> None:
        """Test get_best_run returns None for missing experiment."""
        tracker = mlflow_module.TraigentMLflowTracker()

        result = tracker.get_best_run("nonexistent", "accuracy")

        assert result is None

    def test_get_best_run_returns_none_for_empty_results(
        self, patched_mlflow: MockMLflow
    ) -> None:
        """Test get_best_run returns None when no runs match."""
        import pandas as pd

        tracker = mlflow_module.TraigentMLflowTracker()

        experiment = Mock()
        experiment.experiment_id = "exp_123"
        patched_mlflow.experiments["test_experiment"] = experiment

        # Return empty DataFrame
        patched_mlflow.search_runs = lambda **kwargs: pd.DataFrame()

        result = tracker.get_best_run("test_experiment", "accuracy")

        assert result is None

    def test_get_best_run_handles_errors(self, patched_mlflow: MockMLflow) -> None:
        """Test get_best_run handles errors gracefully."""
        tracker = mlflow_module.TraigentMLflowTracker()

        def raise_error(**kwargs: Any) -> Mock:
            raise RuntimeError("Search failed")

        patched_mlflow.search_runs = raise_error  # type: ignore[assignment]

        result = tracker.get_best_run("test_experiment", "accuracy")

        assert result is None

    def test_get_best_run_when_mlflow_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test get_best_run returns None when MLflow not available."""
        monkeypatch.setattr(mlflow_module, "MLFLOW_AVAILABLE", False)

        with patch.object(mlflow_module, "MLFLOW_AVAILABLE", True):
            tracker = mlflow_module.TraigentMLflowTracker()

        monkeypatch.setattr(mlflow_module, "MLFLOW_AVAILABLE", False)

        result = tracker.get_best_run("test_experiment", "accuracy")

        assert result is None


class TestMLflowOptimizationCallback:
    """Tests for MLflowOptimizationCallback."""

    def test_init(self, patched_mlflow: MockMLflow) -> None:
        """Test callback initialization."""
        tracker = mlflow_module.TraigentMLflowTracker()
        callback = mlflow_module.MLflowOptimizationCallback(tracker)

        assert callback.tracker is tracker
        assert callback.trial_count == 0

    def test_on_optimization_start(self, patched_mlflow: MockMLflow) -> None:
        """Test on_optimization_start callback."""
        tracker = mlflow_module.TraigentMLflowTracker()
        callback = mlflow_module.MLflowOptimizationCallback(tracker)

        callback.on_optimization_start(
            function_name="test_func",
            objectives=["accuracy", "latency"],
            configuration_space={"model": ["gpt-4o"]},
        )

        assert tracker.current_run is not None
        assert patched_mlflow.logged_tags["traigent.function_name"] == "test_func"

    def test_on_optimization_start_with_tags(self, patched_mlflow: MockMLflow) -> None:
        """Test on_optimization_start with custom MLflow tags."""
        tracker = mlflow_module.TraigentMLflowTracker()
        callback = mlflow_module.MLflowOptimizationCallback(tracker)

        callback.on_optimization_start(
            function_name="test_func",
            objectives=["accuracy"],
            configuration_space={},
            mlflow_tags={"custom": "tag"},
        )

        assert patched_mlflow.logged_tags["custom"] == "tag"

    def test_on_trial_complete(
        self, patched_mlflow: MockMLflow, sample_trial: TrialResult
    ) -> None:
        """Test on_trial_complete callback."""
        tracker = mlflow_module.TraigentMLflowTracker()
        callback = mlflow_module.MLflowOptimizationCallback(tracker)

        callback.on_optimization_start(
            function_name="test",
            objectives=["accuracy"],
            configuration_space={},
        )

        patched_mlflow.logged_params.clear()

        with patch("builtins.open", create=True), patch("pathlib.Path.unlink"):
            callback.on_trial_complete(sample_trial)

        assert callback.trial_count == 1
        assert patched_mlflow.logged_params["trial_1_model"] == "gpt-4o"

    def test_on_optimization_complete(
        self, patched_mlflow: MockMLflow, sample_optimization_result: OptimizationResult
    ) -> None:
        """Test on_optimization_complete callback."""
        tracker = mlflow_module.TraigentMLflowTracker()
        callback = mlflow_module.MLflowOptimizationCallback(tracker)

        callback.on_optimization_start(
            function_name="test",
            objectives=["accuracy"],
            configuration_space={},
        )

        callback.on_optimization_complete(sample_optimization_result)

        assert tracker.current_run is None


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_mlflow_tracker(self, patched_mlflow: MockMLflow) -> None:
        """Test create_mlflow_tracker function."""
        tracker = mlflow_module.create_mlflow_tracker(
            tracking_uri="http://localhost:5000",
            experiment_name="test_experiment",
        )

        assert isinstance(tracker, mlflow_module.TraigentMLflowTracker)
        assert patched_mlflow.tracking_uri == "http://localhost:5000"

    def test_enable_mlflow_autolog(self, patched_mlflow: MockMLflow) -> None:
        """Test enable_mlflow_autolog function."""
        callback = mlflow_module.enable_mlflow_autolog(
            tracking_uri="http://localhost:5000",
            experiment_name="test_experiment",
        )

        assert isinstance(callback, mlflow_module.MLflowOptimizationCallback)
        assert isinstance(callback.tracker, mlflow_module.TraigentMLflowTracker)

    def test_log_traigent_optimization(
        self, patched_mlflow: MockMLflow, sample_optimization_result: OptimizationResult
    ) -> None:
        """Test log_traigent_optimization function."""
        patched_mlflow.logged_params.clear()

        with patch("builtins.open", create=True), patch("pathlib.Path.unlink"):
            run_id = mlflow_module.log_traigent_optimization(
                result=sample_optimization_result,
                function_name="test_func",
                objectives=["accuracy"],
                configuration_space={"model": ["gpt-4o"]},
            )

        assert run_id == "mock_run_123"
        assert len(patched_mlflow.logged_params) > 0

    def test_log_traigent_optimization_with_all_params(
        self,
        patched_mlflow: MockMLflow,
        sample_optimization_result: OptimizationResult,
        tmp_path: Path,
    ) -> None:
        """Test log_traigent_optimization with all parameters."""
        dataset_path = tmp_path / "dataset.json"
        dataset_path.write_text('{"data": "test"}')

        with patch("builtins.open", create=True), patch("pathlib.Path.unlink"):
            run_id = mlflow_module.log_traigent_optimization(
                result=sample_optimization_result,
                function_name="test_func",
                objectives=["accuracy"],
                configuration_space={"model": ["gpt-4o"]},
                dataset_path=str(dataset_path),
                tracking_uri="http://localhost:5000",
                experiment_name="custom_exp",
                run_name="custom_run",
            )

        assert run_id is not None

    def test_compare_traigent_runs(self, patched_mlflow: MockMLflow) -> None:
        """Test compare_traigent_runs function."""
        mock_client = MockMLflowClient()
        run1 = MockMLflowRun("run_1")
        run1.data.metrics = {"best_accuracy": 0.85}
        run1.data.params = {"best_model": "gpt-4o"}
        run1.data.tags = {}
        run1.info.run_name = "run_1"
        mock_client.runs["run_1"] = run1

        patched_mlflow.tracking.MlflowClient = lambda: mock_client

        comparison = mlflow_module.compare_traigent_runs(
            run_ids=["run_1"],
            metrics=["accuracy"],
            tracking_uri="http://localhost:5000",
        )

        assert "run_1" in comparison

    def test_get_best_traigent_run(self, patched_mlflow: MockMLflow) -> None:
        """Test get_best_traigent_run function."""
        import pandas as pd

        experiment = Mock()
        experiment.experiment_id = "exp_123"
        patched_mlflow.experiments["test_experiment"] = experiment

        best_run_data = {
            "run_id": ["run_123"],
            "tags.mlflow.runName": ["best_run"],
            "tags.traigent.function_name": ["test_func"],
            "metrics.best_accuracy": [0.95],
            "metrics.total_trials": [20.0],
            "metrics.optimization_duration": [100.0],
        }
        mock_df = pd.DataFrame(best_run_data)

        patched_mlflow.search_runs = lambda **kwargs: mock_df

        result = mlflow_module.get_best_traigent_run(
            experiment_name="test_experiment",
            metric_name="accuracy",
            maximize=True,
            tracking_uri="http://localhost:5000",
        )

        assert result is not None
        assert result["run_id"] == "run_123"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_log_trial_creates_and_deletes_temp_file(
        self, patched_mlflow: MockMLflow, sample_trial: TrialResult
    ) -> None:
        """Test that log_trial creates and deletes temporary trial file."""
        tracker = mlflow_module.TraigentMLflowTracker()
        tracker.start_optimization_run(
            function_name="test",
            objectives=["accuracy"],
            configuration_space={},
        )

        with patch("builtins.open", create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file

            with patch("pathlib.Path.unlink") as mock_unlink:
                tracker.log_trial(sample_trial, trial_number=1)

                # Verify file was attempted to be deleted
                mock_unlink.assert_called_once()

    def test_optimization_result_with_empty_trials(
        self, patched_mlflow: MockMLflow
    ) -> None:
        """Test logging optimization result with empty trials list."""
        tracker = mlflow_module.TraigentMLflowTracker()
        tracker.start_optimization_run(
            function_name="test",
            objectives=["accuracy"],
            configuration_space={},
        )

        result = OptimizationResult(
            trials=[],
            best_config={},
            best_score=0.0,
            optimization_id="opt_123",
            duration=0.0,
            convergence_info={},
            status=OptimizationStatus.FAILED,
            objectives=["accuracy"],
            algorithm="random",
            timestamp=datetime.now(UTC),
        )

        # Should not raise error
        tracker.log_optimization_result(result)

        assert any(k == "total_trials" for k in patched_mlflow.logged_metrics.keys())

    def test_optimization_result_calculates_statistics(
        self, patched_mlflow: MockMLflow
    ) -> None:
        """Test optimization result calculates trial statistics correctly."""
        tracker = mlflow_module.TraigentMLflowTracker()
        tracker.start_optimization_run(
            function_name="test",
            objectives=["accuracy"],
            configuration_space={},
        )

        trials = [
            TrialResult(
                trial_id=f"trial_{i}",
                config={},
                metrics={"accuracy": 0.8 + i * 0.05},
                status=TrialStatus.COMPLETED,
                duration=5.0,
                timestamp=datetime.now(UTC),
            )
            for i in range(5)
        ]

        result = OptimizationResult(
            trials=trials,
            best_config={},
            best_score=0.9,
            optimization_id="opt_123",
            duration=25.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy"],
            algorithm="bayesian",
            timestamp=datetime.now(UTC),
        )

        patched_mlflow.logged_metrics.clear()

        tracker.log_optimization_result(result)

        # Check statistics were calculated
        assert any(k == "accuracy_mean" for k in patched_mlflow.logged_metrics.keys())
        assert any(k == "accuracy_std" for k in patched_mlflow.logged_metrics.keys())

    def test_trial_with_none_timestamp(self, patched_mlflow: MockMLflow) -> None:
        """Test logging trial with None timestamp."""
        tracker = mlflow_module.TraigentMLflowTracker()
        tracker.start_optimization_run(
            function_name="test",
            objectives=["accuracy"],
            configuration_space={},
        )

        # Create trial with None timestamp - use object.__setattr__ to bypass frozen dataclass
        trial = TrialResult(
            trial_id="trial_001",
            config={"model": "gpt-4o"},
            metrics={"accuracy": 0.85},
            status=TrialStatus.COMPLETED,
            duration=5.5,
            timestamp=datetime.now(UTC),
        )
        # Override timestamp to None
        object.__setattr__(trial, "timestamp", None)

        # Should not raise error
        with patch("builtins.open", create=True), patch("pathlib.Path.unlink"):
            result = tracker.log_trial(trial, trial_number=1)
            assert result is None  # Method returns None

    def test_optimization_with_minimize_objective(
        self, patched_mlflow: MockMLflow
    ) -> None:
        """Test get_best_run with minimize objective."""
        import pandas as pd

        tracker = mlflow_module.TraigentMLflowTracker()

        experiment = Mock()
        experiment.experiment_id = "exp_123"
        patched_mlflow.experiments["test_experiment"] = experiment

        best_run_data = {
            "run_id": ["run_123"],
            "tags.mlflow.runName": ["best_run"],
            "tags.traigent.function_name": ["test_func"],
            "metrics.best_latency": [0.05],
            "metrics.total_trials": [20.0],
            "metrics.optimization_duration": [100.0],
        }
        mock_df = pd.DataFrame(best_run_data)

        patched_mlflow.search_runs = lambda **kwargs: mock_df

        result = tracker.get_best_run("test_experiment", "latency", maximize=False)

        assert result is not None
        assert result["best_metric_value"] == 0.05
