"""Unit tests for Weights & Biases integration.

Tests for TraiGent W&B experiment tracking and observability.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Observability
# CONC-Quality-Compatibility FUNC-INTEGRATIONS FUNC-ANALYTICS
# REQ-INT-008 REQ-ANLY-011 SYNC-Observability

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from traigent.api.types import (
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)
from traigent.integrations.observability import wandb as wandb_module


class MockWandBRun:
    """Mock W&B run object."""

    def __init__(self, run_id: str = "mock_run_123") -> None:
        """Initialize mock run."""
        self.id = run_id
        self.name = f"run_{run_id}"
        self.config: dict[str, Any] = {}
        self.summary: dict[str, Any] = {}
        self.tags: list[str] = []


class MockWandB:
    """Mock W&B module."""

    def __init__(self) -> None:
        """Initialize mock wandb."""
        self.current_run: MockWandBRun | None = None
        self.logged_data: list[tuple[dict, int | None]] = []
        self.saved_files: list[str] = []
        self.sweeps: list[tuple[dict, str, str | None]] = []

    def init(
        self,
        project: str | None = None,
        entity: str | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
        config: dict[str, Any] | None = None,
        reinit: bool = False,
        **kwargs,
    ) -> MockWandBRun:
        """Initialize W&B run."""
        self.current_run = MockWandBRun()
        if config:
            self.current_run.config.update(config)
        if tags:
            self.current_run.tags = tags
        return self.current_run

    def log(self, data: dict, step: int | None = None) -> None:
        """Log metrics to W&B."""
        self.logged_data.append((data, step))

    def save(self, path: str) -> None:
        """Save file to W&B."""
        self.saved_files.append(path)

    def finish(self) -> None:
        """Finish current run."""
        self.current_run = None

    def sweep(self, config: dict, project: str, entity: str | None = None) -> str:
        """Create W&B sweep."""
        sweep_id = f"sweep_{len(self.sweeps)}"
        self.sweeps.append((config, project, entity))
        return sweep_id


@pytest.fixture
def mock_wandb() -> MockWandB:
    """Create mock W&B module."""
    return MockWandB()


@pytest.fixture
def mock_wandb_available(monkeypatch, mock_wandb: MockWandB) -> MockWandB:
    """Mock wandb as available and patch the module."""
    monkeypatch.setattr(wandb_module, "wandb", mock_wandb)
    monkeypatch.setattr(wandb_module, "WANDB_AVAILABLE", True)
    return mock_wandb


@pytest.fixture
def mock_wandb_unavailable(monkeypatch) -> None:
    """Mock wandb as unavailable."""
    monkeypatch.setattr(wandb_module, "WANDB_AVAILABLE", False)


@pytest.fixture
def sample_trial() -> TrialResult:
    """Create sample trial result."""
    return TrialResult(
        trial_id="trial_001",
        config={"model": "gpt-4o", "temperature": 0.7, "max_tokens": 100},
        metrics={"accuracy": 0.85, "latency": 1.5, "cost": 0.05},
        status=TrialStatus.COMPLETED,
        duration=10.5,
        timestamp=datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC),
    )


@pytest.fixture
def sample_optimization_result() -> OptimizationResult:
    """Create sample optimization result."""
    trials = [
        TrialResult(
            trial_id=f"trial_{i:03d}",
            config={"temperature": 0.5 + i * 0.1},
            metrics={"accuracy": 0.8 + i * 0.02},
            status=TrialStatus.COMPLETED,
            duration=10.0 + i,
            timestamp=datetime(2025, 1, 15, 12, i, 0, tzinfo=UTC),
        )
        for i in range(5)
    ]

    # Add a failed trial
    trials.append(
        TrialResult(
            trial_id="trial_failed",
            config={"temperature": 1.5},
            metrics={},
            status=TrialStatus.FAILED,
            duration=2.0,
            timestamp=datetime(2025, 1, 15, 12, 5, 0, tzinfo=UTC),
        )
    )

    return OptimizationResult(
        trials=trials,
        best_config={"temperature": 0.9},
        best_score=0.88,
        optimization_id="opt_001",
        duration=65.0,
        convergence_info={"converged": True, "iterations": 10},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="bayesian",
        timestamp=datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC),
        metadata={"optimizer": "bayesian", "version": "1.0"},
    )


class TestTraigentWandBTrackerInit:
    """Tests for TraigentWandBTracker initialization."""

    def test_init_with_defaults(self, mock_wandb_available: MockWandB) -> None:
        """Test tracker initialization with default parameters."""
        tracker = wandb_module.TraigentWandBTracker()

        assert tracker.project == "traigent-optimization"
        assert tracker.entity is None
        assert tracker.tags == []
        assert tracker.notes is None
        assert tracker.auto_log is True
        assert tracker.current_run is None

    def test_init_with_custom_params(self, mock_wandb_available: MockWandB) -> None:
        """Test tracker initialization with custom parameters."""
        tracker = wandb_module.TraigentWandBTracker(
            project="my-project",
            entity="my-team",
            tags=["experiment", "baseline"],
            notes="Initial test run",
            auto_log=False,
        )

        assert tracker.project == "my-project"
        assert tracker.entity == "my-team"
        assert tracker.tags == ["experiment", "baseline"]
        assert tracker.notes == "Initial test run"
        assert tracker.auto_log is False

    def test_init_raises_when_wandb_not_available(
        self, mock_wandb_unavailable: None
    ) -> None:
        """Test initialization raises ImportError when wandb is not available."""
        with pytest.raises(ImportError, match="wandb is not installed"):
            wandb_module.TraigentWandBTracker()


class TestTraigentWandBTrackerStartRun:
    """Tests for starting W&B optimization runs."""

    def test_start_run_with_defaults(
        self, mock_wandb_available: MockWandB, tmp_path: Path, monkeypatch
    ) -> None:
        """Test starting optimization run with default parameters."""
        monkeypatch.chdir(tmp_path)
        tracker = wandb_module.TraigentWandBTracker()

        run_id = tracker.start_optimization_run(
            function_name="test_function",
            objectives=["accuracy", "latency"],
            configuration_space={"temperature": [0.5, 1.0], "model": ["gpt-4o"]},
        )

        assert run_id == "mock_run_123"
        assert tracker.current_run is not None
        assert tracker.current_run.id == "mock_run_123"

        # Verify config was logged
        assert "traigent" in tracker.current_run.config
        traigent_config = tracker.current_run.config["traigent"]
        assert traigent_config["function_name"] == "test_function"
        assert traigent_config["objectives"] == ["accuracy", "latency"]
        assert traigent_config["num_objectives"] == 2

    def test_start_run_with_custom_name(
        self, mock_wandb_available: MockWandB, tmp_path: Path, monkeypatch
    ) -> None:
        """Test starting run with custom run name."""
        monkeypatch.chdir(tmp_path)
        tracker = wandb_module.TraigentWandBTracker()

        run_id = tracker.start_optimization_run(
            function_name="test_function",
            objectives=["accuracy"],
            configuration_space={},
            run_name="custom_run_name",
        )

        assert run_id == "mock_run_123"
        assert tracker.current_run is not None

    def test_start_run_with_additional_tags(
        self, mock_wandb_available: MockWandB, tmp_path: Path, monkeypatch
    ) -> None:
        """Test starting run with additional tags."""
        monkeypatch.chdir(tmp_path)
        tracker = wandb_module.TraigentWandBTracker(tags=["base"])

        run_id = tracker.start_optimization_run(
            function_name="test_function",
            objectives=["accuracy"],
            configuration_space={},
            additional_tags=["extra", "custom"],
        )

        assert run_id == "mock_run_123"
        # Tags should include base tags, additional tags, and auto tags
        assert "base" in tracker.current_run.tags
        assert "extra" in tracker.current_run.tags
        assert "traigent" in tracker.current_run.tags
        assert "test_function" in tracker.current_run.tags

    def test_start_run_with_config(
        self, mock_wandb_available: MockWandB, tmp_path: Path, monkeypatch
    ) -> None:
        """Test starting run with additional config."""
        monkeypatch.chdir(tmp_path)
        tracker = wandb_module.TraigentWandBTracker()

        custom_config = {"learning_rate": 0.001, "batch_size": 32}
        run_id = tracker.start_optimization_run(
            function_name="test_function",
            objectives=["accuracy"],
            configuration_space={},
            config=custom_config,
        )

        assert run_id == "mock_run_123"
        assert "learning_rate" in tracker.current_run.config
        assert tracker.current_run.config["learning_rate"] == 0.001

    def test_start_run_saves_config_artifact(
        self, mock_wandb_available: MockWandB, tmp_path: Path, monkeypatch
    ) -> None:
        """Test that configuration space is saved as artifact."""
        monkeypatch.chdir(tmp_path)
        tracker = wandb_module.TraigentWandBTracker()

        config_space = {"temperature": [0.5, 1.0], "model": ["gpt-4o"]}
        tracker.start_optimization_run(
            function_name="test_function",
            objectives=["accuracy"],
            configuration_space=config_space,
        )

        assert "configuration_space.json" in mock_wandb_available.saved_files

    def test_start_run_handles_artifact_save_error(
        self, mock_wandb_available: MockWandB, tmp_path: Path, monkeypatch
    ) -> None:
        """Test that artifact save errors are handled gracefully."""
        monkeypatch.chdir(tmp_path)

        def save_error(path: str) -> None:
            raise RuntimeError("Save failed")

        # Patch the save method
        monkeypatch.setattr(mock_wandb_available, "save", save_error)
        tracker = wandb_module.TraigentWandBTracker()

        # Should not raise, just log warning
        run_id = tracker.start_optimization_run(
            function_name="test_function",
            objectives=["accuracy"],
            configuration_space={},
        )

        assert run_id == "mock_run_123"


class TestTraigentWandBTrackerLogTrial:
    """Tests for logging individual trials."""

    def test_log_trial_with_valid_data(
        self, mock_wandb_available: MockWandB, sample_trial: TrialResult
    ) -> None:
        """Test logging trial with valid data."""
        tracker = wandb_module.TraigentWandBTracker()
        tracker.current_run = MockWandBRun()

        tracker.log_trial(sample_trial, trial_number=1)

        assert len(mock_wandb_available.logged_data) == 1
        logged_data, step = mock_wandb_available.logged_data[0]

        assert step == 1
        assert logged_data["trial_1/status"] == "completed"
        assert logged_data["trial_1/duration"] == 10.5
        assert logged_data["trial_1/config/model"] == "gpt-4o"
        assert logged_data["trial_1/metrics/accuracy"] == 0.85
        assert logged_data["metrics/accuracy"] == 0.85

    def test_log_trial_with_custom_step(
        self, mock_wandb_available: MockWandB, sample_trial: TrialResult
    ) -> None:
        """Test logging trial with custom step number."""
        tracker = wandb_module.TraigentWandBTracker()
        tracker.current_run = MockWandBRun()

        tracker.log_trial(sample_trial, trial_number=5, step=10)

        assert len(mock_wandb_available.logged_data) == 1
        _, step = mock_wandb_available.logged_data[0]
        assert step == 10

    def test_log_trial_without_active_run(
        self, mock_wandb_available: MockWandB, sample_trial: TrialResult
    ) -> None:
        """Test logging trial without active run logs warning."""
        tracker = wandb_module.TraigentWandBTracker()
        # No current_run set

        tracker.log_trial(sample_trial, trial_number=1)

        # Should not log anything
        assert len(mock_wandb_available.logged_data) == 0

    def test_log_trial_missing_trial_id(self, mock_wandb_available: MockWandB) -> None:
        """Test logging trial without trial_id."""
        tracker = wandb_module.TraigentWandBTracker()
        tracker.current_run = MockWandBRun()

        # Create trial without trial_id
        trial = Mock(spec=[])
        trial.trial_id = None

        tracker.log_trial(trial, trial_number=1)

        # Should not log
        assert len(mock_wandb_available.logged_data) == 0

    def test_log_trial_missing_status(self, mock_wandb_available: MockWandB) -> None:
        """Test logging trial without status."""
        tracker = wandb_module.TraigentWandBTracker()
        tracker.current_run = MockWandBRun()

        trial = Mock()
        trial.trial_id = "test_001"
        trial.status = None

        tracker.log_trial(trial, trial_number=1)

        # Should not log
        assert len(mock_wandb_available.logged_data) == 0

    def test_log_trial_missing_duration(self, mock_wandb_available: MockWandB) -> None:
        """Test logging trial with missing duration defaults to 0."""
        tracker = wandb_module.TraigentWandBTracker()
        tracker.current_run = MockWandBRun()

        trial = Mock()
        trial.trial_id = "test_001"
        trial.status = TrialStatus.COMPLETED
        trial.duration = None
        trial.config = {}
        trial.metrics = {}
        trial.timestamp = None

        tracker.log_trial(trial, trial_number=1)

        assert len(mock_wandb_available.logged_data) == 1
        logged_data, _ = mock_wandb_available.logged_data[0]
        assert logged_data["trial_1/duration"] == 0.0

    def test_log_trial_non_mapping_config(
        self, mock_wandb_available: MockWandB
    ) -> None:
        """Test logging trial with non-mapping config."""
        tracker = wandb_module.TraigentWandBTracker()
        tracker.current_run = MockWandBRun()

        trial = Mock()
        trial.trial_id = "test_001"
        trial.status = TrialStatus.COMPLETED
        trial.duration = 5.0
        trial.config = "not a mapping"
        trial.metrics = {}
        trial.timestamp = None

        tracker.log_trial(trial, trial_number=1)

        assert len(mock_wandb_available.logged_data) == 1

    def test_log_trial_non_mapping_metrics(
        self, mock_wandb_available: MockWandB
    ) -> None:
        """Test logging trial with non-mapping metrics."""
        tracker = wandb_module.TraigentWandBTracker()
        tracker.current_run = MockWandBRun()

        trial = Mock()
        trial.trial_id = "test_001"
        trial.status = TrialStatus.COMPLETED
        trial.duration = 5.0
        trial.config = {}
        trial.metrics = "not a mapping"
        trial.timestamp = None

        tracker.log_trial(trial, trial_number=1)

        assert len(mock_wandb_available.logged_data) == 1

    def test_log_trial_non_numeric_metric_skipped(
        self, mock_wandb_available: MockWandB
    ) -> None:
        """Test that non-numeric metrics are skipped."""
        tracker = wandb_module.TraigentWandBTracker()
        tracker.current_run = MockWandBRun()

        trial = Mock()
        trial.trial_id = "test_001"
        trial.status = TrialStatus.COMPLETED
        trial.duration = 5.0
        trial.config = {}
        trial.metrics = {"valid": 0.5, "invalid": "not a number"}
        trial.timestamp = None

        tracker.log_trial(trial, trial_number=1)

        logged_data, _ = mock_wandb_available.logged_data[0]
        assert "trial_1/metrics/valid" in logged_data
        assert "trial_1/metrics/invalid" not in logged_data

    def test_log_trial_complex_config_values(
        self, mock_wandb_available: MockWandB
    ) -> None:
        """Test logging trial with complex config values."""
        tracker = wandb_module.TraigentWandBTracker()
        tracker.current_run = MockWandBRun()

        trial = Mock()
        trial.trial_id = "test_001"
        trial.status = TrialStatus.COMPLETED
        trial.duration = 5.0
        trial.config = {
            "string": "value",
            "number": 42,
            "bool": True,
            "none": None,
            "complex": {"nested": "object"},
        }
        trial.metrics = {}
        trial.timestamp = None

        tracker.log_trial(trial, trial_number=1)

        logged_data, _ = mock_wandb_available.logged_data[0]
        assert logged_data["trial_1/config/string"] == "value"
        assert logged_data["trial_1/config/number"] == 42
        assert logged_data["trial_1/config/bool"] is True
        assert logged_data["trial_1/config/none"] is None
        assert logged_data["trial_1/config/complex"] == "{'nested': 'object'}"

    def test_log_trial_saves_artifact(
        self,
        mock_wandb_available: MockWandB,
        sample_trial: TrialResult,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        """Test that trial is saved as JSON artifact."""
        monkeypatch.chdir(tmp_path)
        tracker = wandb_module.TraigentWandBTracker()
        tracker.current_run = MockWandBRun()

        tracker.log_trial(sample_trial, trial_number=3)

        assert "trial_3.json" in mock_wandb_available.saved_files

    def test_log_trial_handles_artifact_write_error(
        self, mock_wandb_available: MockWandB, sample_trial: TrialResult, monkeypatch
    ) -> None:
        """Test that artifact write errors are handled gracefully."""
        # Mock open to raise error
        original_open = open

        def mock_open(*args, **kwargs):
            if "trial_" in str(args[0]):
                raise OSError("Write failed")
            return original_open(*args, **kwargs)

        monkeypatch.setattr("builtins.open", mock_open)

        tracker = wandb_module.TraigentWandBTracker()
        tracker.current_run = MockWandBRun()

        # Should not raise
        tracker.log_trial(sample_trial, trial_number=1)

    def test_log_trial_handles_log_error(
        self, mock_wandb_available: MockWandB, sample_trial: TrialResult, monkeypatch
    ) -> None:
        """Test that wandb.log errors are handled gracefully."""

        def log_error(data: dict, step: int | None = None) -> None:
            raise RuntimeError("Log failed")

        # Patch the log method
        monkeypatch.setattr(mock_wandb_available, "log", log_error)

        tracker = wandb_module.TraigentWandBTracker()
        tracker.current_run = MockWandBRun()

        # Should not raise
        tracker.log_trial(sample_trial, trial_number=1)


class TestTraigentWandBTrackerLogOptimizationResult:
    """Tests for logging complete optimization results."""

    def test_log_optimization_result_complete(
        self,
        mock_wandb_available: MockWandB,
        sample_optimization_result: OptimizationResult,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        """Test logging complete optimization result."""
        monkeypatch.chdir(tmp_path)
        tracker = wandb_module.TraigentWandBTracker()
        tracker.current_run = MockWandBRun()

        tracker.log_optimization_result(sample_optimization_result)

        assert len(mock_wandb_available.logged_data) == 1
        logged_data, _ = mock_wandb_available.logged_data[0]

        assert logged_data["optimization/total_trials"] == 6
        assert logged_data["optimization/duration"] == 65.0
        assert logged_data["optimization/status"] == "completed"
        assert logged_data["best_config/temperature"] == 0.9
        assert logged_data["best_metrics/accuracy"] == 0.88

    def test_log_optimization_result_calculates_statistics(
        self,
        mock_wandb_available: MockWandB,
        sample_optimization_result: OptimizationResult,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        """Test that optimization result calculates trial statistics."""
        monkeypatch.chdir(tmp_path)
        tracker = wandb_module.TraigentWandBTracker()
        tracker.current_run = MockWandBRun()

        tracker.log_optimization_result(sample_optimization_result)

        logged_data, _ = mock_wandb_available.logged_data[0]

        # 5 completed trials out of 6
        assert logged_data["optimization/successful_trials"] == 5
        assert logged_data["optimization/success_rate"] == 5 / 6

        # Should have statistics for metrics
        assert "statistics/accuracy_mean" in logged_data
        assert "statistics/accuracy_std" in logged_data
        assert "statistics/accuracy_min" in logged_data
        assert "statistics/accuracy_max" in logged_data

    def test_log_optimization_result_without_active_run(
        self,
        mock_wandb_available: MockWandB,
        sample_optimization_result: OptimizationResult,
    ) -> None:
        """Test logging result without active run logs warning."""
        tracker = wandb_module.TraigentWandBTracker()
        # No current_run set

        tracker.log_optimization_result(sample_optimization_result)

        # Should not log anything
        assert len(mock_wandb_available.logged_data) == 0

    def test_log_optimization_result_saves_artifact(
        self,
        mock_wandb_available: MockWandB,
        sample_optimization_result: OptimizationResult,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        """Test that result is saved as JSON artifact."""
        monkeypatch.chdir(tmp_path)
        tracker = wandb_module.TraigentWandBTracker()
        tracker.current_run = MockWandBRun()

        tracker.log_optimization_result(sample_optimization_result)

        assert "optimization_result.json" in mock_wandb_available.saved_files

    def test_log_optimization_result_with_dataset(
        self,
        mock_wandb_available: MockWandB,
        sample_optimization_result: OptimizationResult,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        """Test logging result with dataset path."""
        monkeypatch.chdir(tmp_path)

        # Create dummy dataset file
        dataset_path = tmp_path / "dataset.json"
        dataset_path.write_text("{}")

        tracker = wandb_module.TraigentWandBTracker()
        tracker.current_run = MockWandBRun()

        tracker.log_optimization_result(sample_optimization_result, str(dataset_path))

        assert str(dataset_path) in mock_wandb_available.saved_files

    def test_log_optimization_result_dataset_not_exists(
        self,
        mock_wandb_available: MockWandB,
        sample_optimization_result: OptimizationResult,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        """Test logging result with non-existent dataset path."""
        monkeypatch.chdir(tmp_path)
        tracker = wandb_module.TraigentWandBTracker()
        tracker.current_run = MockWandBRun()

        tracker.log_optimization_result(sample_optimization_result, "nonexistent.json")

        assert "nonexistent.json" not in mock_wandb_available.saved_files

    def test_log_optimization_result_no_trials(
        self, mock_wandb_available: MockWandB, tmp_path: Path, monkeypatch
    ) -> None:
        """Test logging result with no trials."""
        monkeypatch.chdir(tmp_path)
        result = OptimizationResult(
            trials=[],
            best_config={},
            best_score=0.0,
            optimization_id="opt_empty",
            duration=0.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy"],
            algorithm="test",
            timestamp=datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC),
            metadata={},
        )

        tracker = wandb_module.TraigentWandBTracker()
        tracker.current_run = MockWandBRun()

        tracker.log_optimization_result(result)

        logged_data, _ = mock_wandb_available.logged_data[0]
        assert logged_data["optimization/total_trials"] == 0


class TestTraigentWandBTrackerEndRun:
    """Tests for ending W&B runs."""

    def test_end_optimization_run(self, mock_wandb_available: MockWandB) -> None:
        """Test ending optimization run."""
        tracker = wandb_module.TraigentWandBTracker()
        tracker.current_run = MockWandBRun()

        tracker.end_optimization_run()

        assert tracker.current_run is None
        assert mock_wandb_available.current_run is None

    def test_end_optimization_run_no_active_run(
        self, mock_wandb_available: MockWandB
    ) -> None:
        """Test ending run when no run is active."""
        tracker = wandb_module.TraigentWandBTracker()

        # Should not raise
        tracker.end_optimization_run()


class TestTraigentWandBTrackerHyperparameterSweep:
    """Tests for hyperparameter sweep functionality."""

    def test_log_hyperparameter_sweep(self, mock_wandb_available: MockWandB) -> None:
        """Test creating hyperparameter sweep."""
        tracker = wandb_module.TraigentWandBTracker(
            project="my-project", entity="my-team"
        )

        sweep_config = {
            "method": "bayes",
            "parameters": {"temperature": {"min": 0.5, "max": 1.0}},
        }

        sweep_id = tracker.log_hyperparameter_sweep(sweep_config, "test_function")

        assert sweep_id == "sweep_0"
        assert len(mock_wandb_available.sweeps) == 1
        config, project, entity = mock_wandb_available.sweeps[0]
        assert config["metadata"]["traigent_function"] == "test_function"
        assert config["metadata"]["framework"] == "traigent"

    def test_log_hyperparameter_sweep_unavailable(
        self, mock_wandb_unavailable: None
    ) -> None:
        """Test sweep raises error when wandb unavailable."""
        # First create tracker when available
        with patch.object(wandb_module, "WANDB_AVAILABLE", True):
            with patch.object(wandb_module, "wandb", MockWandB()):
                tracker = wandb_module.TraigentWandBTracker()

        # Now make it unavailable
        with patch.object(wandb_module, "WANDB_AVAILABLE", False):
            with pytest.raises(ImportError, match="wandb not available"):
                tracker.log_hyperparameter_sweep({}, "test_function")


class TestWandBOptimizationCallback:
    """Tests for WandBOptimizationCallback."""

    def test_callback_init(self, mock_wandb_available: MockWandB) -> None:
        """Test callback initialization."""
        tracker = wandb_module.TraigentWandBTracker()
        callback = wandb_module.WandBOptimizationCallback(tracker)

        assert callback.tracker is tracker
        assert callback.trial_count == 0

    def test_on_optimization_start(
        self, mock_wandb_available: MockWandB, tmp_path: Path, monkeypatch
    ) -> None:
        """Test optimization start callback."""
        monkeypatch.chdir(tmp_path)
        tracker = wandb_module.TraigentWandBTracker()
        callback = wandb_module.WandBOptimizationCallback(tracker)

        callback.on_optimization_start(
            function_name="test_function",
            objectives=["accuracy"],
            configuration_space={"temp": [0.5, 1.0]},
            wandb_tags=["custom"],
            wandb_config={"extra": "value"},
        )

        assert tracker.current_run is not None

    def test_on_trial_complete(
        self, mock_wandb_available: MockWandB, sample_trial: TrialResult
    ) -> None:
        """Test trial complete callback."""
        tracker = wandb_module.TraigentWandBTracker()
        tracker.current_run = MockWandBRun()
        callback = wandb_module.WandBOptimizationCallback(tracker)

        callback.on_trial_complete(sample_trial)

        assert callback.trial_count == 1
        assert len(mock_wandb_available.logged_data) == 1

    def test_on_optimization_complete(
        self,
        mock_wandb_available: MockWandB,
        sample_optimization_result: OptimizationResult,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        """Test optimization complete callback."""
        monkeypatch.chdir(tmp_path)
        tracker = wandb_module.TraigentWandBTracker()
        tracker.current_run = MockWandBRun()
        callback = wandb_module.WandBOptimizationCallback(tracker)

        callback.on_optimization_complete(sample_optimization_result)

        assert tracker.current_run is None


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_wandb_tracker(self, mock_wandb_available: MockWandB) -> None:
        """Test create_wandb_tracker function."""
        tracker = wandb_module.create_wandb_tracker(
            project="test-project",
            entity="test-entity",
            tags=["tag1", "tag2"],
            notes="Test notes",
            auto_log=False,
        )

        assert isinstance(tracker, wandb_module.TraigentWandBTracker)
        assert tracker.project == "test-project"
        assert tracker.entity == "test-entity"
        assert tracker.tags == ["tag1", "tag2"]
        assert tracker.notes == "Test notes"
        assert tracker.auto_log is False

    def test_enable_wandb_autolog(self, mock_wandb_available: MockWandB) -> None:
        """Test enable_wandb_autolog function."""
        callback = wandb_module.enable_wandb_autolog(
            project="test-project", entity="test-entity", tags=["test"]
        )

        assert isinstance(callback, wandb_module.WandBOptimizationCallback)
        assert callback.tracker.project == "test-project"

    def test_log_traigent_optimization(
        self,
        mock_wandb_available: MockWandB,
        sample_optimization_result: OptimizationResult,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        """Test log_traigent_optimization function."""
        monkeypatch.chdir(tmp_path)

        run_id = wandb_module.log_traigent_optimization(
            result=sample_optimization_result,
            function_name="test_function",
            objectives=["accuracy"],
            configuration_space={"temp": [0.5, 1.0]},
            project="test-project",
            run_name="test-run",
        )

        assert run_id == "mock_run_123"
        # Should have logged all trials + summary
        assert len(mock_wandb_available.logged_data) > len(
            sample_optimization_result.trials
        )

    def test_create_wandb_sweep_config_categorical(
        self, mock_wandb_available: MockWandB
    ) -> None:
        """Test creating sweep config with categorical parameters."""
        config = wandb_module.create_wandb_sweep_config(
            function_name="test_function",
            configuration_space={"model": ["gpt-4o", "gpt-4-turbo"]},
            objectives=["accuracy"],
            method="grid",
        )

        assert config["method"] == "grid"
        assert config["parameters"]["model"]["values"] == ["gpt-4o", "gpt-4-turbo"]
        assert config["name"] == "test_function_sweep"

    def test_create_wandb_sweep_config_continuous(
        self, mock_wandb_available: MockWandB
    ) -> None:
        """Test creating sweep config with continuous parameters."""
        config = wandb_module.create_wandb_sweep_config(
            function_name="test_function",
            configuration_space={"temperature": (0.5, 1.0), "top_p": (0.8, 1.0)},
            objectives=["accuracy", "latency"],
        )

        assert config["parameters"]["temperature"]["min"] == 0.5
        assert config["parameters"]["temperature"]["max"] == 1.0
        assert config["metric"]["name"] == "best_metrics/accuracy"

    def test_create_wandb_sweep_config_constant(
        self, mock_wandb_available: MockWandB
    ) -> None:
        """Test creating sweep config with constant parameters."""
        config = wandb_module.create_wandb_sweep_config(
            function_name="test_function",
            configuration_space={"model": "gpt-4o", "max_tokens": 100},
            objectives=["accuracy"],
        )

        assert config["parameters"]["model"]["value"] == "gpt-4o"
        assert config["parameters"]["max_tokens"]["value"] == 100

    def test_init_wandb_run_valid_inputs(
        self, mock_wandb_available: MockWandB, tmp_path: Path, monkeypatch
    ) -> None:
        """Test init_wandb_run with valid inputs."""
        monkeypatch.chdir(tmp_path)

        run_id = wandb_module.init_wandb_run(
            function_name="test_function",
            objectives=["accuracy", "latency"],
            configuration_space={"temp": [0.5, 1.0]},
            run_name="custom-run",
            project="test-project",
        )

        assert run_id == "mock_run_123"

    def test_init_wandb_run_invalid_function_name(
        self, mock_wandb_available: MockWandB
    ) -> None:
        """Test init_wandb_run with invalid function name."""
        with pytest.raises(
            ValueError, match="function_name must be a non-empty string"
        ):
            wandb_module.init_wandb_run(
                function_name="",
                objectives=["accuracy"],
                configuration_space={},
            )

    def test_init_wandb_run_invalid_objectives(
        self, mock_wandb_available: MockWandB
    ) -> None:
        """Test init_wandb_run with invalid objectives."""
        with pytest.raises(TypeError, match="objectives must be a sequence"):
            wandb_module.init_wandb_run(
                function_name="test",
                objectives="not a sequence",
                configuration_space={},
            )

    def test_init_wandb_run_empty_objectives(
        self, mock_wandb_available: MockWandB
    ) -> None:
        """Test init_wandb_run with empty objectives."""
        with pytest.raises(
            ValueError, match="objectives must contain at least one item"
        ):
            wandb_module.init_wandb_run(
                function_name="test",
                objectives=[],
                configuration_space={},
            )

    def test_init_wandb_run_invalid_config_space(
        self, mock_wandb_available: MockWandB
    ) -> None:
        """Test init_wandb_run with invalid configuration space."""
        with pytest.raises(TypeError, match="configuration_space must be a mapping"):
            wandb_module.init_wandb_run(
                function_name="test",
                objectives=["accuracy"],
                configuration_space="not a mapping",
            )

    def test_log_trial_to_wandb_alias(
        self, mock_wandb_available: MockWandB, sample_trial: TrialResult
    ) -> None:
        """Test log_trial_to_wandb alias function."""
        tracker = wandb_module.TraigentWandBTracker()
        tracker.current_run = MockWandBRun()

        wandb_module.log_trial_to_wandb(tracker, sample_trial, 1)

        assert len(mock_wandb_available.logged_data) == 1

    def test_log_final_results_to_wandb_alias(
        self,
        mock_wandb_available: MockWandB,
        sample_optimization_result: OptimizationResult,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        """Test log_final_results_to_wandb alias function."""
        monkeypatch.chdir(tmp_path)
        tracker = wandb_module.TraigentWandBTracker()
        tracker.current_run = MockWandBRun()

        wandb_module.log_final_results_to_wandb(tracker, sample_optimization_result)

        assert len(mock_wandb_available.logged_data) == 1

    def test_log_optimization_to_wandb_alias(
        self,
        mock_wandb_available: MockWandB,
        sample_optimization_result: OptimizationResult,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        """Test log_optimization_to_wandb alias."""
        monkeypatch.chdir(tmp_path)

        run_id = wandb_module.log_optimization_to_wandb(
            result=sample_optimization_result,
            function_name="test",
            objectives=["accuracy"],
            configuration_space={},
        )

        assert run_id == "mock_run_123"

    def test_create_optimization_report_alias(
        self, mock_wandb_available: MockWandB
    ) -> None:
        """Test create_optimization_report alias."""
        config = wandb_module.create_optimization_report(
            function_name="test",
            configuration_space={"temp": [0.5, 1.0]},
            objectives=["accuracy"],
        )

        assert config["name"] == "test_sweep"

    def test_wandb_integration_alias(self, mock_wandb_available: MockWandB) -> None:
        """Test WandBIntegration alias."""
        tracker = wandb_module.WandBIntegration()
        assert isinstance(tracker, wandb_module.TraigentWandBTracker)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_trial_with_enum_status(self, mock_wandb_available: MockWandB) -> None:
        """Test trial logging with enum status."""
        tracker = wandb_module.TraigentWandBTracker()
        tracker.current_run = MockWandBRun()

        trial = Mock()
        trial.trial_id = "test_001"
        trial.status = TrialStatus.RUNNING  # Enum with .value
        trial.duration = 5.0
        trial.config = {}
        trial.metrics = {}
        trial.timestamp = None

        tracker.log_trial(trial, trial_number=1)

        logged_data, _ = mock_wandb_available.logged_data[0]
        assert logged_data["trial_1/status"] == "running"

    def test_trial_with_timestamp_isoformat(
        self, mock_wandb_available: MockWandB, tmp_path: Path, monkeypatch
    ) -> None:
        """Test trial artifact includes timestamp in ISO format."""
        monkeypatch.chdir(tmp_path)
        tracker = wandb_module.TraigentWandBTracker()
        tracker.current_run = MockWandBRun()

        timestamp = datetime(2025, 1, 15, 12, 30, 0, tzinfo=UTC)
        trial = Mock()
        trial.trial_id = "test_001"
        trial.status = TrialStatus.COMPLETED
        trial.duration = 5.0
        trial.config = {}
        trial.metrics = {}
        trial.timestamp = timestamp

        tracker.log_trial(trial, trial_number=1)

        # Check artifact was saved (file is deleted after save)
        assert "trial_1.json" in mock_wandb_available.saved_files

    def test_trial_with_no_isoformat_method(
        self, mock_wandb_available: MockWandB, tmp_path: Path, monkeypatch
    ) -> None:
        """Test trial with timestamp that has no isoformat method."""
        monkeypatch.chdir(tmp_path)
        tracker = wandb_module.TraigentWandBTracker()
        tracker.current_run = MockWandBRun()

        trial = Mock()
        trial.trial_id = "test_001"
        trial.status = TrialStatus.COMPLETED
        trial.duration = 5.0
        trial.config = {}
        trial.metrics = {}
        trial.timestamp = "not a datetime"  # String, no isoformat

        tracker.log_trial(trial, trial_number=1)

        assert len(mock_wandb_available.logged_data) == 1

    def test_optimization_result_with_metadata(
        self, mock_wandb_available: MockWandB, tmp_path: Path, monkeypatch
    ) -> None:
        """Test logging optimization result with metadata."""
        monkeypatch.chdir(tmp_path)
        result = OptimizationResult(
            trials=[],
            best_config={},
            best_score=0.0,
            optimization_id="opt_meta",
            duration=10.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy"],
            algorithm="test",
            timestamp=datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC),
            metadata={"key1": "value1", "key2": 42},
        )

        tracker = wandb_module.TraigentWandBTracker()
        tracker.current_run = MockWandBRun()

        tracker.log_optimization_result(result)

        logged_data, _ = mock_wandb_available.logged_data[0]
        assert logged_data["metadata/key1"] == "value1"
        assert logged_data["metadata/key2"] == 42

    def test_optimization_result_no_best_config(
        self, mock_wandb_available: MockWandB, tmp_path: Path, monkeypatch
    ) -> None:
        """Test logging result with no best config."""
        monkeypatch.chdir(tmp_path)
        result = OptimizationResult(
            trials=[],
            best_config={},
            best_score=0.0,
            optimization_id="opt_failed",
            duration=0.0,
            convergence_info={},
            status=OptimizationStatus.FAILED,
            objectives=["accuracy"],
            algorithm="test",
            timestamp=datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC),
            metadata={},
        )

        tracker = wandb_module.TraigentWandBTracker()
        tracker.current_run = MockWandBRun()

        tracker.log_optimization_result(result)

        logged_data, _ = mock_wandb_available.logged_data[0]
        assert "best_config" not in str(logged_data)

    def test_sweep_config_integer_range(self, mock_wandb_available: MockWandB) -> None:
        """Test sweep config with integer ranges."""
        config = wandb_module.create_wandb_sweep_config(
            function_name="test",
            configuration_space={"batch_size": (16, 128)},
            objectives=["accuracy"],
        )

        assert config["parameters"]["batch_size"]["min"] == 16
        assert config["parameters"]["batch_size"]["max"] == 128

    def test_wandb_not_available_mock_module(
        self, mock_wandb_unavailable: None
    ) -> None:
        """Test that mock wandb module methods exist when unavailable."""
        # The mock wandb class should still exist
        assert hasattr(wandb_module, "wandb")
        assert hasattr(wandb_module.wandb, "init")
        assert hasattr(wandb_module.wandb, "log")
        assert hasattr(wandb_module.wandb, "finish")
        assert hasattr(wandb_module.wandb, "save")
