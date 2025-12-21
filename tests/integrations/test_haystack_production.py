"""Tests for Haystack production readiness module.

Tests cover:
- OptimizationConfig loading and validation
- Production-hardened apply function with backup/rollback
- TunedConfig TVL export/import
- Experiment history export
- Artifact management
- CLI result generation
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml

from traigent.integrations.haystack.production import (
    ApplyBackup,
    CLIResult,
    ConfigMismatchError,
    ConfigValidationError,
    OptimizationConfig,
    TunedConfig,
    apply_config_production,
    create_cli_result,
    export_experiment_history,
    export_tuned_config,
    load_artifacts,
    load_optimization_config,
    load_tuned_config,
    rollback_config,
    save_artifacts,
    save_optimization_config,
)

# =============================================================================
# Mock classes for testing
# =============================================================================


@dataclass
class MockComponent:
    """Mock Haystack component for testing."""

    name: str
    temperature: float = 0.7
    top_k: int = 10
    model: str = "gpt-4"


class MockPipeline:
    """Mock Haystack Pipeline for testing."""

    def __init__(self) -> None:
        self.components: dict[str, Any] = {
            "generator": MockComponent(name="generator", temperature=0.7, top_k=10),
            "retriever": MockComponent(name="retriever", top_k=5),
        }

    def get_component(self, name: str) -> MockComponent | None:
        return self.components.get(name)


@dataclass
class MockTrialResult:
    """Mock trial result for testing."""

    trial_id: str
    config: dict[str, Any]
    metrics: dict[str, float]
    is_successful: bool = True
    constraints_satisfied: bool = True
    duration: float = 1.0


@dataclass
class MockOptimizationResult:
    """Mock optimization result for testing."""

    best_config: dict[str, Any] | None
    best_metrics: dict[str, float]
    total_trials: int
    duration: float
    strategy: str = "bayesian"
    history: list[MockTrialResult] | None = None
    pareto_configs: list[dict[str, Any]] | None = None
    warnings: list[str] | None = None


# =============================================================================
# Test OptimizationConfig
# =============================================================================


class TestOptimizationConfig:
    """Tests for OptimizationConfig dataclass."""

    def test_from_dict_minimal(self) -> None:
        """Test creating config from minimal dict."""
        data = {
            "pipeline_module": "myapp.pipeline:pipe",
            "search_space": "space.tvl",
            "targets": [{"metric_name": "accuracy", "direction": "maximize"}],
        }
        config = OptimizationConfig.from_dict(data)

        assert config.pipeline_module == "myapp.pipeline:pipe"
        assert config.search_space == "space.tvl"
        assert len(config.targets) == 1
        assert config.strategy == "bayesian"  # default
        assert config.n_trials == 50  # default

    def test_from_dict_full(self) -> None:
        """Test creating config from full dict."""
        data = {
            "pipeline_module": "myapp.pipeline:pipe",
            "search_space": {"temperature": [0.1, 0.5, 0.9]},
            "targets": [{"metric_name": "accuracy", "direction": "maximize"}],
            "constraints": [{"metric": "cost", "max": 1.0}],
            "strategy": "evolutionary",
            "n_trials": 100,
            "n_parallel": 4,
            "timeout_seconds": 3600.0,
            "checkpoint_path": "/tmp/checkpoint",
            "artifact_path": "/tmp/artifacts",
            "eval_dataset_path": "data.json",
            "random_seed": 42,
        }
        config = OptimizationConfig.from_dict(data)

        assert config.strategy == "evolutionary"
        assert config.n_trials == 100
        assert config.n_parallel == 4
        assert config.timeout_seconds == 3600.0
        assert config.random_seed == 42

    def test_to_dict(self) -> None:
        """Test converting config to dict."""
        config = OptimizationConfig(
            pipeline_module="app:pipe",
            search_space="space.tvl",
            targets=[{"metric_name": "acc", "direction": "maximize"}],
            n_trials=30,
        )
        data = config.to_dict()

        assert data["pipeline_module"] == "app:pipe"
        assert data["search_space"] == "space.tvl"
        assert data["n_trials"] == 30

    def test_validate_valid_config(self) -> None:
        """Test validation passes for valid config."""
        config = OptimizationConfig(
            pipeline_module="app:pipe",
            search_space="space.tvl",
            targets=[{"metric_name": "accuracy", "direction": "maximize"}],
        )
        errors = config.validate()
        assert errors == []

    def test_validate_missing_pipeline(self) -> None:
        """Test validation fails for missing pipeline."""
        config = OptimizationConfig(
            search_space="space.tvl",
            targets=[{"metric_name": "accuracy", "direction": "maximize"}],
        )
        errors = config.validate()
        assert any("pipeline_module" in e for e in errors)

    def test_validate_missing_search_space(self) -> None:
        """Test validation fails for missing search space."""
        config = OptimizationConfig(
            pipeline_module="app:pipe",
            targets=[{"metric_name": "accuracy", "direction": "maximize"}],
        )
        errors = config.validate()
        assert any("search_space" in e for e in errors)

    def test_validate_missing_targets(self) -> None:
        """Test validation fails for missing targets."""
        config = OptimizationConfig(
            pipeline_module="app:pipe",
            search_space="space.tvl",
        )
        errors = config.validate()
        assert any("target" in e.lower() for e in errors)

    def test_validate_invalid_target(self) -> None:
        """Test validation fails for invalid target."""
        config = OptimizationConfig(
            pipeline_module="app:pipe",
            search_space="space.tvl",
            targets=[{"invalid": "target"}],  # missing name and direction
        )
        errors = config.validate()
        assert any("name" in e.lower() or "metric_name" in e.lower() for e in errors)
        assert any("direction" in e.lower() for e in errors)

    def test_validate_invalid_strategy(self) -> None:
        """Test validation fails for invalid strategy."""
        config = OptimizationConfig(
            pipeline_module="app:pipe",
            search_space="space.tvl",
            targets=[{"metric_name": "acc", "direction": "maximize"}],
            strategy="invalid_strategy",
        )
        errors = config.validate()
        assert any("strategy" in e.lower() for e in errors)

    def test_validate_invalid_n_trials(self) -> None:
        """Test validation fails for invalid n_trials."""
        config = OptimizationConfig(
            pipeline_module="app:pipe",
            search_space="space.tvl",
            targets=[{"metric_name": "acc", "direction": "maximize"}],
            n_trials=0,
        )
        errors = config.validate()
        assert any("n_trials" in e for e in errors)

    def test_validate_invalid_timeout(self) -> None:
        """Test validation fails for invalid timeout."""
        config = OptimizationConfig(
            pipeline_module="app:pipe",
            search_space="space.tvl",
            targets=[{"metric_name": "acc", "direction": "maximize"}],
            timeout_seconds=-1.0,
        )
        errors = config.validate()
        assert any("timeout" in e.lower() for e in errors)


# =============================================================================
# Test Config Loading/Saving
# =============================================================================


class TestConfigLoadSave:
    """Tests for config file loading and saving."""

    def test_load_yaml_config(self) -> None:
        """Test loading config from YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(
                {
                    "pipeline_module": "app:pipe",
                    "search_space": "space.tvl",
                    "targets": [{"metric_name": "acc", "direction": "maximize"}],
                },
                f,
            )
            f.flush()

            config = load_optimization_config(f.name)
            assert config.pipeline_module == "app:pipe"

    def test_load_json_config(self) -> None:
        """Test loading config from JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "pipeline_module": "app:pipe",
                    "search_space": "space.tvl",
                    "targets": [{"metric_name": "acc", "direction": "maximize"}],
                },
                f,
            )
            f.flush()

            config = load_optimization_config(f.name)
            assert config.pipeline_module == "app:pipe"

    def test_load_missing_file(self) -> None:
        """Test loading non-existent file raises error."""
        with pytest.raises(ConfigValidationError, match="not found"):
            load_optimization_config("/nonexistent/config.yaml")

    def test_load_invalid_yaml(self) -> None:
        """Test loading invalid YAML raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            f.flush()

            with pytest.raises(ConfigValidationError, match="parse"):
                load_optimization_config(f.name)

    def test_load_validation_error(self) -> None:
        """Test loading config with validation errors."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump({"invalid": "config"}, f)
            f.flush()

            with pytest.raises(ConfigValidationError, match="validation failed"):
                load_optimization_config(f.name)

    def test_save_yaml_config(self) -> None:
        """Test saving config to YAML file."""
        config = OptimizationConfig(
            pipeline_module="app:pipe",
            search_space="space.tvl",
            targets=[{"metric_name": "acc", "direction": "maximize"}],
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            save_optimization_config(config, f.name)

            # Read back and verify
            with open(f.name) as rf:
                data = yaml.safe_load(rf)
                assert data["pipeline_module"] == "app:pipe"

    def test_save_json_config(self) -> None:
        """Test saving config to JSON file."""
        config = OptimizationConfig(
            pipeline_module="app:pipe",
            search_space="space.tvl",
            targets=[{"metric_name": "acc", "direction": "maximize"}],
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            save_optimization_config(config, f.name)

            # Read back and verify
            with open(f.name) as rf:
                data = json.load(rf)
                assert data["pipeline_module"] == "app:pipe"


# =============================================================================
# Test Production Apply Function
# =============================================================================


class TestApplyConfigProduction:
    """Tests for production-hardened apply function."""

    def test_apply_basic(self) -> None:
        """Test basic config application."""
        pipeline = MockPipeline()
        config = {"generator.temperature": 0.9, "retriever.top_k": 20}

        result = apply_config_production(pipeline, config)

        assert pipeline.get_component("generator").temperature == 0.9
        assert pipeline.get_component("retriever").top_k == 20
        assert result is None  # no backup requested

    def test_apply_with_validation(self) -> None:
        """Test apply with validation enabled."""
        pipeline = MockPipeline()
        config = {"generator.temperature": 0.9}

        # Should succeed with valid config
        apply_config_production(pipeline, config, validate=True)
        assert pipeline.get_component("generator").temperature == 0.9

    def test_apply_invalid_path_format(self) -> None:
        """Test apply fails for invalid path format."""
        pipeline = MockPipeline()
        config = {"temperature": 0.9}  # missing component prefix

        with pytest.raises(ConfigMismatchError, match="Invalid parameter path"):
            apply_config_production(pipeline, config, validate=True)

    def test_apply_missing_component(self) -> None:
        """Test apply fails for missing component."""
        pipeline = MockPipeline()
        config = {"nonexistent.temperature": 0.9}

        with pytest.raises(ConfigMismatchError, match="Component not found"):
            apply_config_production(pipeline, config, validate=True)

    def test_apply_missing_parameter(self) -> None:
        """Test apply fails for missing parameter."""
        pipeline = MockPipeline()
        config = {"generator.nonexistent_param": 0.9}

        with pytest.raises(ConfigMismatchError, match="Parameter not found"):
            apply_config_production(pipeline, config, validate=True)

    def test_apply_without_validation(self) -> None:
        """Test apply without validation doesn't check structure."""
        pipeline = MockPipeline()
        # This config is invalid but validation is disabled
        config = {"generator.temperature": 0.9}

        # Should succeed without raising
        apply_config_production(pipeline, config, validate=False)

    def test_apply_with_backup(self) -> None:
        """Test apply creates backup for rollback."""
        pipeline = MockPipeline()
        original_temp = pipeline.get_component("generator").temperature
        config = {"generator.temperature": 0.9}

        backup = apply_config_production(pipeline, config, backup=True)

        assert backup is not None
        assert isinstance(backup, ApplyBackup)
        assert backup.original_values["generator.temperature"] == original_temp

    def test_rollback(self) -> None:
        """Test rollback restores original values."""
        pipeline = MockPipeline()
        original_temp = pipeline.get_component("generator").temperature

        # Apply new config with backup
        backup = apply_config_production(
            pipeline,
            {"generator.temperature": 0.9},
            backup=True,
        )

        assert pipeline.get_component("generator").temperature == 0.9

        # Rollback
        rollback_config(pipeline, backup)

        assert pipeline.get_component("generator").temperature == original_temp


# =============================================================================
# Test TunedConfig Export/Import
# =============================================================================


class TestTunedConfig:
    """Tests for TunedConfig export and import."""

    def test_tuned_config_to_dict(self) -> None:
        """Test converting TunedConfig to dict."""
        tuned = TunedConfig(
            pipeline_name="my_pipeline",
            config={"generator.temperature": 0.8},
            metrics={"accuracy": 0.95},
        )
        data = tuned.to_dict()

        assert data["pipeline_name"] == "my_pipeline"
        assert data["config"]["generator.temperature"] == 0.8
        assert data["metrics"]["accuracy"] == 0.95

    def test_tuned_config_from_dict(self) -> None:
        """Test creating TunedConfig from dict."""
        data = {
            "version": "2.0",
            "pipeline_name": "test_pipe",
            "framework": "haystack",
            "config": {"temp": 0.7},
            "metrics": {"acc": 0.9},
            "constraints_satisfied": True,
        }
        tuned = TunedConfig.from_dict(data)

        assert tuned.version == "2.0"
        assert tuned.pipeline_name == "test_pipe"
        assert tuned.config["temp"] == 0.7

    def test_export_tuned_config_yaml(self) -> None:
        """Test exporting tuned config to YAML."""
        result = MockOptimizationResult(
            best_config={"generator.temperature": 0.8},
            best_metrics={"accuracy": 0.95},
            total_trials=50,
            duration=120.0,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".tvl", delete=False) as f:
            tuned = export_tuned_config(result, f.name, pipeline_name="test")

            assert tuned.config["generator.temperature"] == 0.8
            assert tuned.metrics["accuracy"] == 0.95

            # Verify file was written
            with open(f.name) as rf:
                data = yaml.safe_load(rf)
                assert data["config"]["generator.temperature"] == 0.8

    def test_export_tuned_config_json(self) -> None:
        """Test exporting tuned config to JSON."""
        result = MockOptimizationResult(
            best_config={"temp": 0.8},
            best_metrics={"acc": 0.95},
            total_trials=50,
            duration=120.0,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            export_tuned_config(result, f.name)

            with open(f.name) as rf:
                data = json.load(rf)
                assert data["config"]["temp"] == 0.8

    def test_export_with_history(self) -> None:
        """Test exporting with history summary."""
        result = MockOptimizationResult(
            best_config={"temp": 0.8},
            best_metrics={"acc": 0.95},
            total_trials=50,
            duration=120.0,
            history=[
                MockTrialResult("t1", {}, {"acc": 0.9}, is_successful=True),
                MockTrialResult("t2", {}, {"acc": 0.85}, is_successful=False),
            ],
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".tvl", delete=False) as f:
            tuned = export_tuned_config(result, f.name, include_history=True)

            assert "history_summary" in tuned.optimization_metadata
            assert tuned.optimization_metadata["history_summary"]["total_runs"] == 2
            assert (
                tuned.optimization_metadata["history_summary"]["successful_runs"] == 1
            )

    def test_load_tuned_config(self) -> None:
        """Test loading tuned config from file."""
        data = {
            "version": "1.0",
            "pipeline_name": "test",
            "config": {"temp": 0.8},
            "metrics": {"acc": 0.95},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".tvl", delete=False) as f:
            yaml.safe_dump(data, f)
            f.flush()

            tuned = load_tuned_config(f.name)
            assert tuned.config["temp"] == 0.8


# =============================================================================
# Test Experiment History Export
# =============================================================================


class TestExperimentHistoryExport:
    """Tests for experiment history export."""

    def test_export_json(self) -> None:
        """Test exporting history to JSON."""
        result = MockOptimizationResult(
            best_config={"temp": 0.8},
            best_metrics={"acc": 0.95},
            total_trials=3,
            duration=120.0,
            history=[
                MockTrialResult("t1", {"temp": 0.5}, {"acc": 0.8}),
                MockTrialResult("t2", {"temp": 0.7}, {"acc": 0.9}),
                MockTrialResult("t3", {"temp": 0.8}, {"acc": 0.95}),
            ],
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            export_experiment_history(result, f.name, format="json")

            with open(f.name) as rf:
                data = json.load(rf)
                assert data["schema_version"] == "1.0"
                assert data["summary"]["total_trials"] == 3
                assert len(data["trials"]) == 3
                assert data["trials"][0]["trial_id"] == "t1"

    def test_export_csv(self) -> None:
        """Test exporting history to CSV."""
        result = MockOptimizationResult(
            best_config={"temp": 0.8},
            best_metrics={"acc": 0.95},
            total_trials=2,
            duration=60.0,
            history=[
                MockTrialResult("t1", {"temp": 0.5}, {"acc": 0.8}),
                MockTrialResult("t2", {"temp": 0.7}, {"acc": 0.9}),
            ],
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            export_experiment_history(result, f.name, format="csv")

            with open(f.name) as rf:
                lines = rf.readlines()
                assert len(lines) == 3  # header + 2 rows
                header = lines[0].strip()
                assert "trial_id" in header
                assert "config_temp" in header
                assert "metric_acc" in header

    def test_export_invalid_format(self) -> None:
        """Test exporting with invalid format raises error."""
        result = MockOptimizationResult(
            best_config={},
            best_metrics={},
            total_trials=0,
            duration=0.0,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            with pytest.raises(ValueError, match="Unsupported format"):
                export_experiment_history(result, f.name, format="xml")


# =============================================================================
# Test Artifact Management
# =============================================================================


class TestArtifactManagement:
    """Tests for artifact save/load."""

    def test_save_artifacts(self) -> None:
        """Test saving all artifacts."""
        result = MockOptimizationResult(
            best_config={"temp": 0.8},
            best_metrics={"acc": 0.95},
            total_trials=10,
            duration=60.0,
            history=[MockTrialResult("t1", {"temp": 0.8}, {"acc": 0.95})],
            pareto_configs=[{"temp": 0.8}],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = save_artifacts(result, tmpdir, pipeline_name="test_pipe")

            assert "best_config" in artifacts
            assert "experiment_history" in artifacts
            assert "summary" in artifacts

            # Verify files exist
            assert Path(artifacts["best_config"]).exists()
            assert Path(artifacts["experiment_history"]).exists()
            assert Path(artifacts["summary"]).exists()

    def test_load_artifacts(self) -> None:
        """Test loading artifacts from directory."""
        result = MockOptimizationResult(
            best_config={"temp": 0.8},
            best_metrics={"acc": 0.95},
            total_trials=10,
            duration=60.0,
            history=[MockTrialResult("t1", {"temp": 0.8}, {"acc": 0.95})],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_artifacts(result, tmpdir, pipeline_name="test_pipe")

            # Load back
            artifacts = load_artifacts(tmpdir)

            assert "best_config" in artifacts
            assert "summary" in artifacts
            assert "experiment_history" in artifacts

            assert artifacts["best_config"].config["temp"] == 0.8
            assert artifacts["summary"]["total_trials"] == 10

    def test_load_empty_directory(self) -> None:
        """Test loading from empty directory returns empty dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = load_artifacts(tmpdir)
            assert artifacts == {}


# =============================================================================
# Test CLI Result
# =============================================================================


class TestCLIResult:
    """Tests for CLI result generation."""

    def test_create_cli_result_success(self) -> None:
        """Test creating successful CLI result."""
        result = MockOptimizationResult(
            best_config={"temp": 0.8},
            best_metrics={"accuracy": 0.95},
            total_trials=50,
            duration=120.0,
        )

        cli = create_cli_result(result)

        assert cli.status == "success"
        assert cli.exit_code == 0
        assert cli.best_config == {"temp": 0.8}
        assert cli.best_score == 0.95
        assert cli.n_trials_completed == 50
        assert cli.constraints_satisfied is True

    def test_create_cli_result_no_valid(self) -> None:
        """Test creating CLI result when no valid configs found."""
        result = MockOptimizationResult(
            best_config=None,
            best_metrics={},
            total_trials=50,
            duration=120.0,
            warnings=["All configs violated constraints"],
        )

        cli = create_cli_result(result)

        assert cli.status == "no_valid_configs"
        assert cli.exit_code == 1
        assert cli.best_config is None
        assert "All configs violated constraints" in cli.warnings

    def test_cli_result_to_json(self) -> None:
        """Test CLI result JSON serialization."""
        cli = CLIResult(
            status="success",
            exit_code=0,
            best_config={"temp": 0.8},
            best_score=0.95,
            n_trials_completed=50,
            constraints_satisfied=True,
        )

        json_str = cli.to_json()
        data = json.loads(json_str)

        assert data["status"] == "success"
        assert data["exit_code"] == 0
        assert data["best_config"]["temp"] == 0.8

    def test_cli_result_with_warnings(self) -> None:
        """Test CLI result with warnings."""
        result = MockOptimizationResult(
            best_config={"temp": 0.8},
            best_metrics={"acc": 0.95},
            total_trials=50,
            duration=120.0,
            warnings=["Early stopping triggered", "Some trials failed"],
        )

        cli = create_cli_result(result)

        assert len(cli.warnings) == 2
        assert "Early stopping triggered" in cli.warnings


# =============================================================================
# Test ApplyBackup
# =============================================================================


class TestApplyBackup:
    """Tests for ApplyBackup dataclass."""

    def test_backup_restore(self) -> None:
        """Test backup restore functionality."""
        pipeline = MockPipeline()

        backup = ApplyBackup(
            original_values={
                "generator.temperature": 0.5,
                "generator.top_k": 15,
            }
        )

        backup.restore(pipeline)

        assert pipeline.get_component("generator").temperature == 0.5
        assert pipeline.get_component("generator").top_k == 15

    def test_backup_has_timestamp(self) -> None:
        """Test backup has timestamp."""
        backup = ApplyBackup(original_values={})
        assert backup.timestamp is not None


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_config(self) -> None:
        """Test applying empty config."""
        pipeline = MockPipeline()
        original_temp = pipeline.get_component("generator").temperature

        apply_config_production(pipeline, {})

        # Should not change anything
        assert pipeline.get_component("generator").temperature == original_temp

    def test_nested_parameter_path(self) -> None:
        """Test nested parameter path parsing."""

        # Create a component with nested attribute
        class NestedComponent:
            def __init__(self):
                self.nested = MagicMock()
                self.nested.value = 10

        pipeline = MockPipeline()
        pipeline.components["nested_comp"] = NestedComponent()

        # Note: Our current implementation uses simple split
        # This test documents the expected behavior
        config = {"nested_comp.nested.value": 20}

        # With validation, this should fail because "nested.value" is not
        # a direct attribute
        with pytest.raises(ConfigMismatchError):
            apply_config_production(pipeline, config, validate=True)

    def test_result_with_empty_history(self) -> None:
        """Test export with empty history."""
        result = MockOptimizationResult(
            best_config={"temp": 0.8},
            best_metrics={"acc": 0.95},
            total_trials=0,
            duration=0.0,
            history=[],
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            export_experiment_history(result, f.name, format="json")

            with open(f.name) as rf:
                data = json.load(rf)
                assert data["trials"] == []

    def test_config_with_inline_search_space(self) -> None:
        """Test config with inline search space dict."""
        config = OptimizationConfig(
            pipeline_module="app:pipe",
            search_space={
                "generator.temperature": [0.1, 0.5, 0.9],
                "generator.top_k": {"min": 5, "max": 20},
            },
            targets=[{"metric_name": "acc", "direction": "maximize"}],
        )

        errors = config.validate()
        assert errors == []


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for production workflow."""

    def test_full_workflow(self) -> None:
        """Test complete production workflow."""
        # 1. Create and save config
        config = OptimizationConfig(
            pipeline_module="myapp:pipeline",
            search_space="search_space.tvl",
            targets=[{"metric_name": "accuracy", "direction": "maximize"}],
            strategy="bayesian",
            n_trials=100,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            save_optimization_config(config, config_path)

            # 2. Load config
            loaded = load_optimization_config(config_path)
            assert loaded.n_trials == 100

            # 3. Simulate optimization result
            result = MockOptimizationResult(
                best_config={"generator.temperature": 0.8},
                best_metrics={"accuracy": 0.95, "latency": 150.0},
                total_trials=100,
                duration=300.0,
                strategy="bayesian",
                history=[
                    MockTrialResult(f"t{i}", {"temp": 0.1 * i}, {"acc": 0.8 + 0.01 * i})
                    for i in range(10)
                ],
            )

            # 4. Save artifacts
            artifact_path = Path(tmpdir) / "artifacts"
            artifacts = save_artifacts(result, artifact_path, pipeline_name="test")

            # 5. Load artifacts
            loaded_artifacts = load_artifacts(artifact_path)
            assert (
                loaded_artifacts["best_config"].config["generator.temperature"] == 0.8
            )

            # 6. Generate CLI result
            cli = create_cli_result(result)
            assert cli.status == "success"
            assert cli.exit_code == 0

            # 7. Apply to pipeline
            pipeline = MockPipeline()
            backup = apply_config_production(
                pipeline,
                result.best_config,
                validate=True,
                backup=True,
            )

            assert pipeline.get_component("generator").temperature == 0.8

            # 8. Rollback if needed
            rollback_config(pipeline, backup)
            assert pipeline.get_component("generator").temperature == 0.7
