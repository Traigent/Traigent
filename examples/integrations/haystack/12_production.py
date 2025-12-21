"""Example 12: Production Readiness Features.

This example demonstrates Epic 7 production features:
- Configuration file loading (YAML/JSON)
- Production-hardened apply with validation and rollback
- TVL export for tuned configurations
- Experiment history export
- Artifact management
- CLI result generation for CI/CD integration

These features enable headless optimization runs, CI/CD integration,
and production-grade configuration management.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from traigent.integrations.haystack import (
    ConfigMismatchError,
    ConfigValidationError,
    OptimizationConfig,
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
# Mock classes for demonstration
# =============================================================================


@dataclass
class MockComponent:
    """Mock Haystack component."""

    name: str
    temperature: float = 0.7
    top_k: int = 10
    model: str = "gpt-4"


class MockPipeline:
    """Mock Haystack Pipeline."""

    def __init__(self) -> None:
        self.components: dict[str, Any] = {
            "generator": MockComponent(name="generator", temperature=0.7, top_k=10),
            "retriever": MockComponent(name="retriever", top_k=5),
        }

    def get_component(self, name: str) -> MockComponent | None:
        return self.components.get(name)


@dataclass
class MockTrialResult:
    """Mock trial result."""

    trial_id: str
    config: dict[str, Any]
    metrics: dict[str, float]
    is_successful: bool = True
    constraints_satisfied: bool = True
    duration: float = 1.0


@dataclass
class MockOptimizationResult:
    """Mock optimization result."""

    best_config: dict[str, Any] | None
    best_metrics: dict[str, float]
    total_trials: int
    duration: float
    strategy: str = "bayesian"
    history: list[MockTrialResult] | None = None
    pareto_configs: list[dict[str, Any]] | None = None
    warnings: list[str] | None = None


# =============================================================================
# Example 1: Configuration File Loading
# =============================================================================


def example_1_configuration_loading() -> None:
    """Demonstrate loading optimization config from YAML/JSON files."""
    print("=" * 60)
    print("Example 1: Configuration File Loading")
    print("=" * 60)

    # Create a YAML config file
    yaml_content = {
        "pipeline_module": "myapp.pipeline:create_rag_pipeline",
        "search_space": "search_space.tvl",
        "targets": [
            {"metric_name": "accuracy", "direction": "maximize"},
            {"metric_name": "latency_p95", "direction": "minimize"},
        ],
        "constraints": [
            {"metric": "cost_per_query", "max": 0.05},
        ],
        "strategy": "bayesian",
        "n_trials": 100,
        "n_parallel": 4,
        "timeout_seconds": 3600.0,
        "checkpoint_path": "/tmp/checkpoints",
        "artifact_path": "/tmp/artifacts",
        "random_seed": 42,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(yaml_content, f)
        config_path = f.name

    # Load and validate
    config = load_optimization_config(config_path)

    print(f"Pipeline: {config.pipeline_module}")
    print(f"Strategy: {config.strategy}")
    print(f"Trials: {config.n_trials}")
    print(f"Parallel: {config.n_parallel}")
    print(f"Targets: {len(config.targets)}")

    # Validate programmatically
    errors = config.validate()
    print(f"Validation errors: {errors if errors else 'None'}")
    print()


# =============================================================================
# Example 2: Saving Configuration
# =============================================================================


def example_2_save_configuration() -> None:
    """Demonstrate saving optimization config to file."""
    print("=" * 60)
    print("Example 2: Saving Configuration")
    print("=" * 60)

    # Create config programmatically
    config = OptimizationConfig(
        pipeline_module="myapp:pipeline",
        search_space={
            "generator.temperature": {"min": 0.0, "max": 1.0},
            "generator.top_k": {"values": [5, 10, 20, 50]},
        },
        targets=[
            {"metric_name": "f1_score", "direction": "maximize"},
        ],
        strategy="evolutionary",
        n_trials=200,
        n_parallel=8,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        save_optimization_config(config, f.name)

        # Show saved content
        with open(f.name) as rf:
            content = rf.read()
            print("Saved config:")
            print(content)


# =============================================================================
# Example 3: Production Apply with Validation
# =============================================================================


def example_3_production_apply() -> None:
    """Demonstrate production-hardened config application."""
    print("=" * 60)
    print("Example 3: Production Apply with Validation")
    print("=" * 60)

    pipeline = MockPipeline()
    gen = pipeline.get_component("generator")
    print(f"Original temperature: {gen.temperature}")

    # Apply with validation
    config = {"generator.temperature": 0.85, "generator.top_k": 15}
    apply_config_production(pipeline, config, validate=True)

    print(f"After apply: temperature={gen.temperature}, top_k={gen.top_k}")

    # Invalid config - missing component
    try:
        bad_config = {"nonexistent.param": 0.5}
        apply_config_production(pipeline, bad_config, validate=True)
    except ConfigMismatchError as e:
        print(f"Caught expected error: {e}")

    print()


# =============================================================================
# Example 4: Backup and Rollback
# =============================================================================


def example_4_backup_rollback() -> None:
    """Demonstrate backup and rollback functionality."""
    print("=" * 60)
    print("Example 4: Backup and Rollback")
    print("=" * 60)

    pipeline = MockPipeline()
    gen = pipeline.get_component("generator")
    print(f"Original: temperature={gen.temperature}")

    # Apply with backup
    config = {"generator.temperature": 0.9}
    backup = apply_config_production(pipeline, config, backup=True)

    print(f"After apply: temperature={gen.temperature}")
    print(f"Backup created at: {backup.timestamp}")
    print(f"Backed up values: {backup.original_values}")

    # Rollback
    rollback_config(pipeline, backup)
    print(f"After rollback: temperature={gen.temperature}")
    print()


# =============================================================================
# Example 5: TVL Export
# =============================================================================


def example_5_tvl_export() -> None:
    """Demonstrate exporting tuned config as TVL."""
    print("=" * 60)
    print("Example 5: TVL Export")
    print("=" * 60)

    # Create mock result
    result = MockOptimizationResult(
        best_config={
            "generator.temperature": 0.82,
            "generator.top_k": 15,
            "retriever.top_k": 10,
        },
        best_metrics={
            "accuracy": 0.94,
            "latency_p95": 145.0,
            "cost": 0.032,
        },
        total_trials=100,
        duration=3600.0,
        strategy="bayesian",
        history=[
            MockTrialResult(
                f"trial_{i}",
                {"generator.temperature": 0.1 * i},
                {"accuracy": 0.8 + 0.01 * i},
            )
            for i in range(10)
        ],
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".tvl", delete=False) as f:
        tuned = export_tuned_config(
            result, f.name, pipeline_name="rag_pipeline", include_history=True
        )

        print(f"Exported to: {f.name}")
        print(f"Best config: {tuned.config}")
        print(f"Best metrics: {tuned.metrics}")
        print(f"Metadata: {tuned.optimization_metadata}")

        # Show file content
        with open(f.name) as rf:
            print("\nTVL file content:")
            print(rf.read())


# =============================================================================
# Example 6: Loading TVL Config
# =============================================================================


def example_6_load_tvl() -> None:
    """Demonstrate loading tuned config from TVL file."""
    print("=" * 60)
    print("Example 6: Loading TVL Config")
    print("=" * 60)

    # Create and save TVL
    result = MockOptimizationResult(
        best_config={"generator.temperature": 0.8},
        best_metrics={"accuracy": 0.95},
        total_trials=50,
        duration=1800.0,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".tvl", delete=False) as f:
        export_tuned_config(result, f.name, pipeline_name="my_pipeline")

        # Load it back
        tuned = load_tuned_config(f.name)

        print(f"Version: {tuned.version}")
        print(f"Pipeline: {tuned.pipeline_name}")
        print(f"Framework: {tuned.framework}")
        print(f"Config: {tuned.config}")
        print(f"Metrics: {tuned.metrics}")

        # Apply to pipeline
        pipeline = MockPipeline()
        apply_config_production(pipeline, tuned.config, validate=True)
        print(f"Applied temperature: {pipeline.get_component('generator').temperature}")
        print()


# =============================================================================
# Example 7: Experiment History Export
# =============================================================================


def example_7_history_export() -> None:
    """Demonstrate exporting full experiment history."""
    print("=" * 60)
    print("Example 7: Experiment History Export")
    print("=" * 60)

    result = MockOptimizationResult(
        best_config={"temp": 0.8},
        best_metrics={"accuracy": 0.95},
        total_trials=5,
        duration=300.0,
        history=[
            MockTrialResult(
                f"trial_{i}",
                {"temp": 0.5 + 0.1 * i},
                {"accuracy": 0.85 + 0.02 * i, "latency": 100 + 10 * i},
                is_successful=i != 2,
                constraints_satisfied=i != 3,
            )
            for i in range(5)
        ],
    )

    # Export to JSON
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        export_experiment_history(result, f.name, format="json")

        with open(f.name) as rf:
            data = json.load(rf)
            print("JSON export summary:")
            print(f"  Schema version: {data['schema_version']}")
            print(f"  Total trials: {data['summary']['total_trials']}")
            print(f"  Trials exported: {len(data['trials'])}")

    # Export to CSV
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        export_experiment_history(result, f.name, format="csv")

        with open(f.name) as rf:
            lines = rf.readlines()
            print("\nCSV export:")
            for line in lines[:3]:  # Show header + first 2 rows
                print(f"  {line.strip()}")
            print(f"  ... ({len(lines) - 1} data rows total)")
    print()


# =============================================================================
# Example 8: Artifact Management
# =============================================================================


def example_8_artifacts() -> None:
    """Demonstrate saving and loading all artifacts."""
    print("=" * 60)
    print("Example 8: Artifact Management")
    print("=" * 60)

    result = MockOptimizationResult(
        best_config={"generator.temperature": 0.85},
        best_metrics={"accuracy": 0.96, "latency": 120.0},
        total_trials=100,
        duration=3600.0,
        history=[
            MockTrialResult(
                f"t{i}",
                {"generator.temperature": 0.1 * i},
                {"accuracy": 0.9 + 0.005 * i},
            )
            for i in range(10)
        ],
        pareto_configs=[
            {"temp": 0.8},
            {"temp": 0.85},
        ],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save all artifacts
        artifacts = save_artifacts(result, tmpdir, pipeline_name="production_rag")

        print("Saved artifacts:")
        for name, path in artifacts.items():
            print(f"  {name}: {path}")

        # Load them back
        loaded = load_artifacts(tmpdir)

        print("\nLoaded artifacts:")
        print(f"  Best config: {loaded['best_config'].config}")
        print(f"  Summary trials: {loaded['summary']['total_trials']}")
        print(f"  History trials: {len(loaded['experiment_history']['trials'])}")
    print()


# =============================================================================
# Example 9: CLI Result for CI/CD
# =============================================================================


def example_9_cli_result() -> None:
    """Demonstrate CLI result generation for CI/CD integration."""
    print("=" * 60)
    print("Example 9: CLI Result for CI/CD")
    print("=" * 60)

    # Successful result
    result = MockOptimizationResult(
        best_config={"generator.temperature": 0.8},
        best_metrics={"accuracy": 0.95},
        total_trials=100,
        duration=3600.0,
    )

    cli = create_cli_result(result)
    print("Successful optimization:")
    print(f"  Status: {cli.status}")
    print(f"  Exit code: {cli.exit_code}")
    print(f"  Best score: {cli.best_score}")
    print(f"  Trials completed: {cli.n_trials_completed}")

    # Generate JSON for CI
    print("\nJSON output for CI:")
    print(cli.to_json())

    # Failed result (no valid configs)
    failed_result = MockOptimizationResult(
        best_config=None,
        best_metrics={},
        total_trials=50,
        duration=1800.0,
        warnings=["All configurations violated cost constraint"],
    )

    cli_failed = create_cli_result(failed_result)
    print("\nFailed optimization:")
    print(f"  Status: {cli_failed.status}")
    print(f"  Exit code: {cli_failed.exit_code}")
    print(f"  Warnings: {cli_failed.warnings}")
    print()


# =============================================================================
# Example 10: Full Production Workflow
# =============================================================================


def example_10_full_workflow() -> None:
    """Demonstrate complete production optimization workflow."""
    print("=" * 60)
    print("Example 10: Full Production Workflow")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as workspace:
        workspace_path = Path(workspace)

        # Step 1: Create config file
        print("Step 1: Create optimization config...")
        config = OptimizationConfig(
            pipeline_module="myapp.pipeline:create_rag",
            search_space="search_space.tvl",
            targets=[{"metric_name": "accuracy", "direction": "maximize"}],
            constraints=[{"metric": "cost", "max": 0.05}],
            strategy="bayesian",
            n_trials=100,
        )
        config_path = workspace_path / "optimization_config.yaml"
        save_optimization_config(config, config_path)
        print(f"  Saved config to {config_path}")

        # Step 2: Load config (simulating headless run)
        print("\nStep 2: Load config for optimization...")
        loaded_config = load_optimization_config(config_path)
        print(
            f"  Loaded: {loaded_config.n_trials} trials with {loaded_config.strategy}"
        )

        # Step 3: Simulate optimization
        print("\nStep 3: Running optimization (simulated)...")
        result = MockOptimizationResult(
            best_config={
                "generator.temperature": 0.82,
                "generator.top_k": 12,
            },
            best_metrics={"accuracy": 0.94, "cost": 0.038},
            total_trials=100,
            duration=3600.0,
            strategy="bayesian",
            history=[
                MockTrialResult(
                    f"t{i}",
                    {"generator.temperature": 0.1 * i, "generator.top_k": 10},
                    {"accuracy": 0.8 + 0.01 * i, "cost": 0.05 - 0.001 * i},
                )
                for i in range(10)
            ],
        )
        print(f"  Completed {result.total_trials} trials")
        print(f"  Best accuracy: {result.best_metrics['accuracy']}")

        # Step 4: Save artifacts
        print("\nStep 4: Saving artifacts...")
        artifact_path = workspace_path / "artifacts"
        artifacts = save_artifacts(result, artifact_path, pipeline_name="rag_v2")
        print(f"  Saved {len(artifacts)} artifacts to {artifact_path}")

        # Step 5: Export TVL for deployment
        print("\nStep 5: Export TVL for deployment...")
        tvl_path = workspace_path / "optimized_config.tvl"
        tuned = export_tuned_config(
            result, tvl_path, pipeline_name="rag_v2", include_history=True
        )
        print(f"  Exported to {tvl_path}")

        # Step 6: Apply to production pipeline
        print("\nStep 6: Apply to production pipeline...")
        pipeline = MockPipeline()
        backup = apply_config_production(
            pipeline, tuned.config, validate=True, backup=True
        )
        gen = pipeline.get_component("generator")
        print(f"  Applied: temperature={gen.temperature}, top_k={gen.top_k}")
        print(f"  Backup available: {len(backup.original_values)} values saved")

        # Step 7: Generate CI result
        print("\nStep 7: Generate CI/CD result...")
        cli = create_cli_result(result)
        ci_output = workspace_path / "ci_result.json"
        with open(ci_output, "w") as f:
            f.write(cli.to_json())
        print(f"  CI result saved to {ci_output}")
        print(f"  Exit code: {cli.exit_code}")

        print("\nWorkflow complete!")
        print(f"Workspace contents: {list(workspace_path.glob('*'))}")
    print()


# =============================================================================
# Example 11: Error Handling
# =============================================================================


def example_11_error_handling() -> None:
    """Demonstrate error handling in production scenarios."""
    print("=" * 60)
    print("Example 11: Error Handling")
    print("=" * 60)

    pipeline = MockPipeline()

    # 1. Missing file
    print("1. Missing config file:")
    try:
        load_optimization_config("/nonexistent/config.yaml")
    except ConfigValidationError as e:
        print(f"   Caught: {e}")

    # 2. Invalid config content
    print("\n2. Invalid config content:")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump({"invalid": "config"}, f)
        f.flush()
        try:
            load_optimization_config(f.name)
        except ConfigValidationError:
            print("   Caught: ConfigValidationError")
            print("   Message includes: 'pipeline_module is required'")

    # 3. Invalid parameter path
    print("\n3. Invalid parameter path:")
    try:
        apply_config_production(pipeline, {"invalid_format": 0.5}, validate=True)
    except ConfigMismatchError as e:
        print(f"   Caught: {e}")

    # 4. Missing component
    print("\n4. Missing component:")
    try:
        apply_config_production(
            pipeline, {"nonexistent.temperature": 0.5}, validate=True
        )
    except ConfigMismatchError as e:
        print(f"   Caught: {e}")

    # 5. Missing parameter
    print("\n5. Missing parameter:")
    try:
        apply_config_production(
            pipeline, {"generator.nonexistent_param": 0.5}, validate=True
        )
    except ConfigMismatchError as e:
        print(f"   Caught: {e}")
    print()


# =============================================================================
# Example 12: Integration with Existing Systems
# =============================================================================


def example_12_integration() -> None:
    """Demonstrate integration with existing Traigent systems."""
    print("=" * 60)
    print("Example 12: Integration with Existing Systems")
    print("=" * 60)

    # This example shows how production features integrate with
    # the full Traigent ecosystem

    print("The production module integrates with:")
    print()
    print("1. HaystackOptimizer (Epic 5):")
    print("   - OptimizationResult feeds into export_tuned_config()")
    print("   - TrialResult history feeds into export_experiment_history()")
    print()
    print("2. Attribution System (Epic 6):")
    print("   - ComponentAttribution can be saved with artifacts")
    print("   - Sensitivity scores can be exported alongside config")
    print()
    print("3. Constraint System (Epic 4):")
    print("   - CLI result includes constraints_satisfied flag")
    print("   - Warnings propagate through to CI output")
    print()
    print("4. TVL Spec Loader (Epic 3):")
    print("   - search_space in config can reference TVL files")
    print("   - Tuned configs export as valid TVL format")
    print()

    # Show example integration
    result = MockOptimizationResult(
        best_config={"generator.temperature": 0.8},
        best_metrics={"accuracy": 0.95, "cost": 0.03},
        total_trials=100,
        duration=3600.0,
        warnings=["Early stopping triggered at trial 85"],
    )

    cli = create_cli_result(result)
    print("Example CI/CD output structure:")
    output = json.loads(cli.to_json())
    for key, value in output.items():
        if isinstance(value, dict):
            print(f"  {key}: <dict with {len(value)} keys>")
        elif isinstance(value, list):
            print(f"  {key}: <list with {len(value)} items>")
        else:
            print(f"  {key}: {value}")
    print()


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run all production examples."""
    example_1_configuration_loading()
    example_2_save_configuration()
    example_3_production_apply()
    example_4_backup_rollback()
    example_5_tvl_export()
    example_6_load_tvl()
    example_7_history_export()
    example_8_artifacts()
    example_9_cli_result()
    example_10_full_workflow()
    example_11_error_handling()
    example_12_integration()

    print("=" * 60)
    print("All production examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
