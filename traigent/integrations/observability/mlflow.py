"""MLflow integration for Traigent experiment tracking."""

# Traceability: CONC-Layer-Integration CONC-Quality-Observability CONC-Quality-Compatibility FUNC-INTEGRATIONS FUNC-ANALYTICS REQ-INT-008 REQ-ANLY-011 SYNC-Observability

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, cast

try:
    import mlflow
    import mlflow.sklearn

    # pytorch import is optional - may fail with torch issues
    try:
        import mlflow.pytorch  # noqa: F401
    except (ImportError, RuntimeError):
        pass  # pytorch autologging not available

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

    # Mock mlflow for type hints
    class mlflow:  # type: ignore[no-redef]
        _tracking_uri = None
        _active_experiment = None
        _experiments: dict[str, Any] = {}
        _current_run: Any = None

        class tracking:  # type: ignore[no-redef]
            class MlflowClient:
                def get_run(self, run_id: str) -> Any:
                    raise RuntimeError("MLflow not available")

        @staticmethod
        def start_run(*args, **kwargs) -> Any:
            class MockRun:
                class Info:
                    run_id = "mock_run_id"

                info = Info()

            run = MockRun()
            mlflow._current_run = run
            return run

        @staticmethod
        def set_tracking_uri(uri: str) -> None:
            mlflow._tracking_uri = uri

        @staticmethod
        def get_experiment_by_name(name: str) -> Any | None:
            return mlflow._experiments.get(name)

        @staticmethod
        def create_experiment(name: str) -> str:
            experiment_id = f"exp_{len(mlflow._experiments)}"
            experiment = type("MockExperiment", (), {"experiment_id": experiment_id})()
            mlflow._experiments[name] = experiment
            return experiment_id

        @staticmethod
        def set_experiment(name: str) -> None:
            mlflow._active_experiment = name

        @staticmethod
        def end_run(*args, **kwargs) -> None:
            pass

        @staticmethod
        def finish(*args, **kwargs) -> None:
            pass

        @staticmethod
        def log_param(*args, **kwargs) -> None:
            pass

        @staticmethod
        def log_metric(*args, **kwargs) -> None:
            pass

        @staticmethod
        def log_artifact(*args, **kwargs) -> None:
            pass

        @staticmethod
        def log_dict(*args, **kwargs) -> None:
            pass

        @staticmethod
        def set_tag(*args, **kwargs) -> None:
            pass


from traigent.api.types import OptimizationResult, TrialResult

from ...utils.logging import get_logger
from ...utils.secure_path import safe_write_text

logger = get_logger(__name__)


class TraigentMLflowTracker:
    """MLflow experiment tracker for Traigent optimizations."""

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str = "traigent_optimization",
        auto_log: bool = True,
    ) -> None:
        """Initialize MLflow tracker.

        Args:
            tracking_uri: MLflow tracking URI (defaults to local)
            experiment_name: Name of MLflow experiment
            auto_log: Automatically log optimization results
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError(
                "MLflow is not installed. Please install it with: pip install mlflow"
            )

        self.auto_log = auto_log

        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                self.experiment_id = experiment_id
            else:
                self.experiment_id = experiment.experiment_id
        except Exception as e:
            logger.warning(f"Failed to set MLflow experiment: {e}")
            self.experiment_id = None

        mlflow.set_experiment(experiment_name)

        self.current_run = None

    def start_optimization_run(
        self,
        function_name: str,
        objectives: list[str],
        configuration_space: dict[str, Any],
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        """Start MLflow run for optimization.

        Args:
            function_name: Name of function being optimized
            objectives: Optimization objectives
            configuration_space: Parameter search space
            run_name: Optional run name
            tags: Optional tags for the run

        Returns:
            MLflow run ID
        """
        if not run_name:
            timestamp = int(time.time())
            run_name = f"{function_name}_optimization_{timestamp}"

        self.current_run = mlflow.start_run(run_name=run_name)

        # Log optimization metadata
        mlflow.set_tag("traigent.function_name", function_name)
        mlflow.set_tag("traigent.objectives", ",".join(objectives))
        mlflow.set_tag("traigent.optimization_type", "traigent")

        # Log additional tags
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)

        # Log configuration space
        mlflow.log_dict(configuration_space, "configuration_space.json")

        # Log parameters
        mlflow.log_param("num_objectives", len(objectives))
        mlflow.log_param("config_space_size", len(configuration_space))

        logger.info(f"Started MLflow run: {self.current_run.info.run_id}")  # type: ignore[attr-defined]
        return cast(str, self.current_run.info.run_id)  # type: ignore[attr-defined]

    def log_trial(
        self, trial: TrialResult, trial_number: int, step: int | None = None
    ) -> None:
        """Log individual trial to MLflow.

        Args:
            trial: TrialResult to log
            trial_number: Trial number in optimization
            step: Optional step number for metrics
        """
        if not self.current_run:
            logger.warning("No active MLflow run. Call start_optimization_run first.")
            return

        if step is None:
            step = trial_number

        # Log trial configuration as parameters
        for param_name, param_value in trial.config.items():
            mlflow.log_param(f"trial_{trial_number}_{param_name}", param_value)

        # Log trial metrics
        for metric_name, metric_value in trial.metrics.items():
            mlflow.log_metric(f"trial_{metric_name}", metric_value, step=step)

        # Log trial metadata
        mlflow.log_metric("trial_duration", trial.duration, step=step)
        mlflow.log_param(f"trial_{trial_number}_status", trial.status.value)

        # Log trial as artifact
        trial_data = {
            "trial_id": trial.trial_id,
            "trial_number": trial_number,
            "config": trial.config,
            "metrics": trial.metrics,
            "duration": trial.duration,
            "status": trial.status.value,
            "timestamp": trial.timestamp.isoformat() if trial.timestamp else None,
        }

        # Create temporary file for trial data
        trial_file = f"trial_{trial_number}.json"
        safe_write_text(
            Path(trial_file),
            json.dumps(trial_data, indent=2),
            Path.cwd(),
            encoding="utf-8",
        )

        mlflow.log_artifact(trial_file, "trials")

        # Clean up temporary file
        Path(trial_file).unlink(missing_ok=True)

    def log_optimization_result(
        self, result: OptimizationResult, dataset_path: str | None = None
    ) -> None:
        """Log complete optimization result to MLflow.

        Args:
            result: Optimization result to log
            dataset_path: Optional path to evaluation dataset
        """
        if not self.current_run:
            logger.warning("No active MLflow run. Call start_optimization_run first.")
            return

        # Log best configuration as parameters
        if result.best_config:
            for param_name, param_value in result.best_config.items():
                mlflow.log_param(f"best_{param_name}", param_value)

        # Log best metrics
        if result.best_metrics:
            for metric_name, metric_value in result.best_metrics.items():
                mlflow.log_metric(f"best_{metric_name}", metric_value)

        # Log optimization summary metrics
        mlflow.log_metric("total_trials", len(result.trials))
        mlflow.log_metric("optimization_duration", result.duration)
        mlflow.log_param("optimization_status", result.status.value)

        # Log trial statistics
        if result.trials:
            successful_trials = [
                t
                for t in result.trials
                if hasattr(t, "status") and t.status.value == "completed"
            ]
            mlflow.log_metric("successful_trials", len(successful_trials))
            mlflow.log_metric(
                "success_rate", len(successful_trials) / len(result.trials)
            )

            # Log metric statistics
            if successful_trials and result.best_metrics:
                for metric_name in result.best_metrics.keys():
                    metric_values = [
                        t.metrics.get(metric_name, 0)
                        for t in successful_trials
                        if hasattr(t, "metrics") and metric_name in t.metrics
                    ]
                    if metric_values:
                        mlflow.log_metric(
                            f"{metric_name}_mean",
                            sum(metric_values) / len(metric_values),
                        )
                        mlflow.log_metric(
                            f"{metric_name}_std",
                            (
                                sum(
                                    (x - sum(metric_values) / len(metric_values)) ** 2
                                    for x in metric_values
                                )
                                / len(metric_values)
                            )
                            ** 0.5,
                        )

        # Log metadata if available
        if result.metadata:
            for key, value in result.metadata.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"metadata_{key}", value)
                else:
                    mlflow.log_param(f"metadata_{key}", str(value))

        # Log complete result as artifact
        result_data = {
            "best_config": result.best_config,
            "best_metrics": result.best_metrics,
            "total_trials": len(result.trials),
            "duration": result.duration,
            "status": result.status.value,
            "metadata": result.metadata,
        }

        mlflow.log_dict(result_data, "optimization_result.json")

        # Log dataset if provided
        if dataset_path and Path(dataset_path).exists():
            mlflow.log_artifact(dataset_path, "datasets")

        logger.info(
            f"Logged optimization result to MLflow run: {self.current_run.info.run_id}"
        )

    def end_optimization_run(self) -> None:
        """End current MLflow run."""
        if self.current_run:
            mlflow.finish()
            logger.info(f"Ended MLflow run: {self.current_run.info.run_id}")
            self.current_run = None

    def compare_optimizations(
        self, run_ids: list[str], metrics: list[str] | None = None
    ) -> dict[str, Any]:
        """Compare multiple optimization runs.

        Args:
            run_ids: List of MLflow run IDs to compare
            metrics: Metrics to compare (defaults to all best metrics)

        Returns:
            Comparison results
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow not available for comparison") from None

        client = mlflow.tracking.MlflowClient()
        comparison_data = {}

        for run_id in run_ids:
            try:
                run = client.get_run(run_id)

                # Extract best metrics
                best_metrics = {}
                for key, value in run.data.metrics.items():
                    if key.startswith("best_"):
                        metric_name = key.replace("best_", "")
                        best_metrics[metric_name] = value

                # Extract best config
                best_config = {}
                for key, value in run.data.params.items():
                    if key.startswith("best_"):
                        param_name = key.replace("best_", "")
                        best_config[param_name] = value

                comparison_data[run_id] = {
                    "run_name": run.info.run_name,
                    "function_name": run.data.tags.get("traigent.function_name"),
                    "best_metrics": best_metrics,
                    "best_config": best_config,
                    "total_trials": run.data.metrics.get("total_trials", 0),
                    "duration": run.data.metrics.get("optimization_duration", 0),
                    "status": run.data.params.get("optimization_status", "unknown"),
                }

            except Exception as e:
                logger.error(f"Failed to get run {run_id}: {e}")
                comparison_data[run_id] = {"error": str(e)}

        return comparison_data

    def get_best_run(
        self, experiment_name: str, metric_name: str, maximize: bool = True
    ) -> dict[str, Any] | None:
        """Get best run from experiment based on metric.

        Args:
            experiment_name: Name of experiment
            metric_name: Metric to optimize
            maximize: Whether to maximize or minimize metric

        Returns:
            Best run information
        """
        if not MLFLOW_AVAILABLE:
            return None

        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if not experiment:
                return None

            # Search for runs with the metric
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"metrics.best_{metric_name} IS NOT NULL",
                order_by=[
                    f"metrics.best_{metric_name} {'DESC' if maximize else 'ASC'}"
                ],
                max_results=1,
            )

            if runs.empty:
                return None

            best_run = runs.iloc[0]

            return {
                "run_id": best_run["run_id"],
                "run_name": best_run.get("tags.mlflow.runName"),
                "function_name": best_run.get("tags.traigent.function_name"),
                "best_metric_value": best_run[f"metrics.best_{metric_name}"],
                "total_trials": best_run.get("metrics.total_trials", 0),
                "duration": best_run.get("metrics.optimization_duration", 0),
            }

        except Exception as e:
            logger.error(f"Failed to get best run: {e}")
            return None


class MLflowOptimizationCallback:
    """Callback for automatic MLflow logging during optimization."""

    def __init__(self, tracker: TraigentMLflowTracker) -> None:
        """Initialize callback with MLflow tracker.

        Args:
            tracker: TraigentMLflowTracker instance
        """
        self.tracker = tracker
        self.trial_count = 0

    def on_optimization_start(
        self,
        function_name: str,
        objectives: list[str],
        configuration_space: dict[str, Any],
        **kwargs,
    ) -> None:
        """Called when optimization starts."""
        self.tracker.start_optimization_run(
            function_name=function_name,
            objectives=objectives,
            configuration_space=configuration_space,
            tags=kwargs.get("mlflow_tags", {}),
        )

    def on_trial_complete(self, trial: TrialResult) -> None:
        """Called when trial completes."""
        self.trial_count += 1
        self.tracker.log_trial(trial, self.trial_count)

    def on_optimization_complete(
        self, result: OptimizationResult, dataset_path: str | None = None
    ) -> None:
        """Called when optimization completes."""
        self.tracker.log_optimization_result(result, dataset_path)
        self.tracker.end_optimization_run()


# Convenience functions
def create_mlflow_tracker(
    tracking_uri: str | None = None, experiment_name: str = "traigent_optimization"
) -> TraigentMLflowTracker:
    """Create MLflow tracker for Traigent.

    Args:
        tracking_uri: MLflow tracking URI
        experiment_name: Experiment name

    Returns:
        Configured MLflow tracker
    """
    return TraigentMLflowTracker(tracking_uri, experiment_name)


def enable_mlflow_autolog(
    tracking_uri: str | None = None, experiment_name: str = "traigent_optimization"
) -> MLflowOptimizationCallback:
    """Enable automatic MLflow logging for Traigent optimizations.

    Args:
        tracking_uri: MLflow tracking URI
        experiment_name: Experiment name

    Returns:
        MLflow callback for use with @traigent.optimize
    """
    tracker = create_mlflow_tracker(tracking_uri, experiment_name)
    return MLflowOptimizationCallback(tracker)


def log_traigent_optimization(
    result: OptimizationResult,
    function_name: str,
    objectives: list[str],
    configuration_space: dict[str, Any],
    dataset_path: str | None = None,
    tracking_uri: str | None = None,
    experiment_name: str = "traigent_optimization",
    run_name: str | None = None,
) -> str:
    """Log Traigent optimization result to MLflow.

    Args:
        result: Optimization result
        function_name: Name of optimized function
        objectives: Optimization objectives
        configuration_space: Parameter search space
        dataset_path: Path to evaluation dataset
        tracking_uri: MLflow tracking URI
        experiment_name: Experiment name
        run_name: Run name

    Returns:
        MLflow run ID
    """
    tracker = create_mlflow_tracker(tracking_uri, experiment_name)

    run_id = tracker.start_optimization_run(
        function_name=function_name,
        objectives=objectives,
        configuration_space=configuration_space,
        run_name=run_name,
    )

    # Log all trials
    for i, trial in enumerate(result.trials):
        tracker.log_trial(trial, i + 1)

    # Log final result
    tracker.log_optimization_result(result, dataset_path)
    tracker.end_optimization_run()

    return run_id


def compare_traigent_runs(
    run_ids: list[str],
    metrics: list[str] | None = None,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Compare multiple Traigent optimization runs.

    Args:
        run_ids: MLflow run IDs to compare
        metrics: Metrics to compare
        tracking_uri: MLflow tracking URI

    Returns:
        Comparison results
    """
    tracker = TraigentMLflowTracker(tracking_uri=tracking_uri)
    return tracker.compare_optimizations(run_ids, metrics)


def get_best_traigent_run(
    experiment_name: str,
    metric_name: str,
    maximize: bool = True,
    tracking_uri: str | None = None,
) -> dict[str, Any] | None:
    """Get best Traigent run from experiment.

    Args:
        experiment_name: Experiment name
        metric_name: Metric to optimize
        maximize: Whether to maximize metric
        tracking_uri: MLflow tracking URI

    Returns:
        Best run information
    """
    tracker = TraigentMLflowTracker(tracking_uri=tracking_uri)
    return tracker.get_best_run(experiment_name, metric_name, maximize)
