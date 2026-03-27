"""Analysis utilities for optimization runs.

This module provides comprehensive analysis capabilities for logged optimization runs,
including data loading, comparison, visualization, and export functionality.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Performance CONC-Quality-Observability FUNC-ANALYTICS FUNC-STORAGE REQ-ANLY-011 REQ-STOR-007 SYNC-StorageLogging

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from traigent.utils.file_versioning import FileVersionManager
from traigent.utils.logging import get_logger
from traigent.utils.optimization_logger import OptimizationLogger
from traigent.utils.secure_path import safe_open, validate_path

logger = get_logger(__name__)


class OptimizationAnalyzer:
    """Utilities for analyzing logged optimization runs."""

    def __init__(
        self,
        base_path: Path | None = None,
        file_version: str = "2",
    ) -> None:
        """Initialize analyzer.

        Args:
            base_path: Base directory for logs (defaults to ~/.traigent/optimization_logs)
            file_version: Preferred file naming version to look for when loading runs
        """
        self.base_path = base_path or OptimizationLogger._resolve_default_base_path()
        self.file_manager = FileVersionManager(version=file_version)
        self.legacy_file_manager = FileVersionManager(use_legacy=True)

    def list_experiments(self) -> pd.DataFrame:
        """List all experiments with summary statistics.

        Returns:
            DataFrame with aggregated experiment information
        """
        return self._list_experiments_impl(
            self.base_path, self.file_manager, self.legacy_file_manager
        )

    @staticmethod
    def _candidate_files(
        run_path: Path,
        subdir: str,
        file_type: str,
        file_manager: FileVersionManager,
        legacy_manager: FileVersionManager,
        **kwargs,
    ) -> list[Path]:
        base_dir = run_path / subdir if subdir else run_path
        candidates: list[Path] = []

        try:
            candidates.append(base_dir / file_manager.get_filename(file_type, **kwargs))
        except ValueError:
            pass

        try:
            candidates.append(
                base_dir / legacy_manager.get_filename(file_type, **kwargs)
            )
        except ValueError:
            pass

        unique: list[Path] = []
        seen = set()
        for candidate in candidates:
            key = str(candidate)
            if key not in seen:
                unique.append(candidate)
                seen.add(key)
        return unique

    @staticmethod
    def _resolve_existing_file(
        run_path: Path,
        subdir: str,
        file_type: str,
        file_manager: FileVersionManager,
        legacy_manager: FileVersionManager,
        **kwargs,
    ) -> Path | None:
        for candidate in OptimizationAnalyzer._candidate_files(
            run_path, subdir, file_type, file_manager, legacy_manager, **kwargs
        ):
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _load_json_file(
        run_path: Path,
        subdir: str,
        file_type: str,
        file_manager: FileVersionManager,
        legacy_manager: FileVersionManager,
        **kwargs,
    ) -> Any | None:
        file_path = OptimizationAnalyzer._resolve_existing_file(
            run_path, subdir, file_type, file_manager, legacy_manager, **kwargs
        )
        if not file_path:
            return None
        validated_path = validate_path(file_path, run_path, must_exist=True)
        with safe_open(validated_path, run_path, mode="r") as f:
            return json.load(f)

    @staticmethod
    def _load_jsonl_file(
        run_path: Path,
        subdir: str,
        file_type: str,
        file_manager: FileVersionManager,
        legacy_manager: FileVersionManager,
        **kwargs,
    ) -> list[Any]:
        file_path = OptimizationAnalyzer._resolve_existing_file(
            run_path, subdir, file_type, file_manager, legacy_manager, **kwargs
        )
        if not file_path:
            return []

        entries: list[Any] = []
        validated_path = validate_path(file_path, run_path, must_exist=True)
        with safe_open(validated_path, run_path, mode="r") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    entries.append(json.loads(stripped))
        return entries

    @staticmethod
    def _list_experiments_impl(
        base_path: Path,
        file_manager: FileVersionManager,
        legacy_manager: FileVersionManager,
    ) -> pd.DataFrame:
        """Implementation of list_experiments."""
        if not base_path.exists():
            logger.warning(f"No optimization logs found at {base_path}")
            return pd.DataFrame(columns=["experiment", "runs", "latest_run", "run_ids"])

        # Get all runs data
        runs_df = OptimizationAnalyzer._get_all_runs(
            base_path, file_manager, legacy_manager
        )

        if runs_df.empty:
            return pd.DataFrame(columns=["experiment", "runs", "latest_run", "run_ids"])

        # Aggregate by experiment
        experiments = []
        for exp_name, group in runs_df.groupby("experiment"):
            sorted_group = group.sort_values("timestamp", ascending=False)
            experiments.append(
                {
                    "experiment": exp_name,
                    "runs": len(group),
                    "latest_run": sorted_group.iloc[0]["run_id"],
                    "latest_timestamp": sorted_group.iloc[0]["timestamp"],
                    "run_ids": sorted_group["run_id"].tolist(),
                    "execution_modes": sorted_group["execution_mode"].unique().tolist(),
                }
            )

        df = pd.DataFrame(experiments)
        if not df.empty:
            df = df.sort_values("latest_timestamp", ascending=False).drop(
                columns=["latest_timestamp"]
            )
        return df

    @staticmethod
    def _get_all_runs(
        base_path: Path,
        file_manager: FileVersionManager,
        legacy_manager: FileVersionManager,
    ) -> pd.DataFrame:
        """Get all run information from index or directory."""
        # Load index file if it exists
        index_file = base_path / "index.json"
        if not index_file.exists():
            # Build index from directory structure
            return OptimizationAnalyzer._build_experiment_index(
                base_path, file_manager, legacy_manager
            )

        validated_index = validate_path(index_file, base_path, must_exist=True)
        with safe_open(validated_index, base_path, mode="r") as f:
            index_data = json.load(f)

        # Convert to DataFrame
        runs = []
        for exp_name, exp_data in index_data.get("experiments", {}).items():
            for run in exp_data.get("runs", []):
                runs.append(
                    {
                        "experiment": exp_name,
                        "run_id": run["run_id"],
                        "session_id": run["session_id"],
                        "timestamp": run["timestamp"],
                        "execution_mode": run.get("execution_mode", "unknown"),
                        "path": run["path"],
                    }
                )

        if not runs:
            return pd.DataFrame()

        df = pd.DataFrame(runs)
        # Timestamps are emitted via datetime.isoformat(), so parse accordingly without warnings
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], format="ISO8601", errors="coerce"
        )
        return df

    @staticmethod
    def _build_experiment_index(
        base_path: Path,
        file_manager: FileVersionManager,
        legacy_manager: FileVersionManager,
    ) -> pd.DataFrame:
        """Build experiment index from directory structure."""
        runs = []
        experiments_dir = base_path / "experiments"

        if not experiments_dir.exists():
            return pd.DataFrame()

        for exp_dir in experiments_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            runs_dir = exp_dir / "runs"
            if not runs_dir.exists():
                continue

            for run_dir in runs_dir.iterdir():
                if not run_dir.is_dir():
                    continue

                session_data = OptimizationAnalyzer._load_json_file(
                    run_dir,
                    "meta",
                    "session",
                    file_manager,
                    legacy_manager,
                )

                if session_data is None:
                    continue

                runs.append(
                    {
                        "experiment": exp_dir.name,
                        "run_id": run_dir.name,
                        "session_id": session_data.get("session_id", "unknown"),
                        "timestamp": session_data.get("start_time", ""),
                        "execution_mode": session_data.get("execution_mode", "unknown"),
                        "path": str(run_dir),
                        "status": session_data.get("status", "unknown"),
                    }
                )

        if not runs:
            return pd.DataFrame()

        df = pd.DataFrame(runs)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df

    def load_run(self, experiment_name: str, run_id: str) -> dict[str, Any]:
        """Load all data for a specific run."""
        return self._load_run_impl(
            experiment_name,
            run_id,
            self.base_path,
            self.file_manager,
            self.legacy_file_manager,
        )

    @staticmethod
    def _load_run_impl(
        experiment_name: str,
        run_id: str,
        base_path: Path,
        file_manager: FileVersionManager,
        legacy_manager: FileVersionManager,
    ) -> dict[str, Any]:
        """Implementation of load_run."""
        run_path = base_path / "experiments" / experiment_name / "runs" / run_id

        if not run_path.exists():
            logger.warning(f"Run not found: {experiment_name}/{run_id}")
            return {}

        result = {}

        session = OptimizationAnalyzer._load_json_file(
            run_path, "meta", "session", file_manager, legacy_manager
        )
        if session is not None:
            result["session"] = session

        config = OptimizationAnalyzer._load_json_file(
            run_path, "meta", "config", file_manager, legacy_manager
        )
        if config is not None:
            result["config"] = config

        objectives = OptimizationAnalyzer._load_json_file(
            run_path, "meta", "objectives", file_manager, legacy_manager
        )
        if objectives is not None:
            result["objectives"] = objectives

        best_config = OptimizationAnalyzer._load_json_file(
            run_path, "artifacts", "best_config", file_manager, legacy_manager
        )
        if best_config is not None:
            result["best_config"] = best_config

        weighted = OptimizationAnalyzer._load_json_file(
            run_path, "artifacts", "weighted_results", file_manager, legacy_manager
        )
        if weighted is not None:
            result["weighted_results"] = weighted

        trials = OptimizationAnalyzer._load_jsonl_file(
            run_path, "trials", "trials_stream", file_manager, legacy_manager
        )
        if trials:
            result["trials"] = trials

        metrics_summary = OptimizationAnalyzer._load_json_file(
            run_path, "metrics", "metrics_summary", file_manager, legacy_manager
        )
        if metrics_summary is not None:
            result["metrics_summary"] = metrics_summary

        return result

    def compare_runs(
        self, experiment_name: str, run_ids: list[str] | None = None
    ) -> pd.DataFrame:
        """Compare multiple runs side by side."""
        return self._compare_runs_impl(
            experiment_name,
            run_ids,
            self.base_path,
            self.file_manager,
            self.legacy_file_manager,
        )

    @staticmethod
    def _compare_runs_impl(
        experiment_name: str,
        run_ids: list[str] | None,
        base_path: Path,
        file_manager: FileVersionManager,
        legacy_manager: FileVersionManager,
    ) -> pd.DataFrame:
        """Implementation of compare_runs."""
        # If no run_ids specified, get all runs for the experiment
        if run_ids is None:
            runs_df = OptimizationAnalyzer._get_all_runs(
                base_path, file_manager, legacy_manager
            )
            if not runs_df.empty:
                exp_runs = runs_df[runs_df["experiment"] == experiment_name]
                run_ids = exp_runs["run_id"].tolist()
            else:
                return pd.DataFrame()

        comparisons = []
        for run_id in run_ids:
            data = OptimizationAnalyzer._load_run_impl(
                experiment_name,
                run_id,
                base_path,
                file_manager,
                legacy_manager,
            )

            if not data:
                continue

            comparison = {
                "run_id": run_id,
                "status": data.get("session", {}).get("status", "unknown"),
                "duration": data.get("metrics_summary", {}).get("duration", 0),
                "total_trials": data.get("metrics_summary", {}).get("total_trials", 0),
                "successful_trials": data.get("metrics_summary", {}).get(
                    "successful_trials", 0
                ),
                "algorithm": data.get("metrics_summary", {}).get(
                    "algorithm", "unknown"
                ),
            }

            # Add best metrics
            best_metrics = data.get("metrics_summary", {}).get("best_metrics", {})
            for metric, value in best_metrics.items():
                comparison[f"best_{metric}"] = value

            # Add weighted results if available
            if "weighted_results" in data:
                weighted_config = data["weighted_results"].get(
                    "best_weighted_config", {}
                )
                weighted_score = data["weighted_results"].get(
                    "best_weighted_score", None
                )
                comparison["weighted_score"] = weighted_score
                comparison["weighted_best_model"] = weighted_config.get(
                    "model", "unknown"
                )

            comparisons.append(comparison)

        if not comparisons:
            return pd.DataFrame()

        return pd.DataFrame(comparisons)

    def plot_convergence(
        self,
        experiment_name: str,
        run_id: str,
        objective: str | None = None,
        show: bool = True,
    ):
        """Plot optimization convergence."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error(
                "Matplotlib not available. Install with: pip install matplotlib"
            )
            return None

        data = self.load_run(experiment_name, run_id)

        if not data or "trials" not in data:
            logger.warning(f"No trial data found for {experiment_name}/{run_id}")
            return None

        trials = data["trials"]
        if not trials:
            return None

        # Determine objective to plot
        if objective is None:
            objectives_data = data.get("objectives", {})
            if isinstance(objectives_data, dict):
                objectives_list = objectives_data.get("objectives", [])
            else:
                objectives_list = (
                    objectives_data if isinstance(objectives_data, list) else []
                )

            # Extract objective name if it's a dict
            if objectives_list and isinstance(objectives_list[0], dict):
                objective = objectives_list[0].get("name", "score")
            elif objectives_list and isinstance(objectives_list[0], str):
                objective = objectives_list[0]
            else:
                objective = "score"

        # Extract metric values
        values = []
        best_values: list[float] = []
        for trial in trials:
            metrics = trial.get("metrics", {})
            if objective in metrics:
                value = metrics[objective]
                values.append(value)
                if best_values:
                    best_values.append(max(value, best_values[-1]))
                else:
                    best_values.append(value)

        if not values:
            logger.warning("No values found for the requested objective")
            return None

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        x = range(1, len(values) + 1)
        ax.plot(x, values, "o-", alpha=0.5, label="Trial values")
        ax.plot(x, best_values, "-", linewidth=2, label="Best so far")

        ax.set_xlabel("Trial Number")
        ax.set_ylabel(objective.capitalize())
        ax.set_title(f"Optimization Convergence: {experiment_name}/{run_id}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if show:
            plt.show()

        return fig

    def plot_pareto_front(
        self,
        experiment_name: str,
        run_id: str,
        objectives: list[str] | None = None,
        show: bool = True,
    ):
        """Plot Pareto front for multi-objective optimization."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error(
                "Matplotlib not available. Install with: pip install matplotlib"
            )
            return None

        data = self.load_run(experiment_name, run_id)

        if not data or "trials" not in data:
            logger.warning(f"No trial data found for {experiment_name}/{run_id}")
            return None

        trials = data["trials"]
        if not trials:
            return None

        # Determine objectives to plot
        if objectives is None or len(objectives) < 2:
            objectives_data = data.get("objectives", {})
            if isinstance(objectives_data, dict):
                all_objectives = objectives_data.get("objectives", [])
            else:
                all_objectives = (
                    objectives_data if isinstance(objectives_data, list) else []
                )

            # Extract objective names if they're dicts
            objective_names = []
            for obj in all_objectives:
                if isinstance(obj, dict):
                    objective_names.append(obj.get("name", ""))
                elif isinstance(obj, str):
                    objective_names.append(obj)

            if len(objective_names) >= 2:
                objectives = objective_names[:2]
            else:
                logger.warning("Need at least 2 objectives for Pareto front")
                return None

        # Extract objective values
        obj1_values = []
        obj2_values = []
        for trial in trials:
            metrics = trial.get("metrics", {})
            if objectives[0] in metrics and objectives[1] in metrics:
                obj1_values.append(metrics[objectives[0]])
                obj2_values.append(metrics[objectives[1]])

        if not obj1_values:
            logger.warning("No values found for the requested objectives")
            return None

        # Compute Pareto front
        points = np.array(list(zip(obj1_values, obj2_values, strict=False)))
        pareto_points = []

        for i, point in enumerate(points):
            dominated = False
            for j, other in enumerate(points):
                if i != j:
                    # Check if 'other' dominates 'point' (assuming maximization)
                    if (
                        other[0] >= point[0]
                        and other[1] >= point[1]
                        and (other[0] > point[0] or other[1] > point[1])
                    ):
                        dominated = True
                        break
            if not dominated:
                pareto_points.append(point)

        pareto_points = np.array(pareto_points)

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot all points
        ax.scatter(obj1_values, obj2_values, alpha=0.5, label="All trials")

        # Plot Pareto front
        if len(pareto_points) > 0:
            # Sort for line plot
            sorted_indices = np.argsort(pareto_points[:, 0])  # type: ignore[call-overload]
            pareto_sorted = pareto_points[sorted_indices]

            ax.scatter(
                pareto_sorted[:, 0],
                pareto_sorted[:, 1],
                color="red",
                s=100,
                marker="*",
                label="Pareto front",
                zorder=5,
            )
            ax.plot(
                pareto_sorted[:, 0], pareto_sorted[:, 1], "r--", alpha=0.5, linewidth=1
            )

        ax.set_xlabel(objectives[0].capitalize())
        ax.set_ylabel(objectives[1].capitalize())
        ax.set_title(f"Pareto Front: {experiment_name}/{run_id}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if show:
            plt.show()

        return fig

    def export_for_analysis(
        self, experiment_name: str, format: str = "csv", output_dir: Path | None = None
    ) -> str | None:
        """Export experiment data for external analysis."""
        output_dir = output_dir or Path.cwd()

        # Get all runs for the experiment
        runs_df = OptimizationAnalyzer._get_all_runs(
            self.base_path, self.file_manager, self.legacy_file_manager
        )
        if runs_df.empty:
            return None

        exp_runs = runs_df[runs_df["experiment"] == experiment_name]
        if exp_runs.empty:
            logger.warning(f"No runs found for experiment: {experiment_name}")
            return None

        # Collect all trial data
        all_trials = []
        for _, run_info in exp_runs.iterrows():
            run_id = run_info["run_id"]
            data = self.load_run(experiment_name, run_id)

            if "trials" in data:
                for trial in data["trials"]:
                    trial_data = {
                        "run_id": run_id,
                        "trial_id": trial.get("trial_id"),
                        "status": trial.get("status"),
                        "duration": trial.get("duration"),
                    }

                    # Add config parameters
                    config = trial.get("config", {})
                    for key, value in config.items():
                        trial_data[f"config_{key}"] = value

                    # Add metrics
                    metrics = trial.get("metrics", {})
                    for key, value in metrics.items():
                        trial_data[f"metric_{key}"] = value

                    all_trials.append(trial_data)

        if not all_trials:
            logger.warning(f"No trial data found for experiment: {experiment_name}")
            return None

        df = pd.DataFrame(all_trials)

        # Export based on format
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_{timestamp}"

        if format == "csv":
            output_file = output_dir / f"{filename}.csv"
            df.to_csv(output_file, index=False)
        elif format == "json":
            output_file = output_dir / f"{filename}.json"
            df.to_json(output_file, orient="records", indent=2)
        elif format == "excel":
            try:
                output_file = output_dir / f"{filename}.xlsx"
                df.to_excel(output_file, index=False)
            except ImportError:
                logger.error(
                    "Excel export requires openpyxl. Install with: pip install openpyxl"
                )
                return None
        else:
            logger.warning(f"Unsupported format: {format}")
            return None

        logger.info(f"Exported {len(df)} trials to {output_file}")
        return str(output_file)

    def get_run_summary(self, experiment_name: str, run_id: str) -> str:
        """Generate a text summary of a run."""
        data = self.load_run(experiment_name, run_id)

        if not data:
            return f"No data found for {experiment_name}/{run_id}"

        summary = []
        summary.append(f"{'=' * 60}")
        summary.append("Optimization Run Summary")
        summary.append(f"{'=' * 60}")
        summary.append(f"Experiment: {experiment_name}")
        summary.append(f"Run ID: {run_id}")

        # Session info
        if "session" in data:
            session = data["session"]
            summary.append(f"Session ID: {session.get('session_id', 'unknown')}")
            summary.append(f"Status: {session.get('status', 'unknown')}")
            summary.append(f"Duration: {session.get('duration', 0):.2f} seconds")

        # Metrics summary
        if "metrics_summary" in data:
            metrics = data["metrics_summary"]
            summary.append("\nTrials:")
            summary.append(f"  Total: {metrics.get('total_trials', 0)}")
            summary.append(f"  Successful: {metrics.get('successful_trials', 0)}")
            summary.append(f"  Failed: {metrics.get('failed_trials', 0)}")
            summary.append(f"  Success Rate: {metrics.get('success_rate', 0):.1%}")

            summary.append(f"\nAlgorithm: {metrics.get('algorithm', 'unknown')}")

            if "best_metrics" in metrics:
                summary.append("\nBest Metrics:")
                for metric, value in metrics["best_metrics"].items():
                    if isinstance(value, float):
                        summary.append(f"  {metric}: {value:.4f}")
                    else:
                        summary.append(f"  {metric}: {value}")

        # Best configuration
        if "best_config" in data:
            config = data["best_config"]
            summary.append("\nBest Configuration:")
            for key, value in config.get("config", {}).items():
                summary.append(f"  {key}: {value}")

        # Weighted results
        if "weighted_results" in data:
            weighted = data["weighted_results"]
            summary.append("\nWeighted Scoring Results:")
            summary.append(
                f"  Best Weighted Score: {weighted.get('best_weighted_score', 0):.4f}"
            )

            if "best_weighted_config" in weighted:
                summary.append("  Best Weighted Config:")
                for key, value in weighted["best_weighted_config"].items():
                    summary.append(f"    {key}: {value}")

            if "objective_weights" in weighted:
                summary.append("  Objective Weights:")
                for obj, weight in weighted["objective_weights"].items():
                    summary.append(f"    {obj}: {weight}")

        summary.append(f"{'=' * 60}")

        return "\n".join(summary)
