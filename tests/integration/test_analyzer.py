#!/usr/bin/env python3
"""
Test Analysis utilities functionality.
Tests the OptimizationAnalyzer for post-run analysis capabilities.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from traigent.api.types import OptimizationResult, TrialResult, TrialStatus
from traigent.utils.optimization_analyzer import OptimizationAnalyzer
from traigent.utils.optimization_logger import OptimizationLogger


@pytest.fixture
def base_path():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


class TestOptimizationAnalyzer:
    """Test OptimizationAnalyzer functionality."""

    @pytest.fixture(autouse=True)
    def setup(self, base_path):
        """Setup test data for analyzer tests."""
        self.base_path = base_path
        self._create_test_data(base_path)
        self.analyzer = OptimizationAnalyzer(base_path=base_path)

    def _create_test_data(self, base_path: Path):
        """Create test optimization runs for analysis."""
        # Create multiple experiments and runs
        experiments = ["test_exp_1", "test_exp_2"]

        for exp_name in experiments:
            # Create 2 runs per experiment
            for run_idx in range(2):
                session_id = f"session_{exp_name}_{run_idx}"

                # Create logger
                logger = OptimizationLogger(
                    experiment_name=exp_name,
                    session_id=session_id,
                    execution_mode="edge_analytics",
                    base_path=base_path,
                )

                # Log session start
                logger.log_session_start(
                    config={"test": "config", "experiment": exp_name},
                    objectives=["accuracy", "cost", "latency"],
                    algorithm="GridSearch",
                    dataset_info={"size": 100, "name": f"dataset_{exp_name}"},
                )

                # Create trials with varying performance
                trials = []
                for i in range(10):
                    trial = TrialResult(
                        trial_id=f"trial_{i:03d}",
                        config={"param": f"value_{i}", "model": f"model_{i % 3}"},
                        metrics={
                            "accuracy": 0.7
                            + (i * 0.02)
                            + np.random.uniform(-0.01, 0.01),
                            "cost": 0.001 * (10 - i) + np.random.uniform(0, 0.0001),
                            "latency": 1.0 + (i * 0.1) + np.random.uniform(-0.05, 0.05),
                        },
                        status=TrialStatus.COMPLETED,
                        duration=1.0 + np.random.uniform(0, 0.5),
                        timestamp=None,
                    )
                    trials.append(trial)
                    logger.log_trial_result(trial)

                # Create optimization result
                best_trial = max(trials, key=lambda t: t.metrics["accuracy"])
                opt_result = OptimizationResult(
                    trials=trials,
                    best_config=best_trial.config,
                    best_score=best_trial.metrics["accuracy"],
                    optimization_id=f"opt_{exp_name}_{run_idx}",
                    duration=15.0,
                    convergence_info={"converged": True, "iteration": 8},
                    status="completed",
                    objectives=["accuracy", "cost", "latency"],
                    algorithm="GridSearch",
                    timestamp=None,
                )

                # Calculate weighted results
                weighted_results = opt_result.calculate_weighted_scores(
                    objective_weights={"accuracy": 0.5, "cost": 0.3, "latency": 0.2},
                    minimize_objectives=["cost", "latency"],
                )

                # Log session end
                logger.log_session_end(
                    optimization_result=opt_result, weighted_results=weighted_results
                )

                print(f"Created run: {exp_name}/{logger.run_id}")

    def test_list_experiments(self):
        """Test listing experiments."""
        experiments_df = self.analyzer.list_experiments()

        assert isinstance(experiments_df, pd.DataFrame)
        assert len(experiments_df) > 0, "Should find experiments"
        assert all(
            col in experiments_df.columns
            for col in ["experiment", "runs", "latest_run"]
        )

    def test_load_specific_run(self):
        """Test loading a specific run."""
        experiments_df = self.analyzer.list_experiments()
        assert len(experiments_df) > 0

        exp_name = experiments_df.iloc[0]["experiment"]
        latest_run = experiments_df.iloc[0]["latest_run"]

        run_data = self.analyzer.load_run(exp_name, latest_run)

        assert isinstance(run_data, dict)
        assert "session" in run_data
        assert all(key in run_data for key in ["session", "config", "objectives"])

    def test_compare_runs(self):
        """Test comparing multiple runs."""
        experiments_df = self.analyzer.list_experiments()
        assert len(experiments_df) > 0

        exp_name = experiments_df.iloc[0]["experiment"]
        run_ids = (
            experiments_df.iloc[0]["run_ids"][:2]
            if len(experiments_df.iloc[0]["run_ids"]) >= 2
            else experiments_df.iloc[0]["run_ids"]
        )

        if len(run_ids) > 0:
            comparison_df = self.analyzer.compare_runs(exp_name, run_ids)
            assert isinstance(comparison_df, pd.DataFrame)

    @pytest.mark.skipif(
        not pytest.importorskip("matplotlib", reason="matplotlib not available"),
        reason="matplotlib not available",
    )
    def test_plot_convergence(self):
        """Test plotting convergence."""
        import matplotlib

        matplotlib.use("Agg")  # Use non-interactive backend
        import matplotlib.pyplot as plt

        experiments_df = self.analyzer.list_experiments()
        assert len(experiments_df) > 0

        exp_name = experiments_df.iloc[0]["experiment"]
        latest_run = experiments_df.iloc[0]["latest_run"]

        fig = self.analyzer.plot_convergence(exp_name, latest_run, show=False)
        assert fig is not None
        plt.close(fig)

    def test_export_for_analysis(self):
        """Test exporting data for analysis."""
        experiments_df = self.analyzer.list_experiments()
        assert len(experiments_df) > 0

        exp_name = experiments_df.iloc[0]["experiment"]

        with tempfile.TemporaryDirectory() as export_dir:
            export_path = Path(export_dir)

            # Test CSV export
            csv_file = self.analyzer.export_for_analysis(
                exp_name, format="csv", output_dir=export_path
            )
            assert csv_file is not None
            assert Path(csv_file).exists()

            # Test JSON export
            json_file = self.analyzer.export_for_analysis(
                exp_name, format="json", output_dir=export_path
            )
            assert json_file is not None
            assert Path(json_file).exists()

    def test_get_run_summary(self):
        """Test getting run summary."""
        experiments_df = self.analyzer.list_experiments()
        assert len(experiments_df) > 0

        exp_name = experiments_df.iloc[0]["experiment"]
        latest_run = experiments_df.iloc[0]["latest_run"]

        summary = self.analyzer.get_run_summary(exp_name, latest_run)

        assert isinstance(summary, str)
        assert len(summary) > 0
        assert all(
            keyword in summary for keyword in ["Experiment", "Duration", "Trials"]
        )

    @pytest.mark.skipif(
        not pytest.importorskip("matplotlib", reason="matplotlib not available"),
        reason="matplotlib not available",
    )
    def test_plot_pareto_front(self):
        """Test Pareto front analysis."""
        import matplotlib.pyplot as plt

        experiments_df = self.analyzer.list_experiments()
        assert len(experiments_df) > 0

        exp_name = experiments_df.iloc[0]["experiment"]
        latest_run = experiments_df.iloc[0]["latest_run"]

        fig = self.analyzer.plot_pareto_front(
            exp_name, latest_run, objectives=["accuracy", "cost"], show=False
        )
        assert fig is not None
        plt.close(fig)


class TestAnalyzerEdgeCases:
    """Test edge cases and error handling for OptimizationAnalyzer."""

    @pytest.fixture(autouse=True)
    def setup(self, base_path):
        """Setup for edge case tests."""
        self.base_path = base_path
        self.analyzer = OptimizationAnalyzer(base_path=base_path)

    def test_non_existent_experiment(self):
        """Test handling non-existent experiment."""
        run_data = self.analyzer.load_run("non_existent_exp", "non_existent_run")
        assert run_data is None or run_data == {}

    def test_empty_base_path(self):
        """Test handling empty experiments directory."""
        with tempfile.TemporaryDirectory() as empty_dir:
            empty_analyzer = OptimizationAnalyzer(base_path=Path(empty_dir))
            empty_df = empty_analyzer.list_experiments()

            assert isinstance(empty_df, pd.DataFrame)
            assert len(empty_df) == 0

    def test_export_invalid_format(self):
        """Test export with invalid format."""
        # First create some test data
        TestOptimizationAnalyzer()._create_test_data(self.base_path)
        analyzer_with_data = OptimizationAnalyzer(base_path=self.base_path)
        experiments_df = analyzer_with_data.list_experiments()

        if len(experiments_df) > 0:
            exp_name = experiments_df.iloc[0]["experiment"]

            # Should either return None or raise an exception
            result = analyzer_with_data.export_for_analysis(
                exp_name, format="invalid_format"
            )
            assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
