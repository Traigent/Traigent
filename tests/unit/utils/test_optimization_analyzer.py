"""Unit tests for optimization_analyzer.

Tests for optimization run analysis capabilities including data loading,
comparison, visualization, and export functionality.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Performance CONC-Quality-Observability FUNC-ANALYTICS FUNC-STORAGE REQ-ANLY-011 REQ-STOR-007 SYNC-StorageLogging

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from traigent.utils.file_versioning import FileVersionManager
from traigent.utils.optimization_analyzer import OptimizationAnalyzer

try:
    import matplotlib.pyplot  # noqa: F401

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class TestOptimizationAnalyzer:
    """Tests for OptimizationAnalyzer class."""

    @pytest.fixture
    def temp_base_path(self) -> Path:
        """Create temporary base path for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def analyzer(self, temp_base_path: Path) -> OptimizationAnalyzer:
        """Create test analyzer instance."""
        return OptimizationAnalyzer(base_path=temp_base_path, file_version="2")

    @pytest.fixture
    def mock_run_structure(self, temp_base_path: Path) -> Path:
        """Create mock run directory structure."""
        exp_dir = temp_base_path / "experiments" / "test_exp" / "runs" / "run_001"
        meta_dir = exp_dir / "meta"
        trials_dir = exp_dir / "trials"
        artifacts_dir = exp_dir / "artifacts"
        metrics_dir = exp_dir / "metrics"

        meta_dir.mkdir(parents=True)
        trials_dir.mkdir(parents=True)
        artifacts_dir.mkdir(parents=True)
        metrics_dir.mkdir(parents=True)

        # Create session file
        session_data = {
            "session_id": "sess_001",
            "start_time": "2025-01-01T00:00:00Z",
            "execution_mode": "edge_analytics",
            "status": "completed",
            "duration": 120.5,
        }
        (meta_dir / "session_v2.json").write_text(json.dumps(session_data))

        # Create config file
        config_data = {"model": "gpt-4", "temperature": 0.7}
        (meta_dir / "config_v2.json").write_text(json.dumps(config_data))

        # Create objectives file
        objectives_data = {
            "objectives": [
                {"name": "accuracy", "direction": "maximize"},
                {"name": "latency", "direction": "minimize"},
            ]
        }
        (meta_dir / "objectives_v2.json").write_text(json.dumps(objectives_data))

        # Create trials file
        trials_data = [
            {
                "trial_id": "trial_001",
                "status": "success",
                "config": {"model": "gpt-3.5-turbo"},
                "metrics": {"accuracy": 0.85, "latency": 100},
                "duration": 5.0,
            },
            {
                "trial_id": "trial_002",
                "status": "success",
                "config": {"model": "gpt-4"},
                "metrics": {"accuracy": 0.92, "latency": 200},
                "duration": 8.0,
            },
        ]
        trials_file = trials_dir / "trials_v2.jsonl"
        with open(trials_file, "w") as f:
            for trial in trials_data:
                f.write(json.dumps(trial) + "\n")

        # Create best_config file
        best_config_data = {
            "config": {"model": "gpt-4", "temperature": 0.7},
            "metrics": {"accuracy": 0.92, "latency": 200},
        }
        (artifacts_dir / "best_config_v2.json").write_text(json.dumps(best_config_data))

        # Create weighted_results file
        weighted_results_data = {
            "best_weighted_score": 0.88,
            "best_weighted_config": {"model": "gpt-4"},
            "objective_weights": {"accuracy": 0.7, "latency": 0.3},
        }
        (artifacts_dir / "weighted_results_v2.json").write_text(
            json.dumps(weighted_results_data)
        )

        # Create metrics_summary file
        metrics_summary_data = {
            "total_trials": 2,
            "successful_trials": 2,
            "failed_trials": 0,
            "success_rate": 1.0,
            "duration": 120.5,
            "algorithm": "bayesian",
            "best_metrics": {"accuracy": 0.92, "latency": 200},
        }
        (metrics_dir / "metrics_summary_v2.json").write_text(
            json.dumps(metrics_summary_data)
        )

        return exp_dir

    # Initialization tests
    def test_initialization_default(self) -> None:
        """Test analyzer initialization with default parameters."""
        analyzer = OptimizationAnalyzer()
        assert analyzer.base_path is not None
        assert analyzer.file_manager.version == "2"
        assert analyzer.legacy_file_manager.use_legacy is True

    def test_initialization_custom_path(self, temp_base_path: Path) -> None:
        """Test analyzer initialization with custom base path."""
        analyzer = OptimizationAnalyzer(base_path=temp_base_path)
        assert analyzer.base_path == temp_base_path

    def test_initialization_custom_file_version(self) -> None:
        """Test analyzer initialization with custom file version."""
        analyzer = OptimizationAnalyzer(file_version="3")
        assert analyzer.file_manager.version == "3"

    # list_experiments tests
    def test_list_experiments_empty_directory(
        self, analyzer: OptimizationAnalyzer
    ) -> None:
        """Test list_experiments with no experiments."""
        df = analyzer.list_experiments()
        assert isinstance(df, pd.DataFrame)
        assert df.empty
        assert "experiment" in df.columns
        assert "runs" in df.columns

    def test_list_experiments_nonexistent_path(self) -> None:
        """Test list_experiments with nonexistent base path."""
        analyzer = OptimizationAnalyzer(base_path=Path("/nonexistent/path"))
        df = analyzer.list_experiments()
        assert df.empty

    def test_list_experiments_with_runs(
        self, analyzer: OptimizationAnalyzer, mock_run_structure: Path
    ) -> None:
        """Test list_experiments with existing runs."""
        df = analyzer.list_experiments()
        assert not df.empty
        assert len(df) == 1
        assert df.iloc[0]["experiment"] == "test_exp"
        assert df.iloc[0]["runs"] == 1
        assert df.iloc[0]["latest_run"] == "run_001"

    def test_list_experiments_with_index_file(
        self, analyzer: OptimizationAnalyzer, temp_base_path: Path
    ) -> None:
        """Test list_experiments using index file."""
        # Create index file
        index_data = {
            "experiments": {
                "exp1": {
                    "runs": [
                        {
                            "run_id": "run_001",
                            "session_id": "sess_001",
                            "timestamp": "2025-01-01T00:00:00Z",
                            "execution_mode": "edge_analytics",
                            "path": "/path/to/run",
                        }
                    ]
                }
            }
        }
        index_file = temp_base_path / "index.json"
        index_file.write_text(json.dumps(index_data))

        df = analyzer.list_experiments()
        assert not df.empty
        assert len(df) == 1
        assert df.iloc[0]["experiment"] == "exp1"

    def test_list_experiments_multiple_runs(
        self, analyzer: OptimizationAnalyzer, mock_run_structure: Path
    ) -> None:
        """Test list_experiments with multiple runs."""
        # Create second run
        run2_dir = mock_run_structure.parent / "run_002"
        run2_meta = run2_dir / "meta"
        run2_meta.mkdir(parents=True)

        session_data = {
            "session_id": "sess_002",
            "start_time": "2025-01-02T00:00:00Z",
            "execution_mode": "edge_analytics",
            "status": "completed",
        }
        (run2_meta / "session_v2.json").write_text(json.dumps(session_data))

        df = analyzer.list_experiments()
        assert len(df) == 1
        assert df.iloc[0]["runs"] == 2

    # load_run tests
    def test_load_run_success(
        self, analyzer: OptimizationAnalyzer, mock_run_structure: Path
    ) -> None:
        """Test successful load_run."""
        data = analyzer.load_run("test_exp", "run_001")
        assert data is not None
        assert "session" in data
        assert "config" in data
        assert "objectives" in data
        assert "trials" in data
        assert "best_config" in data
        assert "weighted_results" in data
        assert "metrics_summary" in data

    def test_load_run_nonexistent(self, analyzer: OptimizationAnalyzer) -> None:
        """Test load_run with nonexistent run."""
        data = analyzer.load_run("nonexistent_exp", "run_999")
        assert data == {}

    def test_load_run_missing_files(
        self, analyzer: OptimizationAnalyzer, temp_base_path: Path
    ) -> None:
        """Test load_run with missing optional files."""
        # Create minimal run structure
        exp_dir = temp_base_path / "experiments" / "test_exp" / "runs" / "run_001"
        meta_dir = exp_dir / "meta"
        meta_dir.mkdir(parents=True)

        session_data = {"session_id": "sess_001"}
        (meta_dir / "session_v2.json").write_text(json.dumps(session_data))

        data = analyzer.load_run("test_exp", "run_001")
        assert "session" in data
        assert "config" not in data
        assert "trials" not in data

    def test_load_run_legacy_files(
        self, analyzer: OptimizationAnalyzer, temp_base_path: Path
    ) -> None:
        """Test load_run with legacy file names."""
        exp_dir = temp_base_path / "experiments" / "test_exp" / "runs" / "run_001"
        meta_dir = exp_dir / "meta"
        meta_dir.mkdir(parents=True)

        session_data = {"session_id": "sess_001"}
        (meta_dir / "session.json").write_text(json.dumps(session_data))

        data = analyzer.load_run("test_exp", "run_001")
        assert "session" in data
        assert data["session"]["session_id"] == "sess_001"

    # compare_runs tests
    def test_compare_runs_success(
        self, analyzer: OptimizationAnalyzer, mock_run_structure: Path
    ) -> None:
        """Test successful compare_runs."""
        df = analyzer.compare_runs("test_exp", ["run_001"])
        assert not df.empty
        assert len(df) == 1
        assert "run_id" in df.columns
        assert "status" in df.columns
        assert "duration" in df.columns
        assert df.iloc[0]["run_id"] == "run_001"

    def test_compare_runs_no_run_ids(
        self, analyzer: OptimizationAnalyzer, mock_run_structure: Path
    ) -> None:
        """Test compare_runs without specifying run_ids."""
        df = analyzer.compare_runs("test_exp")
        assert not df.empty
        assert len(df) == 1

    def test_compare_runs_nonexistent_experiment(
        self, analyzer: OptimizationAnalyzer
    ) -> None:
        """Test compare_runs with nonexistent experiment."""
        df = analyzer.compare_runs("nonexistent_exp")
        assert df.empty

    def test_compare_runs_multiple_runs(
        self, analyzer: OptimizationAnalyzer, mock_run_structure: Path
    ) -> None:
        """Test compare_runs with multiple runs."""
        # Create second run
        run2_dir = mock_run_structure.parent / "run_002"
        meta_dir = run2_dir / "meta"
        metrics_dir = run2_dir / "metrics"
        meta_dir.mkdir(parents=True)
        metrics_dir.mkdir(parents=True)

        session_data = {
            "session_id": "sess_002",
            "start_time": "2025-01-02T00:00:00Z",
            "status": "completed",
        }
        (meta_dir / "session_v2.json").write_text(json.dumps(session_data))

        metrics_summary_data = {
            "total_trials": 3,
            "successful_trials": 3,
            "algorithm": "random",
            "best_metrics": {"accuracy": 0.95},
        }
        (metrics_dir / "metrics_summary_v2.json").write_text(
            json.dumps(metrics_summary_data)
        )

        df = analyzer.compare_runs("test_exp")
        assert len(df) == 2

    def test_compare_runs_with_weighted_results(
        self, analyzer: OptimizationAnalyzer, mock_run_structure: Path
    ) -> None:
        """Test compare_runs includes weighted results."""
        df = analyzer.compare_runs("test_exp", ["run_001"])
        assert "weighted_score" in df.columns
        assert "weighted_best_model" in df.columns
        assert df.iloc[0]["weighted_score"] == 0.88

    # plot_convergence tests
    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.show")
    def test_plot_convergence_success(
        self,
        mock_show: MagicMock,
        mock_subplots: MagicMock,
        analyzer: OptimizationAnalyzer,
        mock_run_structure: Path,
    ) -> None:
        """Test successful plot_convergence."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        fig = analyzer.plot_convergence("test_exp", "run_001", show=False)

        assert fig is not None
        mock_subplots.assert_called_once()
        assert mock_ax.plot.called
        assert mock_ax.set_xlabel.called
        assert mock_ax.set_ylabel.called

    def test_plot_convergence_no_matplotlib(
        self, analyzer: OptimizationAnalyzer, mock_run_structure: Path
    ) -> None:
        """Test plot_convergence without matplotlib."""
        with patch.dict("sys.modules", {"matplotlib.pyplot": None}):
            result = analyzer.plot_convergence("test_exp", "run_001", show=False)
            assert result is None

    def test_plot_convergence_no_trials(
        self, analyzer: OptimizationAnalyzer, temp_base_path: Path
    ) -> None:
        """Test plot_convergence with no trial data."""
        # Create run with no trials
        exp_dir = temp_base_path / "experiments" / "test_exp" / "runs" / "run_001"
        meta_dir = exp_dir / "meta"
        meta_dir.mkdir(parents=True)

        session_data = {"session_id": "sess_001"}
        (meta_dir / "session_v2.json").write_text(json.dumps(session_data))

        result = analyzer.plot_convergence("test_exp", "run_001", show=False)
        assert result is None

    def test_plot_convergence_no_objective_values(
        self, analyzer: OptimizationAnalyzer, temp_base_path: Path
    ) -> None:
        """Test plot_convergence with trials but no objective values."""
        exp_dir = temp_base_path / "experiments" / "test_exp" / "runs" / "run_001"
        meta_dir = exp_dir / "meta"
        trials_dir = exp_dir / "trials"
        meta_dir.mkdir(parents=True)
        trials_dir.mkdir(parents=True)

        session_data = {"session_id": "sess_001"}
        (meta_dir / "session_v2.json").write_text(json.dumps(session_data))

        # Trials with no metrics
        trials_data = [{"trial_id": "trial_001", "config": {}, "metrics": {}}]
        trials_file = trials_dir / "trials_v2.jsonl"
        with open(trials_file, "w") as f:
            for trial in trials_data:
                f.write(json.dumps(trial) + "\n")

        result = analyzer.plot_convergence("test_exp", "run_001", show=False)
        assert result is None

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.show")
    def test_plot_convergence_with_explicit_objective(
        self,
        mock_plt: MagicMock,
        analyzer: OptimizationAnalyzer,
        mock_run_structure: Path,
    ) -> None:
        """Test plot_convergence with explicit objective specified."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        fig = analyzer.plot_convergence(
            "test_exp", "run_001", objective="accuracy", show=False
        )

        assert fig is not None

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.show")
    def test_plot_convergence_objectives_as_list(
        self, mock_plt: MagicMock, analyzer: OptimizationAnalyzer, temp_base_path: Path
    ) -> None:
        """Test plot_convergence with objectives as list of strings."""
        exp_dir = temp_base_path / "experiments" / "test_exp" / "runs" / "run_001"
        meta_dir = exp_dir / "meta"
        trials_dir = exp_dir / "trials"
        meta_dir.mkdir(parents=True)
        trials_dir.mkdir(parents=True)

        session_data = {"session_id": "sess_001"}
        (meta_dir / "session_v2.json").write_text(json.dumps(session_data))

        objectives_data = ["accuracy", "latency"]
        (meta_dir / "objectives_v2.json").write_text(json.dumps(objectives_data))

        trials_data = [{"trial_id": "trial_001", "metrics": {"accuracy": 0.85}}]
        trials_file = trials_dir / "trials_v2.jsonl"
        with open(trials_file, "w") as f:
            for trial in trials_data:
                f.write(json.dumps(trial) + "\n")

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        fig = analyzer.plot_convergence("test_exp", "run_001", show=False)
        assert fig is not None

    # plot_pareto_front tests
    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.show")
    def test_plot_pareto_front_success(
        self,
        mock_show: MagicMock,
        mock_subplots: MagicMock,
        analyzer: OptimizationAnalyzer,
        mock_run_structure: Path,
    ) -> None:
        """Test successful plot_pareto_front."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        fig = analyzer.plot_pareto_front("test_exp", "run_001", show=False)

        assert fig is not None
        assert mock_ax.scatter.called
        assert mock_ax.plot.called

    def test_plot_pareto_front_no_matplotlib(
        self, analyzer: OptimizationAnalyzer, mock_run_structure: Path
    ) -> None:
        """Test plot_pareto_front without matplotlib."""
        with patch.dict("sys.modules", {"matplotlib.pyplot": None}):
            result = analyzer.plot_pareto_front("test_exp", "run_001", show=False)
            assert result is None

    def test_plot_pareto_front_no_trials(
        self, analyzer: OptimizationAnalyzer, temp_base_path: Path
    ) -> None:
        """Test plot_pareto_front with no trial data."""
        exp_dir = temp_base_path / "experiments" / "test_exp" / "runs" / "run_001"
        meta_dir = exp_dir / "meta"
        meta_dir.mkdir(parents=True)

        session_data = {"session_id": "sess_001"}
        (meta_dir / "session_v2.json").write_text(json.dumps(session_data))

        result = analyzer.plot_pareto_front("test_exp", "run_001", show=False)
        assert result is None

    def test_plot_pareto_front_insufficient_objectives(
        self, analyzer: OptimizationAnalyzer, temp_base_path: Path
    ) -> None:
        """Test plot_pareto_front with fewer than 2 objectives."""
        exp_dir = temp_base_path / "experiments" / "test_exp" / "runs" / "run_001"
        meta_dir = exp_dir / "meta"
        trials_dir = exp_dir / "trials"
        meta_dir.mkdir(parents=True)
        trials_dir.mkdir(parents=True)

        session_data = {"session_id": "sess_001"}
        (meta_dir / "session_v2.json").write_text(json.dumps(session_data))

        objectives_data = {"objectives": [{"name": "accuracy"}]}
        (meta_dir / "objectives_v2.json").write_text(json.dumps(objectives_data))

        trials_data = [{"trial_id": "trial_001", "metrics": {"accuracy": 0.85}}]
        trials_file = trials_dir / "trials_v2.jsonl"
        with open(trials_file, "w") as f:
            for trial in trials_data:
                f.write(json.dumps(trial) + "\n")

        result = analyzer.plot_pareto_front("test_exp", "run_001", show=False)
        assert result is None

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.show")
    def test_plot_pareto_front_with_explicit_objectives(
        self,
        mock_plt: MagicMock,
        analyzer: OptimizationAnalyzer,
        mock_run_structure: Path,
    ) -> None:
        """Test plot_pareto_front with explicit objectives."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        fig = analyzer.plot_pareto_front(
            "test_exp", "run_001", objectives=["accuracy", "latency"], show=False
        )

        assert fig is not None

    def test_plot_pareto_front_no_matching_metrics(
        self, analyzer: OptimizationAnalyzer, temp_base_path: Path
    ) -> None:
        """Test plot_pareto_front with trials but no matching metrics."""
        exp_dir = temp_base_path / "experiments" / "test_exp" / "runs" / "run_001"
        meta_dir = exp_dir / "meta"
        trials_dir = exp_dir / "trials"
        meta_dir.mkdir(parents=True)
        trials_dir.mkdir(parents=True)

        session_data = {"session_id": "sess_001"}
        (meta_dir / "session_v2.json").write_text(json.dumps(session_data))

        objectives_data = {"objectives": [{"name": "accuracy"}, {"name": "latency"}]}
        (meta_dir / "objectives_v2.json").write_text(json.dumps(objectives_data))

        trials_data = [{"trial_id": "trial_001", "metrics": {"other_metric": 0.85}}]
        trials_file = trials_dir / "trials_v2.jsonl"
        with open(trials_file, "w") as f:
            for trial in trials_data:
                f.write(json.dumps(trial) + "\n")

        result = analyzer.plot_pareto_front("test_exp", "run_001", show=False)
        assert result is None

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.show")
    def test_plot_pareto_front_pareto_calculation(
        self, mock_plt: MagicMock, analyzer: OptimizationAnalyzer, temp_base_path: Path
    ) -> None:
        """Test plot_pareto_front Pareto front calculation."""
        exp_dir = temp_base_path / "experiments" / "test_exp" / "runs" / "run_001"
        meta_dir = exp_dir / "meta"
        trials_dir = exp_dir / "trials"
        meta_dir.mkdir(parents=True)
        trials_dir.mkdir(parents=True)

        session_data = {"session_id": "sess_001"}
        (meta_dir / "session_v2.json").write_text(json.dumps(session_data))

        objectives_data = {"objectives": [{"name": "accuracy"}, {"name": "latency"}]}
        (meta_dir / "objectives_v2.json").write_text(json.dumps(objectives_data))

        # Create trials where some dominate others
        trials_data = [
            {"trial_id": "trial_001", "metrics": {"accuracy": 0.8, "latency": 100}},
            {"trial_id": "trial_002", "metrics": {"accuracy": 0.9, "latency": 90}},
            {"trial_id": "trial_003", "metrics": {"accuracy": 0.85, "latency": 95}},
        ]
        trials_file = trials_dir / "trials_v2.jsonl"
        with open(trials_file, "w") as f:
            for trial in trials_data:
                f.write(json.dumps(trial) + "\n")

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        fig = analyzer.plot_pareto_front("test_exp", "run_001", show=False)
        assert fig is not None

    # export_for_analysis tests
    def test_export_for_analysis_csv(
        self, analyzer: OptimizationAnalyzer, mock_run_structure: Path
    ) -> None:
        """Test export_for_analysis with CSV format."""
        output_file = analyzer.export_for_analysis("test_exp", format="csv")
        assert output_file is not None
        assert Path(output_file).exists()
        assert Path(output_file).suffix == ".csv"

        # Verify CSV content
        df = pd.read_csv(output_file)
        assert not df.empty
        assert "run_id" in df.columns
        assert "trial_id" in df.columns

        # Cleanup
        Path(output_file).unlink()

    def test_export_for_analysis_json(
        self, analyzer: OptimizationAnalyzer, mock_run_structure: Path
    ) -> None:
        """Test export_for_analysis with JSON format."""
        output_file = analyzer.export_for_analysis("test_exp", format="json")
        assert output_file is not None
        assert Path(output_file).exists()
        assert Path(output_file).suffix == ".json"

        # Verify JSON content
        with open(output_file) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) > 0

        # Cleanup
        Path(output_file).unlink()

    @patch("traigent.utils.optimization_analyzer.pd.DataFrame.to_excel")
    def test_export_for_analysis_excel(
        self,
        mock_to_excel: MagicMock,
        analyzer: OptimizationAnalyzer,
        mock_run_structure: Path,
    ) -> None:
        """Test export_for_analysis with Excel format."""
        output_file = analyzer.export_for_analysis("test_exp", format="excel")
        assert output_file is not None
        mock_to_excel.assert_called_once()

    def test_export_for_analysis_excel_no_openpyxl(
        self, analyzer: OptimizationAnalyzer, mock_run_structure: Path
    ) -> None:
        """Test export_for_analysis Excel without openpyxl."""
        with patch(
            "traigent.utils.optimization_analyzer.pd.DataFrame.to_excel",
            side_effect=ImportError,
        ):
            output_file = analyzer.export_for_analysis("test_exp", format="excel")
            assert output_file is None

    def test_export_for_analysis_unsupported_format(
        self, analyzer: OptimizationAnalyzer, mock_run_structure: Path
    ) -> None:
        """Test export_for_analysis with unsupported format."""
        output_file = analyzer.export_for_analysis("test_exp", format="xml")
        assert output_file is None

    def test_export_for_analysis_nonexistent_experiment(
        self, analyzer: OptimizationAnalyzer
    ) -> None:
        """Test export_for_analysis with nonexistent experiment."""
        output_file = analyzer.export_for_analysis("nonexistent_exp")
        assert output_file is None

    def test_export_for_analysis_custom_output_dir(
        self,
        analyzer: OptimizationAnalyzer,
        mock_run_structure: Path,
        temp_base_path: Path,
    ) -> None:
        """Test export_for_analysis with custom output directory."""
        output_dir = temp_base_path / "exports"
        output_dir.mkdir()

        output_file = analyzer.export_for_analysis(
            "test_exp", format="csv", output_dir=output_dir
        )
        assert output_file is not None
        assert Path(output_file).parent == output_dir

        # Cleanup
        Path(output_file).unlink()

    def test_export_for_analysis_no_trials(
        self, analyzer: OptimizationAnalyzer, temp_base_path: Path
    ) -> None:
        """Test export_for_analysis with no trial data."""
        # Create run with no trials
        exp_dir = temp_base_path / "experiments" / "test_exp" / "runs" / "run_001"
        meta_dir = exp_dir / "meta"
        meta_dir.mkdir(parents=True)

        session_data = {"session_id": "sess_001", "start_time": "2025-01-01T00:00:00Z"}
        (meta_dir / "session_v2.json").write_text(json.dumps(session_data))

        output_file = analyzer.export_for_analysis("test_exp")
        assert output_file is None

    # get_run_summary tests
    def test_get_run_summary_success(
        self, analyzer: OptimizationAnalyzer, mock_run_structure: Path
    ) -> None:
        """Test successful get_run_summary."""
        summary = analyzer.get_run_summary("test_exp", "run_001")
        assert isinstance(summary, str)
        assert "test_exp" in summary
        assert "run_001" in summary
        assert "sess_001" in summary
        assert "completed" in summary

    def test_get_run_summary_nonexistent_run(
        self, analyzer: OptimizationAnalyzer
    ) -> None:
        """Test get_run_summary with nonexistent run."""
        summary = analyzer.get_run_summary("nonexistent_exp", "run_999")
        assert "No data found" in summary

    def test_get_run_summary_includes_all_sections(
        self, analyzer: OptimizationAnalyzer, mock_run_structure: Path
    ) -> None:
        """Test get_run_summary includes all expected sections."""
        summary = analyzer.get_run_summary("test_exp", "run_001")
        assert "Session ID" in summary
        assert "Status" in summary
        assert "Duration" in summary
        assert "Trials:" in summary
        assert "Algorithm:" in summary
        assert "Best Metrics:" in summary
        assert "Best Configuration:" in summary
        assert "Weighted Scoring Results:" in summary

    def test_get_run_summary_minimal_data(
        self, analyzer: OptimizationAnalyzer, temp_base_path: Path
    ) -> None:
        """Test get_run_summary with minimal run data."""
        exp_dir = temp_base_path / "experiments" / "test_exp" / "runs" / "run_001"
        meta_dir = exp_dir / "meta"
        meta_dir.mkdir(parents=True)

        session_data = {"session_id": "sess_001"}
        (meta_dir / "session_v2.json").write_text(json.dumps(session_data))

        summary = analyzer.get_run_summary("test_exp", "run_001")
        assert "sess_001" in summary
        assert isinstance(summary, str)

    # Static method tests
    def test_candidate_files(self) -> None:
        """Test _candidate_files static method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_path = Path(tmpdir)
            file_manager = FileVersionManager(version="2")
            legacy_manager = FileVersionManager(use_legacy=True)

            candidates = OptimizationAnalyzer._candidate_files(
                run_path, "meta", "session", file_manager, legacy_manager
            )

            assert len(candidates) == 2
            assert any("session_v2.json" in str(c) for c in candidates)
            assert any("session.json" in str(c) for c in candidates)

    def test_resolve_existing_file(self) -> None:
        """Test _resolve_existing_file static method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_path = Path(tmpdir)
            meta_dir = run_path / "meta"
            meta_dir.mkdir()

            # Create versioned file
            session_file = meta_dir / "session_v2.json"
            session_file.write_text("{}")

            file_manager = FileVersionManager(version="2")
            legacy_manager = FileVersionManager(use_legacy=True)

            resolved = OptimizationAnalyzer._resolve_existing_file(
                run_path, "meta", "session", file_manager, legacy_manager
            )

            assert resolved is not None
            assert resolved.exists()
            assert "session_v2.json" in str(resolved)

    def test_resolve_existing_file_not_found(self) -> None:
        """Test _resolve_existing_file with no matching files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_path = Path(tmpdir)
            file_manager = FileVersionManager(version="2")
            legacy_manager = FileVersionManager(use_legacy=True)

            resolved = OptimizationAnalyzer._resolve_existing_file(
                run_path, "meta", "session", file_manager, legacy_manager
            )

            assert resolved is None

    def test_load_json_file(self) -> None:
        """Test _load_json_file static method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_path = Path(tmpdir)
            meta_dir = run_path / "meta"
            meta_dir.mkdir()

            data = {"test": "value"}
            session_file = meta_dir / "session_v2.json"
            session_file.write_text(json.dumps(data))

            file_manager = FileVersionManager(version="2")
            legacy_manager = FileVersionManager(use_legacy=True)

            loaded = OptimizationAnalyzer._load_json_file(
                run_path, "meta", "session", file_manager, legacy_manager
            )

            assert loaded == data

    def test_load_json_file_not_found(self) -> None:
        """Test _load_json_file with nonexistent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_path = Path(tmpdir)
            file_manager = FileVersionManager(version="2")
            legacy_manager = FileVersionManager(use_legacy=True)

            loaded = OptimizationAnalyzer._load_json_file(
                run_path, "meta", "session", file_manager, legacy_manager
            )

            assert loaded is None

    def test_load_jsonl_file(self) -> None:
        """Test _load_jsonl_file static method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_path = Path(tmpdir)
            trials_dir = run_path / "trials"
            trials_dir.mkdir()

            trials_file = trials_dir / "trials_v2.jsonl"
            with open(trials_file, "w") as f:
                f.write(json.dumps({"trial": 1}) + "\n")
                f.write(json.dumps({"trial": 2}) + "\n")

            file_manager = FileVersionManager(version="2")
            legacy_manager = FileVersionManager(use_legacy=True)

            loaded = OptimizationAnalyzer._load_jsonl_file(
                run_path, "trials", "trials_stream", file_manager, legacy_manager
            )

            assert len(loaded) == 2
            assert loaded[0] == {"trial": 1}
            assert loaded[1] == {"trial": 2}

    def test_load_jsonl_file_empty_lines(self) -> None:
        """Test _load_jsonl_file with empty lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_path = Path(tmpdir)
            trials_dir = run_path / "trials"
            trials_dir.mkdir()

            trials_file = trials_dir / "trials_v2.jsonl"
            with open(trials_file, "w") as f:
                f.write(json.dumps({"trial": 1}) + "\n")
                f.write("\n")
                f.write(json.dumps({"trial": 2}) + "\n")

            file_manager = FileVersionManager(version="2")
            legacy_manager = FileVersionManager(use_legacy=True)

            loaded = OptimizationAnalyzer._load_jsonl_file(
                run_path, "trials", "trials_stream", file_manager, legacy_manager
            )

            assert len(loaded) == 2

    def test_load_jsonl_file_not_found(self) -> None:
        """Test _load_jsonl_file with nonexistent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_path = Path(tmpdir)
            file_manager = FileVersionManager(version="2")
            legacy_manager = FileVersionManager(use_legacy=True)

            loaded = OptimizationAnalyzer._load_jsonl_file(
                run_path, "trials", "trials_stream", file_manager, legacy_manager
            )

            assert loaded == []

    def test_build_experiment_index_no_experiments_dir(self) -> None:
        """Test _build_experiment_index with no experiments directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            file_manager = FileVersionManager(version="2")
            legacy_manager = FileVersionManager(use_legacy=True)

            df = OptimizationAnalyzer._build_experiment_index(
                base_path, file_manager, legacy_manager
            )

            assert df.empty

    def test_get_all_runs_no_index(self) -> None:
        """Test _get_all_runs without index file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            file_manager = FileVersionManager(version="2")
            legacy_manager = FileVersionManager(use_legacy=True)

            df = OptimizationAnalyzer._get_all_runs(
                base_path, file_manager, legacy_manager
            )

            assert df.empty

    def test_build_experiment_index_skips_non_directories(self) -> None:
        """Test _build_experiment_index skips non-directory files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            experiments_dir = base_path / "experiments"
            experiments_dir.mkdir()

            # Create a file instead of directory
            (experiments_dir / "not_a_dir.txt").write_text("test")

            file_manager = FileVersionManager(version="2")
            legacy_manager = FileVersionManager(use_legacy=True)

            df = OptimizationAnalyzer._build_experiment_index(
                base_path, file_manager, legacy_manager
            )

            assert df.empty

    def test_list_experiments_sorts_by_latest_timestamp(
        self, analyzer: OptimizationAnalyzer, mock_run_structure: Path
    ) -> None:
        """Test list_experiments sorts by latest timestamp."""
        # Create second experiment with earlier timestamp
        exp2_dir = analyzer.base_path / "experiments" / "test_exp2" / "runs" / "run_001"
        meta_dir = exp2_dir / "meta"
        meta_dir.mkdir(parents=True)

        session_data = {
            "session_id": "sess_002",
            "start_time": "2024-12-31T00:00:00Z",
            "execution_mode": "edge_analytics",
            "status": "completed",
        }
        (meta_dir / "session_v2.json").write_text(json.dumps(session_data))

        df = analyzer.list_experiments()
        assert len(df) == 2
        assert df.iloc[0]["experiment"] == "test_exp"  # Later timestamp first

    def test_compare_runs_includes_execution_modes(
        self, analyzer: OptimizationAnalyzer, mock_run_structure: Path
    ) -> None:
        """Test compare_runs includes execution modes."""
        df = analyzer.compare_runs("test_exp", ["run_001"])
        assert "algorithm" in df.columns
        assert df.iloc[0]["algorithm"] == "bayesian"

    def test_export_for_analysis_includes_config_and_metrics(
        self, analyzer: OptimizationAnalyzer, mock_run_structure: Path
    ) -> None:
        """Test export_for_analysis includes config and metric columns."""
        output_file = analyzer.export_for_analysis("test_exp", format="csv")
        df = pd.read_csv(output_file)

        # Check for config columns
        assert any(col.startswith("config_") for col in df.columns)
        # Check for metric columns
        assert any(col.startswith("metric_") for col in df.columns)

        # Cleanup
        Path(output_file).unlink()

    def test_candidate_files_handles_value_error(self) -> None:
        """Test _candidate_files handles ValueError from get_filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_path = Path(tmpdir)
            file_manager = FileVersionManager(version="2")
            legacy_manager = FileVersionManager(use_legacy=True)

            # Use invalid file type to trigger ValueError
            candidates = OptimizationAnalyzer._candidate_files(
                run_path, "meta", "invalid_type", file_manager, legacy_manager
            )

            assert len(candidates) == 0

    def test_candidate_files_deduplicates(self) -> None:
        """Test _candidate_files removes duplicates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_path = Path(tmpdir)
            # Create two managers with same version
            file_manager = FileVersionManager(version="2")
            legacy_manager = FileVersionManager(version="2")

            candidates = OptimizationAnalyzer._candidate_files(
                run_path, "meta", "session", file_manager, legacy_manager
            )

            # Should deduplicate identical paths
            assert len(candidates) == len({str(c) for c in candidates})
